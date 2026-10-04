//! Constraint evaluator for the RV32IM AIR over a 31-bit field.
//!
//! The same function runs prover-side on every LDE row (base field) and
//! verifier-side at the out-of-domain point (extension field); it must be a
//! pure polynomial function of the row values.
//!
//! Every machine word is a pair of 16-bit limbs; wrap-around arithmetic is
//! expressed limb by limb with explicit boolean carries/borrows, and every
//! limb is range-checked (see `lookups::r16_checked`), which is what makes
//! each limb equation exact over the integers.

use crate::layout::*;
use crate::lookups::{row_terms_with, Enc, BATCH};
use zkb_field::{ExtLike, FieldLike, Fp4};
use zkb_stark::air::Ext;
use zkb_stark::{ConstraintSink, RowAccess, RowView};

const B16: u64 = 1 << 16;

/// A 32-bit value as (lo, hi) limb expressions.
#[derive(Clone, Copy)]
pub struct Lw<F>(pub F, pub F);

impl<F: FieldLike> Lw<F> {
    #[inline(always)]
    fn add(self, o: Lw<F>) -> Lw<F> {
        Lw(self.0 + o.0, self.1 + o.1)
    }
    #[inline(always)]
    fn sub(self, o: Lw<F>) -> Lw<F> {
        Lw(self.0 - o.0, self.1 - o.1)
    }
    #[inline(always)]
    fn scale(self, s: F) -> Lw<F> {
        Lw(self.0 * s, self.1 * s)
    }
}

/// Decoded instruction fields as linear expressions in the bit columns.
pub struct Decoded<F> {
    pub instr: Lw<F>,
    pub opcode: F,
    pub rd: F,
    pub funct3: F,
    pub rs1: F,
    pub rs2: F,
    pub funct7: F,
    pub b25: F,
    pub b30: F,
    pub imm_i: Lw<F>,
    pub imm_s: Lw<F>,
    pub imm_b: Lw<F>,
    pub imm_u: Lw<F>,
    pub imm_j: Lw<F>,
}

pub fn decode<F: FieldLike, M: RowAccess<F>>(l: &Layout, m: M) -> Decoded<F> {
    let c = |v: u64| F::from_u64(v);
    let b = |i: usize| m.at(l.bits.start + i);
    // Σ_{i=lo..=hi} b_i · 2^(i + shift)
    let sum = |lo: usize, hi: usize, shift: i32| -> F {
        let mut acc = F::ZERO;
        for i in lo..=hi {
            let e = i as i32 + shift;
            acc += b(i) * c(1u64 << e);
        }
        acc
    };
    let instr = Lw(sum(0, 15, 0), sum(16, 31, -16));
    let opcode = sum(0, 6, 0);
    let rd = sum(7, 11, -7);
    let funct3 = sum(12, 14, -12);
    let rs1 = sum(15, 19, -15);
    let rs2 = sum(20, 24, -20);
    let funct7 = sum(25, 31, -25);
    let b25 = b(25);
    let b30 = b(30);
    let sign = b(31);
    let ones16 = c(B16 - 1);
    // sign-extended immediates: the high limb is all sign bits
    let imm_i = Lw(sum(20, 30, -20) + sign * c(B16 - (1 << 11)), sign * ones16);
    let imm_s = Lw(sum(7, 11, -7) + sum(25, 30, -20) + sign * c(B16 - (1 << 11)), sign * ones16);
    let imm_b = Lw(b(7) * c(1 << 11) + sum(8, 11, -7) + sum(25, 30, -20) + sign * c(B16 - (1 << 12)), sign * ones16);
    let imm_u = Lw(sum(12, 15, 0), sum(16, 31, -16));
    let imm_j = Lw(sum(21, 30, -20) + b(20) * c(1 << 11) + sum(12, 15, 0), sum(16, 19, -16) + sign * c(B16 - (1 << 4)));
    Decoded { instr, opcode, rd, funct3, rs1, rs2, funct7, b25, b30, imm_i, imm_s, imm_b, imm_u, imm_j }
}

/// Memory-table constraints between a previous slot `p` and current slot `s`.
///
/// The table is sorted in three stages: access groups (an init entry followed
/// by the real accesses of that address, ordered by clock), then the closing
/// entries (strictly increasing addresses), then inert padding. Within a
/// group `same = 1`; a new address or a new stage has `same = 0` and the
/// ordering `diff ≥ 0` is checked except at the two stage boundaries.
fn slot_pair<F: FieldLike, M: RowAccess<F>>(m_p: M, p: &SlotCols, m_s: M, s: &SlotCols, close_lo: F::Ext, close_hi: F::Ext, sink: &mut ConstraintSink<F>) {
    let one = F::ONE;
    let same = m_s.at(s.same);
    let (close_p, close_s) = (m_p.at(p.close), m_s.at(s.close));
    let pad_p = one - m_p.at(p.real) - m_p.at(p.init) - close_p;
    let pad_s = one - m_s.at(s.real) - m_s.at(s.init) - close_s;
    // stage ∈ {0: access, 1: closing, 2: padding} never decreases
    let d = (close_s + pad_s * F::from_u64(2)) - (close_p + pad_p * F::from_u64(2));
    sink.transition_hi(|| d * (d - one) * (d - F::from_u64(2)));
    sink.transition(|| same * (m_s.at(s.addr) - m_p.at(p.addr)));
    let first_close = close_s * (one - close_p);
    let first_pad = pad_s * (one - pad_p);
    let order = same * (m_s.at(s.clk) - m_p.at(p.clk) - one) + (one - same) * (m_s.at(s.addr) - m_p.at(p.addr) - one);
    sink.transition_hi(|| m_s.at(s.diff) - (one - first_close - first_pad) * order);
    // reads carry the previous value of the address
    sink.transition_hi(|| same * (one - m_s.at(s.w)) * (m_s.at(s.val.lo) - m_p.at(p.val.lo)));
    sink.transition_hi(|| same * (one - m_s.at(s.w)) * (m_s.at(s.val.hi) - m_p.at(p.val.hi)));
    // the closing range of the segment is public: first and last closing entry
    sink.transition_hi(|| Ext(first_close.to_ext() * (m_s.at(s.addr).to_ext() - close_lo)));
    sink.transition_hi(|| Ext((close_p * (one - close_s)).to_ext() * (m_p.at(p.addr).to_ext() - close_hi)));
}

pub fn eval<F: FieldLike, M: RowAccess<F>, A: RowAccess<F::Ext>>(l: &Layout, row: &RowView<F, M, A>, ch: &[Fp4], public: &[Fp4], aux_public: &[Fp4], sink: &mut ConstraintSink<F>) {
    let ext = |v: Fp4| F::Ext::from(v);
    let m = row.main;
    let mn = row.main_next;
    let one = F::ONE;
    let c = |v: u64| F::from_u64(v);
    let w = |wd: Word| Lw(m.at(wd.lo), m.at(wd.hi));
    let d = decode(l, m);
    let fl = |k: usize| m.at(l.flags.start + k);
    let f3 = |k: usize| m.at(l.f3.start + k);
    let active = m.at(l.active);
    let (f_lui, f_auipc, f_jal, f_jalr, f_br, f_load, f_store, f_alui, f_alur, f_ecall) =
        (fl(F_LUI), fl(F_AUIPC), fl(F_JAL), fl(F_JALR), fl(F_BR), fl(F_LOAD), fl(F_STORE), fl(F_ALUI), fl(F_ALUR), fl(F_ECALL));
    let f_alu = f_alui + f_alur;
    let f_mem = f_load + f_store;
    let is_m = m.at(l.is_m);
    let boolean = |x: F| x * (x - one);
    let (h0, h1, h2, h3, h4) = (m.at(l.h0), m.at(l.h1), m.at(l.h2), m.at(l.h3), m.at(l.h4));
    let is_halt = f_ecall * h0;
    let suspend = m.at(l.suspend);
    let cv = |k: usize| w(l.cpu_val[k]);
    let (sa, sb) = (m.at(l.a_sign), m.at(l.b_sign));
    let a = Lw(m.at(l.a_lo16), m.at(l.a_hi15) + sa * c(1 << 15));
    let b = Lw(m.at(l.b_lo16), m.at(l.b_hi15) + sb * c(1 << 15));
    let b_reg = cv(2);
    let m_word = cv(3);
    let pc = m.at(l.pc);
    let (pcl14, pc_hi) = (m.at(l.pcl14), m.at(l.pc_hi));
    let pc_lo = pcl14 * c(4);

    // ---- clock / activity / segment boundary
    sink.transition(|| mn.at(l.clk) - m.at(l.clk) - one);
    sink.transition(|| boolean(active));
    sink.transition(|| boolean(suspend));
    sink.transition(|| suspend * (one - active));
    sink.transition_hi(|| suspend * is_halt);
    sink.transition(|| mn.at(l.active) - active + is_halt + suspend);
    // at the active->inactive transition row: next pc / jcount and the halt flag match the public outputs
    let ending = active - mn.at(l.active);
    sink.transition(|| Ext((mn.at(l.pc).to_ext() - ext(public[PUB_PC_OUT])).mul_base(ending)));
    sink.transition(|| Ext((mn.at(l.jcount).to_ext() - ext(public[PUB_JCOUNT_OUT])).mul_base(ending)));
    sink.transition_hi(|| Ext((is_halt.to_ext() - ext(public[PUB_HALTED])).mul_base(ending)));
    // pc limbs: pc = 4·pcl14 + pc_hi·2^16 (< 2^31)
    sink.transition(|| pc - pc_lo - pc_hi * c(B16));

    // ---- instruction bits and decode flags
    for i in 0..32 {
        sink.transition(|| boolean(m.at(l.bits.start + i)));
    }
    let mut flag_sum = F::ZERO;
    for k in 0..11 {
        sink.transition(|| boolean(fl(k)));
        sink.transition(|| fl(k) * (d.opcode - c(OPCODES[k])));
        flag_sum += fl(k);
    }
    sink.transition(|| flag_sum - active);
    let mut f3_sum = F::ZERO;
    let mut f3_val = F::ZERO;
    for k in 0..8 {
        sink.transition(|| boolean(f3(k)));
        f3_sum += f3(k);
        f3_val += f3(k) * c(k as u64);
    }
    sink.transition(|| f3_sum - active);
    sink.transition(|| f3_val - d.funct3);
    // encoding validity (mirrors the VM's strict decoder)
    sink.transition(|| f_jalr * d.funct3);
    sink.transition(|| f_br * (f3(2) + f3(3)));
    sink.transition(|| f_load * (f3(3) + f3(6) + f3(7)));
    sink.transition(|| f_store * (f3(3) + f3(4) + f3(5) + f3(6) + f3(7)));
    sink.transition(|| f_alur * (d.funct7 - d.b30 * c(32) - d.b25));
    sink.transition_hi(|| f_alur * d.b25 * d.b30);
    sink.transition_hi(|| f_alur * d.b30 * (one - f3(0) - f3(5)));
    sink.transition_hi(|| f_alui * f3(1) * d.funct7);
    sink.transition_hi(|| f_alui * f3(5) * (d.funct7 - d.b30 * c(32)));
    sink.transition(|| f_ecall * (d.instr.0 - c(0x73)));
    sink.transition(|| f_ecall * d.instr.1);
    sink.transition(|| is_m - f_alur * d.b25);

    // ---- operands: a = rs1 value, op_b = rs2 value or immediate
    sink.transition(|| boolean(sa));
    sink.transition(|| boolean(sb));
    sink.transition(|| cv(1).0 - a.0);
    sink.transition(|| cv(1).1 - a.1);
    let imm_sel = d.imm_i.scale(f_alui + f_load + f_jalr).add(d.imm_s.scale(f_store));
    let op_b = b_reg.scale(f_alur + f_br).add(imm_sel);
    sink.transition(|| b.0 - op_b.0);
    sink.transition(|| b.1 - op_b.1);

    // ---- adder / comparator: a ± b with explicit carries
    let use_sub = m.at(l.use_sub);
    sink.transition_hi(|| use_sub - (f_alur * f3(0) * d.b30 + f_alu * (f3(2) + f3(3)) + f_br));
    let in_lo = b.0 + use_sub * (c(B16 - 1) - b.0 * c(2));
    let in_hi = b.1 + use_sub * (c(B16 - 1) - b.1 * c(2));
    let add = w(l.add);
    let (add_c0, add_carry) = (m.at(l.add_c0), m.at(l.add_carry));
    sink.transition(|| boolean(add_c0));
    sink.transition(|| boolean(add_carry));
    sink.transition(|| a.0 + in_lo + use_sub - add.0 - add_c0 * c(B16));
    sink.transition(|| a.1 + in_hi + add_c0 - add.1 - add_carry * c(B16));
    let ltu = one - add_carry;
    sink.transition_hi(|| m.at(l.lt_s) - (sa * (one - sb) + (one - sa - sb + sa * sb * c(2)) * ltu));
    // a == b, limb by limb
    let (eq_lo, eq_hi) = (m.at(l.eq_lo), m.at(l.eq_hi));
    sink.transition(|| eq_lo * (a.0 - b.0));
    sink.transition(|| (a.0 - b.0) * m.at(l.inv_lo) + eq_lo - one);
    sink.transition(|| eq_hi * (a.1 - b.1));
    sink.transition(|| (a.1 - b.1) * m.at(l.inv_hi) + eq_hi - one);
    let eq = eq_lo * eq_hi;
    let lt_s = m.at(l.lt_s);
    sink.transition_hi(|| m.at(l.taken) - (f3(0) * eq + f3(1) * (one - eq) + f3(4) * lt_s + f3(5) * (one - lt_s) + f3(6) * ltu + f3(7) * (one - ltu)));

    // ---- pc arithmetic and control flow
    let imm_pc = d.imm_j.scale(f_jal).add(d.imm_b.scale(f_br)).add(d.imm_u.scale(f_auipc));
    let pt = w(l.pt);
    let (pt_c0, pt_c1) = (m.at(l.pt_c0), m.at(l.pt_c1));
    sink.transition(|| boolean(pt_c0));
    sink.transition(|| boolean(pt_c1));
    sink.transition(|| pc_lo + imm_pc.0 - pt.0 - pt_c0 * c(B16));
    sink.transition(|| pc_hi + imm_pc.1 + pt_c0 - pt.1 - pt_c1 * c(B16));
    let (pc4_lo, pc4_c) = (m.at(l.pc4_lo), m.at(l.pc4_c));
    sink.transition(|| boolean(pc4_c));
    sink.transition(|| pc_lo + c(4) - pc4_lo - pc4_c * c(B16));
    let pc4 = Lw(pc4_lo, pc_hi + pc4_c);
    let taken = m.at(l.taken);
    let tgt_lo = pc4.0 + f_jal * (pt.0 - pc4.0) + f_br * taken * (pt.0 - pc4.0) + f_jalr * (add.0 - pc4.0);
    let tgt_hi = pc4.1 + f_jal * (pt.1 - pc4.1) + f_br * taken * (pt.1 - pc4.1) + f_jalr * (add.1 - pc4.1);
    // the next pc is the target, limb for limb (so it is < 2^31 and 4-aligned)
    sink.transition_hi(|| tgt_lo - mn.at(l.pcl14) * c(4));
    sink.transition_hi(|| tgt_hi - mn.at(l.pc_hi));

    // ---- rd == 0 gadget
    sink.transition(|| m.at(l.rd_is_zero) * d.rd);
    sink.transition(|| d.rd * m.at(l.inv_rd) + m.at(l.rd_is_zero) - one);

    // ---- shared multiplier: mA (= a, or the quotient on division rows) × mB
    sink.transition(|| m.at(l.b_lo16) - (m.at(l.sh_s) + m.at(l.b_rest) * c(32)));
    let f_sll = (f_alui + f_alur * (one - d.b25)) * f3(1);
    let f_mul = is_m * (f3(0) + f3(1) + f3(2) + f3(3));
    let f_div = is_m * (f3(4) + f3(5) + f3(6) + f3(7));
    let nib = |r: &std::ops::Range<usize>, i: usize| m.at(r.start + i);
    let limbs_of = |r: &std::ops::Range<usize>| -> Lw<F> {
        let mut lo = F::ZERO;
        let mut hi = F::ZERO;
        for i in 0..4 {
            lo += nib(r, i) * c(1u64 << (4 * i));
            hi += nib(r, i + 4) * c(1u64 << (4 * i));
        }
        Lw(lo, hi)
    };
    let m_a = limbs_of(&l.nib_a);
    let nb = limbs_of(&l.nib_b);
    let bw = limbs_of(&l.nib_c);
    let dq = w(l.dq);
    sink.transition_hi(|| m_a.0 - (a.0 + f_div * (dq.0 - a.0)));
    sink.transition_hi(|| m_a.1 - (a.1 + f_div * (dq.1 - a.1)));
    sink.transition(|| nb.0 - b.0);
    sink.transition(|| nb.1 - b.1);
    // mB = q (shift right), p (shift left), op_b (multiply) or the divisor (divide)
    let mbb = |j: usize| m.at(l.mb_byte.start + j);
    let mb = Lw(mbb(0) + mbb(1) * c(256), mbb(2) + mbb(3) * c(256));
    let mb_32 = m.at(l.mb_32);
    let (pow_p, pow_q, q32) = (w(l.pow_p), w(l.pow_q), m.at(l.pow_q32));
    let is_sdiv = f3(4) + f3(6);
    let abs_a = w(l.abs_a);
    let abs_b = w(l.abs_b);
    let dv_d = a.add(abs_a.sub(a).scale(is_sdiv));
    let dv_v = b.add(abs_b.sub(b).scale(is_sdiv));
    sink.transition_hi(|| mb.0 - (pow_q.0 + f_sll * (pow_p.0 - pow_q.0) + f_mul * (b.0 - pow_q.0) + f_div * (dv_v.0 - pow_q.0)));
    sink.transition_hi(|| mb.1 - (pow_q.1 + f_sll * (pow_p.1 - pow_q.1) + f_mul * (b.1 - pow_q.1) + f_div * (dv_v.1 - pow_q.1)));
    sink.transition(|| boolean(mb_32));
    sink.transition_hi(|| mb_32 - q32 * (one - f_sll - f_mul - f_div));
    // schoolbook: bytes of mA × (bytes of mB + bit 32), grouped into 16-bit limbs
    let ab = |i: usize| nib(&l.nib_a, 2 * i) + nib(&l.nib_a, 2 * i + 1) * c(16);
    let bb = |j: usize| if j == 4 { mb_32 } else { mbb(j) };
    let p = |i: usize, j: usize| ab(i) * bb(j);
    let prod = |k: usize| m.at(l.prod[k]);
    let pcar = |k: usize| m.at(l.pcarry[k]);
    let s0 = p(0, 0) + (p(0, 1) + p(1, 0)) * c(256);
    let s1 = p(0, 2) + p(1, 1) + p(2, 0) + (p(0, 3) + p(1, 2) + p(2, 1) + p(3, 0)) * c(256);
    let s2 = p(0, 4) + p(1, 3) + p(2, 2) + p(3, 1) + (p(1, 4) + p(2, 3) + p(3, 2)) * c(256);
    let s3 = p(2, 4) + p(3, 3) + p(3, 4) * c(256);
    sink.transition(|| s0 - prod(0) - pcar(0) * c(B16));
    sink.transition(|| s1 + pcar(0) - prod(1) - pcar(1) * c(B16));
    sink.transition(|| s2 + pcar(1) - prod(2) - pcar(2) * c(B16));
    sink.transition(|| s3 + pcar(2) - prod(3));
    let sh_lo = Lw(prod(0), prod(1));
    let sh_hi = Lw(prod(2), prod(3));
    // arithmetic-shift fill: the top s bits when a is negative
    let fill = w(l.fill);
    let sra_fill = w(l.sra_fill);
    sink.transition_hi(|| sra_fill.0 - sa * d.b30 * fill.0);
    sink.transition_hi(|| sra_fill.1 - sa * d.b30 * fill.1);

    // ---- signed high products: mulh (f3=1): hi − sa·b − sb·a ; mulhsu (f3=2): hi − sa·b   (mod 2^32)
    let g1 = is_m * (f3(1) + f3(2)) * sa;
    let g2 = is_m * f3(1) * sb;
    let adj = w(l.adj);
    let (adj_c0, adj_c1) = (m.at(l.adj_c0), m.at(l.adj_c1));
    sink.transition(|| boolean(adj_c0));
    sink.transition(|| boolean(adj_c1));
    sink.transition_hi(|| g1 * b.0 + g2 * a.0 - adj.0 - adj_c0 * c(B16));
    sink.transition_hi(|| g1 * b.1 + g2 * a.1 + adj_c0 - adj.1 - adj_c1 * c(B16));
    let mh = w(l.mh);
    let (mh_b0, mh_b1) = (m.at(l.mh_b0), m.at(l.mh_b1));
    sink.transition(|| boolean(mh_b0));
    sink.transition(|| boolean(mh_b1));
    sink.transition(|| mh.0 - (sh_hi.0 - adj.0 + mh_b0 * c(B16)));
    sink.transition(|| mh.1 - (sh_hi.1 - adj.1 - mh_b0 + mh_b1 * c(B16)));

    // ---- division: dv_d = dq·dv_v + dr, dr < dv_v (or dv_v = 0 => dq = 2^32-1, dr = dv_d)
    // |x| = x or ~x + 1 (two's complement; the carry out of the low limb is a witness bit)
    let (abs_a_c, abs_b_c) = (m.at(l.abs_a_c), m.at(l.abs_b_c));
    sink.transition(|| boolean(abs_a_c));
    sink.transition(|| boolean(abs_b_c));
    sink.transition(|| abs_a.0 - (a.0 + sa * (c(B16) - a.0 * c(2) - abs_a_c * c(B16))));
    sink.transition(|| abs_a.1 - (a.1 + sa * (c(B16 - 1) - a.1 * c(2) + abs_a_c)));
    sink.transition(|| abs_b.0 - (b.0 + sb * (c(B16) - b.0 * c(2) - abs_b_c * c(B16))));
    sink.transition(|| abs_b.1 - (b.1 + sb * (c(B16 - 1) - b.1 * c(2) + abs_b_c)));
    let (vz_lo, vz_hi, vz) = (m.at(l.vz_lo), m.at(l.vz_hi), m.at(l.vz));
    sink.transition_hi(|| vz_lo * dv_v.0);
    sink.transition_hi(|| dv_v.0 * m.at(l.vinv_lo) + vz_lo - one);
    sink.transition_hi(|| vz_hi * dv_v.1);
    sink.transition_hi(|| dv_v.1 * m.at(l.vinv_hi) + vz_hi - one);
    sink.transition(|| vz - vz_lo * vz_hi);
    let dr = w(l.dr);
    let dcarry = m.at(l.dcarry);
    sink.transition(|| boolean(dcarry));
    // on division rows the multiplier computed dq · dv_v (64 bits, must fit 32)
    sink.transition_hi(|| f_div * (prod(0) + dr.0 - dv_d.0 - dcarry * c(B16)));
    sink.transition_hi(|| f_div * (prod(1) + dr.1 + dcarry - dv_d.1));
    sink.transition_hi(|| f_div * prod(2));
    sink.transition_hi(|| f_div * prod(3));
    sink.transition(|| vz * (dq.0 - c(B16 - 1)));
    sink.transition(|| vz * (dq.1 - c(B16 - 1)));
    // dr < dv_v  ⇔  rlt = dv_v − dr − 1 ≥ 0 (range-checked limbs)
    let rlt = w(l.rlt);
    let rlt_b = m.at(l.rlt_b);
    sink.transition(|| boolean(rlt_b));
    sink.transition_hi(|| (one - vz) * (dv_v.0 - dr.0 - one + rlt_b * c(B16) - rlt.0));
    sink.transition_hi(|| (one - vz) * (dv_v.1 - dr.1 - rlt_b - rlt.1));
    let neg_q = m.at(l.neg_q);
    sink.transition_hi(|| neg_q - is_sdiv * (sa + sb - sa * sb * c(2)) * (one - vz));
    let div = w(l.div);
    let (div_c, div_c2) = (m.at(l.div_c), m.at(l.div_c2));
    sink.transition(|| boolean(div_c));
    sink.transition(|| boolean(div_c2));
    sink.transition(|| div.0 - (dq.0 + neg_q * (c(B16) - dq.0 * c(2) - div_c * c(B16))));
    sink.transition(|| div.1 - (dq.1 + neg_q * (c(B16 - 1) - dq.1 * c(2) + div_c - div_c2 * c(B16))));
    let neg_r = is_sdiv * sa;
    let rem = w(l.rem);
    let (rem_c, rem_c2) = (m.at(l.rem_c), m.at(l.rem_c2));
    sink.transition(|| boolean(rem_c));
    sink.transition(|| boolean(rem_c2));
    sink.transition_hi(|| rem.0 - (dr.0 + neg_r * (c(B16) - dr.0 * c(2) - rem_c * c(B16))));
    sink.transition_hi(|| rem.1 - (dr.1 + neg_r * (c(B16 - 1) - dr.1 * c(2) + rem_c - rem_c2 * c(B16))));

    // ---- ALU result
    let f_bw = f3(4) + f3(6) + f3(7);
    let alu_base = Lw(
        f3(0) * add.0 + f3(1) * sh_lo.0 + f3(2) * lt_s + f3(3) * ltu + f_bw * bw.0 + f3(5) * (sh_hi.0 + sra_fill.0),
        f3(0) * add.1 + f3(1) * sh_lo.1 + f_bw * bw.1 + f3(5) * (sh_hi.1 + sra_fill.1),
    );
    let alu_m = Lw(
        f3(0) * sh_lo.0 + (f3(1) + f3(2)) * mh.0 + f3(3) * sh_hi.0 + f3(4) * div.0 + f3(5) * dq.0 + f3(6) * rem.0 + f3(7) * dr.0,
        f3(0) * sh_lo.1 + (f3(1) + f3(2)) * mh.1 + f3(3) * sh_hi.1 + f3(4) * div.1 + f3(5) * dq.1 + f3(6) * rem.1 + f3(7) * dr.1,
    );
    let alu = w(l.alu);
    sink.transition_hi(|| alu.0 - ((one - is_m) * alu_base.0 + is_m * alu_m.0));
    sink.transition_hi(|| alu.1 - ((one - is_m) * alu_base.1 + is_m * alu_m.1));

    // ---- loads / stores
    let mb_ = |i: usize| m.at(l.m_byte.start + i);
    sink.transition(|| m_word.0 - (mb_(0) + mb_(1) * c(1 << 8)));
    sink.transition(|| m_word.1 - (mb_(2) + mb_(3) * c(1 << 8)));
    let (o0, o1) = (m.at(l.o0), m.at(l.o1));
    sink.transition(|| boolean(o0));
    sink.transition(|| boolean(o1));
    // byte address = add (a + imm); word address < 2^29
    sink.transition(|| f_mem * (add.0 - m.at(l.wl14) * c(4) - o0 - o1 * c(2)));
    sink.transition(|| f_mem * (add.1 - m.at(l.mw_hi15)));
    let mem_waddr = m.at(l.wl14) + m.at(l.mw_hi15) * c(1 << 14);
    let sel = [(one - o0) * (one - o1), o0 * (one - o1), (one - o0) * o1, o0 * o1];
    let sel_byte = m.at(l.sel_byte);
    sink.transition_hi(|| sel_byte - (sel[0] * mb_(0) + sel[1] * mb_(1) + sel[2] * mb_(2) + sel[3] * mb_(3)));
    sink.transition(|| sel_byte - (m.at(l.sb_lo7) + m.at(l.sb7) * c(128)));
    sink.transition(|| boolean(m.at(l.sb7)));
    let sel_half = m.at(l.sel_half);
    sink.transition(|| sel_half - ((one - o1) * (mb_(0) + mb_(1) * c(256)) + o1 * (mb_(2) + mb_(3) * c(256))));
    sink.transition(|| sel_half - (m.at(l.sh_lo15) + m.at(l.h15) * c(1 << 15)));
    sink.transition(|| boolean(m.at(l.h15)));
    let (sb7, h15) = (m.at(l.sb7), m.at(l.h15));
    let load = Lw(
        f3(0) * (sel_byte + sb7 * c(B16 - 256)) + f3(1) * sel_half + f3(2) * m_word.0 + f3(4) * sel_byte + f3(5) * sel_half,
        f3(0) * sb7 * c(B16 - 1) + f3(1) * h15 * c(B16 - 1) + f3(2) * m_word.1,
    );
    sink.transition_hi(|| f_mem * (f3(1) + f3(5)) * o0);
    sink.transition_hi(|| f_mem * f3(2) * (o0 + o1));
    let (st_b0, st_b1) = (m.at(l.st_b0), m.at(l.st_b1));
    sink.transition(|| b_reg.0 - (st_b0 + st_b1 * c(256)));
    // merged word for byte / halfword stores
    let nw_sb = |i: usize| mb_(i) + sel[i] * (st_b0 - mb_(i));
    let nw_sh = [mb_(0) + (one - o1) * (st_b0 - mb_(0)), mb_(1) + (one - o1) * (st_b1 - mb_(1)), mb_(2) + o1 * (st_b0 - mb_(2)), mb_(3) + o1 * (st_b1 - mb_(3))];
    let new_word = w(l.new_word);
    sink.transition_hi(|| new_word.0 - (f3(0) * (nw_sb(0) + nw_sb(1) * c(256)) + f3(1) * (nw_sh[0] + nw_sh[1] * c(256)) + f3(2) * b_reg.0));
    sink.transition_hi(|| new_word.1 - (f3(0) * (nw_sb(2) + nw_sb(3) * c(256)) + f3(1) * (nw_sh[2] + nw_sh[3] * c(256)) + f3(2) * b_reg.1));

    // ---- result written to rd
    let eo = w(l.eo);
    let result = w(l.result);
    let res = Lw(
        f_lui * d.imm_u.0 + f_auipc * pt.0 + (f_jal + f_jalr) * pc4.0 + f_load * load.0 + f_alu * alu.0 + f_ecall * eo.0,
        f_lui * d.imm_u.1 + f_auipc * pt.1 + (f_jal + f_jalr) * pc4.1 + f_load * load.1 + f_alu * alu.1 + f_ecall * eo.1,
    );
    sink.transition_hi(|| result.0 - res.0);
    sink.transition_hi(|| result.1 - res.1);

    // ---- ecall: a7 in {0: halt, 1: read witness, 2: write journal, 3: write private output word,
    //                    4: write private output buffer (a0 = address of [byte_len, data...]; unconstrained like 3)}
    sink.transition(|| boolean(h0));
    sink.transition(|| boolean(h1));
    sink.transition(|| boolean(h2));
    sink.transition(|| boolean(h3));
    sink.transition(|| boolean(h4));
    sink.transition(|| f_ecall * (h0 + h1 + h2 + h3 + h4 - one));
    sink.transition(|| f_ecall * (h1 + h2 * c(2) + h3 * c(3) + h4 * c(4) - a.0));
    sink.transition(|| f_ecall * a.1);
    sink.transition_hi(|| f_ecall * (one - h1) * (eo.0 - b_reg.0));
    sink.transition_hi(|| f_ecall * (one - h1) * (eo.1 - b_reg.1));
    sink.transition_hi(|| is_halt * b_reg.0);
    sink.transition_hi(|| is_halt * b_reg.1);
    sink.transition(|| mn.at(l.jcount) - m.at(l.jcount) - f_ecall * h2);

    // ---- CPU-side memory slots
    let rb = c(REG_BASE);
    let writes_rd = f_lui + f_auipc + f_jal + f_jalr + f_load + f_alu;
    sink.transition(|| m.at(l.cpu_addr[0]) - (pcl14 + pc_hi * c(1 << 14)));
    sink.transition(|| cv(0).0 - d.instr.0);
    sink.transition(|| cv(0).1 - d.instr.1);
    sink.transition(|| m.at(l.cpu_addr[1]) - (rb + d.rs1 + f_ecall * (c(17) - d.rs1)));
    sink.transition(|| m.at(l.cpu_addr[2]) - (rb + d.rs2 + f_ecall * (c(10) - d.rs2)));
    sink.transition(|| m.at(l.cpu_addr[3]) - (rb + f_mem * (mem_waddr - rb)));
    sink.transition(|| m.at(l.cpu_addr[4]) - (rb + writes_rd * d.rd + f_store * (mem_waddr - rb) + f_ecall * c(10)));
    let rd_nz = one - m.at(l.rd_is_zero);
    sink.transition_hi(|| cv(4).0 - (f_store * new_word.0 + writes_rd * rd_nz * result.0 + f_ecall * eo.0));
    sink.transition_hi(|| cv(4).1 - (f_store * new_word.1 + writes_rd * rd_nz * result.1 + f_ecall * eo.1));

    // ---- sorted memory table
    let has_close = ext(public[PUB_HAS_CLOSE]);
    for s in &l.mem {
        sink.transition(|| boolean(m.at(s.w)));
        sink.transition(|| boolean(m.at(s.init)));
        sink.transition(|| boolean(m.at(s.real)));
        sink.transition(|| boolean(m.at(s.same)));
        sink.transition(|| boolean(m.at(s.close)));
        sink.transition(|| m.at(s.real) * m.at(s.init));
        sink.transition(|| m.at(s.close) * m.at(s.real));
        sink.transition(|| m.at(s.close) * m.at(s.init));
        sink.transition(|| (one - m.at(s.real)) * m.at(s.w));
        // every address group starts with its init entry; closing entries stand alone
        sink.transition(|| m.at(s.init) * m.at(s.same));
        sink.transition(|| m.at(s.real) * (one - m.at(s.same)));
        sink.transition(|| m.at(s.close) * m.at(s.same));
        sink.transition(|| m.at(s.init) * m.at(s.clk));
        sink.transition(|| m.at(s.diff) - (m.at(s.d_lo) + m.at(s.d_hi) * c(B16)));
        sink.transition(|| m.at(s.age) - m.at(s.init) * (m.at(l.seg_idx) - m.at(s.t)));
        // a segment that closes addresses says so publicly (the verifier chains the ranges)
        sink.transition(|| Ext(m.at(s.close).to_ext() * (F::Ext::ONE - has_close)));
    }
    let (close_lo, close_hi) = (ext(public[PUB_CLOSE_LO]), ext(public[PUB_CLOSE_HI]));
    for k in 1..NUM_SLOTS {
        slot_pair(m, &l.mem[k - 1], m, &l.mem[k], close_lo, close_hi, sink);
    }
    slot_pair(m, &l.mem[NUM_SLOTS - 1], mn, &l.mem[0], close_lo, close_hi, sink);
    sink.transition(|| m.at(l.same_next0) - mn.at(l.mem[0].same));
    // segment index: bound to the public value, constant down the trace
    sink.transition(|| mn.at(l.seg_idx) - m.at(l.seg_idx));

    // ---- LogUp aux columns
    let e = Enc::new(ch[0], ch[1]);
    // Terms are streamed in groups of BATCH straight into the batched
    // constraint (no per-row term buffer).
    let mut pend: [(F, F::Ext); BATCH] = [(F::ZERO, F::Ext::ONE); BATCH];
    let mut np = 0usize;
    let mut col = 0usize;
    let emit = |pend: &[(F, F::Ext); BATCH], col: usize, sink: &mut ConstraintSink<F>| {
        let (m1, d1) = pend[0];
        let (m2, d2) = pend[1];
        let (m3, d3) = pend[2];
        let s = row.aux.at(col);
        let sn = row.aux_next.at(col);
        let d12 = d1 * d2;
        let d123 = d12 * d3;
        // K = m1·D2·D3 + m2·D1·D3 + m3·D1·D2 = D3·(m1·D2 + m2·D1) + m3·D1·D2
        let k = d3 * (d2.mul_base(m1) + d1.mul_base(m2)) + d12.mul_base(m3);
        // (s' - s)·D1·D2·D3 = K
        sink.transition_hi(|| Ext((sn - s) * d123 - k));
        sink.first(|| Ext(s));
        sink.last_hi(|| Ext((ext(aux_public[col]) - s) * d123 - k));
    };
    row_terms_with(l, m, row.fixed, &e, |mult, denom| {
        pend[np] = (mult, denom);
        np += 1;
        if np == BATCH {
            emit(&pend, col, sink);
            np = 0;
            col += 1;
        }
    });
    if np != 0 {
        for p in pend.iter_mut().skip(np) {
            *p = (F::ZERO, F::Ext::ONE);
        }
        emit(&pend, col, sink);
    }

    // ---- boundaries
    sink.first(|| m.at(l.clk));
    sink.first(|| active - one);
    sink.first(|| Ext(pc.to_ext() - ext(public[PUB_PC_IN])));
    sink.first(|| Ext(m.at(l.jcount).to_ext() - ext(public[PUB_JCOUNT_IN])));
    let s0 = &l.mem[0];
    sink.first(|| m.at(s0.same));
    sink.first(|| Ext(m.at(l.seg_idx).to_ext() - ext(public[PUB_SEG])));

    sink.last(|| active);
    sink.last(|| flag_sum);
    sink.last(|| m.at(l.same_next0));
    // the last real row is padding: every table stage has ended before it
    for s in &l.mem {
        sink.last(|| m.at(s.real));
        sink.last(|| m.at(s.init));
        sink.last(|| m.at(s.close));
    }
}
