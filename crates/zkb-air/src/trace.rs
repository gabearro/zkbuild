//! Main-trace generation from a VM segment record, and LogUp aux generation.

use crate::layout::*;
use crate::lookups::{row_terms, Enc, Term, TermBuf, BATCH};
use anyhow::{ensure, Result};
use rayon::prelude::*;
use zkb_field::columns::ColumnStore;
use zkb_stark::RowAccess;
use zkb_field::simd::{Fp16, Fp4x16, LANES};
use zkb_field::{Fp, Fp4};
use zkb_vm::{CycleRecord, WriteTarget};

const TWO32: u64 = 1 << 32;

#[inline]
fn f(v: u64) -> Fp {
    Fp::new(v)
}
#[inline]
fn fb(v: bool) -> Fp {
    Fp::new(v as u64)
}

/// One segment of execution, as the trace generator needs it.
pub struct Segment<'a> {
    /// Index of the segment in the execution (its writes carry stamp k+1).
    pub index: usize,
    /// Init entries: every address the segment touches, with its value at
    /// segment start and the stamp of the last segment that touched it
    /// (0 = initial memory). Sorted by address.
    pub init_entries: &'a [(u64, u32, u32)],
    /// This segment's share of the execution's closing table: (address,
    /// final value, stamp of the last toucher), sorted; may be empty.
    pub closing: &'a [(u64, u32, u32)],
    pub records: &'a [CycleRecord],
    pub pc_in: u32,
    pub pc_out: u32,
    pub jcount_in: u32,
    pub jcount_out: u32,
    pub halted: bool,
}

impl Segment<'_> {
    pub fn public_values(&self) -> Vec<Fp4> {
        let (lo, hi) = match (self.closing.first(), self.closing.last()) {
            (Some(a), Some(b)) => (a.0, b.0),
            _ => (0, 0),
        };
        vec![
            Fp4::from(self.pc_in as u64),
            Fp4::from(self.pc_out as u64),
            Fp4::from(self.jcount_in as u64),
            Fp4::from(self.jcount_out as u64),
            Fp4::from(self.halted as u64),
            Fp4::from(self.index as u64),
            Fp4::from(lo),
            Fp4::from(hi),
            Fp4::from(!self.closing.is_empty() as u64),
        ]
    }
}

/// Minimum log2 trace length for a segment of `cycles` cycles whose memory
/// table holds `extra_entries` init and closing entries, leaving
/// `random_rows` rows for the prover's zero-knowledge padding.
pub fn min_log_n(cycles: usize, extra_entries: usize, random_rows: usize) -> u32 {
    let need = (cycles + 1).max(cycles + extra_entries.div_ceil(NUM_SLOTS) + 1).max(1 << 16) + random_rows;
    need.next_power_of_two().trailing_zeros()
}

/// Memory-table slots available to a segment of 2^log_n rows (the last real
/// row is padding): 5·cycles + init entries + closing entries must fit.
pub fn slot_budget(log_n: u32, random_rows: usize) -> usize {
    let n = (1usize << log_n).saturating_sub(random_rows);
    if n < 2 {
        return 0;
    }
    NUM_SLOTS * (n - 1)
}

#[derive(Clone, Copy)]
struct Access {
    addr: u64,
    clk: u64,
    val: u64,
    w: bool,
    init: bool,
    /// cited / carried stamp (init and closing entries)
    t: u64,
    close: bool,
}

fn opcode_flag(opcode: u32) -> Option<usize> {
    OPCODES.iter().position(|&o| o == opcode as u64)
}

/// Build the main trace of a segment: `real_rows` rows are filled (cycles,
/// then inactive padding); rows beyond are left zero for the prover's
/// zero-knowledge random rows. Columns have length `2^log_n`.
pub fn generate_main(l: &Layout, seg: &Segment, log_n: u32, real_rows: usize) -> Result<ColumnStore> {
    let full = 1usize << log_n;
    let n = real_rows;
    let records = seg.records;
    let cycles = records.len();
    ensure!(n >= 1 << 16 && n <= full, "need at least 2^16 real rows (16-bit range table)");
    ensure!(cycles >= 1, "empty segment");
    ensure!(cycles + 1 <= n, "trace too short for {cycles} cycles");
    ensure!(NUM_SLOTS * cycles + seg.init_entries.len() + seg.closing.len() <= NUM_SLOTS * (n - 1), "trace too short for memory table");
    ensure!(records[0].pc == seg.pc_in, "segment pc_in mismatch");
    let w = l.main_width;
    let timing = std::env::var("ZKB_TIMING").is_ok();
    let t0 = std::time::Instant::now();
    // Row-major scratch (row i = `rows.col(i)`), recycled across segments.
    let mut rows = ColumnStore::zeros(full, w);
    let tlog = |what: &str| if timing { eprintln!("      trace: {what} at {:?}", t0.elapsed()) };
    tlog("rows allocated");
    let mut accesses: Vec<Access> = Vec::with_capacity(NUM_SLOTS * cycles + seg.init_entries.len() + seg.closing.len());
    for &(a, v, t) in seg.init_entries {
        ensure!((t as usize) <= seg.index, "init entry cites a stamp from the future");
        accesses.push(Access { addr: a, clk: 0, val: v as u64, w: false, init: true, t: t as u64, close: false });
    }
    for &(a, v, t) in seg.closing {
        accesses.push(Access { addr: a, clk: 0, val: v as u64, w: false, init: false, t: t as u64, close: true });
    }

    // journal counter per row (prefix over ecall/write_journal rows)
    let mut jc = vec![0u64; cycles];
    let mut jcount: u64 = seg.jcount_in as u64;
    for (i, rec) in records.iter().enumerate() {
        jc[i] = jcount;
        if rec.instr == 0x73 && rec.r1.1 == 2 {
            jcount += 1;
        }
    }
    rows.par_cols_mut().take(cycles).enumerate().for_each(|(i, r)| {
        let rec = &records[i];
        let instr = rec.instr;
        let pc = rec.pc as u64;
        r[l.clk] = f(i as u64);
        r[l.active] = Fp::ONE;
        r[l.pc] = f(pc);
        r[l.pcl14] = f((pc & 0xffff) / 4);
        r[l.pc_hi] = f(pc >> 16);
        for k in 0..32 {
            r[l.bits.start + k] = fb((instr >> k) & 1 == 1);
        }
        let opcode = instr & 0x7f;
        let flag = opcode_flag(opcode).expect("VM executed an instruction the AIR does not know");
        r[l.flags.start + flag] = Fp::ONE;
        let funct3 = ((instr >> 12) & 7) as usize;
        r[l.f3.start + funct3] = Fp::ONE;
        let rd = ((instr >> 7) & 0x1f) as u64;
        let rs1 = ((instr >> 15) & 0x1f) as u64;
        let rs2 = ((instr >> 20) & 0x1f) as u64;
        let b25 = ((instr >> 25) & 1) as u64;
        let b30 = ((instr >> 30) & 1) as u64;
        let is = |k: usize| flag == k;
        let is_m = is(F_ALUR) && b25 == 1;
        r[l.is_m] = fb(is_m);
        let sext = |v: u32, bits: u32| -> u64 { ((((v << (32 - bits)) as i32) >> (32 - bits)) as u32) as u64 };
        let imm_i = sext(instr >> 20, 12);
        let imm_s = sext(((instr >> 25) << 5) | ((instr >> 7) & 0x1f), 12);
        let imm_b = sext(((instr >> 31) << 12) | (((instr >> 7) & 1) << 11) | (((instr >> 25) & 0x3f) << 5) | (((instr >> 8) & 0xf) << 1), 13);
        let imm_u = (instr & 0xffff_f000) as u64;
        let imm_j = sext(((instr >> 31) << 20) | (((instr >> 12) & 0xff) << 12) | (((instr >> 20) & 1) << 11) | (((instr >> 21) & 0x3ff) << 1), 21);
        let lo = |v: u64| v & 0xffff;
        let hi = |v: u64| (v >> 16) & 0xffff;
        let set_word = |r: &mut [Fp], wd: Word, v: u64| {
            r[wd.lo] = f(lo(v));
            r[wd.hi] = f(hi(v));
        };

        let a_val = rec.r1.1 as u64;
        let b_reg = rec.r2.1 as u64;
        let m_word = rec.mem_read.map(|(_, v)| v as u64).unwrap_or(0);
        set_word(r, l.cpu_val[1], a_val);
        set_word(r, l.cpu_val[2], b_reg);
        set_word(r, l.cpu_val[3], m_word);

        let op_b: u64 = if is(F_ALUR) || is(F_BR) {
            b_reg
        } else if is(F_ALUI) || is(F_LOAD) || is(F_JALR) {
            imm_i
        } else if is(F_STORE) {
            imm_s
        } else {
            0
        };
        let decomp = |v: u64| (v & 0xffff, (v >> 16) & 0x7fff, v >> 31);
        let (alo, ahi15, asg) = decomp(a_val);
        let (blo, bhi15, bsg) = decomp(op_b);
        r[l.a_lo16] = f(alo);
        r[l.a_hi15] = f(ahi15);
        r[l.a_sign] = f(asg);
        r[l.b_lo16] = f(blo);
        r[l.b_hi15] = f(bhi15);
        r[l.b_sign] = f(bsg);
        let (a_lo, a_hi) = (lo(a_val), hi(a_val));
        let (b_lo, b_hi) = (lo(op_b), hi(op_b));

        // adder with explicit carries
        let is_alu = is(F_ALUI) || is(F_ALUR);
        let use_sub = (is(F_ALUR) && funct3 == 0 && b30 == 1) || (is_alu && (funct3 == 2 || funct3 == 3)) || is(F_BR);
        r[l.use_sub] = fb(use_sub);
        let (in_lo, in_hi, cin) = if use_sub { (0xffff - b_lo, 0xffff - b_hi, 1) } else { (b_lo, b_hi, 0) };
        let t0 = a_lo + in_lo + cin;
        let add_lo = t0 & 0xffff;
        let c0 = t0 >> 16;
        let t1 = a_hi + in_hi + c0;
        let add_hi = t1 & 0xffff;
        let add_carry = t1 >> 16;
        let add_res = add_lo | (add_hi << 16);
        debug_assert_eq!(add_res, (a_val + if use_sub { TWO32 - op_b } else { op_b }) & 0xffff_ffff);
        r[l.add.lo] = f(add_lo);
        r[l.add.hi] = f(add_hi);
        r[l.add_c0] = f(c0);
        r[l.add_carry] = f(add_carry);
        let ltu = 1 - add_carry;
        let lt_s = if asg != bsg { asg } else { ltu };
        r[l.lt_s] = f(lt_s);
        let eq_lo = a_lo == b_lo;
        let eq_hi = a_hi == b_hi;
        r[l.eq_lo] = fb(eq_lo);
        r[l.eq_hi] = fb(eq_hi);
        // inverses for all gadgets of this row are computed in one batch below
        let mut inv_batch = [f(a_lo) - f(b_lo), f(a_hi) - f(b_hi), f(rd), Fp::ZERO, Fp::ZERO];
        let eqv = (eq_lo && eq_hi) as u64;
        let taken = match funct3 {
            0 => eqv,
            1 => 1 - eqv,
            4 => lt_s,
            5 => 1 - lt_s,
            6 => ltu,
            7 => 1 - ltu,
            _ => 0,
        };
        r[l.taken] = f(taken);

        // pc arithmetic
        let imm_pc = if is(F_JAL) {
            imm_j
        } else if is(F_BR) {
            imm_b
        } else if is(F_AUIPC) {
            imm_u
        } else {
            0
        };
        let pc_lo = lo(pc);
        let pc_hi = hi(pc);
        let t = pc_lo + lo(imm_pc);
        let pt_lo = t & 0xffff;
        let pt_c0 = t >> 16;
        let t = pc_hi + hi(imm_pc) + pt_c0;
        let pt_hi = t & 0xffff;
        let pt_c1 = t >> 16;
        r[l.pt.lo] = f(pt_lo);
        r[l.pt.hi] = f(pt_hi);
        r[l.pt_c0] = f(pt_c0);
        r[l.pt_c1] = f(pt_c1);
        let t = pc_lo + 4;
        r[l.pc4_lo] = f(t & 0xffff);
        r[l.pc4_c] = f(t >> 16);
        r[l.rd_is_zero] = fb(rd == 0);

        // shared multiplier
        let s = blo & 31;
        let b_rest = blo >> 5;
        let pow_p = 1u64 << s;
        let pow_q = 1u64 << (32 - s);
        let fill = if s == 0 { 0 } else { (TWO32 - pow_q) & 0xffff_ffff };
        r[l.sh_s] = f(s);
        r[l.b_rest] = f(b_rest);
        set_word(r, l.pow_p, pow_p);
        set_word(r, l.pow_q, pow_q & 0xffff_ffff);
        r[l.pow_q32] = fb(s == 0);
        set_word(r, l.fill, fill);
        let sra_fill = if asg == 1 && b30 == 1 { fill } else { 0 };
        set_word(r, l.sra_fill, sra_fill);
        let is_sll = is_alu && !is_m && funct3 == 1;
        let is_mul = is_m && funct3 < 4;
        let is_div = is_m && funct3 >= 4;
        let is_sdiv = funct3 == 4 || funct3 == 6;
        let abs_a = if asg == 1 { (TWO32 - a_val) & 0xffff_ffff } else { a_val };
        let abs_b = if bsg == 1 { (TWO32 - op_b) & 0xffff_ffff } else { op_b };
        let dv_d = if is_sdiv { abs_a } else { a_val };
        let dv_v = if is_sdiv { abs_b } else { op_b };
        let (dq, dr) = if dv_v == 0 { (TWO32 - 1, dv_d) } else { (dv_d / dv_v, dv_d % dv_v) };
        let m_a = if is_div { dq } else { a_val };
        let m_b: u64 = if is_sll {
            pow_p
        } else if is_mul {
            op_b
        } else if is_div {
            dv_v
        } else {
            pow_q
        };
        for i in 0..8 {
            r[l.nib_a.start + i] = f((m_a >> (4 * i)) & 15);
            r[l.nib_b.start + i] = f((op_b >> (4 * i)) & 15);
        }
        for j in 0..4 {
            r[l.mb_byte.start + j] = f((m_b >> (8 * j)) & 0xff);
        }
        r[l.mb_32] = f(m_b >> 32);
        let prod = (m_a as u128) * (m_b as u128);
        debug_assert!(prod < (1u128 << 64));
        // limb equations with carries: S_k + carry_{k-1} = prod_k + 2^16·carry_k
        let ab: [u64; 4] = core::array::from_fn(|i| (m_a >> (8 * i)) & 0xff);
        let bb: [u64; 5] = core::array::from_fn(|j| (m_b >> (8 * j)) & 0xff);
        let pp = |i: usize, j: usize| ab[i] * bb[j];
        let sums = [
            pp(0, 0) + 256 * (pp(0, 1) + pp(1, 0)),
            pp(0, 2) + pp(1, 1) + pp(2, 0) + 256 * (pp(0, 3) + pp(1, 2) + pp(2, 1) + pp(3, 0)),
            pp(0, 4) + pp(1, 3) + pp(2, 2) + pp(3, 1) + 256 * (pp(1, 4) + pp(2, 3) + pp(3, 2)),
            pp(2, 4) + pp(3, 3) + 256 * pp(3, 4),
        ];
        let mut carry = 0u64;
        for k in 0..4 {
            let t = sums[k] + carry;
            r[l.prod[k]] = f(t & 0xffff);
            carry = t >> 16;
            if k < 3 {
                r[l.pcarry[k]] = f(carry);
            }
        }
        debug_assert_eq!(carry, 0);
        let sh_lo = (prod & 0xffff_ffff) as u64;
        let sh_hi = ((prod >> 32) & 0xffff_ffff) as u64;
        debug_assert_eq!(sh_lo, r[l.prod[0]].as_u64() | (r[l.prod[1]].as_u64() << 16));
        // signed high product adjustment: adj = g1·b + g2·a (mod 2^32), mh = sh_hi − adj (mod 2^32)
        let (g1, g2) = if is_m && funct3 == 1 { (asg, bsg) } else if is_m && funct3 == 2 { (asg, 0) } else { (0, 0) };
        let t = g1 * b_lo + g2 * a_lo;
        let adj_lo = t & 0xffff;
        let adj_c0 = t >> 16;
        let t = g1 * b_hi + g2 * a_hi + adj_c0;
        let adj_hi = t & 0xffff;
        let adj_c1 = t >> 16;
        r[l.adj.lo] = f(adj_lo);
        r[l.adj.hi] = f(adj_hi);
        r[l.adj_c0] = f(adj_c0);
        r[l.adj_c1] = f(adj_c1);
        let (mh_lo, mh_b0) = if lo(sh_hi) >= adj_lo { (lo(sh_hi) - adj_lo, 0) } else { (lo(sh_hi) + 65536 - adj_lo, 1) };
        let t = hi(sh_hi) as i64 - adj_hi as i64 - mh_b0 as i64;
        let (mh_hi, mh_b1) = if t >= 0 { (t as u64, 0) } else { ((t + 65536) as u64, 1) };
        r[l.mh.lo] = f(mh_lo);
        r[l.mh.hi] = f(mh_hi);
        r[l.mh_b0] = f(mh_b0);
        r[l.mh_b1] = f(mh_b1);
        debug_assert_eq!(mh_lo | (mh_hi << 16), (sh_hi + TWO32 * 2 - (g1 * op_b + g2 * a_val)) & 0xffff_ffff);

        // division gadget (always evaluated)
        set_word(r, l.abs_a, abs_a);
        set_word(r, l.abs_b, abs_b);
        r[l.abs_a_c] = fb(asg == 1 && a_lo == 0);
        r[l.abs_b_c] = fb(bsg == 1 && b_lo == 0);
        set_word(r, l.dq, dq);
        set_word(r, l.dr, dr);
        let vz_lo = lo(dv_v) == 0;
        let vz_hi = hi(dv_v) == 0;
        let vz = dv_v == 0;
        r[l.vz_lo] = fb(vz_lo);
        r[l.vz_hi] = fb(vz_hi);
        r[l.vz] = fb(vz);
        inv_batch[3] = f(lo(dv_v));
        inv_batch[4] = f(hi(dv_v));
        // dv_d = prod + dr on division rows (prod = dq·dv_v < 2^32)
        let dcarry = if is_div { (sh_lo & 0xffff) + lo(dr) >> 16 } else { 0 };
        r[l.dcarry] = f(dcarry);
        let rlt = if vz { 0 } else { dv_v - dr - 1 };
        let (rlt_lo, rlt_b) = if vz { (0, 0) } else if lo(dv_v) >= lo(dr) + 1 { (lo(dv_v) - lo(dr) - 1, 0) } else { (lo(dv_v) + 65536 - lo(dr) - 1, 1) };
        r[l.rlt.lo] = f(rlt_lo);
        r[l.rlt.hi] = f(if vz { 0 } else { hi(dv_v) - hi(dr) - rlt_b });
        r[l.rlt_b] = f(rlt_b);
        debug_assert!(vz || rlt == rlt_lo | (r[l.rlt.hi].as_u64() << 16));
        let neg_q = is_sdiv && (asg != bsg) && !vz;
        r[l.neg_q] = fb(neg_q);
        let neg_word = |r: &mut [Fp], wd: Word, c1: usize, c2: usize, v: u64, neg: bool| {
            if neg {
                let t = 65536 - lo(v);
                let c = t >> 16; // 1 iff lo == 0
                r[wd.lo] = f(t & 0xffff);
                let t = 65535 - hi(v) + c;
                r[c1] = f(c);
                r[c2] = f(t >> 16);
                r[wd.hi] = f(t & 0xffff);
                debug_assert_eq!(r[wd.lo].as_u64() | (r[wd.hi].as_u64() << 16), (TWO32 - v) & 0xffff_ffff);
            } else {
                r[wd.lo] = f(lo(v));
                r[wd.hi] = f(hi(v));
            }
        };
        neg_word(r, l.div, l.div_c, l.div_c2, dq, neg_q);
        let neg_r = is_sdiv && asg == 1;
        neg_word(r, l.rem, l.rem_c, l.rem_c2, dr, neg_r);
        let div_res = if neg_q { (TWO32 - dq) & 0xffff_ffff } else { dq };
        let rem_res = if neg_r { (TWO32 - dr) & 0xffff_ffff } else { dr };

        // nibbles of the bitwise result (op from f3: 6 => or, 7 => and, else xor)
        let bw = match funct3 {
            6 => m_a | op_b,
            7 => m_a & op_b,
            _ => m_a ^ op_b,
        };
        for i in 0..8 {
            r[l.nib_c.start + i] = f((bw >> (4 * i)) & 15);
        }
        let mh_res = mh_lo | (mh_hi << 16);
        let alu_val: u64 = if is_m {
            match funct3 {
                0 => sh_lo,
                1 | 2 => mh_res,
                3 => sh_hi,
                4 => div_res,
                5 => dq,
                6 => rem_res,
                7 => dr,
                _ => unreachable!(),
            }
        } else {
            match funct3 {
                0 => add_res,
                1 => sh_lo,
                2 => lt_s,
                3 => ltu,
                4 | 6 | 7 => bw,
                5 => sh_hi + sra_fill,
                _ => unreachable!(),
            }
        };
        set_word(r, l.alu, alu_val);

        // loads / stores
        let mb: [u64; 4] = [m_word & 0xff, (m_word >> 8) & 0xff, (m_word >> 16) & 0xff, (m_word >> 24) & 0xff];
        for i in 0..4 {
            r[l.m_byte.start + i] = f(mb[i]);
        }
        let is_mem = is(F_LOAD) || is(F_STORE);
        let o0 = add_res & 1;
        let o1 = (add_res >> 1) & 1;
        let off = (add_res & 3) as usize;
        r[l.o0] = f(o0);
        r[l.o1] = f(o1);
        if is_mem {
            r[l.wl14] = f((add_res & 0xffff) >> 2);
            r[l.mw_hi15] = f(add_res >> 16);
        }
        let mem_waddr = add_res >> 2;
        let sel_byte = mb[off];
        r[l.sel_byte] = f(sel_byte);
        r[l.sb_lo7] = f(sel_byte & 0x7f);
        r[l.sb7] = f(sel_byte >> 7);
        let sel_half = if o1 == 1 { mb[2] + 256 * mb[3] } else { mb[0] + 256 * mb[1] };
        r[l.sel_half] = f(sel_half);
        r[l.sh_lo15] = f(sel_half & 0x7fff);
        r[l.h15] = f(sel_half >> 15);
        let load_val: u64 = match funct3 {
            0 => sel_byte | ((sel_byte >> 7) * 0xffff_ff00),
            1 => sel_half | ((sel_half >> 15) * 0xffff_0000),
            2 => m_word,
            4 => sel_byte,
            5 => sel_half,
            _ => 0,
        };
        let st_b0 = b_reg & 0xff;
        let st_b1 = (b_reg >> 8) & 0xff;
        r[l.st_b0] = f(st_b0);
        r[l.st_b1] = f(st_b1);
        // merged word exactly as the constraint forms it (halfword merges only
        // look at o1; a misaligned halfword store traps in the VM anyway)
        let new_word: u64 = match funct3 {
            0 => (m_word & !(0xff << (8 * off))) | (st_b0 << (8 * off)),
            1 => {
                let h = st_b0 | (st_b1 << 8);
                if o1 == 1 {
                    (m_word & 0xffff) | (h << 16)
                } else {
                    (m_word & 0xffff_0000) | h
                }
            }
            2 => b_reg,
            _ => 0,
        };
        set_word(r, l.new_word, new_word);

        // ecall
        let mut ecall_out = 0u64;
        if is(F_ECALL) {
            match a_val {
                0 => r[l.h0] = Fp::ONE,
                1 => r[l.h1] = Fp::ONE,
                2 => r[l.h2] = Fp::ONE,
                3 => r[l.h3] = Fp::ONE,
                4 => r[l.h4] = Fp::ONE,
                _ => unreachable!("VM accepted an unknown syscall"),
            }
            ecall_out = match rec.witness_in {
                Some(wv) => wv as u64,
                None => b_reg,
            };
        }
        set_word(r, l.eo, ecall_out);
        r[l.jcount] = f(jc[i]);
        // suspend on the last row of a non-halting segment
        if i + 1 == cycles && !rec.halted {
            r[l.suspend] = Fp::ONE;
        }

        // result
        let pc4 = pc + 4;
        let result: u64 = if is(F_LUI) {
            imm_u
        } else if is(F_AUIPC) {
            pt_lo | (pt_hi << 16)
        } else if is(F_JAL) || is(F_JALR) {
            pc4 & 0xffff_ffff
        } else if is(F_LOAD) {
            load_val
        } else if is_alu {
            alu_val
        } else if is(F_ECALL) {
            ecall_out
        } else {
            0
        };
        set_word(r, l.result, result);

        // CPU-side slots
        let rb = REG_BASE;
        let writes_rd = is(F_LUI) || is(F_AUIPC) || is(F_JAL) || is(F_JALR) || is(F_LOAD) || is_alu;
        let addr0 = pc / 4;
        let addr1 = if is(F_ECALL) { rb + 17 } else { rb + rs1 };
        let addr2 = if is(F_ECALL) { rb + 10 } else { rb + rs2 };
        let addr3 = if is_mem { mem_waddr } else { rb };
        let addr4 = if is(F_STORE) {
            mem_waddr
        } else if is(F_ECALL) {
            rb + 10
        } else if writes_rd {
            rb + rd
        } else {
            rb
        };
        let val4: u64 = if is(F_STORE) {
            new_word
        } else if is(F_ECALL) {
            ecall_out
        } else if writes_rd && rd != 0 {
            result
        } else {
            0
        };
        match rec.write {
            (WriteTarget::Reg(rr), v) => {
                debug_assert_eq!(addr4, rb + rr as u64, "rd slot mismatch at cycle {i}");
                debug_assert_eq!(val4, v as u64, "rd value mismatch at cycle {i} instr={instr:#x}");
            }
            (WriteTarget::Mem(wa), v) => {
                debug_assert_eq!(addr4, wa as u64);
                debug_assert_eq!(val4, v as u64, "store value mismatch at cycle {i}");
            }
        }
        if let Some((wa, _)) = rec.mem_read {
            debug_assert_eq!(addr3, wa as u64);
        }
        let addrs = [addr0, addr1, addr2, addr3, addr4];
        set_word(r, l.cpu_val[0], instr as u64);
        set_word(r, l.cpu_val[4], val4);
        for k in 0..NUM_SLOTS {
            r[l.cpu_addr[k]] = f(addrs[k]);
        }
        Fp::batch_inverse(&mut inv_batch);
        r[l.inv_lo] = inv_batch[0];
        r[l.inv_hi] = inv_batch[1];
        r[l.inv_rd] = inv_batch[2];
        r[l.vinv_lo] = inv_batch[3];
        r[l.vinv_hi] = inv_batch[4];
    });
    for i in 0..cycles {
        let r = rows.col(i);
        for k in 0..NUM_SLOTS {
            let v = r[l.cpu_val[k].lo].as_u64() | (r[l.cpu_val[k].hi].as_u64() << 16);
            accesses.push(Access { addr: r[l.cpu_addr[k]].as_u64(), clk: 5 * i as u64 + k as u64 + 1, val: v, w: k == 4, init: false, t: 0, close: false });
        }
    }
    let last_rec = &records[cycles - 1];
    ensure!(last_rec.halted == seg.halted, "segment halted flag mismatch");
    ensure!(last_rec.next_pc == seg.pc_out, "segment pc_out mismatch");
    ensure!(jcount == seg.jcount_out as u64, "segment jcount_out mismatch");

    // padding CPU rows: clk increments, pc increments by 4, everything else zero
    let mut pc = last_rec.next_pc as u64;
    for i in cycles..n {
        let r = rows.col_mut(i);
        r[l.clk] = f(i as u64);
        r[l.pc] = f(pc);
        r[l.pcl14] = f((pc & 0xffff) / 4);
        r[l.pc_hi] = f(pc >> 16);
        // pc + imm (imm = 0) and pc + 4 limbs
        r[l.pt.lo] = f(pc & 0xffff);
        r[l.pt.hi] = f(pc >> 16);
        let t = (pc & 0xffff) + 4;
        r[l.pc4_lo] = f(t & 0xffff);
        r[l.pc4_c] = f(t >> 16);
        r[l.eq_lo] = Fp::ONE;
        r[l.eq_hi] = Fp::ONE;
        r[l.lt_s] = Fp::ONE;
        r[l.rd_is_zero] = Fp::ONE;
        // s = 0: p = 1, q = 2^32 (bit 32), fill = 0; mB = q
        r[l.pow_p.lo] = Fp::ONE;
        r[l.pow_q32] = Fp::ONE;
        r[l.mb_32] = Fp::ONE;
        r[l.jcount] = f(jcount);
        // division gadget on zeros: v = 0 => q = 2^32-1, r = 0
        r[l.vz_lo] = Fp::ONE;
        r[l.vz_hi] = Fp::ONE;
        r[l.vz] = Fp::ONE;
        r[l.dq.lo] = f(0xffff);
        r[l.dq.hi] = f(0xffff);
        r[l.div.lo] = f(0xffff);
        r[l.div.hi] = f(0xffff);
        for k in 0..NUM_SLOTS {
            r[l.cpu_addr[k]] = if k == 0 { f(pc / 4) } else { f(REG_BASE) };
        }
        pc += 4;
        ensure!(pc < (1 << 31), "pc overflow in padding");
    }

    tlog("rows filled");
    // ---- sorted memory table: access groups, then closing entries, then padding
    accesses.par_sort_unstable_by_key(|a| (a.close, a.addr, a.clk));
    tlog("accesses sorted");
    let total_slots = NUM_SLOTS * n;
    ensure!(accesses.len() <= NUM_SLOTS * (n - 1), "memory table overflow");
    let last = *accesses.last().unwrap();
    let mut prev: Option<Access> = None;
    let seg_idx = f(seg.index as u64);
    for idx in 0..total_slots {
        let (row, k) = (idx / NUM_SLOTS, idx % NUM_SLOTS);
        let s = &l.mem[k];
        let (acc, real, pad) = if idx < accesses.len() {
            (accesses[idx], !accesses[idx].init && !accesses[idx].close, false)
        } else {
            // inert padding: new address each, zero value
            (Access { addr: last.addr + 1 + (idx - accesses.len()) as u64, clk: 0, val: 0, w: false, init: false, t: 0, close: false }, false, true)
        };
        let r = rows.col_mut(row);
        r[s.addr] = f(acc.addr);
        r[s.clk] = f(acc.clk);
        r[s.val.lo] = f(acc.val & 0xffff);
        r[s.val.hi] = f(acc.val >> 16);
        r[s.w] = fb(acc.w);
        r[s.init] = fb(acc.init);
        r[s.real] = fb(real);
        r[s.close] = fb(acc.close);
        r[s.t] = f(acc.t);
        r[s.age] = if acc.init { f(seg.index as u64 - acc.t) } else { Fp::ZERO };
        r[l.seg_idx] = seg_idx;
        let (same, diff) = match prev {
            None => (false, 0u64),
            Some(p) => {
                let p_pad = !p.init && !p.close && !(idx - 1 < accesses.len());
                let first_close = acc.close && !p.close;
                let first_pad = pad && !p_pad;
                if first_close || first_pad {
                    (false, 0)
                } else if p.addr == acc.addr {
                    ensure!(acc.clk > p.clk, "memory table clk order violated");
                    ensure!(!acc.init && !acc.close, "duplicate init/closing entry for address {:#x}", acc.addr);
                    (true, acc.clk - p.clk - 1)
                } else {
                    ensure!(acc.addr > p.addr, "memory table addr order violated");
                    ensure!(!real, "access without init entry for address {:#x}", acc.addr);
                    (false, acc.addr - p.addr - 1)
                }
            }
        };
        ensure!(diff < 1 << 30, "memory table diff out of range");
        r[s.same] = fb(same);
        r[s.diff] = f(diff);
        r[s.d_lo] = f(diff & 0xffff);
        r[s.d_hi] = f(diff >> 16);
        prev = Some(acc);
    }
    tlog("memory table filled");

    // ---- next-row copy of slot-0 `same` (0 on the last real row)
    for i in 0..n - 1 {
        let v = rows.col(i + 1)[l.mem[0].same];
        rows.col_mut(i)[l.same_next0] = v;
    }

    // ---- lookup multiplicities (per-chunk histograms, then reduced)
    {
        let chunk = 4096;
        let hists: Vec<Result<(Vec<u32>, Vec<u32>, Vec<u32>)>> = (0..n)
            .into_par_iter()
            .step_by(chunk)
            .map(|start| {
                let mut h16 = vec![0u32; 1 << 16];
                let mut hnib = vec![0u32; 3 * 256];
                let mut hpow = vec![0u32; 64];
                for i in start..(start + chunk).min(n) {
                    let row = rows.col(i);
                    for v in r16_values(l, row) {
                        ensure!(v < 65536, "range-check value {v} out of range (trace bug)");
                        h16[v as usize] += 1;
                    }
                    let bw_op = row[l.f3.start + 6].as_u64() + 2 * row[l.f3.start + 7].as_u64();
                    for k in 0..8 {
                        let a = row[l.nib_a.start + k].as_u64();
                        let b = row[l.nib_b.start + k].as_u64();
                        hnib[(bw_op * 256 + a * 16 + b) as usize] += 1;
                    }
                    hpow[row[l.sh_s].as_u64() as usize] += 1;
                }
                Ok((h16, hnib, hpow))
            })
            .collect();
        let mut m_r16 = vec![0u64; n];
        let mut m_nib = vec![0u64; n];
        let mut m_pow = vec![0u64; n];
        for h in hists {
            let (h16, hnib, hpow) = h?;
            for (i, v) in h16.iter().enumerate() {
                m_r16[i] += *v as u64;
            }
            for (i, v) in hnib.iter().enumerate() {
                m_nib[i] += *v as u64;
            }
            for (i, v) in hpow.iter().enumerate() {
                m_pow[i] += *v as u64;
            }
        }
        for i in 0..n {
            let r = rows.col_mut(i);
            r[l.m_r16] = f(m_r16[i]);
            r[l.m_nib] = f(m_nib[i]);
            r[l.m_pow] = f(m_pow[i]);
        }
    }

    // Transpose rows -> columns in blocks of 64 rows: each block's rows are
    // read once (sequentially) and every column receives a contiguous run.
    // Threads own disjoint row ranges, so the column writes never overlap.
    tlog("multiplicities");
    let mut cols = ColumnStore::pooled(w, full);
    {
        struct ColPtrs(Vec<*mut Fp>);
        unsafe impl Sync for ColPtrs {}
        unsafe impl Send for ColPtrs {}
        let ptrs = ColPtrs(cols.col_ptrs());
        let block = 64;
        debug_assert_eq!(full % block, 0);
        let w16 = w / LANES * LANES;
        (0..full).into_par_iter().step_by(block).for_each(|r0| {
            let ptrs = &ptrs;
            // 16×16 tiles through the in-register transpose: each row read as
            // whole cache lines, each column written as whole cache lines
            for t0 in (r0..r0 + block).step_by(LANES) {
                let rws: [&[Fp]; LANES] = core::array::from_fn(|i| rows.col(t0 + i));
                for c0 in (0..w16).step_by(LANES) {
                    let mut v: [Fp16; LANES] = core::array::from_fn(|i| Fp16::from_slice(&rws[i][c0..]));
                    zkb_field::simd::transpose16(&mut v);
                    for j in 0..LANES {
                        // SAFETY: rows t0..t0+16 of column c0+j are written by exactly one thread.
                        let dst = unsafe { std::slice::from_raw_parts_mut(ptrs.0[c0 + j].add(t0), LANES) };
                        v[j].store_to(dst);
                    }
                }
                for (i, row) in rws.iter().enumerate() {
                    for c in w16..w {
                        unsafe { *ptrs.0[c].add(t0 + i) = row[c] };
                    }
                }
            }
        });
    }
    tlog("transposed");
    Ok(cols)
}

/// The 16-bit range-checked values of a row, in the same order as `lookups::row_terms`.
pub fn r16_values(l: &Layout, row: &[Fp]) -> Vec<u64> {
    crate::lookups::r16_checked::<Fp, &[Fp]>(l, row).into_iter().map(|v| v.as_u64()).collect()
}

pub fn fixed_columns(l: &Layout, n: usize) -> Vec<Vec<Fp>> {
    let mut cols = vec![vec![Fp::ZERO; n]; l.fixed_width];
    for i in 0..n {
        cols[l.t_r16][i] = f((i & 0xffff) as u64);
        if i < 768 {
            let op = (i / 256) as u64;
            let a = ((i / 16) % 16) as u64;
            let b = (i % 16) as u64;
            let c = match op {
                0 => a ^ b,
                1 => a | b,
                _ => a & b,
            };
            cols[l.t_nib_op][i] = f(op);
            cols[l.t_nib_a][i] = f(a);
            cols[l.t_nib_b][i] = f(b);
            cols[l.t_nib_c][i] = f(c);
        }
        // power table: rows >= 32 repeat the s = 0 entry (multiplicity 0)
        let s = if i < 32 { i as u64 } else { 0 };
        let p = 1u64 << s;
        let q = 1u64 << (32 - s);
        let fill = if s == 0 { 0 } else { (TWO32 - q) & 0xffff_ffff };
        cols[l.t_pow_s][i] = f(s);
        cols[l.t_pow_p.lo][i] = f(p & 0xffff);
        cols[l.t_pow_p.hi][i] = f((p >> 16) & 0xffff);
        cols[l.t_pow_q.lo][i] = f(q & 0xffff);
        cols[l.t_pow_q.hi][i] = f((q >> 16) & 0xffff);
        cols[l.t_pow_q32][i] = f(q >> 32);
        cols[l.t_fill.lo][i] = f(fill & 0xffff);
        cols[l.t_fill.hi][i] = f(fill >> 16);
    }
    cols
}

/// LogUp aux columns: running sums of paired fractions. Returns the columns
/// and the per-column final sums (aux public values).
pub fn generate_aux(l: &Layout, main: &[&[Fp]], fixed: &[&[Fp]], ch: &[Fp4], real_rows: usize) -> (Vec<Vec<Fp4>>, Vec<Fp4>) {
    let full = main[0].len();
    let n = real_rows;
    let e = Enc::new(ch[0], ch[1]);
    let num_terms = crate::lookups::num_terms(l);
    let num_aux = num_terms.div_ceil(BATCH);
    let chunk = 256;
    // Per-row sums of the batched fractions, stored row-major
    // (row i, aux a) at fracs[i * num_aux + a].
    let mut fracs = vec![Fp4::ZERO; n * num_aux];
    fracs
        .par_chunks_mut(chunk * num_aux)
        .enumerate()
        .for_each(|(ci, out)| {
            let start = ci * chunk;
            let end = (start + chunk).min(n);
            let rows = end - start;
            let vec_rows = rows / LANES * LANES;
            let mut done = 0;
            if vec_rows > 0 {
                if aux_chunk_vec(l, main, fixed, &e, start, vec_rows, num_terms, num_aux, out) {
                    done = vec_rows;
                }
            }
            if done < rows {
                aux_chunk_scalar(l, main, fixed, &e, start + done, rows - done, num_terms, num_aux, &mut out[done * num_aux..]);
            }
        });
    let mut cols = vec![vec![Fp4::ZERO; full]; num_aux];
    let mut acc = vec![Fp4::ZERO; num_aux];
    for i in 0..n {
        for a in 0..num_aux {
            cols[a][i] = acc[a];
            acc[a] += fracs[i * num_aux + a];
        }
    }
    (cols, acc)
}

/// Row `i` of a set of columns, read in place.
#[derive(Clone, Copy)]
struct RowAt<'a> {
    cols: &'a [&'a [Fp]],
    i: usize,
}
impl RowAccess<Fp> for RowAt<'_> {
    #[inline(always)]
    fn at(&self, c: usize) -> Fp {
        self.cols[c][self.i]
    }
}

/// Rows `i..i+16` of a set of columns as one vector per column, read in place.
#[derive(Clone, Copy)]
struct Rows16At<'a> {
    cols: &'a [&'a [Fp]],
    i: usize,
}
impl RowAccess<Fp16> for Rows16At<'_> {
    #[inline(always)]
    fn at(&self, c: usize) -> Fp16 {
        Fp16::from_slice(&self.cols[c][self.i..])
    }
}

/// Scalar fallback: rows [start, start+rows) → out (row-major, num_aux per row).
#[allow(clippy::too_many_arguments)]
fn aux_chunk_scalar(
    l: &Layout,
    main: &[&[Fp]],
    fixed: &[&[Fp]],
    e: &Enc,
    start: usize,
    rows: usize,
    num_terms: usize,
    num_aux: usize,
    out: &mut [Fp4],
) {
    let mut terms: TermBuf<Fp> = TermBuf::default();
    let mut all_terms: Vec<Term<Fp>> = Vec::with_capacity(rows * num_terms);
    for i in start..start + rows {
        terms.clear();
        row_terms(l, RowAt { cols: main, i }, RowAt { cols: fixed, i }, e, &mut terms);
        all_terms.extend_from_slice(terms.as_slice());
    }
    let mut denoms: Vec<Fp4> = all_terms.iter().map(|t| t.denom).collect();
    Fp4::batch_inverse(&mut denoms);
    for ri in 0..rows {
        let row = &mut out[ri * num_aux..(ri + 1) * num_aux];
        for t in 0..num_terms {
            let idx = ri * num_terms + t;
            row[t / BATCH] += all_terms[idx].mul(denoms[idx]);
        }
    }
}

/// Sixteen rows at a time with the AVX-512 field kernel. `rows` must be a
/// multiple of `LANES`. Returns false (leaving `out` untouched) if a lane-wise
/// batch inversion hit a zero denominator, so the caller can fall back.
#[allow(clippy::too_many_arguments)]
fn aux_chunk_vec(
    l: &Layout,
    main: &[&[Fp]],
    fixed: &[&[Fp]],
    e: &Enc,
    start: usize,
    rows: usize,
    num_terms: usize,
    num_aux: usize,
    out: &mut [Fp4],
) -> bool {
    let blocks = rows / LANES;
    let mut terms: TermBuf<Fp16> = TermBuf::default();
    let mut all_terms: Vec<Term<Fp16>> = Vec::with_capacity(blocks * num_terms);
    for b in 0..blocks {
        let i = start + b * LANES;
        terms.clear();
        row_terms(l, Rows16At { cols: main, i }, Rows16At { cols: fixed, i }, e, &mut terms);
        all_terms.extend_from_slice(terms.as_slice());
    }
    let mut denoms: Vec<Fp4x16> = all_terms.iter().map(|t| t.denom).collect();
    if !Fp4x16::batch_inverse(&mut denoms) {
        return false;
    }
    let mut sums = vec![Fp4x16::ZERO; num_aux];
    for b in 0..blocks {
        for s in sums.iter_mut() {
            *s = Fp4x16::ZERO;
        }
        for t in 0..num_terms {
            let idx = b * num_terms + t;
            sums[t / BATCH] += all_terms[idx].mul(denoms[idx]);
        }
        for a in 0..num_aux {
            let lanes = sums[a].to_lanes();
            for k in 0..LANES {
                out[(b * LANES + k) * num_aux + a] = lanes[k];
            }
        }
    }
    true
}
