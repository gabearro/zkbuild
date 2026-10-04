//! LogUp lookup terms. This is the single source of truth shared by the
//! constraint evaluator (over generic field rows) and the aux-column
//! generator (concrete values), so the two can never disagree.
//!
//! Every term is a fraction `mult / (γ - enc)`. Terms are batched into aux
//! columns in order; the first [`AUX_CHAIN`] columns hold the segment's
//! contribution to the execution-wide memory argument (offline memory
//! checking across segments: +init reads, +closing reads, −final writes,
//! −closing initial writes, all stamped), and the rest the balance of all
//! other lookups (which must equal the public journal contribution).

use crate::layout::*;
use std::ops::Range;
use zkb_stark::RowAccess;
use zkb_field::{ExtLike, FieldLike, Fp, Fp4};

/// Terms per aux column (batched fractions). With multiplicities of degree
/// ≤ 2 and linear denominators the batched constraint has degree 4.
pub const BATCH: usize = 3;
/// Aux columns holding the memory-chain terms (4 per slot = 20 terms + 1 pad = 7 columns).
pub const AUX_CHAIN: Range<usize> = 0..7;

#[derive(Clone, Copy, Debug)]
pub struct Term<F: FieldLike> {
    pub mult: F::Ext,
    /// The multiplicity as a base-field value when the row is base-field
    /// (prover side): lets the batched constraint use `Ext × F` products.
    pub mult_base: Option<F>,
    /// γ - enc
    pub denom: F::Ext,
}

impl<F: FieldLike> Term<F> {
    #[inline(always)]
    pub fn new(mult: F, denom: F::Ext) -> Term<F> {
        Term { mult: mult.to_ext(), mult_base: mult.as_base(), denom }
    }
    #[inline(always)]
    pub fn zero() -> Term<F> {
        Term { mult: F::Ext::ZERO, mult_base: Some(F::ZERO), denom: F::Ext::ONE }
    }
    /// mult · x
    #[inline(always)]
    pub fn mul(&self, x: F::Ext) -> F::Ext {
        match self.mult_base {
            Some(b) => x.mul_base(b),
            None => self.mult * x,
        }
    }
}

/// Fixed-capacity, stack-allocated term buffer (no per-row heap allocation).
pub const MAX_TERMS: usize = 128;
pub struct TermBuf<F: FieldLike> {
    pub items: [Term<F>; MAX_TERMS],
    pub len: usize,
}
impl<F: FieldLike> Default for TermBuf<F> {
    fn default() -> Self {
        TermBuf { items: [Term::zero(); MAX_TERMS], len: 0 }
    }
}
impl<F: FieldLike> TermBuf<F> {
    #[inline(always)]
    pub fn push(&mut self, t: Term<F>) {
        self.items[self.len] = t;
        self.len += 1;
    }
    #[inline(always)]
    pub fn clear(&mut self) {
        self.len = 0;
    }
    #[inline(always)]
    pub fn as_slice(&self) -> &[Term<F>] {
        &self.items[..self.len]
    }
}

/// Fingerprint encodings: `γ − (tag + Σ βⁱ·fieldᵢ)` with β, γ ∈ F_{p^4}.
pub struct Enc {
    /// β^1 .. β^8
    pub beta: [Fp4; 8],
    pub gamma: Fp4,
}

impl Enc {
    pub fn new(beta: Fp4, gamma: Fp4) -> Enc {
        let mut b = [beta; 8];
        for i in 1..8 {
            b[i] = b[i - 1] * beta;
        }
        Enc { beta: b, gamma }
    }
    #[inline(always)]
    fn tag<F: FieldLike>(t: u64) -> F::Ext {
        F::Ext::from(Fp4::from_base(Fp::new(t)))
    }
    #[inline(always)]
    fn fold<F: FieldLike>(&self, tag: u64, fields: &[F]) -> F::Ext {
        let mut acc = Self::tag::<F>(tag);
        for (i, v) in fields.iter().enumerate() {
            acc += v.mul_ext(self.beta[i]);
        }
        F::Ext::from(self.gamma) - acc
    }
    #[inline]
    pub fn r16<F: FieldLike>(&self, v: F) -> F::Ext {
        self.fold(TAG_R16, &[v])
    }
    #[inline]
    pub fn nib<F: FieldLike>(&self, op: F, a: F, b: F, c: F) -> F::Ext {
        self.fold(TAG_NIB, &[op, a, b, c])
    }
    /// Power-table entry: (s, p = 2^s, q = 2^(32−s) with its bit 32, fill = 2^32 − q).
    #[inline]
    pub fn pow<F: FieldLike>(&self, s: F, p_lo: F, p_hi: F, q_lo: F, q_hi: F, q32: F, fill_lo: F, fill_hi: F) -> F::Ext {
        self.fold(TAG_POW, &[s, p_lo, p_hi, q_lo, q_hi, q32, fill_lo, fill_hi])
    }
    #[inline]
    pub fn mem<F: FieldLike>(&self, addr: F, clk: F, val_lo: F, val_hi: F, w: F) -> F::Ext {
        self.fold(TAG_MEM, &[addr, clk, val_lo, val_hi, w])
    }
    /// Stamped memory-state entry `(addr, value, stamp)` of the execution-wide
    /// memory argument (reads and writes use the same encoding so they cancel).
    #[inline]
    pub fn state<F: FieldLike>(&self, addr: F, val_lo: F, val_hi: F, t: F) -> F::Ext {
        self.fold(TAG_INIT, &[addr, val_lo, val_hi, t])
    }
    #[inline]
    pub fn journal<F: FieldLike>(&self, idx: F, val_lo: F, val_hi: F) -> F::Ext {
        self.fold(TAG_JOURNAL, &[idx, val_lo, val_hi])
    }
}

/// Emit all lookup terms of one row, in a fixed order, with the execution's
/// global challenges `e`.
pub fn row_terms<F: FieldLike, M: RowAccess<F>>(l: &Layout, m: M, f: M, e: &Enc, out: &mut TermBuf<F>) {
    row_terms_with(l, m, f, e, |mult, denom| out.push(Term::new(mult, denom)));
}

/// [`row_terms`] streamed to a callback `(multiplicity, denominator)`, so a
/// consumer can batch terms on the fly without materializing a buffer.
pub fn row_terms_with<F: FieldLike, M: RowAccess<F>>(l: &Layout, m: M, f: M, e: &Enc, mut push: impl FnMut(F, F::Ext)) {
    let c = |v: u64| F::from_u64(v);
    let one = F::ONE;

    // ---- memory chain (offline memory checking across segments), 4 terms per slot + 1 pad:
    //   +init:  read  (addr, value at segment start, cited stamp t ≤ k)
    //   +close: read  (addr, final value, stamp of the last toucher)
    //   −last entry of an address group (real or init): write (addr, value, k+1)
    //   −close: write (addr, 0, 0)   (the address's initial state; image words corrected by the verifier)
    let stamp_out = m.at(l.seg_idx) + one;
    for k in 0..NUM_SLOTS {
        let s = &l.mem[k];
        let same_next = if k + 1 < NUM_SLOTS { m.at(l.mem[k + 1].same) } else { m.at(l.same_next0) };
        push(m.at(s.init), e.state(m.at(s.addr), m.at(s.val.lo), m.at(s.val.hi), m.at(s.t)));
        push(m.at(s.close), e.state(m.at(s.addr), m.at(s.val.lo), m.at(s.val.hi), m.at(s.t)));
        push(-(one - same_next) * (m.at(s.real) + m.at(s.init)), e.state(m.at(s.addr), m.at(s.val.lo), m.at(s.val.hi), stamp_out));
        push(-m.at(s.close), e.state(m.at(s.addr), F::ZERO, F::ZERO, F::ZERO));
    }
    push(F::ZERO, F::Ext::ONE);

    // ---- 16-bit range checks, multiplicity 1 (same order as `trace::r16_values`)
    for v in r16_checked(l, m) {
        push(one, e.r16(v));
    }
    // ---- nibble bitwise table: op = f3[6] + 2 f3[7]  (xor=0, or=1, and=2)
    let bw_op = m.at(l.f3.start + 6) + m.at(l.f3.start + 7) * c(2);
    for i in 0..8 {
        push(one, e.nib(bw_op, m.at(l.nib_a.start + i), m.at(l.nib_b.start + i), m.at(l.nib_c.start + i)));
    }
    // ---- power table
    push(one, e.pow(m.at(l.sh_s), m.at(l.pow_p.lo), m.at(l.pow_p.hi), m.at(l.pow_q.lo), m.at(l.pow_q.hi), m.at(l.pow_q32), m.at(l.fill.lo), m.at(l.fill.hi)));
    // ---- journal writes: multiplicity f_ecall * h2, entry (jcount, a0 = cpu_val[2])
    push(m.at(l.flags.start + F_ECALL) * m.at(l.h2), e.journal(m.at(l.jcount), m.at(l.cpu_val[2].lo), m.at(l.cpu_val[2].hi)));
    // ---- CPU memory accesses (+active)
    let clk5 = m.at(l.clk) * c(5);
    for k in 0..NUM_SLOTS {
        let w = if k == NUM_SLOTS - 1 { F::ONE } else { F::ZERO };
        push(m.at(l.active), e.mem(m.at(l.cpu_addr[k]), clk5 + c(k as u64 + 1), m.at(l.cpu_val[k].lo), m.at(l.cpu_val[k].hi), w));
    }
    // ---- memory table rows (-real)
    for s in &l.mem {
        push(-m.at(s.real), e.mem(m.at(s.addr), m.at(s.clk), m.at(s.val.lo), m.at(s.val.hi), m.at(s.w)));
    }
    // ---- table multiplicities (-m)
    push(-m.at(l.m_r16), e.r16(f.at(l.t_r16)));
    push(-m.at(l.m_nib), e.nib(f.at(l.t_nib_op), f.at(l.t_nib_a), f.at(l.t_nib_b), f.at(l.t_nib_c)));
    push(-m.at(l.m_pow), e.pow(f.at(l.t_pow_s), f.at(l.t_pow_p.lo), f.at(l.t_pow_p.hi), f.at(l.t_pow_q.lo), f.at(l.t_pow_q.hi), f.at(l.t_pow_q32), f.at(l.t_fill.lo), f.at(l.t_fill.hi)));
}

/// The values a row range-checks against the 16-bit table, as row expressions.
/// A k-bit limb is checked as `limb · 2^(16−k)`.
pub fn r16_checked<F: FieldLike, M: RowAccess<F>>(l: &Layout, m: M) -> Vec<F> {
    let c = |v: u64| F::from_u64(v);
    let mut v = vec![
        m.at(l.a_lo16),
        m.at(l.a_hi15),
        m.at(l.a_hi15) * c(2),
        m.at(l.b_lo16),
        m.at(l.b_hi15),
        m.at(l.b_hi15) * c(2),
        m.at(l.add.lo),
        m.at(l.add.hi),
        m.at(l.pt.lo),
        m.at(l.pt.hi),
        m.at(l.pc4_lo),
        m.at(l.pcl14) * c(4),
        m.at(l.pc_hi) * c(2),
        m.at(l.sh_s) * c(2048),
        m.at(l.b_rest) * c(32),
        m.at(l.mb_byte.start) * c(256),
        m.at(l.mb_byte.start + 1) * c(256),
        m.at(l.mb_byte.start + 2) * c(256),
        m.at(l.mb_byte.start + 3) * c(256),
        m.at(l.prod[0]),
        m.at(l.prod[1]),
        m.at(l.prod[2]),
        m.at(l.prod[3]),
        m.at(l.pcarry[0]) * c(32),
        m.at(l.pcarry[1]) * c(32),
        m.at(l.pcarry[2]) * c(32),
        m.at(l.adj.lo),
        m.at(l.adj.hi),
        m.at(l.mh.lo),
        m.at(l.mh.hi),
        m.at(l.abs_a.lo),
        m.at(l.abs_a.hi),
        m.at(l.abs_b.lo),
        m.at(l.abs_b.hi),
        m.at(l.dq.lo),
        m.at(l.dq.hi),
        m.at(l.dr.lo),
        m.at(l.dr.hi),
        m.at(l.rlt.lo),
        m.at(l.rlt.hi),
        m.at(l.div.lo),
        m.at(l.div.hi),
        m.at(l.rem.lo),
        m.at(l.rem.hi),
        m.at(l.m_byte.start) * c(256),
        m.at(l.m_byte.start + 1) * c(256),
        m.at(l.m_byte.start + 2) * c(256),
        m.at(l.m_byte.start + 3) * c(256),
        m.at(l.sb_lo7) * c(512),
        m.at(l.sh_lo15) * c(2),
        m.at(l.wl14) * c(4),
        m.at(l.mw_hi15) * c(2),
        m.at(l.st_b0) * c(256),
        m.at(l.st_b1) * c(256),
        m.at(l.eo.lo),
        m.at(l.eo.hi),
    ];
    for s in &l.mem {
        v.push(m.at(s.d_lo));
        v.push(m.at(s.d_hi) * c(4));
        // an init entry may only cite a stamp of an earlier segment: age = init·(k − t) ∈ [0, 2^16)
        v.push(m.at(s.age));
    }
    v
}

pub fn num_terms(l: &Layout) -> usize {
    let m = vec![Fp::ZERO; l.main_width];
    let f = vec![Fp::ZERO; l.fixed_width];
    let e = Enc::new(Fp4::ONE, Fp4::ONE);
    let mut out: TermBuf<Fp> = TermBuf::default();
    row_terms(l, m.as_slice(), f.as_slice(), &e, &mut out);
    out.len
}

/// Number of Fp4 aux columns: `BATCH` fractions per column.
pub fn num_aux(l: &Layout) -> usize {
    num_terms(l).div_ceil(BATCH)
}
