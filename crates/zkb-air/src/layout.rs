//! Column layout of the RV32IM AIR over a 31-bit field. Every index is
//! computed by a builder so the trace generator and the constraint
//! evaluator can never disagree.
//!
//! Machine words do not fit a field element, so every 32-bit quantity is a
//! pair of 16-bit limbs `(lo, hi)`; quantities known to be below 2^30
//! (clock, word addresses, the program counter, counters) are single
//! elements. Every limb is range-checked with the 16-bit table (a k-bit
//! limb by checking `limb · 2^(16−k)`).

use std::ops::Range;

pub const NUM_SLOTS: usize = 5;
/// Register file lives at *word* addresses REG_BASE..REG_BASE+32; data memory
/// is word addresses below REG_BASE (byte addresses below 2^31).
pub const REG_BASE: u64 = 1 << 29;

/// Opcode flags (one-hot per active row), in this order.
pub const OPCODES: [u64; 11] = [0x37, 0x17, 0x6f, 0x67, 0x63, 0x03, 0x23, 0x13, 0x33, 0x73, 0x0f];
pub const F_LUI: usize = 0;
pub const F_AUIPC: usize = 1;
pub const F_JAL: usize = 2;
pub const F_JALR: usize = 3;
pub const F_BR: usize = 4;
pub const F_LOAD: usize = 5;
pub const F_STORE: usize = 6;
pub const F_ALUI: usize = 7;
pub const F_ALUR: usize = 8;
pub const F_ECALL: usize = 9;
pub const F_FENCE: usize = 10;

/// Public values of a segment, by index.
pub const PUB_PC_IN: usize = 0;
pub const PUB_PC_OUT: usize = 1;
pub const PUB_JCOUNT_IN: usize = 2;
pub const PUB_JCOUNT_OUT: usize = 3;
pub const PUB_HALTED: usize = 4;
/// Index k of the segment (its writes carry stamp k+1; init entries may only cite stamps ≤ k).
pub const PUB_SEG: usize = 5;
/// Closing-table range of the segment: first and last closed address, and
/// whether the segment closes anything (ranges must be disjoint and
/// increasing across segments, checked by the execution verifier).
pub const PUB_CLOSE_LO: usize = 6;
pub const PUB_CLOSE_HI: usize = 7;
pub const PUB_HAS_CLOSE: usize = 8;
pub const NUM_PUBLIC: usize = 9;

/// Lookup table tags (constant term of the encodings).
pub const TAG_R16: u64 = 1;
pub const TAG_NIB: u64 = 2;
pub const TAG_POW: u64 = 3;
pub const TAG_MEM: u64 = 4;
pub const TAG_INIT: u64 = 5;
pub const TAG_JOURNAL: u64 = 6;

/// A 32-bit word as two 16-bit limb columns.
#[derive(Clone, Copy, Debug)]
pub struct Word {
    pub lo: usize,
    pub hi: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct SlotCols {
    pub addr: usize,
    pub clk: usize,
    pub val: Word,
    pub w: usize,
    pub init: usize,
    pub real: usize,
    pub same: usize,
    pub diff: usize,
    pub d_lo: usize,
    /// 14-bit high limb of the (< 2^30) difference.
    pub d_hi: usize,
    /// Stamp cited by an init entry (segment index + 1 of the last toucher,
    /// 0 = initial memory) or carried by a closing entry.
    pub t: usize,
    /// Closing-table entry: contributes the address's final (value, stamp)
    /// as a read and its initial (0, 0) as a write to the global memory argument.
    pub close: usize,
    /// `init · (k − t)`: the cited stamp's age, range-checked so init entries only cite the past.
    pub age: usize,
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub clk: usize,
    pub active: usize,
    /// pc = 4·pcl14 + pc_hi·2^16 (< 2^31)
    pub pc: usize,
    pub pcl14: usize,
    pub pc_hi: usize,
    pub bits: Range<usize>,
    pub flags: Range<usize>,
    pub f3: Range<usize>,
    pub a_lo16: usize,
    pub a_hi15: usize,
    pub a_sign: usize,
    pub b_lo16: usize,
    pub b_hi15: usize,
    pub b_sign: usize,
    pub use_sub: usize,
    pub add: Word,
    pub add_c0: usize,
    pub add_carry: usize,
    pub eq_lo: usize,
    pub inv_lo: usize,
    pub eq_hi: usize,
    pub inv_hi: usize,
    pub lt_s: usize,
    pub taken: usize,
    /// pc + imm (32-bit wrap) as limbs with its two carries
    pub pt: Word,
    pub pt_c0: usize,
    pub pt_c1: usize,
    /// pc + 4: low limb and carry into the high limb
    pub pc4_lo: usize,
    pub pc4_c: usize,
    pub rd_is_zero: usize,
    pub inv_rd: usize,
    // ---- shared multiplier: mA (bytes = nibble pairs) × mB (4 bytes + bit 32)
    pub sh_s: usize,
    pub b_rest: usize,
    pub pow_p: Word,
    pub pow_q: Word,
    pub pow_q32: usize,
    pub fill: Word,
    pub sra_fill: Word,
    pub nib_a: Range<usize>,
    pub nib_b: Range<usize>,
    pub nib_c: Range<usize>,
    pub mb_byte: Range<usize>,
    pub mb_32: usize,
    /// 64-bit product as four 16-bit limbs and three carries
    pub prod: [usize; 4],
    pub pcarry: [usize; 3],
    // ---- RV32M
    pub is_m: usize,
    pub adj: Word,
    pub adj_c0: usize,
    pub adj_c1: usize,
    pub mh: Word,
    pub mh_b0: usize,
    pub mh_b1: usize,
    pub abs_a: Word,
    pub abs_a_c: usize,
    pub abs_b: Word,
    pub abs_b_c: usize,
    pub dq: Word,
    pub dr: Word,
    pub rlt: Word,
    pub rlt_b: usize,
    pub dcarry: usize,
    pub vz_lo: usize,
    pub vinv_lo: usize,
    pub vz_hi: usize,
    pub vinv_hi: usize,
    pub vz: usize,
    pub neg_q: usize,
    pub div: Word,
    pub div_c: usize,
    pub div_c2: usize,
    pub rem: Word,
    pub rem_c: usize,
    pub rem_c2: usize,
    pub alu: Word,
    // ---- loads / stores
    pub m_byte: Range<usize>,
    pub o0: usize,
    pub o1: usize,
    /// byte address = 4·wl14 + o0 + 2·o1 (low limb), mw_hi15 (high limb, < 2^15)
    pub wl14: usize,
    pub mw_hi15: usize,
    pub sel_byte: usize,
    pub sb_lo7: usize,
    pub sb7: usize,
    pub sel_half: usize,
    pub sh_lo15: usize,
    pub h15: usize,
    pub st_b0: usize,
    pub st_b1: usize,
    pub new_word: Word,
    pub result: Word,
    pub jcount: usize,
    pub h0: usize,
    pub h1: usize,
    pub h2: usize,
    pub h3: usize,
    /// ecall a7 = 4: write a length-prefixed private output buffer (same AIR semantics as 3)
    pub h4: usize,
    pub eo: Word,
    /// Segment suspend flag (continuations): set on the last executed row of a non-final segment.
    pub suspend: usize,
    /// CPU-side memory slot address and value columns.
    pub cpu_addr: [usize; NUM_SLOTS],
    pub cpu_val: [Word; NUM_SLOTS],
    /// Sorted memory table slots.
    pub mem: [SlotCols; NUM_SLOTS],
    /// Copy of the next row's slot-0 `same` flag (0 on the last real row), so
    /// final-state lookup terms never reference the next row.
    pub same_next0: usize,
    /// The segment index k, constant down the column (bound to PUB_SEG on the first row).
    pub seg_idx: usize,
    pub m_r16: usize,
    pub m_nib: usize,
    pub m_pow: usize,
    pub main_width: usize,
    // fixed columns
    pub t_r16: usize,
    pub t_nib_op: usize,
    pub t_nib_a: usize,
    pub t_nib_b: usize,
    pub t_nib_c: usize,
    pub t_pow_s: usize,
    pub t_pow_p: Word,
    pub t_pow_q: Word,
    pub t_pow_q32: usize,
    pub t_fill: Word,
    pub fixed_width: usize,
}

struct B(usize);
impl B {
    fn one(&mut self) -> usize {
        let i = self.0;
        self.0 += 1;
        i
    }
    fn word(&mut self) -> Word {
        Word { lo: self.one(), hi: self.one() }
    }
    fn range(&mut self, n: usize) -> Range<usize> {
        let s = self.0;
        self.0 += n;
        s..self.0
    }
}

impl Layout {
    pub fn new() -> Layout {
        let mut b = B(0);
        let clk = b.one();
        let active = b.one();
        let pc = b.one();
        let pcl14 = b.one();
        let pc_hi = b.one();
        let bits = b.range(32);
        let flags = b.range(11);
        let f3 = b.range(8);
        let a_lo16 = b.one();
        let a_hi15 = b.one();
        let a_sign = b.one();
        let b_lo16 = b.one();
        let b_hi15 = b.one();
        let b_sign = b.one();
        let use_sub = b.one();
        let add = b.word();
        let add_c0 = b.one();
        let add_carry = b.one();
        let eq_lo = b.one();
        let inv_lo = b.one();
        let eq_hi = b.one();
        let inv_hi = b.one();
        let lt_s = b.one();
        let taken = b.one();
        let pt = b.word();
        let pt_c0 = b.one();
        let pt_c1 = b.one();
        let pc4_lo = b.one();
        let pc4_c = b.one();
        let rd_is_zero = b.one();
        let inv_rd = b.one();
        let sh_s = b.one();
        let b_rest = b.one();
        let pow_p = b.word();
        let pow_q = b.word();
        let pow_q32 = b.one();
        let fill = b.word();
        let sra_fill = b.word();
        let nib_a = b.range(8);
        let nib_b = b.range(8);
        let nib_c = b.range(8);
        let mb_byte = b.range(4);
        let mb_32 = b.one();
        let prod = [b.one(), b.one(), b.one(), b.one()];
        let pcarry = [b.one(), b.one(), b.one()];
        let is_m = b.one();
        let adj = b.word();
        let adj_c0 = b.one();
        let adj_c1 = b.one();
        let mh = b.word();
        let mh_b0 = b.one();
        let mh_b1 = b.one();
        let abs_a = b.word();
        let abs_a_c = b.one();
        let abs_b = b.word();
        let abs_b_c = b.one();
        let dq = b.word();
        let dr = b.word();
        let rlt = b.word();
        let rlt_b = b.one();
        let dcarry = b.one();
        let vz_lo = b.one();
        let vinv_lo = b.one();
        let vz_hi = b.one();
        let vinv_hi = b.one();
        let vz = b.one();
        let neg_q = b.one();
        let div = b.word();
        let div_c = b.one();
        let div_c2 = b.one();
        let rem = b.word();
        let rem_c = b.one();
        let rem_c2 = b.one();
        let alu = b.word();
        let m_byte = b.range(4);
        let o0 = b.one();
        let o1 = b.one();
        let wl14 = b.one();
        let mw_hi15 = b.one();
        let sel_byte = b.one();
        let sb_lo7 = b.one();
        let sb7 = b.one();
        let sel_half = b.one();
        let sh_lo15 = b.one();
        let h15 = b.one();
        let st_b0 = b.one();
        let st_b1 = b.one();
        let new_word = b.word();
        let result = b.word();
        let jcount = b.one();
        let h0 = b.one();
        let h1 = b.one();
        let h2 = b.one();
        let h3 = b.one();
        let h4 = b.one();
        let eo = b.word();
        let suspend = b.one();
        let mut cpu_addr = [0; NUM_SLOTS];
        let mut cpu_val = [Word { lo: 0, hi: 0 }; NUM_SLOTS];
        for k in 0..NUM_SLOTS {
            cpu_addr[k] = b.one();
            cpu_val[k] = b.word();
        }
        let zero_slot = SlotCols { addr: 0, clk: 0, val: Word { lo: 0, hi: 0 }, w: 0, init: 0, real: 0, same: 0, diff: 0, d_lo: 0, d_hi: 0, t: 0, close: 0, age: 0 };
        let mut mem = [zero_slot; NUM_SLOTS];
        for s in mem.iter_mut() {
            *s = SlotCols {
                addr: b.one(),
                clk: b.one(),
                val: b.word(),
                w: b.one(),
                init: b.one(),
                real: b.one(),
                same: b.one(),
                diff: b.one(),
                d_lo: b.one(),
                d_hi: b.one(),
                t: b.one(),
                close: b.one(),
                age: b.one(),
            };
        }
        let same_next0 = b.one();
        let seg_idx = b.one();
        let m_r16 = b.one();
        let m_nib = b.one();
        let m_pow = b.one();
        let main_width = b.0;
        let mut f = B(0);
        let t_r16 = f.one();
        let t_nib_op = f.one();
        let t_nib_a = f.one();
        let t_nib_b = f.one();
        let t_nib_c = f.one();
        let t_pow_s = f.one();
        let t_pow_p = f.word();
        let t_pow_q = f.word();
        let t_pow_q32 = f.one();
        let t_fill = f.word();
        let fixed_width = f.0;
        Layout {
            clk, active, pc, pcl14, pc_hi, bits, flags, f3, a_lo16, a_hi15, a_sign, b_lo16, b_hi15, b_sign, use_sub, add, add_c0, add_carry,
            eq_lo, inv_lo, eq_hi, inv_hi, lt_s, taken, pt, pt_c0, pt_c1, pc4_lo, pc4_c, rd_is_zero, inv_rd, sh_s, b_rest, pow_p, pow_q, pow_q32,
            fill, sra_fill, nib_a, nib_b, nib_c, mb_byte, mb_32, prod, pcarry, is_m, adj, adj_c0, adj_c1, mh, mh_b0, mh_b1, abs_a, abs_a_c, abs_b,
            abs_b_c, dq, dr, rlt, rlt_b, dcarry, vz_lo, vinv_lo, vz_hi, vinv_hi, vz, neg_q, div, div_c, div_c2, rem, rem_c, rem_c2, alu, m_byte,
            o0, o1, wl14, mw_hi15, sel_byte, sb_lo7, sb7, sel_half, sh_lo15, h15, st_b0, st_b1, new_word, result, jcount, h0, h1, h2, h3, h4, eo,
            suspend, cpu_addr, cpu_val, mem, same_next0, seg_idx, m_r16, m_nib, m_pow, main_width, t_r16, t_nib_op, t_nib_a, t_nib_b, t_nib_c, t_pow_s,
            t_pow_p, t_pow_q, t_pow_q32, t_fill, fixed_width,
        }
    }
}

impl Default for Layout {
    fn default() -> Self {
        Layout::new()
    }
}
