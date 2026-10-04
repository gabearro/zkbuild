//! RV32I guest virtual machine with a deterministic, hermetic syscall layer
//! (the "ZK build OS"). The guest has no clock, no hostname, no filesystem,
//! no network and no threads: its only I/O is a private witness stream and a
//! public journal.
//!
//! Every executed cycle is recorded in enough detail for the AIR trace
//! generator (`zkb-air`) to reproduce all memory accesses deterministically.

pub mod elf;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Syscall numbers (in `a7`).
pub const SYS_HALT: u32 = 0;
pub const SYS_READ_WITNESS: u32 = 1;
pub const SYS_WRITE_JOURNAL: u32 = 2;
/// Private output stream: the host records it, the AIR does not constrain it.
pub const SYS_WRITE_OUTPUT: u32 = 3;
/// a0 = byte address (word-aligned) of `[byte_len: u32, data words...]`; appends the
/// same `[len][words]` frame to the private output as `len/4` SYS_WRITE_OUTPUT calls would.
pub const SYS_WRITE_OUTPUT_BUF: u32 = 4;

pub const REG_A0: usize = 10;
pub const REG_A7: usize = 17;

/// A loaded program: a sparse word-addressed memory image plus an entry point.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Image {
    pub entry: u32,
    /// (word address, value) for every nonzero initial word, sorted by address.
    pub words: Vec<(u32, u32)>,
}

impl Image {
    pub fn from_bytes(entry: u32, segments: &[(u32, Vec<u8>)]) -> Result<Image> {
        let mut map: BTreeMap<u32, u32> = BTreeMap::new();
        for (base, bytes) in segments {
            if base % 4 != 0 {
                bail!("segment base {base:#x} not word aligned");
            }
            for (i, b) in bytes.iter().enumerate() {
                let addr = *base as u64 + i as u64;
                if addr >= ADDR_LIMIT as u64 {
                    bail!("segment exceeds the 2^31-byte address space");
                }
                let w = (addr / 4) as u32;
                let shift = ((addr % 4) * 8) as u32;
                let cur = map.entry(w).or_insert(0);
                *cur |= (*b as u32) << shift;
            }
        }
        let words = map.into_iter().filter(|(_, v)| *v != 0).collect();
        Ok(Image { entry, words })
    }

    /// Canonical digest of the image; this is what a certificate's
    /// `toolchain_root` ultimately commits to for the guest program.
    pub fn digest(&self) -> [u8; 32] {
        let mut h = blake3::Hasher::new();
        h.update(b"zkb-image-v0");
        h.update(&self.entry.to_le_bytes());
        h.update(&(self.words.len() as u64).to_le_bytes());
        for (a, v) in &self.words {
            h.update(&a.to_le_bytes());
            h.update(&v.to_le_bytes());
        }
        *h.finalize().as_bytes()
    }
}

/// What the CPU wrote in a cycle's write slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteTarget {
    /// Register index (0 means "x0", value is forced to 0).
    Reg(u8),
    /// Word address in memory.
    Mem(u32),
}

/// Full record of one executed instruction.
#[derive(Clone, Copy, Debug)]
pub struct CycleRecord {
    pub pc: u32,
    pub instr: u32,
    /// Register read in slot 1 (rs1, or a7 for ecall) and its value.
    pub r1: (u8, u32),
    /// Register read in slot 2 (rs2 field, or a0 for ecall) and its value.
    pub r2: (u8, u32),
    /// Memory read in slot 3: word address and value (dummy read of x0 => None).
    pub mem_read: Option<(u32, u32)>,
    /// Write in slot 4.
    pub write: (WriteTarget, u32),
    /// Witness word consumed by SYS_READ_WITNESS (unconstrained in the AIR).
    pub witness_in: Option<u32>,
    pub next_pc: u32,
    pub halted: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trap {
    IllegalInstruction(u32, u32),
    Misaligned(u32),
    /// Byte address at or above 2^31 (outside the provable address space).
    AddressOutOfRange(u32),
    UnknownSyscall(u32),
    WitnessExhausted,
    CycleLimit,
    PcOverflow,
}

impl std::fmt::Display for Trap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Trap::IllegalInstruction(pc, i) => write!(f, "illegal instruction {i:#010x} at pc={pc:#x}"),
            Trap::Misaligned(a) => write!(f, "misaligned access at {a:#x}"),
            Trap::AddressOutOfRange(a) => write!(f, "address {a:#x} outside the 2^31-byte address space"),
            Trap::UnknownSyscall(n) => write!(f, "unknown syscall {n}"),
            Trap::WitnessExhausted => write!(f, "witness stream exhausted"),
            Trap::CycleLimit => write!(f, "cycle limit reached"),
            Trap::PcOverflow => write!(f, "pc overflow"),
        }
    }
}
impl std::error::Error for Trap {}

/// Sparse flat memory: 2^29 words addressed through a page table of
/// 4096-word (16 KB) pages allocated on first write. Reads of unallocated
/// pages return 0, matching the zkVM's "untouched memory is zero" rule.
pub struct Memory {
    pages: Vec<Option<Box<[u32; PAGE_WORDS]>>>,
}
const PAGE_BITS: u32 = 12;
const PAGE_WORDS: usize = 1 << PAGE_BITS;
impl Memory {
    pub fn new() -> Memory {
        Memory { pages: vec![None; (1usize << 29) >> PAGE_BITS] }
    }
    #[inline(always)]
    pub fn get(&self, waddr: u32) -> u32 {
        match &self.pages[(waddr >> PAGE_BITS) as usize] {
            Some(p) => p[(waddr as usize) & (PAGE_WORDS - 1)],
            None => 0,
        }
    }
    #[inline(always)]
    pub fn set(&mut self, waddr: u32, v: u32) {
        let slot = &mut self.pages[(waddr >> PAGE_BITS) as usize];
        if slot.is_none() {
            if v == 0 {
                return;
            }
            *slot = Some(Box::new([0u32; PAGE_WORDS]));
        }
        slot.as_mut().unwrap()[(waddr as usize) & (PAGE_WORDS - 1)] = v;
    }
    /// All nonzero words in address order.
    pub fn iter_nonzero(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.pages.iter().enumerate().filter_map(|(pi, p)| p.as_ref().map(|p| (pi, p))).flat_map(|(pi, p)| {
            let base = (pi as u32) << PAGE_BITS;
            p.iter().enumerate().filter(|(_, &v)| v != 0).map(move |(i, &v)| (base + i as u32, v))
        })
    }
}
impl Default for Memory {
    fn default() -> Self {
        Memory::new()
    }
}

pub struct Vm {
    pub regs: [u32; 32],
    pub pc: u32,
    pub mem: Memory,
    /// Per word: stamp (segment index + 1) of the last segment that touched it, 0 = never.
    pub stamps: Memory,
    pub reg_stamps: [u32; 32],
    /// Stamp of the segment being executed (see [`Vm::begin_segment`]).
    pub seg_stamp: u32,
    /// Addresses first touched in the current segment, in touch order:
    /// (field address, value before the segment, previous stamp). These are
    /// the segment's memory-table init entries.
    pub touched: Vec<(u64, u32, u32)>,
    pub witness: Vec<u32>,
    pub witness_pos: usize,
    pub journal: Vec<u32>,
    pub output: Vec<u32>,
    pub cycles: u64,
    pub halted: bool,
    pub exit_code: u32,
}

#[derive(Clone, Debug, Default)]
pub struct ExecutionResult {
    pub journal: Vec<u32>,
    pub output: Vec<u32>,
    pub exit_code: u32,
    pub cycles: u64,
    pub records: Vec<CycleRecord>,
}

#[inline]
fn sext(v: u32, bits: u32) -> u32 {
    let shift = 32 - bits;
    (((v << shift) as i32) >> shift) as u32
}

impl Vm {
    pub fn new(image: &Image, witness: Vec<u32>) -> Vm {
        let mut mem = Memory::new();
        for &(a, v) in &image.words {
            mem.set(a, v);
        }
        Vm { regs: [0; 32], pc: image.entry, mem, stamps: Memory::new(), reg_stamps: [0; 32], seg_stamp: 1, touched: Vec::new(), witness, witness_pos: 0, journal: Vec::new(), output: Vec::new(), cycles: 0, halted: false, exit_code: 0 }
    }

    #[inline]
    pub fn load_word(&self, waddr: u32) -> u32 {
        self.mem.get(waddr)
    }
    #[inline]
    fn store_word(&mut self, waddr: u32, v: u32) {
        self.mem.set(waddr, v);
    }

    /// Execute one instruction and return its record.
    pub fn step(&mut self) -> Result<CycleRecord, Trap> {
        let pc = self.pc;
        if pc % 4 != 0 {
            return Err(Trap::Misaligned(pc));
        }
        if pc >= ADDR_LIMIT {
            return Err(Trap::AddressOutOfRange(pc));
        }
        let instr = self.load_word(pc / 4);
        let opcode = instr & 0x7f;
        let rd = ((instr >> 7) & 0x1f) as u8;
        let f3 = (instr >> 12) & 7;
        let rs1 = ((instr >> 15) & 0x1f) as u8;
        let rs2 = ((instr >> 20) & 0x1f) as u8;
        let f7 = instr >> 25;
        let imm_i = sext(instr >> 20, 12);
        let imm_s = sext(((instr >> 25) << 5) | ((instr >> 7) & 0x1f), 12);
        let imm_b = sext(((instr >> 31) << 12) | (((instr >> 7) & 1) << 11) | (((instr >> 25) & 0x3f) << 5) | (((instr >> 8) & 0xf) << 1), 13);
        let imm_u = instr & 0xffff_f000;
        let imm_j = sext(((instr >> 31) << 20) | (((instr >> 12) & 0xff) << 12) | (((instr >> 20) & 1) << 11) | (((instr >> 21) & 0x3ff) << 1), 21);

        let a = self.regs[rs1 as usize];
        let b = self.regs[rs2 as usize];
        let mut rec = CycleRecord {
            pc,
            instr,
            r1: (rs1, a),
            r2: (rs2, b),
            mem_read: None,
            write: (WriteTarget::Reg(0), 0),
            witness_in: None,
            next_pc: pc.wrapping_add(4),
            halted: false,
        };
        let illegal = Err(Trap::IllegalInstruction(pc, instr));
        let set_rd = |rec: &mut CycleRecord, v: u32| {
            rec.write = (WriteTarget::Reg(rd), if rd == 0 { 0 } else { v });
        };
        match opcode {
            0x37 => set_rd(&mut rec, imm_u),                       // LUI
            0x17 => set_rd(&mut rec, pc.wrapping_add(imm_u)),       // AUIPC
            0x6f => {
                // JAL
                set_rd(&mut rec, pc.wrapping_add(4));
                rec.next_pc = pc.wrapping_add(imm_j);
            }
            0x67 => {
                if f3 != 0 {
                    return illegal;
                }
                set_rd(&mut rec, pc.wrapping_add(4));
                rec.next_pc = a.wrapping_add(imm_i); // misaligned targets trap at fetch (AIR-compatible)
            }
            0x63 => {
                let taken = match f3 {
                    0 => a == b,
                    1 => a != b,
                    4 => (a as i32) < (b as i32),
                    5 => (a as i32) >= (b as i32),
                    6 => a < b,
                    7 => a >= b,
                    _ => return illegal,
                };
                if taken {
                    rec.next_pc = pc.wrapping_add(imm_b);
                }
            }
            0x03 => {
                let addr = a.wrapping_add(imm_i);
                if addr >= ADDR_LIMIT {
                    return Err(Trap::AddressOutOfRange(addr));
                }
                let waddr = addr / 4;
                let off = addr % 4;
                let word = self.load_word(waddr);
                rec.mem_read = Some((waddr, word));
                let v = match f3 {
                    0 => sext(word >> (off * 8), 8),
                    1 => {
                        if off % 2 != 0 {
                            return Err(Trap::Misaligned(addr));
                        }
                        sext(word >> (off * 8), 16)
                    }
                    2 => {
                        if off != 0 {
                            return Err(Trap::Misaligned(addr));
                        }
                        word
                    }
                    4 => (word >> (off * 8)) & 0xff,
                    5 => {
                        if off % 2 != 0 {
                            return Err(Trap::Misaligned(addr));
                        }
                        (word >> (off * 8)) & 0xffff
                    }
                    _ => return illegal,
                };
                set_rd(&mut rec, v);
            }
            0x23 => {
                let addr = a.wrapping_add(imm_s);
                if addr >= ADDR_LIMIT {
                    return Err(Trap::AddressOutOfRange(addr));
                }
                let waddr = addr / 4;
                let off = addr % 4;
                let old = self.load_word(waddr);
                let new = match f3 {
                    0 => (old & !(0xff << (off * 8))) | ((b & 0xff) << (off * 8)),
                    1 => {
                        if off % 2 != 0 {
                            return Err(Trap::Misaligned(addr));
                        }
                        (old & !(0xffff << (off * 8))) | ((b & 0xffff) << (off * 8))
                    }
                    2 => {
                        if off != 0 {
                            return Err(Trap::Misaligned(addr));
                        }
                        b
                    }
                    _ => return illegal,
                };
                rec.mem_read = Some((waddr, old));
                rec.write = (WriteTarget::Mem(waddr), new);
            }
            0x33 if f7 == 1 => {
                // RV32M
                let (sa, sb) = (a as i32 as i64, b as i32 as i64);
                let v = match f3 {
                    0 => a.wrapping_mul(b),
                    1 => ((sa * sb) >> 32) as u32,
                    2 => ((sa * (b as i64)) >> 32) as u32,
                    3 => (((a as u64) * (b as u64)) >> 32) as u32,
                    4 => {
                        if b == 0 {
                            u32::MAX
                        } else if a == 0x8000_0000 && b == u32::MAX {
                            0x8000_0000
                        } else {
                            ((a as i32) / (b as i32)) as u32
                        }
                    }
                    5 => {
                        if b == 0 {
                            u32::MAX
                        } else {
                            a / b
                        }
                    }
                    6 => {
                        if b == 0 {
                            a
                        } else if a == 0x8000_0000 && b == u32::MAX {
                            0
                        } else {
                            ((a as i32) % (b as i32)) as u32
                        }
                    }
                    7 => {
                        if b == 0 {
                            a
                        } else {
                            a % b
                        }
                    }
                    _ => unreachable!(),
                };
                set_rd(&mut rec, v);
            }
            0x13 | 0x33 => {
                let is_reg = opcode == 0x33;
                let bb = if is_reg { b } else { imm_i };
                let sh = bb & 31;
                let v = match f3 {
                    0 => {
                        if is_reg && f7 == 0x20 {
                            a.wrapping_sub(bb)
                        } else if !is_reg || f7 == 0 {
                            a.wrapping_add(bb)
                        } else {
                            return illegal;
                        }
                    }
                    1 => {
                        if f7 != 0 {
                            return illegal;
                        }
                        a << sh
                    }
                    2 => ((a as i32) < (bb as i32)) as u32,
                    3 => (a < bb) as u32,
                    4 => a ^ bb,
                    5 => {
                        let f7m = if is_reg { f7 } else { (instr >> 25) & 0x7f };
                        if f7m == 0x20 {
                            ((a as i32) >> sh) as u32
                        } else if f7m == 0 {
                            a >> sh
                        } else {
                            return illegal;
                        }
                    }
                    6 => a | bb,
                    7 => a & bb,
                    _ => unreachable!(),
                };
                // strict encoding check so the VM accepts exactly what the AIR proves
                if is_reg {
                    let ok = f7 == 0 || (f7 == 0x20 && (f3 == 0 || f3 == 5));
                    if !ok {
                        return illegal;
                    }
                }
                set_rd(&mut rec, v);
            }
            0x0f => {
                // FENCE: no-op
            }
            0x73 => {
                if instr != 0x0000_0073 {
                    return illegal; // only ECALL; no EBREAK/CSR
                }
                let a7 = self.regs[REG_A7];
                let a0 = self.regs[REG_A0];
                rec.r1 = (REG_A7 as u8, a7);
                rec.r2 = (REG_A0 as u8, a0);
                match a7 {
                    SYS_HALT => {
                        rec.halted = true;
                        rec.write = (WriteTarget::Reg(REG_A0 as u8), a0);
                        self.halted = true;
                        self.exit_code = a0;
                    }
                    SYS_READ_WITNESS => {
                        if self.witness_pos >= self.witness.len() {
                            return Err(Trap::WitnessExhausted);
                        }
                        let w = self.witness[self.witness_pos];
                        self.witness_pos += 1;
                        rec.witness_in = Some(w);
                        rec.write = (WriteTarget::Reg(REG_A0 as u8), w);
                    }
                    SYS_WRITE_JOURNAL => {
                        self.journal.push(a0);
                        rec.write = (WriteTarget::Reg(REG_A0 as u8), a0);
                    }
                    SYS_WRITE_OUTPUT => {
                        self.output.push(a0);
                        rec.write = (WriteTarget::Reg(REG_A0 as u8), a0);
                    }
                    SYS_WRITE_OUTPUT_BUF => {
                        if a0 % 4 != 0 {
                            return Err(Trap::Misaligned(a0));
                        }
                        let n = self.load_word(a0 / 4);
                        let words = n.div_ceil(4);
                        let end = (a0 / 4).checked_add(1 + words).filter(|&e| e <= ADDR_LIMIT / 4);
                        let Some(end) = end else { return Err(Trap::AddressOutOfRange(a0)) };
                        self.output.push(n);
                        for w in a0 / 4 + 1..end {
                            self.output.push(self.load_word(w));
                        }
                        rec.write = (WriteTarget::Reg(REG_A0 as u8), a0);
                    }
                    n => return Err(Trap::UnknownSyscall(n)),
                }
            }
            _ => return illegal,
        }
        // first-touch bookkeeping (values before this cycle's write), then commit
        for addr in slot_addrs(&rec) {
            self.touch(addr);
        }
        match rec.write {
            (WriteTarget::Reg(r), v) => {
                if r != 0 {
                    self.regs[r as usize] = v;
                }
            }
            (WriteTarget::Mem(waddr), v) => self.store_word(waddr, v),
        }
        if rec.next_pc >= ADDR_LIMIT {
            return Err(Trap::PcOverflow);
        }
        self.pc = rec.next_pc;
        self.cycles += 1;
        Ok(rec)
    }

    /// Run to halt. If `record` is true, keep every cycle record.
    pub fn run(&mut self, max_cycles: u64, record: bool) -> Result<ExecutionResult> {
        let mut records = Vec::new();
        while !self.halted {
            if self.cycles >= max_cycles {
                bail!(Trap::CycleLimit);
            }
            let rec = self.step().map_err(|t| anyhow::anyhow!("guest trap after {} cycles: {}", self.cycles, t))?;
            if record {
                records.push(rec);
            }
        }
        Ok(ExecutionResult { journal: self.journal.clone(), output: self.output.clone(), exit_code: self.exit_code, cycles: self.cycles, records })
    }
}

/// One proven segment of a longer execution.
#[derive(Clone, Debug, Default)]
pub struct SegmentRun {
    pub records: Vec<CycleRecord>,
    pub cycles: u64,
    pub pc_in: u32,
    pub pc_out: u32,
    pub jcount_in: u32,
    pub jcount_out: u32,
    pub halted: bool,
}

/// The five memory-argument slots of a cycle, as field addresses (word
/// addresses; registers at [`REG_BASE`] + i): fetch, rs1 (a7 for ecall),
/// rs2 (a0 for ecall), memory read (x0 when none), write (rd, x0 when none,
/// or the stored word). Shared by the VM's touch tracking and the trace.
pub fn slot_addrs(rec: &CycleRecord) -> [u64; 5] {
    let rb = REG_BASE;
    [
        (rec.pc / 4) as u64,
        rb + rec.r1.0 as u64,
        rb + rec.r2.0 as u64,
        rec.mem_read.map_or(rb, |(w, _)| w as u64),
        match rec.write.0 {
            WriteTarget::Reg(r) => rb + r as u64,
            WriteTarget::Mem(w) => w as u64,
        },
    ]
}

/// Register-file *word* address space inside the memory argument (see zkb-air):
/// data memory is byte addresses below [`ADDR_LIMIT`] (word addresses below 2^29).
pub const REG_BASE: u64 = 1 << 29;
/// First byte address outside the provable address space.
pub const ADDR_LIMIT: u32 = 1 << 31;

impl Vm {
    #[inline(always)]
    fn touch(&mut self, addr: u64) {
        if addr >= REG_BASE {
            let i = (addr - REG_BASE) as usize;
            if self.reg_stamps[i] != self.seg_stamp {
                self.touched.push((addr, self.regs[i], self.reg_stamps[i]));
                self.reg_stamps[i] = self.seg_stamp;
            }
        } else {
            let w = addr as u32;
            let st = self.stamps.get(w);
            if st != self.seg_stamp {
                self.touched.push((addr, self.mem.get(w), st));
                self.stamps.set(w, self.seg_stamp);
            }
        }
    }

    /// Start segment `k` (stamp k+1): clears the first-touch list.
    pub fn begin_segment(&mut self, k: usize) {
        self.seg_stamp = k as u32 + 1;
        self.touched.clear();
    }

    /// The segment's init entries, sorted by address: (addr, value at segment start, stamp of the last toucher).
    pub fn init_entries(&self) -> Vec<(u64, u32, u32)> {
        let mut v = self.touched.clone();
        v.sort_unstable();
        v
    }

    /// Closing entries of the whole execution (one per address ever touched,
    /// plus every untouched image word): (addr, final value, stamp of the last
    /// toucher; 0 for untouched image words), sorted by address.
    pub fn closing_entries(&self, image: &Image) -> Vec<(u64, u32, u32)> {
        let mut v: Vec<(u64, u32, u32)> = self.stamps.iter_nonzero().map(|(a, st)| (a as u64, self.mem.get(a), st)).collect();
        for (i, &st) in self.reg_stamps.iter().enumerate() {
            if st != 0 {
                v.push((REG_BASE + i as u64, self.regs[i], st));
            }
        }
        for &(a, val) in &image.words {
            if self.stamps.get(a) == 0 {
                v.push((a as u64, val, 0));
            }
        }
        v.sort_unstable();
        v
    }

    /// Nonzero memory words and registers, as (field address, value), sorted.
    /// This is the state carried between segments.
    pub fn state_nonzero(&self) -> Vec<(u64, u32)> {
        let mut v: Vec<(u64, u32)> = self.mem.iter_nonzero().map(|(a, x)| (a as u64, x)).collect();
        for (i, &r) in self.regs.iter().enumerate() {
            if r != 0 {
                v.push((REG_BASE + i as u64, r));
            }
        }
        v
    }

    /// Run up to `max_cycles` more cycles (or until halt).
    pub fn run_segment(&mut self, max_cycles: u64, record: bool) -> Result<SegmentRun> {
        self.run_segment_budget(max_cycles, usize::MAX, record)
    }

    /// Run up to `max_cycles` more cycles, stopping early so that the
    /// segment's memory table (5 slots per cycle plus one init entry per
    /// first-touched address) stays within `slot_budget` slots.
    pub fn run_segment_budget(&mut self, max_cycles: u64, slot_budget: usize, record: bool) -> Result<SegmentRun> {
        let mut seg = SegmentRun { pc_in: self.pc, jcount_in: self.journal.len() as u32, ..Default::default() };
        if self.halted {
            bail!("vm already halted");
        }
        // a cycle adds 5 slots and at most 5 init entries
        while !self.halted && seg.cycles < max_cycles && 5 * (seg.cycles as usize + 1) + self.touched.len() + 5 <= slot_budget {
            let rec = self.step().map_err(|t| anyhow::anyhow!("guest trap after {} cycles: {}", self.cycles, t))?;
            seg.cycles += 1;
            if record {
                seg.records.push(rec);
            }
        }
        seg.pc_out = self.pc;
        seg.jcount_out = self.journal.len() as u32;
        seg.halted = self.halted;
        Ok(seg)
    }
}

/// Convenience: load an ELF, run it, return the result.
pub fn execute(elf_bytes: &[u8], witness: Vec<u32>, max_cycles: u64, record: bool) -> Result<(Image, ExecutionResult)> {
    let image = elf::load_elf(elf_bytes)?;
    let mut vm = Vm::new(&image, witness);
    let res = vm.run(max_cycles, record)?;
    Ok((image, res))
}

/// Pack bytes into little-endian words for the witness stream (length-prefixed).
pub fn pack_witness_bytes(data: &[u8]) -> Vec<u32> {
    let mut out = vec![data.len() as u32];
    for chunk in data.chunks(4) {
        let mut w = [0u8; 4];
        w[..chunk.len()].copy_from_slice(chunk);
        out.push(u32::from_le_bytes(w));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asm(words: &[u32]) -> Image {
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        Image::from_bytes(0x1000, &[(0x1000, bytes)]).unwrap()
    }

    #[test]
    fn add_and_halt() {
        // li a0, 5 ; li t0, 7 ; add a0, a0, t0 ; li a7, 2 ; ecall ; li a7, 0; li a0, 0; ecall
        let prog = asm(&[
            0x00500513, // addi a0, x0, 5
            0x00700293, // addi t0, x0, 7
            0x00550533, // add a0, a0, t0
            0x00200893, // addi a7, x0, 2
            0x00000073, // ecall (write journal)
            0x00000893, // addi a7, x0, 0
            0x00000513, // addi a0, x0, 0
            0x00000073, // ecall (halt)
        ]);
        let mut vm = Vm::new(&prog, vec![]);
        let r = vm.run(100, true).unwrap();
        assert_eq!(r.journal, vec![12]);
        assert_eq!(r.exit_code, 0);
        assert_eq!(r.cycles, 8);
    }
}
