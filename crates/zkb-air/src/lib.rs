//! The RV32IM zkVM: an AIR for `zkb-vm` executions, proven with `zkb-stark`.
//!
//! Statement proven by [`prove_execution`] / checked by [`verify_execution`]:
//!
//! > Running the program `image` (public) from its entry point, with *some*
//! > private witness stream, halts with exit code 0 and writes exactly
//! > `journal` (public) to the output.
//!
//! Long executions are split into segments ("continuations"). Every
//! segment is a STARK over its own trace; all segments share one set of
//! LogUp challenges, drawn after every main trace is committed. Memory is
//! carried across segments by offline memory checking (Blum et al.): every
//! address a segment touches gets an *init* entry citing the value and the
//! stamp (segment index + 1) of the last segment that touched it, and a
//! *final* entry stamped with this segment; a *closing* table (one entry per
//! address ever touched or in the image, spread over the segments in
//! disjoint increasing address ranges) reads each address's final state and
//! writes its initial `(0, 0)`. With `Σ reads = Σ writes` (the verifier adds
//! the image's initial values) and init stamps forced below the segment's
//! own, every init value is the previous toucher's final value. Registers
//! live in memory, so the only other state is (pc, journal length), exposed
//! as public values.
//!
//! The witness is never revealed: `SYS_READ_WITNESS` values are
//! unconstrained columns (only range-checked to 32 bits), the private
//! output stream is unconstrained, and in `zk` mode every committed
//! polynomial is blinded.

pub mod constraints;
pub mod layout;
pub mod lookups;
pub mod trace;

use anyhow::{ensure, Context, Result};
use layout::{Layout, NUM_PUBLIC, PUB_CLOSE_HI, PUB_CLOSE_LO, PUB_HALTED, PUB_HAS_CLOSE, PUB_JCOUNT_IN, PUB_JCOUNT_OUT, PUB_PC_IN, PUB_PC_OUT, PUB_SEG};
use lookups::{Enc, AUX_CHAIN};
use serde::{Deserialize, Serialize};
use zkb_field::{FieldLike, Fp, Fp4};
use zkb_stark::prover::ProvingKey;
use zkb_stark::{Air, AirShape, ConstraintSink, Proof, RowAccess, RowView, StarkConfig, VerifyingKey};
use zkb_vm::{Image, Vm};

pub use trace::Segment;

pub struct Rv32Air {
    pub l: Layout,
    num_transition: usize,
    num_first: usize,
    num_last: usize,
}

impl Default for Rv32Air {
    fn default() -> Self {
        Rv32Air::new()
    }
}

impl Rv32Air {
    pub fn new() -> Rv32Air {
        let l = Layout::new();
        let m = vec![Fp::ZERO; l.main_width];
        let f = vec![Fp::ZERO; l.fixed_width];
        let a = vec![Fp4::ZERO; lookups::num_aux(&l)];
        let row = RowView::new(&m[..], &m[..], &f[..], &f[..], &a[..], &a[..]);
        let mut sink: ConstraintSink<Fp> = ConstraintSink::default();
        let public = vec![Fp4::ZERO; NUM_PUBLIC];
        constraints::eval(&l, &row, &[Fp4::ONE; 4], &public, &a, &mut sink);
        let (num_transition, num_first, num_last) = sink.counts();
        Rv32Air { num_transition, num_first, num_last, l }
    }
}

impl Air for Rv32Air {
    fn shape(&self) -> AirShape {
        AirShape {
            main_width: self.l.main_width,
            fixed_width: self.l.fixed_width,
            aux_width: lookups::num_aux(&self.l),
            num_challenges: 2, // (β, γ), global to the execution
            num_transition: self.num_transition,
            num_first: self.num_first,
            num_last: self.num_last,
            max_degree: 4,
        }
    }
    fn fixed_columns(&self, n: usize) -> Vec<Vec<Fp>> {
        trace::fixed_columns(&self.l, n)
    }
    fn generate_aux(&self, main: &[&[Fp]], fixed: &[&[Fp]], challenges: &[Fp4], real_rows: usize) -> (Vec<Vec<Fp4>>, Vec<Fp4>) {
        trace::generate_aux(&self.l, main, fixed, challenges, real_rows)
    }
    fn eval<F: FieldLike, M: RowAccess<F>, A: RowAccess<F::Ext>>(&self, row: &RowView<F, M, A>, ch: &[Fp4], public: &[Fp4], aux_public: &[Fp4], sink: &mut ConstraintSink<F>) {
        constraints::eval(&self.l, row, ch, public, aux_public, sink)
    }
}

/// Public statement of an execution proof.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExecutionClaim {
    pub image_digest: [u8; 32],
    pub entry: u32,
    pub journal: Vec<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutionProof {
    pub claim: ExecutionClaim,
    pub vk: VerifyingKey,
    pub proof: Proof,
}

/// What the prover learns besides the proof.
#[derive(Clone, Debug, Default)]
pub struct ExecutionOutput {
    pub journal: Vec<u32>,
    /// Private output stream (SYS_WRITE_OUTPUT), not attested by itself.
    pub output: Vec<u32>,
    pub cycles: u64,
    pub segments: usize,
}

/// Σ 1/(γ - enc_state(addr, val, stamp)) over stamped entries.
pub fn state_sum(state: &[(u64, u32, u32)], e: &Enc) -> Fp4 {
    let mut denoms: Vec<Fp4> = state.iter().map(|&(a, v, t)| e.state::<Fp>(Fp::new(a), Fp::new((v & 0xffff) as u64), Fp::new((v >> 16) as u64), Fp::new(t as u64))).collect();
    Fp4::batch_inverse(&mut denoms);
    denoms.into_iter().sum()
}

/// The verifier's side of the memory argument: the image's initial words
/// enter the write set as `(addr, v0, 0)` in place of the `(addr, 0, 0)` the
/// closing table writes: Σ_image [1/(γ − enc(a, v0, 0)) − 1/(γ − enc(a, 0, 0))].
pub fn image_correction(image: &Image, e: &Enc) -> Fp4 {
    let with: Vec<(u64, u32, u32)> = image.words.iter().map(|&(a, v)| (a as u64, v, 0)).collect();
    let zero: Vec<(u64, u32, u32)> = image.words.iter().map(|&(a, _)| (a as u64, 0, 0)).collect();
    state_sum(&with, e) - state_sum(&zero, e)
}

/// Σ 1/(γ - enc_journal(i, val)) over journal[lo..hi].
pub fn journal_sum(journal: &[u32], lo: usize, hi: usize, e: &Enc) -> Fp4 {
    let mut denoms: Vec<Fp4> = (lo..hi).map(|i| e.journal::<Fp>(Fp::new(i as u64), Fp::new((journal[i] & 0xffff) as u64), Fp::new((journal[i] >> 16) as u64))).collect();
    Fp4::batch_inverse(&mut denoms);
    denoms.into_iter().sum()
}


/// Key generation is deterministic in (log_n, config); cache per size.
pub fn setup(air: &Rv32Air, log_n: u32, config: StarkConfig) -> ProvingKey {
    zkb_stark::prover::setup(air, log_n, config)
}

/// Plan of a segmented execution: cycle counts per segment, the closing
/// table and its distribution, public values.
struct Plan {
    cycles: Vec<u64>,
    /// closing-table slice of each segment (indices into `closing`)
    close_ranges: Vec<(usize, usize)>,
    closing: Vec<(u64, u32, u32)>,
    publics: Vec<Vec<Fp4>>,
    journal: Vec<u32>,
    output: Vec<u32>,
    total_cycles: u64,
}

struct SegPlan {
    cycles: u64,
    inits: usize,
    pc_in: u32,
    pc_out: u32,
    jcount_in: u32,
    jcount_out: u32,
    halted: bool,
}

/// One planning pass: run the execution with `budget` memory-table slots per
/// segment (closing entries are placed afterwards, into the room each
/// segment leaves).
fn plan_pass(image: &Image, witness: &[u32], budget: usize, max_total_cycles: u64, segment_cap: Option<u64>) -> Result<(Vec<SegPlan>, Vm)> {
    let mut vm = Vm::new(image, witness.to_vec());
    let mut segs = Vec::new();
    loop {
        vm.begin_segment(segs.len());
        let cap = segment_cap.unwrap_or(u64::MAX);
        let seg = vm.run_segment_budget(cap, budget, false)?;
        ensure!(seg.cycles >= 1, "memory table budget too small for a single cycle");
        segs.push(SegPlan { cycles: seg.cycles, inits: vm.touched.len(), pc_in: seg.pc_in, pc_out: seg.pc_out, jcount_in: seg.jcount_in, jcount_out: seg.jcount_out, halted: seg.halted });
        ensure!(vm.cycles <= max_total_cycles, "cycle limit {max_total_cycles} exceeded");
        if seg.halted {
            break;
        }
    }
    ensure!(vm.exit_code == 0, "guest exited with code {}", vm.exit_code);
    Ok((segs, vm))
}

fn plan_execution(image: &Image, witness: &[u32], log_n: u32, random_rows: usize, max_total_cycles: u64, segment_cap: Option<u64>) -> Result<Plan> {
    let full_budget = trace::slot_budget(log_n, random_rows);
    ensure!(full_budget > 10, "segment too small");
    // The closing table (every address ever touched, plus the image) is only
    // known after a full run; reserve room for it by shrinking the per-segment
    // budget until it fits into the room the segments leave.
    let mut budget = full_budget;
    for _ in 0..8 {
        let (segs, vm) = plan_pass(image, witness, budget, max_total_cycles, segment_cap)?;
        let closing = vm.closing_entries(image);
        let room: Vec<usize> = segs.iter().map(|s| full_budget - NUM_SLOTS_USIZE * s.cycles as usize - s.inits).collect();
        let total_room: usize = room.iter().sum();
        if total_room >= closing.len() {
            // distribute from the last segment backwards (closing entries can live anywhere)
            let mut close_ranges = vec![(0usize, 0usize); segs.len()];
            let mut end = closing.len();
            for k in (0..segs.len()).rev() {
                let take = room[k].min(end);
                close_ranges[k] = (end - take, end);
                end -= take;
                if end == 0 {
                    break;
                }
            }
            let publics: Vec<Vec<Fp4>> = segs
                .iter()
                .enumerate()
                .map(|(k, s)| {
                    let seg = Segment { index: k, init_entries: &[], closing: &closing[close_ranges[k].0..close_ranges[k].1], records: &[], pc_in: s.pc_in, pc_out: s.pc_out, jcount_in: s.jcount_in, jcount_out: s.jcount_out, halted: s.halted };
                    seg.public_values()
                })
                .collect();
            return Ok(Plan { cycles: segs.iter().map(|s| s.cycles).collect(), close_ranges, closing, publics, journal: vm.journal.clone(), output: vm.output.clone(), total_cycles: vm.cycles });
        }
        let missing = closing.len() - total_room;
        let shrink = missing.div_ceil(segs.len()) + 16;
        ensure!(shrink < budget / 2, "closing table ({} entries) does not fit the execution's segments", closing.len());
        budget -= shrink;
    }
    anyhow::bail!("could not fit the closing table in 8 planning passes")
}

const NUM_SLOTS_USIZE: usize = layout::NUM_SLOTS;

/// Sequential trace generator: the prover asks for segments in increasing
/// order (twice: the commitment phase and the proving phase), so one VM is
/// advanced segment by segment and restarted from the image when the
/// requested segment is behind it.
struct SegmentGen<'a> {
    air: &'a Rv32Air,
    image: &'a Image,
    witness: &'a [u32],
    plan: &'a Plan,
    log_n: u32,
    real_rows: usize,
    vm: std::sync::Mutex<(Vm, usize)>,
}

impl SegmentGen<'_> {
    fn trace(&self, k: usize) -> Result<zkb_field::columns::ColumnStore> {
        let mut guard = self.vm.lock().unwrap();
        let (vm, next) = &mut *guard;
        if k < *next {
            *vm = Vm::new(self.image, self.witness.to_vec());
            *next = 0;
        }
        let budget = trace::slot_budget(self.log_n, (1usize << self.log_n) - self.real_rows);
        while *next < k {
            vm.begin_segment(*next);
            vm.run_segment_budget(self.plan.cycles[*next], budget, false)?;
            *next += 1;
        }
        vm.begin_segment(k);
        let seg = vm.run_segment_budget(self.plan.cycles[k], budget, true)?;
        ensure!(seg.cycles == self.plan.cycles[k], "segment {k}: re-execution diverged (cycles)");
        let init_entries = vm.init_entries();
        *next = k + 1;
        let (c0, c1) = self.plan.close_ranges[k];
        let s = Segment { index: k, init_entries: &init_entries, closing: &self.plan.closing[c0..c1], records: &seg.records, pc_in: seg.pc_in, pc_out: seg.pc_out, jcount_in: seg.jcount_in, jcount_out: seg.jcount_out, halted: seg.halted };
        ensure!(s.public_values() == self.plan.publics[k], "segment {k}: re-execution diverged");
        drop(guard);
        trace::generate_main(&self.air.l, &s, self.log_n, self.real_rows)
    }
}

/// Prove a complete execution of `image` on `witness`, splitting it into
/// segments of 2^log_n rows.
pub fn prove_execution(air: &Rv32Air, pk: &ProvingKey, image: &Image, witness: &[u32], max_total_cycles: u64) -> Result<(ExecutionProof, ExecutionOutput)> {
    prove_execution_capped(air, pk, image, witness, max_total_cycles, None)
}

/// Like [`prove_execution`] with an explicit upper bound on cycles per segment (testing).
pub fn prove_execution_capped(air: &Rv32Air, pk: &ProvingKey, image: &Image, witness: &[u32], max_total_cycles: u64, segment_cap: Option<u64>) -> Result<(ExecutionProof, ExecutionOutput)> {
    let log_n = pk.vk.log_n;
    let random_rows = pk.vk.config.random_rows();
    let real_rows = (1usize << log_n) - random_rows;
    let plan = plan_execution(image, witness, log_n, random_rows, max_total_cycles, segment_cap)?;
    let sg = SegmentGen { air, image, witness, plan: &plan, log_n, real_rows, vm: std::sync::Mutex::new((Vm::new(image, witness.to_vec()), 0)) };
    let gen = |k: usize| sg.trace(k);
    let proof = zkb_stark::prover::prove(air, pk, &plan.publics, &gen)?;
    let ep = ExecutionProof { claim: ExecutionClaim { image_digest: image.digest(), entry: image.entry, journal: plan.journal.clone() }, vk: pk.vk.clone(), proof };
    let out = ExecutionOutput { journal: plan.journal, output: plan.output, cycles: plan.total_cycles, segments: plan.cycles.len() };
    Ok((ep, out))
}

/// Pick the smallest trace size that fits the whole execution in one
/// segment, capped at `max_log_n` (larger executions use several segments).
pub fn choose_log_n(image: &Image, witness: &[u32], max_log_n: u32, max_total_cycles: u64, config: &StarkConfig) -> Result<u32> {
    let mut vm = Vm::new(image, witness.to_vec());
    let res = vm.run(max_total_cycles, false)?;
    let extra = vm.touched.len() + vm.closing_entries(image).len();
    Ok(trace::min_log_n(res.cycles as usize, extra, config.random_rows()).min(max_log_n))
}

/// Verify an execution proof against the program image the verifier trusts.
pub fn verify_execution(air: &Rv32Air, ep: &ExecutionProof, image: &Image) -> Result<()> {
    ensure!(ep.claim.image_digest == image.digest(), "proof is for a different program image");
    ensure!(ep.claim.entry == image.entry, "entry point mismatch");
    let segs = &ep.proof.segments;
    ensure!(!segs.is_empty(), "no segments");
    let publics: Vec<Vec<Fp4>> = segs.iter().map(|s| s.public.clone()).collect();
    for (k, p) in publics.iter().enumerate() {
        ensure!(p.len() == NUM_PUBLIC, "segment {k}: bad public values");
    }
    let ch = zkb_stark::verifier::verify(air, &ep.vk, &ep.proof, &publics).context("STARK verification")?;
    let journal = &ep.claim.journal;
    let num_aux = lookups::num_aux(&air.l);
    let e = Enc::new(ch[0], ch[1]);
    let mut chain = Fp4::ZERO;
    let mut pc_expect = Fp4::from(image.entry as u64);
    let mut jc_expect = Fp4::ZERO;
    let mut close_next_min: usize = 0; // closing ranges are disjoint and increasing
    for (k, seg) in segs.iter().enumerate() {
        let p = &seg.public;
        ensure!(seg.aux_public.len() == num_aux, "segment {k}: bad aux public length");
        ensure!(p[PUB_PC_IN] == pc_expect, "segment {k}: pc does not continue from the previous segment");
        ensure!(p[PUB_JCOUNT_IN] == jc_expect, "segment {k}: journal count does not continue");
        ensure!(p[PUB_SEG] == Fp4::from(k as u64), "segment {k}: wrong segment index");
        let has_close = fp4_to_usize(p[PUB_HAS_CLOSE])?;
        ensure!(has_close <= 1, "segment {k}: bad closing flag");
        if has_close == 1 {
            let lo = fp4_to_usize(p[PUB_CLOSE_LO])?;
            let hi = fp4_to_usize(p[PUB_CLOSE_HI])?;
            ensure!(lo >= close_next_min && lo <= hi, "segment {k}: closing range not increasing");
            close_next_min = hi + 1;
        }
        chain += seg.aux_public[AUX_CHAIN].iter().cloned().sum::<Fp4>();
        let rest: Fp4 = seg.aux_public[AUX_CHAIN.end..].iter().cloned().sum();
        let jin = fp4_to_usize(p[PUB_JCOUNT_IN])?;
        let jout = fp4_to_usize(p[PUB_JCOUNT_OUT])?;
        ensure!(jin <= jout && jout <= journal.len(), "segment {k}: journal range out of bounds");
        ensure!(rest == journal_sum(journal, jin, jout, &e), "segment {k}: LogUp balance failed (journal or memory inconsistency)");
        let last = k + 1 == segs.len();
        let halted = p[PUB_HALTED];
        ensure!(halted == Fp4::from(last as u64), "segment {k}: unexpected halt flag");
        pc_expect = p[PUB_PC_OUT];
        jc_expect = p[PUB_JCOUNT_OUT];
    }
    ensure!(jc_expect == Fp4::from(journal.len() as u64), "journal length does not match the proof");
    // execution-wide memory argument: Σ reads − Σ writes over all segments = image correction
    ensure!(chain == image_correction(image, &e), "memory state is inconsistent across segments");
    Ok(())
}

fn fp4_to_usize(v: Fp4) -> Result<usize> {
    ensure!(v.is_base(), "public value is not a small integer");
    Ok(v.0[0].as_u64() as usize)
}

#[cfg(test)]
mod memory_argument_tests {
    //! A dishonest prover's segments are individually valid STARKs; the
    //! execution-level checks must reject them.
    use super::*;

    fn guest() -> Vec<u8> {
        let p = format!("{}/../../guest/build/echo-guest.elf", env!("CARGO_MANIFEST_DIR"));
        std::fs::read(&p).expect("run guest/build.sh first")
    }

    fn prove_with(mutate: impl Fn(&mut Plan)) -> (Rv32Air, ExecutionProof, Image) {
        let image = zkb_vm::elf::load_elf(&guest()).unwrap();
        let air = Rv32Air::new();
        let cfg = StarkConfig::fast();
        let pk = setup(&air, 17, cfg);
        let random_rows = cfg.random_rows();
        let real_rows = (1usize << 17) - random_rows;
        let witness = [41u32];
        let mut plan = plan_execution(&image, &witness, 17, random_rows, 1 << 24, Some(30000)).unwrap();
        assert_eq!(plan.cycles.len(), 3);
        mutate(&mut plan);
        let sg = SegmentGen { air: &air, image: &image, witness: &witness, plan: &plan, log_n: 17, real_rows, vm: std::sync::Mutex::new((Vm::new(&image, witness.to_vec()), 0)) };
        let gen = |k: usize| sg.trace(k);
        let proof = zkb_stark::prover::prove(&air, &pk, &plan.publics, &gen).unwrap();
        let ep = ExecutionProof { claim: ExecutionClaim { image_digest: image.digest(), entry: image.entry, journal: plan.journal.clone() }, vk: pk.vk.clone(), proof };
        (air, ep, image)
    }

    fn republish(plan: &mut Plan) {
        // recompute public values after changing the closing distribution
        for k in 0..plan.cycles.len() {
            let (c0, c1) = plan.close_ranges[k];
            let p = &mut plan.publics[k];
            let closing = &plan.closing[c0..c1];
            p[PUB_CLOSE_LO] = Fp4::from(closing.first().map_or(0, |e| e.0));
            p[PUB_CLOSE_HI] = Fp4::from(closing.last().map_or(0, |e| e.0));
            p[PUB_HAS_CLOSE] = Fp4::from(!closing.is_empty() as u64);
        }
    }

    #[test]
    fn honest_plan_verifies() {
        let (air, ep, image) = prove_with(|_| {});
        verify_execution(&air, &ep, &image).unwrap();
    }

    #[test]
    fn duplicated_closing_entries_are_rejected() {
        // the "fork" attack: the same addresses closed twice, in two segments
        let (air, ep, image) = prove_with(|plan| {
            let n = plan.closing.len();
            // give every segment the whole closing table (ranges overlap)
            for k in 0..plan.cycles.len() {
                plan.close_ranges[k] = (0, n.min(1000));
            }
            republish(plan);
        });
        let err = verify_execution(&air, &ep, &image).unwrap_err().to_string();
        assert!(err.contains("closing range not increasing"), "{err}");
    }

    #[test]
    fn missing_closing_entries_are_rejected() {
        let (air, ep, image) = prove_with(|plan| {
            for k in 0..plan.cycles.len() {
                plan.close_ranges[k] = (0, 0);
            }
            republish(plan);
        });
        let err = verify_execution(&air, &ep, &image).unwrap_err().to_string();
        assert!(err.contains("inconsistent"), "{err}");
    }
}
