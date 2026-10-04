# zkbuild — zero-knowledge reproducible-build attestation

A from-scratch zkVM and build-provenance system that proves statements of the form

> *Binary `B` was produced by the exact toolchain `C` with build configuration `F`
> from a source tree committed as `R`* — without revealing the source.

Everything is implemented here, in Rust and C, with no external proof system —
and the toolchain that runs inside the VM is a **real C compiler** (chibicc).

| crate / dir | what it is |
|---|---|
| `zkb-field` | BabyBear field (p = 2³¹−2²⁷+1, Montgomery form in `u32`), quartic extension F_{p⁴} = F_p[u]/(u⁴−11), 16-lane AVX-512 kernels, NTT |
| `zkb-stark` | STARK prover/verifier: blake3 Merkle commitments, Fiat–Shamir, LogUp aux columns, DEEP-ALI composition, FRI, **zero-knowledge blinding**, **multi-segment proofs** |
| `zkb-vm` | RV32IM guest VM with the deterministic syscall layer (the "ZK build OS"), ELF loader, segmented execution |
| `zkb-air` | The RV32IM AIR over 16-bit limbs: 286 witness columns, 5-slot memory argument, lookup tables, **continuations** (memory carried across segments by offline memory checking under one global LogUp challenge) |
| `zkb-ca` | Content addressing (source tree / flags / object commitments, BLAKE3) shared with the guests |
| `zkb-proof` | `ObjectCertificate`, `LinkCertificate`, `BuildCertificate`, toolchain commitment, prover driver |
| `zkb-cli` | `zkb` command line: `commit`, `prove-tu`, `link`, `bundle`, `verify`, `run`, `info` |
| `guest/rt`, `guest/libc` | Freestanding runtime and a deterministic mini libc (stdio, malloc, strings, VFS over the witness, 64-bit and soft-float builtins) |
| `guest/zkcc` | **chibicc** (upstream, MIT) built as the RV32IM translation-unit guest — the attested compiler |

## Quick start

```sh
guest/build.sh && guest/build-zkcc.sh     # clang with the RISC-V backend + ld.lld
cargo build --release
./demo.sh          # identity toolchain: 2 TUs + link, bundle, verify (minutes)
./bootstrap.sh     # real compiler: prove fib.c's compilation, self-host check (≈2 min)
./attest-self.sh   # prove the compiler compiling itself: 9 TUs + link, one certificate (≈4 h; JOBS=n on big machines)
```

## What was demonstrated on this machine (2 cores, 7 GB)

**Attested compilation with a real compiler.** `zkcc` (chibicc inside the zkVM)
compiled `examples/fib/fib.c` to x86-64 assembly in 0.74 M cycles. The output
is byte-identical to native chibicc; `gcc` assembles it and the program runs.
The compilation was proven in zero knowledge across 7 segments of 2¹⁷ rows
(18–22 s), linked (2 segments, 6.6 s), bundled and **verified in 66 ms**
from a 5.1 MB build certificate. The verifier learned `source_root`,
`flags_root`, `toolchain_root`, `object_hash` and `output_hash` — not the source.

**Bootstrap fixpoint.** All nine translation units of chibicc compile inside
the zkVM (13.3 M – 149 M cycles each, 470 M total; the whole `bootstrap.sh`
run, proofs included, takes 49 s), each output identical to a native chibicc
build against the same sysroot, and the resulting stage-2 compiler
reproduces stage-1 output exactly.

**The compiler's own build, attested end to end** (`./attest-self.sh`,
≈4 h here). chibicc inside the zkVM compiled all nine of its own translation
units (470 M cycles, 3 630 segments) and the link guest combined the nine
certified objects (67 M cycles, 515 segments), every step proven in zero
knowledge; the ten certificates were bundled into one 2.37 GB
`BuildCertificate` (`build_root = 6782f18d…ba87`, `output_hash =
84ce13f1…ce46`) that **verifies in 21 s** (4 145 segment proofs, checked
in parallel after a sequential transcript replay). The nine
certified objects are byte-for-byte what a native chibicc build of the same
sources produces, and assembled natively they are a compiler that
reproduces stage-1 output (the fixpoint). What the certificate states: with
toolchain `16890d31…4203`, these nine `source_root`s under these
`flags_root`s compile to these nine `object_hash`es, which link to
`output_hash`. What it does not state: that the toolchain image itself came
from those objects — chibicc emits x86-64, so the loop closes onto a native
binary, not onto the RV32 guest image (roadmap item 5). Toolchain root of
this run: `7883e1b3…4867`; the verifier transcript is
`docs/self-attestation/verify.txt`.

| step | cycles | segments | prove | verify |
|---|---|---|---|---|
| chibicc compiles `fib.c` (ZK) | 0.74 M | 7 × 2¹⁷ | 18–22 s | — |
| chibicc compiles its own `strings.c` (ZK) | 13.3 M | 103 × 2¹⁷ | 5 m 25 s | — |
| chibicc compiles its own `parse.c` (ZK) | 149 M | 1 147 × 2¹⁷ | 59 min | — |
| chibicc compiles all of itself + link (ZK) | 537 M | 4 145 × 2¹⁷ | 3.8 h | — |
| `zkb verify` of the fib bundle | | 9 | | 66 ms |
| `zkb verify` of the compiler's own build | | 4 145 | | 21 s |

(Proofs and certificates are not compatible across the field change (third
pass), the hash change (fourth pass) or the memory-argument change (fifth
pass); everything above was produced by the current binary.)

Prover throughput is ≈2.5 s per 2¹⁷-row zero-knowledge segment on 2 cores
(≈53 k cycles/s), after the four optimization passes described below (the
`strings.c` numbers above were measured with the 11 s/segment prover and the
old guest).
Clang-scale workloads (10⁹–10¹⁰ cycles per TU) still need ~30× more
(see roadmap). Wall-clock numbers on this VM vary by ±25 % between runs:
the host backs guest memory lazily and a cold run spends most of its extra
time in page faults (`sys` time), not in the prover.

### Where the prover's time goes (and what was done about it)

`tools/bpfprof.py` is an eBPF sampling profiler written directly against the
`bpf(2)`/`perf_event_open(2)` syscalls (the container has no `perf` or
`bpftrace`): a `BPF_PROG_TYPE_PERF_EVENT` program records user stacks into a
`STACK_TRACE` map at 1 kHz per thread, symbolized against the binary. Build
with `RUSTFLAGS="-C force-frame-pointers=yes"` and run
`python3 tools/bpfprof.py <pid> <seconds>`.

The profile drove these changes (per 2¹⁷-row ZK segment, 2 cores):

| step | per segment |
|---|---|
| baseline: blinding by `Z_H·r` (degree 2N → blowup 8) | 43 s |
| ethSTARK random rows instead → trace degree < N, blowup 4 | 19 s |
| α powers precomputed once; base-field constraints combined with `Fp2×Fp` | 16 s |
| DEEP quotient: one coefficient-space combination + NTT instead of 344 terms × 4N points | 15.5 s |
| one-segment challenge lookahead so every main trace is committed once | 11.9 s |
| Merkle leaves: one hash call per serialized row, blocked column reads (was 250 `update`s per leaf) | 11 s |
| LDE as 4 size-N transforms (coset-major), buffer pool, parallel trace generation | ≈10.7 s |
| 3-way batched LogUp fractions, degree-aware composition split (degree ≤ 2 constraints on the 2N subdomain), power tables | ≈9.3 s |
| AVX-512 field kernel (`Fp8`/`Fp2x8`), radix-4 NTT passes, 8-row vectorized constraint evaluation | ≈8.2 s |

Second pass — NTT and constraint evaluation (`crates/zkb-field/src/{simd,ntt,columns}.rs`,
`crates/zkb-stark/src/{prover,air,blake3x16,merkle}.rs`):

| change | effect |
|---|---|
| carry-free 32-bit-limb 64×64 product + lean Goldilocks reduction (26 vector ops per multiply) | NTT 2¹⁷: 2.4 → 1.8 ms |
| DIF/DIT transforms with no bit-reversal: `inverse_to_bitrev` → scale → `forward_from_bitrev`; coefficients kept bit-reversed everywhere (OOD/DEEP use bit-reversed power tables) | 1.8 → 1.08 ms per NTT, LDE 10 → 6 ms per column |
| first three NTT stages via an 8×8 in-register transpose (every butterfly is now an 8-lane op), coset scaling fused into that pass | included above |
| power-of-two root convention (ω₄ = 2⁴⁸, ω₈ = 2²⁴): true radix-4 butterflies (3 multiplies + a shift), shift-only small stages | ≈1.0 ms per NTT (at the port-throughput floor: ~165 vector ops per 4-element butterfly at 2.4 ops/cycle) |
| 16-way AVX-512 BLAKE3 (`blake3x16`, bit-exact with the `blake3` crate) for leaves, inner nodes and salts; rows serialized straight into the transposed lane layout | Merkle: 0.80 → 0.21 s per main commit |
| lazy α-combination: Σ αᵢ·cᵢ accumulated as 22-bit × 32-bit limb products in u64 lanes, one reduction per 512 terms (`LazyFp2Acc`); same primitive for OOD dot products and the DEEP coefficient combination | OOD 0.27 → 0.10 s, DEEP 0.39 → 0.15 s |
| accumulating `ConstraintSink` (constraints folded into the combination as they are declared; nothing stored) and streamed LogUp terms (no 48 KB per-row term buffer) | 8-row eval 19 → 17 µs (All) / 13 → 10.6 µs (HighOnly) |
| **staggered column storage** (`ColumnStore`, each column offset by one cache line): with one `Vec` per column, row *i* of all ~300 columns mapped to the same L1/L2 set and every load went to L3 | composition 0.85 → 0.62 s |
| row views read the LDE in place (`RowAccess` trait) instead of gathering 632 vectors per step; cached vanishing-polynomial tables; sequential segment re-execution (was quadratic); parallel histograms/sorts, chunked batch inversions, pooled recycled buffers (mimalloc, no purging) | trace generation 0.45 → 0.27 s per segment |

After the second pass a 2¹⁷-row ZK segment cost ≈3.0 s on 2 cores (trace
0.27 s, main commit 1.0 s, aux 0.5 s, composition 0.65 s, OOD/DEEP/FRI
0.35 s), with the NTT at 45 % of CPU time and at the instruction-throughput
floor for 64-bit Goldilocks on this core.

Third pass — the 31-bit field. Goldilocks is a Solinas prime, not a
Mersenne one, and its 64-bit multiply costs ~26 vector ops per 8 lanes. A
31-bit prime fits a Montgomery multiply in ~10 ops per **16** lanes
(`vpmuludq` on even/odd lanes, one `blend`, one `min`). The Mersenne prime
2³¹−1 has no 2-adic subgroup (it needs the circle-group FFT); BabyBear
(p = 15·2²⁷+1) keeps the ordinary NTT, so the whole system was moved to it
(`zkb-field` rewritten; `zkb-air` layout, constraints and trace rewritten):

| change | effect |
|---|---|
| `Fp` = BabyBear in Montgomery form (`u32`), `Fp16` = 16 lanes; challenges, LogUp fingerprints and the DEEP/FRI combination in F_{p⁴} (`Fp4`, `Fp4x16`; ≈2¹²⁴ for soundness) with Frobenius-norm inversion | NTT 2¹⁷: 1.0 → 0.4 ms (dif) / 0.5 ms (dit); LDE ×4: 6.0 → 2.1 ms per column |
| first four NTT stages via a 16×16 in-register transpose (`transpose16x16`), two interleaved radix-4 butterflies per pass | included above |
| **16-bit-limbed AIR**: every 32-bit word is (lo, hi) limbs; limb adder with boolean carries, per-limb equality/zero gadgets, `pc = 4·pcl14 + 2¹⁶·pc_hi`, the shared multiplier as 4×(4 bytes + bit32) nibble-pair products grouped into four 16-bit product limbs with range-checked carries; signed `mulh`/`div`/`rem` via two's-complement carry bits and the division identity checked through the multiplier; memory addresses < 2³¹ bytes (registers at word 2²⁹+i), memory `diff` split as `d_lo + 2¹⁶·d_hi` | 294 main / 34 aux / 13 fixed columns, 344 transition constraints, 102 LogUp terms (was 249 / 43 / 13, 344 constraints over 64-bit words) |
| Merkle leaves 4 bytes per element (`put_u32x16`), transcript/openings in canonical `u32` | main commit 1.0 → 0.49 s per segment; proof 7.9 → 7.2 MB for 13 segments |
| lazy-reduction F_{p⁴} multiply (`avx::mul4`: the four 64-bit products feeding each output coefficient summed two at a time, 8 Montgomery reductions instead of 19) | `Fp4x16` mul 45.6 → 27.7 ns; 16-row eval 36 → 27.7 µs (All) / 25.8 → 18.8 µs (HighOnly); composition 0.68 → 0.53 s |
| DEEP denominators `(x−z)`, `(x−gz)` inverted in 16-lane batches per chunk, in parallel (was two serial scalar `Fp4::batch_inverse` over 2¹⁹ points) | DEEP 0.34 → 0.22 s |
| the second pass's lazy limb accumulators (`LazyFp2Acc`) are gone: with 31-bit elements a plain `Fp4x16` product per term is cheaper than 64-bit limb products plus a deferred reduction | OOD/DEEP combination unchanged in time, less code |

Per 2¹⁷-row ZK segment on 2 cores today: trace 0.21 s, main commit 0.49 s
(294 columns, LDE 0.27 s + Merkle 0.16 s), aux 0.28 + 0.32 s, composition
0.53 s, OOD 0.12 s, DEEP 0.22 s, FRI/queries 0.08 s ≈ **2.45 s** (was 3.0 s
after the second pass, 43 s originally). The NTT is now ≈22 % of CPU time;
constraint evaluation (≈25 %, dominated by the 34 batched LogUp constraints
whose products are degree-4 extension multiplies) and the 16-way BLAKE3
(≈10 %) are the next targets. Goldilocks-era proofs do not verify with this
binary (different field, transcript and AIR).

### Fourth pass — cycles, not rows (the guest side)

A JIT for the executor was considered and rejected on the numbers: the
interpreter already runs at 21 M cycles/s (99 M after replacing its
`BTreeMap` memory with a page table), i.e. ~6 ms per 2¹⁷-cycle segment
against 2.45 s to prove it. Trace generation is 0.2 s of that. What a
"RISC-V → trace" view *does* buy is the realisation that proof time is
linear in guest cycles, so `zkb run --profile` (per-function cycles, calls,
cycles/call from the ELF symbol table) was pointed at chibicc:

| what chibicc's cycles went to | fix (all in `guest/libc`, `guest/rt`; chibicc untouched) | parse.c cycles |
|---|---|---|
| 30 % SHA-256 of the 918 KB output (63 cycles/byte: no rotate instruction, 8 rotates per round) | **BLAKE3** as the system's commitment hash, guest and host (`guest/rt/blake3.c`, 7 rounds fully unrolled with the message permutation resolved at compile time: 1408 cycles per 64-byte block = 22 cycles/byte; bit-exact with the `blake3` crate on multi-chunk inputs) | 298 M → 211 M (with the next three rows) |
| 16 % 64-bit `%` — chibicc's hash map does `uint64 hash % capacity` per probe, and the generic `udivmod64` cost 130 cycles | capacity is always a power of two: mask-and-shift fast path (9 cycles); an `asm("")` barrier keeps clang from rewriting the power-of-two test into a 20-instruction popcount | |
| 11 % `fputc`/`fwrite`: every output byte went emit → put_file → fputc → out_byte | `printf` writes straight into the sink's own buffer (the 4 KB stdout frame or the memstream's buffer, grown ahead of the writer) and flushes in blocks; `%s`/`%d`/`%c`/`%%` fast paths; 32-bit number formatting (one-cycle `div`/`rem`) when the value fits | |
| byte loops in `memcpy`/`memset`/`memmove` | word-wise bodies behind alignment checks (`strlen`/`memcmp` stay byte loops: chibicc's strings average 4 bytes and the alignment prologue cost more than it saved — measured, not assumed) | |
| 3 % one `ecall` per output word (230 k ecalls for parse.c) | `SYS_WRITE_OUTPUT_BUF`: one ecall hands the VM a word-aligned `[len, data…]` frame (one new one-hot column `h4` in the AIR, same pass-through semantics as the per-word syscall; the output stream is private and unconstrained either way) | 211 M → **149 M** |

Every output stayed byte-identical to native chibicc (`bootstrap.sh`
checks all nine TUs) and the AIR validates every cycle regardless, so none
of this is in the trusted base. Net: parse.c 298 M → 149 M cycles (−50 %),
strings.c 19.7 M → 13.3 M, fib.c 1.53 M → 0.74 M; the fib proof 32 s → ≈20 s
(13 → 7 segments), the link step 782 k → 248 k cycles (6 → 2 segments), and
the whole bootstrap run 1 m 49 s → 49 s. `object_hash`/`output_hash` are now
`b3sum` values rather than `sha256sum` values.

### Fifth pass — memory carried by offline memory checking (one argument across all segments)

The continuation used to re-state every live nonzero word in every segment
(5 words per row): 143 k words after `strings.c`, 1.0 M after `parse.c` —
more than a 2¹⁷-row segment holds, so the compiler's largest TU could not
be proven at all. It is now carried the way Blum–Evans–Gemmell–Kannan–Naor
offline memory checking does it, with the *segment index* as the clock:

* **One challenge set for the whole execution.** The prover commits every
  segment's main trace first (phase 1), the transcript then yields (β, γ),
  and every segment is proven with them (phase 2; each main trace is
  re-derived from the VM — traces are deterministic and the blinding is
  derived from a master seed and the segment label, so the phase-2
  commitment is checked to be bit-identical). This is what lets a multiset
  term in segment 5 cancel one in segment 1.
* **Init / final entries per touched address.** Segment k's memory table
  holds, for every address it touches, an *init* entry `(addr, value at
  segment start, t)` citing the stamp `t` = index+1 of the last segment that
  touched it (0 = initial memory), and the address's last entry in the
  segment is its *final* write `(addr, value, k+1)`. The cited stamp is
  forced into the past by range-checking `init·(k − t)` (`age` column).
  A segment that touches 30 k distinct words spends 6 k rows on them instead
  of 29 k–200 k on the whole live state.
* **Closing table.** Every address ever touched (and every image word) is
  *closed* exactly once somewhere in the execution: a closing entry reads
  the address's final `(value, stamp)` and writes its initial `(0, 0)`. The
  entries are sorted strictly within a segment, and each segment publishes
  its closing range `[lo, hi]` (bound to its first and last closing entry);
  the verifier requires the ranges to be disjoint and increasing, which is
  what rules out the fork attack (closing an address twice to launder a
  stale value). The verifier adds the image's `(addr, v0, 0)` in place of
  the closing table's `(addr, 0, 0)` and checks
  `Σ_k Σ (reads − writes) = Σ_image [1/(γ−enc(a,v0,0)) − 1/(γ−enc(a,0,0))]`.
* **Why it is sound.** Per address, write stamps are unique (0, k+1 for each
  toucher k) and every init must cite a smaller stamp than its own segment,
  so the only bijection between the read set and the write set is the
  chronological one: the earliest toucher reads the initial value, every
  later toucher reads its predecessor's final value, and the closing entry
  reads the last. Inside a segment the existing clock-sorted table still
  enforces read-after-write. The dishonest-prover tests
  (`zkb-air/src/lib.rs::memory_argument_tests`) prove three segments with a
  duplicated closing range and with the closing table left out; both
  verify as STARKs and are rejected by the execution verifier.

Follow-ups in the same pass: the link guest hashes every object once (its
`output_hash` commits to the executable through the per-object hashes, so
`inputs_root` and `output_hash` share the work: 133 M → 67 M cycles for
the self-build's link, 1 024 → ≈515 segments); the trace's row→column
transposition goes through 16×16 register tiles (100 → 23 ms); 2¹⁸-row
segments were measured and rejected (strings.c 345 s vs 325 s: the NTT's
n log n and cache footprint outweigh the halved segment count, though the
proof halves in size).

Cost: the value-is-zero gadget (5 columns per slot) is gone, the stamp,
closing flag and age columns are new, and the chain group has 4 LogUp
terms per slot (286 main / 39 aux columns, 366 transition constraints).
Phase 1 costs a trace generation, LDE and Merkle tree per segment
(≈1.0 s), so a 2¹⁷-row segment is now ≈3.2 s instead of 2.45 s, but
segments hold ≈129 k cycles whatever the heap: `strings.c` 103 segments
(5 m 25 s, was ≈120 segments), `parse.c` ≈1150 segments and finally
provable, the fib compile 6 segments (≈22 s).

## The statement, exactly

An `ExecutionProof` (`zkb-air`) attests:

> Running program image `I` (public: the toolchain guest, committed by its
> content digest) from its entry point, with *some* private witness stream,
> halts with exit code 0 having written exactly `journal` (public).

The guest is the toolchain. The TU-guest contract is:

```text
witness: flags | n | (name | content)*          (private)
journal: source_root[8] | flags_root[8] | object_hash[8] | n   (public)
output : the object bytes                       (private stream; host checks its hash)
```

`zkcc` implements it with chibicc's unmodified `main()` in `-cc1` mode: the
libc builds `argv` from `flags`, serves the source files (and the toolchain's
own headers, baked into the image) through an in-memory VFS, hashes everything
it was given into `source_root`, and hashes `stdout` into `object_hash`.
Swapping in another compiler keeps every format unchanged.

`BuildCertificate::verify` checks every execution proof against the trusted
toolchain images, then the chain: `link.inputs_root == H(object_hash_0 ‖ …)`,
one `toolchain_root` everywhere, and optionally `blake3(binary) == output_hash`.

## The zkVM

* **ISA**: RV32IM. `mul/mulh/mulhsu/mulhu` share one 32×32 multiplier gadget
  with the shifter; `div/divu/rem/remu` use `d = q·v + r, r < v` with the
  spec's divide-by-zero and overflow cases; all verified against the VM on
  edge cases in the clear.
* **CPU**: one row per cycle; 32 instruction-bit columns, one-hot opcode /
  funct3 selectors, immediates as linear combinations of bits; all of
  `lb/lh/lbu/lhu/sb/sh/lw/sw`, shifts, signed/unsigned compares, `ecall`,
  `fence`. The VM's decoder is strict so VM and AIR accept exactly the same
  programs.
* **Memory**: byte addresses are < 2³¹ (the VM traps above that; words are
  stored as (lo, hi) 16-bit limbs); registers live at word address 2²⁹+i; five accesses per cycle
  (fetch, rs1, rs2, mem-read, write). A sorted `(addr, clk)` table enforces
  read-after-write consistency and is tied to the CPU by a LogUp multiset.
  First reads of untouched words are forced to 0.
* **Continuations**: an execution is cut into segments of ≤ 2¹⁷ rows in one
  Fiat–Shamir transcript with one global LogUp challenge set. Memory is
  carried by offline memory checking with the segment index as clock (fifth
  pass above): per touched address an init entry citing the last toucher's
  stamp and a final write, plus a closing table spread over the segments in
  disjoint address ranges; the verifier checks one execution-wide multiset
  equation and chains `pc`, journal length, the halt flag and the closing
  ranges through public values. No Merkle memory tree is needed; the cost is
  one row per 5 addresses a segment touches.
* **Syscalls**: `halt`, `read_witness` (unconstrained), `write_journal`
  (looked up against a verifier-side sum), `write_output` and
  `write_output_buf` (private, not constrained — only the output's BLAKE3,
  computed by the guest, is attested).
* **Lookups**: 16-bit limb range checks, 4-bit nibble tables for
  `xor/or/and`, a 32-row power table for shifts; everything in one LogUp
  balance (three batched fractions per aux column, 34 aux columns).

## The STARK

* BabyBear (p = 2³¹−2²⁷+1) with challenges, LogUp fingerprints and the
  DEEP/FRI combination in F_{p⁴} = F_p[u]/(u⁴−11) (every extension column is
  committed as four base columns); blake3 Merkle trees; DEEP-ALI with
  composition chunks of degree < `bound`; FRI folding by 2 to a 32-point
  final layer; max constraint degree 4, checked at setup against the blowup.
  The 2-adic roots are ω₂ₖ = 31^((p−1)/2ᵏ), which prover and verifier share
  through `Fp::two_adic_root`; the LDE coset is shifted by the generator 31;
  the prover keeps every polynomial in bit-reversed coefficient order
  (`Committed::coeffs`, scaled lazily by `Committed::scale`) so that no
  transform ever permutes.
* **Zero knowledge** (ethSTARK style): the last `4·queries + 8` rows of
  every trace are uniformly random, transition constraints vanish only on the
  real rows (`Z_T = (xᴺ−1)/Π(x−gⁱ)` over the excluded rows, which the verifier
  evaluates in O(rows)), Merkle leaves are salted, and no two proofs of the
  same trace share a commitment. Trace degree stays < N, so blowup stays 4.
* **Challenge schedule**: two-phase. All main traces are committed, then
  one (β, γ) is drawn for the execution, then every segment's aux,
  composition, OOD, DEEP and FRI follow. No trace can adapt to a challenge
  it influenced, and multiset terms cancel across any pair of segments.
* Parameters: `zk()` = blowup 4, 40 queries, 16 bits of grinding (≈96 bits
  of FRI soundness); `zk_secure()` = blowup 8, 44 queries, 20 bits of
  grinding; `fast()` = no zero knowledge, no grinding (tests).

Negative tests (`cargo test --release`): forged register reads, ALU results,
journal entries, skipped halts, wrong image, wrong journal, dropped /
reordered segments, reordered link inputs, swapped toolchains, bit-flipped
proofs, tampered OOD values and openings are all rejected.

## Honest status

1. **The compiler is chibicc, not clang.** It is a real C11 compiler
   (preprocessor, structs, VLAs, floats, …) and it self-hosts inside the
   VM, but it targets x86-64 assembly only; the "link" guest is still the v0
   concatenation stand-in. Porting LLVM is the next toolchain step (below).
2. **Three host-portability patches to chibicc** (`guest/zkcc/zkbuild-port.patch`,
   207 lines): LP64 assumptions made explicit (`long long`, `%lld`,
   `strtoull`), and `long double` constants held as `double` with the x87
   80-bit encoding synthesized on output (RV32 has no 80-bit type; literals
   beyond double precision are the only observable difference). Everything
   else is upstream.
3. **No recursive aggregation.** A `BuildCertificate` bundles child proofs;
   verification time and size are linear in their number (0.83 s and 132 MB
   for 205 segments). Recursion needs a field-friendly hash (see roadmap);
   with blake3, verifying one segment proof inside the VM would cost ≈50 M
   cycles, so it is not worth building yet.
4. **Prover speed.** ≈2.5 s per 2¹⁷-row ZK segment on 2 cores; chibicc's
   largest TU (`parse.c`, 149 M cycles) would take ≈50 min here — except
   that it currently cannot be segmented at 2¹⁷ rows at all, see 6. The
   AVX-512 paths (`Fp16`/`Fp4x16`, `blake3x16`) are compiled in
   only with `-C target-cpu=native` on an AVX-512 machine (`.cargo/config.toml`);
   elsewhere the portable scalar code runs, several times slower.
5. Unaudited; development parameters; no grinding; single machine.
6. **Proof size, no recursion.** A 2¹⁷-row ZK segment proof is ≈570 KB
   (40 queries × openings of 467 base columns plus Merkle paths); the
   compiler's own build is 4 145 segments, a 2.37 GB `BuildCertificate`
   (a streaming container — members are verified one at a time) that
   verifies in 21 s. Compressing that to one small proof needs the
   recursion stack described in roadmap item 1.
7. **The link guest is the v0 stand-in**: 67 M cycles to hash 2.6 MB of
   objects at 22 cycles/byte (12 % of the self-build's proving time); a
   real linker guest will replace it with the LLVM work.

## Roadmap to an attested LLVM

1. **Recursion, then a SNARK wrap — one small proof for the whole build.**
   The numbers rule out the naive route: verifying one 2¹⁷-row segment
   proof *inside the RV32 VM* costs ≈25 M cycles with BLAKE3 Merkle trees
   (≈190 segments of proving per segment verified — it cannot converge),
   and still ≈3–4 M cycles with a Poseidon2 hash chip (DEEP: 40 queries ×
   467 columns of F_{p⁴} arithmetic emulated in RV32 instructions; ratio
   ≈30:1). Every production zkVM therefore has a second proof system for
   recursion. The staged plan, each step a prerequisite of the next, with
   the first brick already in the tree:
   * **Poseidon2 over BabyBear — done as a primitive** (`zkb-stark/src/poseidon2.rs`):
     RISC Zero's width-24 instance (rate 16, x⁷, 8+21 rounds), bit-exact
     with their published test vectors, scalar and 16-way AVX-512
     (≈0.25 µs per permutation uncontended). Cost of switching the base
     prover's commitments to it: a 2¹⁹-leaf main commitment is ≈10 M
     permutations ≈ 1.3 s on two cores against BLAKE3's 0.16 s, i.e. the
     segment would go from ≈3.0 s to ≈5.5 s (two phases). Hence it is a
     config switch to add, not the default, until the recursion layer
     needs it — and the switch is the next step: `Committed`/`MerkleTree`
     over `[Fp; 8]` digests, a Poseidon2 sponge transcript, and a
     permutation AIR table (24 cells × 29 rounds of columns, ≈1 row per
     permutation) so a verifier checks paths natively.
   * **A native-field recursion AIR**: a straight-line "circuit" machine —
     the verifier of a fixed proof shape has no data-dependent control
     flow — whose rows are F_{p⁴} multiply-adds, Poseidon2 permutations
     (via the chip) and witness lookups, with a LogUp memory for
     intermediate values. The verifier program is generated by symbolic
     execution: `verifier.rs` rewritten over a trait, with one
     implementation that computes and one that records; the OOD constraint
     check already is generic (`FieldLike`). Cost model per base segment
     verified: ≈21 k permutations (4 trees × 40 queries × 2 openings × 19
     levels + leaf absorptions + FRI) ≈ 37 k row-equivalents, ≈37 k F_{p⁴}
     mul-adds for DEEP, ≈5 k for OOD and FRI folds — ≈60 k rows of a
     ≈300-column AIR, i.e. ≈0.5 of the base segment. The same machine
     verifies its own (smaller) proofs, giving a binary tree whose root
     proof's journal is the build claim; total recursion cost ≈1× the base
     proving, so the chibicc self-build would take ≈8 h with Poseidon2
     commitments and ≈4 h more of recursion.
   * **SNARK wrap**: a Groth16/PLONK circuit over BN254 verifying the root
     recursion proof. Honest sizing: BabyBear and Poseidon2 emulated in
     BN254 cost ≈40 constraints per base multiplication, so even a
     deliberately small root proof (blowup 16, 20 queries, ≈200 columns)
     is ≈5–10 M R1CS constraints — arkworks Groth16 at that size needs
     ≈30 GB of RAM and a trusted setup; it does not fit this 7 GB machine.
     The output would be ≈200–300 bytes, verifiable in milliseconds,
     on-chain if wanted. Wrapping segment proofs directly, without the
     recursion stage, would keep thousands of proofs at minutes each.
2. **Prover throughput**: 2¹⁷-row segments cost ≈3.2 s today (phase 1 ≈1.0 s, then constraints
   ≈25%, NTT ≈22%, hashing ≈10%, trace ≈6%). The levers now: fewer LogUp
   terms (a pair-of-bytes table for the bitwise ops and the multiplier
   instead of 16-bit limbs + nibbles), fewer columns (byte-oriented decode
   tables instead of 32 bit columns, a "null" last row that turns the
   degree-4 last-row LogUp constraint into a degree-1 one), a Karatsuba
   F_{p⁴} multiply, then GPU NTT/hashing.
3. **Phase-1 cost of the global challenge** (≈1.0 s of the ≈3.2 s per
   segment): cache the phase-1 LDE/Merkle trees of as many segments as
   memory allows (≈0.8 GB each), or move main commitments to Poseidon2 on
   the GPU together with item 1.
4. **The LLVM guest**: clang+lld as one static RV32IM program on a
   freestanding libc++ subset, `fs::`/`Process::` mapped onto the witness VFS
   (preprocessed TU + exact header set as `header_dependency_root`),
   `SOURCE_DATE_EPOCH`, locale and seed fixed by the OS layer — exactly as
   `guest/libc` does for chibicc. Expect 10⁹–10¹⁰ cycles per TU.
5. **Bootstrap**: prove the build of the guest compiler itself from source
   with a previous (or diverse-double-compiled) attested toolchain so
   `toolchain_root` carries its own certificate — the Level-2 picture of the
   design note. `bootstrap.sh --prove-self` is that step for chibicc.

## Certificates

```text
ObjectCertificate { source_root, flags_root, toolchain_root, object_hash, n_files, proof }
LinkCertificate   { inputs_root, flags_root, toolchain_root, output_hash, n_objects, proof }
BuildCertificate  { objects: [ObjectCertificate], link: LinkCertificate }   // root = H(digests)
```

`toolchain_root = H("zkb-toolchain-v1" ‖ digest(tu image) ‖ digest(link image))`,
where an image digest commits to the entry point and every nonzero word of
the loaded ELF — for `zkcc` that includes the compiler, its libc and the
sysroot headers. The verifier must trust those images (it is "the known
compiler C"); everything else is proven.
