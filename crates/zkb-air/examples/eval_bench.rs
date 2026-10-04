//! Micro-benchmark of the 8-row constraint evaluation (the composition inner loop).
use std::sync::Arc;
use std::time::Instant;
use zkb_air::Rv32Air;
use zkb_field::simd::{Fp4x16, Fp16};
use zkb_field::{Fp, Fp4};
use zkb_stark::{Air, CombinationWeights, ConstraintSink, RowView, SinkMode};

fn main() {
    let air = Rv32Air::new();
    let shape = air.shape();
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        Fp::new(seed)
    };
    let mk = |n: usize, rnd: &mut dyn FnMut() -> Fp| -> Vec<Fp16> { (0..n).map(|_| Fp16::from_lanes(core::array::from_fn(|_| rnd().0))).collect() };
    let mn = mk(shape.main_width, &mut rnd);
    let mnx = mk(shape.main_width, &mut rnd);
    let fx = mk(shape.fixed_width, &mut rnd);
    let fxn = mk(shape.fixed_width, &mut rnd);
    let mke = |n: usize, rnd: &mut dyn FnMut() -> Fp| -> Vec<Fp4x16> { (0..n).map(|_| Fp4x16 { c: core::array::from_fn(|_| Fp16::from_lanes(core::array::from_fn(|_| rnd().0))) }).collect() };
    let ax = mke(shape.aux_width, &mut rnd);
    let axn = mke(shape.aux_width, &mut rnd);
    let ch: Vec<Fp4> = (0..4).map(|_| Fp4([rnd(), rnd(), rnd(), rnd()])).collect();
    let public: Vec<Fp4> = (0..8).map(|_| Fp4([rnd(), rnd(), rnd(), rnd()])).collect();
    let aux_public: Vec<Fp4> = (0..shape.aux_width).map(|_| Fp4([rnd(), rnd(), rnd(), rnd()])).collect();
    let alphas: Vec<Fp4> = (0..shape.num_transition + shape.num_first + shape.num_last).map(|_| Fp4([rnd(), rnd(), rnd(), rnd()])).collect();
    let weights = Arc::new(CombinationWeights::<Fp16>::new(alphas, shape.num_transition, shape.num_first));
    let row = RowView::new(&mn[..], &mnx[..], &fx[..], &fxn[..], &ax[..], &axn[..]);
    for (label, mode) in [("All", SinkMode::All), ("HighOnly", SinkMode::HighOnly)] {
        let mut sink: ConstraintSink<Fp16> = ConstraintSink::accumulating(mode, weights.clone());
        let reps = 20000;
        let mut acc = Fp4x16::ZERO;
        let t0 = Instant::now();
        for _ in 0..reps {
            sink.clear();
            air.eval(&row, &ch, &public, &aux_public, &mut sink);
            let (lo, hi) = sink.combined(Fp16::ONE, Fp16::ONE, Fp16::ONE);
            acc += lo + hi;
        }
        let dt = t0.elapsed();
        println!("{label:8}: {:.2} us per 16-row eval ({:.0} cycles @2.1GHz)  [{}]", dt.as_secs_f64() * 1e6 / reps as f64, dt.as_secs_f64() * 2.1e9 / reps as f64, acc.lane(0).0[0].0 % 7);
    }
    {
        let mut sink: ConstraintSink<Fp16> = ConstraintSink::new(SinkMode::HighOnly);
        air.eval(&row, &ch, &public, &aux_public, &mut sink);
        println!("HighOnly evaluates {} transition + {} first + {} last values", sink.transition.len(), sink.first.len(), sink.last.len());
        let t0 = Instant::now();
        let mut acc = Fp4x16::ZERO;
        for _ in 0..20000 {
            let tb: zkb_air::lookups::TermBuf<Fp16> = zkb_air::lookups::TermBuf::default();
            acc += std::hint::black_box(&tb).items[3].denom;
        }
        println!("TermBuf::default: {:.2} us [{}]", t0.elapsed().as_secs_f64() * 1e6 / 20000.0, acc.lane(0).0[0].0 % 7);
        let t0 = Instant::now();
        let mut acc = Fp16::ZERO;
        for _ in 0..20000 {
            let d = zkb_air::constraints::decode(&air.l, std::hint::black_box(&mn[..]));
            acc += d.instr.0 + d.rs1 + d.rd;
        }
        println!("decode: {:.2} us [{}]", t0.elapsed().as_secs_f64() * 1e6 / 20000.0, acc.lane(0).0 % 7);
    }
    // ---- parts: row_terms alone, batch arithmetic alone
    use zkb_air::lookups::{row_terms, Enc, TermBuf, BATCH};
    let e = Enc::new(ch[0], ch[1]);
    let reps = 20000;
    let mut terms: TermBuf<Fp16> = TermBuf::default();
    let t0 = Instant::now();
    let mut acc = Fp4x16::ZERO;
    for _ in 0..reps {
        terms.clear();
        row_terms(&air.l, &mn[..], &fx[..], &e, &mut terms);
        acc += terms.as_slice()[5].denom;
    }
    while terms.len % BATCH != 0 {
        terms.push(zkb_air::lookups::Term::new(Fp16::ZERO, Fp4x16::ONE));
    }
    let dt = t0.elapsed();
    println!("row_terms: {:.2} us ({} terms) [{}]", dt.as_secs_f64() * 1e6 / reps as f64, terms.len, acc.lane(0).0[0].0 % 7);
    let t0 = Instant::now();
    let mut acc = Fp4x16::ZERO;
    for _ in 0..reps {
        for (i, tr) in terms.as_slice().chunks(BATCH).enumerate() {
            let (t1, t2, t3) = (tr[0], tr[1], tr[2]);
            let s = ax[i];
            let sn = axn[i];
            let d12 = t1.denom * t2.denom;
            let d123 = d12 * t3.denom;
            let k = t3.denom * (t1.mul(t2.denom) + t2.mul(t1.denom)) + t3.mul(d12);
            acc += (sn - s) * d123 - k;
            acc += (Fp4x16::splat(aux_public[i]) - s) * d123 - k;
        }
    }
    let dt = t0.elapsed();
    println!("batch arithmetic: {:.2} us [{}]", dt.as_secs_f64() * 1e6 / reps as f64, acc.lane(0).0[0].0 % 7);
    let t0 = Instant::now();
    let mut acc = Fp4x16::ZERO;
    let x = ax[1];
    let y = axn[2];
    for i in 0..reps * 100 {
        acc += x * y;
        acc.c[0] = acc.c[0] + Fp16::splat(Fp::new(i as u64));
    }
    let dt = t0.elapsed();
    println!("Fp4x16 mul (latency-bound chain): {:.1} ns [{}]", dt.as_secs_f64() * 1e9 / (reps * 100) as f64, acc.lane(0).0[0].0 % 7);
    let mut accs = [Fp4x16::ZERO; 8];
    let t0 = Instant::now();
    for i in 0..reps * 100 {
        let v = Fp4x16::splat(Fp4([Fp::new(i as u64), Fp::new(3), Fp::ZERO, Fp::ONE]));
        for k in 0..8 {
            accs[k] += ax[k] * v;
        }
    }
    let dt = t0.elapsed();
    let mut tot = Fp4x16::ZERO;
    for a in accs {
        tot += a;
    }
    println!("Fp4x16 mul (throughput, 8 independent): {:.1} ns [{}]", dt.as_secs_f64() * 1e9 / (reps * 800) as f64, tot.lane(0).0[0].0 % 7);
}

#[allow(dead_code)]
pub fn parts() {}
