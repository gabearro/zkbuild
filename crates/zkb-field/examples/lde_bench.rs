use std::time::Instant;
use zkb_field::ntt::{coset_evaluate, lde_coset_major, NttPlan};
use zkb_field::Fp;
fn main() {
    let log = 17;
    let n = 1usize << log;
    let plan_n = NttPlan::new(n);
    let plan_m = NttPlan::new(n * 4);
    let coeffs: Vec<Fp> = (0..n).map(|i| Fp::new(i as u64 * 7919 + 13)).collect();
    let reps = 10;
    let t0 = Instant::now();
    for _ in 0..reps { let v = coset_evaluate(&plan_m, &coeffs, Fp::new(7)); std::hint::black_box(v); }
    println!("big 4N NTT:      {:?}", t0.elapsed() / reps);
    let t0 = Instant::now();
    for _ in 0..reps { let v = lde_coset_major(&plan_n, &coeffs, Fp::new(7), 2); std::hint::black_box(v); }
    println!("4 x N NTT:       {:?}", t0.elapsed() / reps);
    // 250 columns in parallel like the prover, then again while holding 1.5 GB
    use rayon::prelude::*;
    let cols: Vec<Vec<Fp>> = (0..250).map(|c| (0..n).map(|i| Fp::new((i * 31 + c) as u64)).collect()).collect();
    let t0 = Instant::now();
    let l1: Vec<Vec<Fp>> = cols.par_iter().map(|c| lde_coset_major(&plan_n, c, Fp::new(7), 2)).collect();
    println!("250-col LDE (parallel):            {:?}", t0.elapsed());
    let hold: Vec<Vec<Fp>> = (0..375).map(|c| vec![Fp::new(c as u64); n * 4]).collect();
    let t0 = Instant::now();
    let l2: Vec<Vec<Fp>> = cols.par_iter().map(|c| lde_coset_major(&plan_n, c, Fp::new(7), 2)).collect();
    println!("250-col LDE while holding 1.5 GB:  {:?}", t0.elapsed());
    drop(l1);
    let t0 = Instant::now();
    let l3: Vec<Vec<Fp>> = cols.par_iter().map(|c| lde_coset_major(&plan_n, c, Fp::new(7), 2)).collect();
    println!("250-col LDE after freeing 1 GB:    {:?}", t0.elapsed());
    std::hint::black_box((&hold, &l2, &l3));
    let mut v = coeffs.clone();
    let t0 = Instant::now();
    for _ in 0..reps { plan_n.inverse(&mut v); }
    println!("iNTT N:          {:?}", t0.elapsed() / reps);
    // hashing cost of one column-equivalent: 4N * 8 bytes
    let bytes = vec![1u8; n * 4 * 8];
    let t0 = Instant::now();
    for _ in 0..reps { std::hint::black_box(blake3::hash(&bytes)); }
    println!("blake3 of 4N*8B: {:?}", t0.elapsed() / reps);
}
#[allow(dead_code)]
fn unused() {}
