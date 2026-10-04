use std::time::Instant;
use zkb_field::ntt::{lde_from_bitrev_coeffs, CosetPowers, NttPlan};
use zkb_field::Fp;
fn main() {
    println!("avx512: {}", zkb_field::simd::HAS_AVX512);
    for log in [17u32, 20] {
        let n = 1usize << log;
        let plan = NttPlan::new(n);
        let mut v: Vec<Fp> = (0..n).map(|i| Fp::new(i as u64 * 7919 + 13)).collect();
        let reps = 10;
        let t0 = Instant::now();
        for _ in 0..reps {
            plan.forward(&mut v);
        }
        println!("NTT 2^{log} forward (natural):      {:?}", t0.elapsed() / reps);
        let t0 = Instant::now();
        for _ in 0..reps {
            plan.forward_from_bitrev(&mut v);
        }
        println!("NTT 2^{log} dit (from bitrev):      {:?}", t0.elapsed() / reps);
        let t0 = Instant::now();
        for _ in 0..reps {
            plan.inverse_to_bitrev_unscaled(&mut v);
        }
        println!("NTT 2^{log} dif (to bitrev):        {:?}", t0.elapsed() / reps);
        if log == 17 {
            let pows = CosetPowers::get(log, Fp::new(7), 2, plan.n_inv());
            let mut out = vec![Fp::ZERO; 4 * n];
            let t0 = Instant::now();
            for _ in 0..reps {
                plan.inverse_to_bitrev_unscaled(&mut v);
                lde_from_bitrev_coeffs(&plan, &v, &pows, &mut out);
            }
            println!("LDE 2^{log} x4 (dif + 4 dit):      {:?}", t0.elapsed() / reps);
        }
    }
}
