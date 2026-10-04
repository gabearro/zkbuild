use std::time::Instant;
use zkb_field::simd::{Fp16, LANES};
use zkb_field::Fp;
use zkb_stark::poseidon2::*;

fn main() {
    let mut st: [Fp; CELLS] = core::array::from_fn(|i| Fp::new(i as u64 + 1));
    let n = 200_000;
    let t0 = Instant::now();
    for _ in 0..n {
        permute(&mut st);
    }
    let scalar = t0.elapsed().as_secs_f64() / n as f64;
    println!("scalar permutation: {:.0} ns [{}]", scalar * 1e9, st[0].as_canonical() % 7);
    let mut st16: [Fp16; CELLS] = core::array::from_fn(|i| Fp16::splat(Fp::new(i as u64 + 1)));
    let t0 = Instant::now();
    for _ in 0..n {
        permute16(&mut st16);
    }
    let vec = t0.elapsed().as_secs_f64() / (n * LANES) as f64;
    println!("16-way permutation: {:.0} ns per permutation [{}]", vec * 1e9, st16[0].lane(0).as_canonical() % 7);
    // a 2^19-leaf commitment of 286 columns: ceil(286/16) = 18 permutations per leaf + 2^19 tree nodes
    let leaves = 1u64 << 19;
    let perms = leaves * 18 + leaves;
    println!("main-trace commitment (2^19 leaves x 286 cols): {} permutations = {:.2} s on one core (16-way), vs BLAKE3x16 0.16 s on two cores", perms, perms as f64 * vec);
}
