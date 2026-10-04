//! Number-theoretic transform over Fp and Fp4 (coefficient-wise).
//!
//! Two cores are provided:
//! * `dit`: decimation-in-time, bit-reversed input → natural output;
//! * `dif`: decimation-in-frequency, natural input → bit-reversed output.
//!
//! An LDE therefore needs no bit-reversal permutations at all: `dif` on the
//! trace evaluations yields bit-reversed coefficients, which `dit` turns into
//! coset evaluations after a point-wise multiplication by a (bit-reversed)
//! power table. `forward` / `inverse` keep the natural-order API for callers
//! that need it.
//!
//! Stages are fused in radix-4 pairs; the four stages that act inside
//! 16-element blocks are done with a 16×16 in-register transpose so every
//! butterfly of a 2^k ≥ 256 transform is a 16-lane AVX-512 operation.

use crate::simd::{butterfly_slices, dif_butterfly_slices, mul_slices, Fp16, Small16Twiddles, LANES};
use crate::{Fp, Fp4};

/// Log2 of the block size handled by the transposed small-stage kernels.
const SMALL_LOG: usize = 4;
const SMALL_MIN_LOG: u32 = 8;

#[inline(always)]
pub fn bit_reverse_index(i: usize, bits: u32) -> usize {
    if bits == 0 {
        0
    } else {
        ((i as u64).reverse_bits() >> (64 - bits)) as usize
    }
}

pub fn bit_reverse_permute<T: Copy>(v: &mut [T]) {
    let n = v.len();
    if n <= 2 {
        return;
    }
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = bit_reverse_index(i, bits);
        if j > i {
            v.swap(i, j);
        }
    }
}

pub struct NttPlan {
    pub n: usize,
    pub log_n: u32,
    /// stage_tw[s] holds the `2^s` twiddles of stage s (ω_{2^(s+1)}^k, k < 2^s).
    stage_tw: Vec<Vec<Fp>>,
    stage_tw_inv: Vec<Vec<Fp>>,
    small: Option<Small16Twiddles>,
    small_inv: Option<Small16Twiddles>,
    n_inv: Fp,
    omega: Fp,
}

impl NttPlan {
    pub fn new(n: usize) -> NttPlan {
        assert!(n.is_power_of_two());
        let log_n = n.trailing_zeros();
        let omega = Fp::two_adic_root(log_n);
        let mut stage_tw = Vec::with_capacity(log_n as usize);
        let mut stage_tw_inv = Vec::with_capacity(log_n as usize);
        for s in 0..log_n {
            let half = 1usize << s;
            let w_len = Fp::two_adic_root(s + 1);
            let w_len_inv = w_len.inverse();
            let mut tw = Vec::with_capacity(half);
            let mut twi = Vec::with_capacity(half);
            let mut w = Fp::ONE;
            let mut wi = Fp::ONE;
            for _ in 0..half {
                tw.push(w);
                twi.push(wi);
                w *= w_len;
                wi *= w_len_inv;
            }
            stage_tw.push(tw);
            stage_tw_inv.push(twi);
        }
        let small = if log_n >= SMALL_MIN_LOG { Some(Small16Twiddles::new(&stage_tw)) } else { None };
        let small_inv = if log_n >= SMALL_MIN_LOG { Some(Small16Twiddles::new(&stage_tw_inv)) } else { None };
        NttPlan { n, log_n, stage_tw, stage_tw_inv, small, small_inv, n_inv: Fp::new(n as u64).inverse(), omega }
    }

    pub fn omega(&self) -> Fp {
        self.omega
    }

    pub fn n_inv(&self) -> Fp {
        self.n_inv
    }

    /// Scalar DIT reference core (tiny transforms and the non-AVX-512
    /// fallback): bit-reversed input → natural output.
    fn dit_scalar(v: &mut [Fp], stages: &[Vec<Fp>]) {
        let mut half = 1usize;
        for tw in stages {
            for blk in v.chunks_exact_mut(half * 2) {
                let (lo, hi) = blk.split_at_mut(half);
                for k in 0..half {
                    let a = lo[k];
                    let b = hi[k] * tw[k];
                    lo[k] = a + b;
                    hi[k] = a - b;
                }
            }
            half *= 2;
        }
    }

    fn dif_scalar(v: &mut [Fp], stages: &[Vec<Fp>]) {
        let mut half = v.len() / 2;
        for tw in stages.iter().rev() {
            for blk in v.chunks_exact_mut(half * 2) {
                let (lo, hi) = blk.split_at_mut(half);
                for k in 0..half {
                    let a = lo[k];
                    let b = hi[k];
                    lo[k] = a + b;
                    hi[k] = (a - b) * tw[k];
                }
            }
            half /= 2;
        }
    }

    /// Radix-4 DIT pass fusing stages `s` and `s+1` (half = 2^s ≥ LANES).
    #[inline]
    fn dit_radix4_pass(v: &mut [Fp], s: usize, twa: &[Fp], twb: &[Fp]) {
        let half = 1usize << s;
        let len = half * 4;
        for blk in v.chunks_exact_mut(len) {
            let (q0, rest) = blk.split_at_mut(half);
            let (q1, rest) = rest.split_at_mut(half);
            let (q2, q3) = rest.split_at_mut(half);
            let twb_lo = &twb[..half];
            let twb_hi = &twb[half..2 * half];
            let twa = &twa[..half];
            #[inline(always)]
            fn bfly(q0: &mut [Fp], q1: &mut [Fp], q2: &mut [Fp], q3: &mut [Fp], twa: &[Fp], twb_lo: &[Fp], twb_hi: &[Fp], k: usize) {
                let wa = Fp16::from_slice(&twa[k..k + LANES]);
                let x0 = Fp16::from_slice(&q0[k..k + LANES]);
                let x1 = Fp16::from_slice(&q1[k..k + LANES]) * wa;
                let x2 = Fp16::from_slice(&q2[k..k + LANES]);
                let x3 = Fp16::from_slice(&q3[k..k + LANES]) * wa;
                let y0 = x0 + x1;
                let y1 = x0 - x1;
                let y2 = x2 + x3;
                let y3 = x2 - x3;
                let z2 = y2 * Fp16::from_slice(&twb_lo[k..k + LANES]);
                let z3 = y3 * Fp16::from_slice(&twb_hi[k..k + LANES]);
                (y0 + z2).store_to(&mut q0[k..]);
                (y0 - z2).store_to(&mut q2[k..]);
                (y1 + z3).store_to(&mut q1[k..]);
                (y1 - z3).store_to(&mut q3[k..]);
            }
            // two independent butterflies per iteration: the multiply chains
            // are latency-bound, so let them overlap
            let mut k = 0;
            while k + 2 * LANES <= half {
                bfly(q0, q1, q2, q3, twa, twb_lo, twb_hi, k);
                bfly(q0, q1, q2, q3, twa, twb_lo, twb_hi, k + LANES);
                k += 2 * LANES;
            }
            while k < half {
                bfly(q0, q1, q2, q3, twa, twb_lo, twb_hi, k);
                k += LANES;
            }
        }
    }

    /// Radix-4 DIF pass fusing stages `s` (outer, half = 2^s) and `s-1`.
    #[inline]
    fn dif_radix4_pass(v: &mut [Fp], s: usize, twa: &[Fp], twb: &[Fp]) {
        let half = 1usize << (s - 1);
        let len = half * 4;
        for blk in v.chunks_exact_mut(len) {
            let (q0, rest) = blk.split_at_mut(half);
            let (q1, rest) = rest.split_at_mut(half);
            let (q2, q3) = rest.split_at_mut(half);
            let twa_lo = &twa[..half];
            let twa_hi = &twa[half..2 * half];
            let twb = &twb[..half];
            #[inline(always)]
            fn bfly(q0: &mut [Fp], q1: &mut [Fp], q2: &mut [Fp], q3: &mut [Fp], twa_lo: &[Fp], twa_hi: &[Fp], twb: &[Fp], k: usize) {
                let x0 = Fp16::from_slice(&q0[k..k + LANES]);
                let x1 = Fp16::from_slice(&q1[k..k + LANES]);
                let x2 = Fp16::from_slice(&q2[k..k + LANES]);
                let x3 = Fp16::from_slice(&q3[k..k + LANES]);
                let y0 = x0 + x2;
                let y2 = (x0 - x2) * Fp16::from_slice(&twa_lo[k..k + LANES]);
                let y1 = x1 + x3;
                let y3 = (x1 - x3) * Fp16::from_slice(&twa_hi[k..k + LANES]);
                let wb = Fp16::from_slice(&twb[k..k + LANES]);
                (y0 + y1).store_to(&mut q0[k..]);
                ((y0 - y1) * wb).store_to(&mut q1[k..]);
                (y2 + y3).store_to(&mut q2[k..]);
                ((y2 - y3) * wb).store_to(&mut q3[k..]);
            }
            let mut k = 0;
            while k + 2 * LANES <= half {
                bfly(q0, q1, q2, q3, twa_lo, twa_hi, twb, k);
                bfly(q0, q1, q2, q3, twa_lo, twa_hi, twb, k + LANES);
                k += 2 * LANES;
            }
            while k < half {
                bfly(q0, q1, q2, q3, twa_lo, twa_hi, twb, k);
                k += LANES;
            }
        }
    }

    /// DIT core: bit-reversed input → natural output. With `scaled = Some((src, scale))`
    /// the first pass reads `src[i]·scale[i]` instead of `v` (fused LDE multiply).
    fn dit(&self, v: &mut [Fp], stages: &[Vec<Fp>], small: &Option<Small16Twiddles>, scaled: Option<(&[Fp], &[Fp])>) {
        assert_eq!(v.len(), self.n);
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        if let Some(sm) = small {
            match scaled {
                Some((src, scale)) => crate::simd::dit_small_stages_scaled(src, scale, v, sm),
                None => crate::simd::dit_small_stages(v, sm),
            }
            let mut s = SMALL_LOG;
            while s + 1 < stages.len() {
                Self::dit_radix4_pass(v, s, &stages[s], &stages[s + 1]);
                s += 2;
            }
            if s < stages.len() {
                let half = 1usize << s;
                for blk in v.chunks_exact_mut(half * 2) {
                    let (lo, hi) = blk.split_at_mut(half);
                    butterfly_slices(lo, hi, &stages[s][..half]);
                }
            }
            return;
        }
        let _ = small;
        if let Some((src, scale)) = scaled {
            mul_slices(v, src, scale);
        }
        Self::dit_scalar(v, stages);
    }

    /// DIF core: natural input → bit-reversed output (unscaled).
    fn dif(&self, v: &mut [Fp], stages: &[Vec<Fp>], small: &Option<Small16Twiddles>) {
        assert_eq!(v.len(), self.n);
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        if let Some(sm) = small {
            let mut s = stages.len() - 1; // top stage index
            while s >= SMALL_LOG + 1 {
                Self::dif_radix4_pass(v, s, &stages[s], &stages[s - 1]);
                s -= 2;
            }
            if s == SMALL_LOG {
                let half = 1usize << SMALL_LOG;
                for blk in v.chunks_exact_mut(2 * half) {
                    let (lo, hi) = blk.split_at_mut(half);
                    dif_butterfly_slices(lo, hi, &stages[SMALL_LOG][..half]);
                }
            }
            crate::simd::dif_small_stages(v, sm);
            return;
        }
        let _ = small;
        Self::dif_scalar(v, stages);
    }

    /// Forward transform: coefficients (natural order) -> evaluations on <omega>.
    pub fn forward(&self, v: &mut [Fp]) {
        bit_reverse_permute(v);
        self.dit(v, &self.stage_tw, &self.small, None);
    }

    /// Inverse transform: evaluations -> coefficients (natural order).
    pub fn inverse(&self, v: &mut [Fp]) {
        self.dif(v, &self.stage_tw_inv, &self.small_inv);
        bit_reverse_permute(v);
        crate::simd::scale_slice(v, self.n_inv);
    }

    /// Forward transform from bit-reversed coefficients to natural-order evaluations.
    pub fn forward_from_bitrev(&self, v: &mut [Fp]) {
        self.dit(v, &self.stage_tw, &self.small, None);
    }

    /// `dst = NTT(src · scale)` from bit-reversed coefficients, with the
    /// point-wise multiplication fused into the first pass.
    pub fn forward_from_bitrev_scaled(&self, src: &[Fp], scale: &[Fp], dst: &mut [Fp]) {
        assert_eq!(src.len(), self.n);
        assert_eq!(scale.len(), self.n);
        self.dit(dst, &self.stage_tw, &self.small, Some((src, scale)));
    }

    /// Inverse transform from natural-order evaluations to bit-reversed
    /// coefficients, *without* the final 1/n scaling (fold `n_inv()` in later).
    pub fn inverse_to_bitrev_unscaled(&self, v: &mut [Fp]) {
        self.dif(v, &self.stage_tw_inv, &self.small_inv);
    }

    pub fn forward_ext(&self, v: &mut [Fp4]) {
        let mut parts = ext_split(v);
        for p in parts.iter_mut() {
            self.forward(p);
        }
        for (i, x) in v.iter_mut().enumerate() {
            *x = Fp4([parts[0][i], parts[1][i], parts[2][i], parts[3][i]]);
        }
    }

    pub fn inverse_ext(&self, v: &mut [Fp4]) {
        let mut parts = ext_split(v);
        for p in parts.iter_mut() {
            self.inverse(p);
        }
        for (i, x) in v.iter_mut().enumerate() {
            *x = Fp4([parts[0][i], parts[1][i], parts[2][i], parts[3][i]]);
        }
    }
}

/// Evaluate polynomial `coeffs` (degree < n) on the coset `shift * <omega_n>`.
pub fn coset_evaluate(plan: &NttPlan, coeffs: &[Fp], shift: Fp) -> Vec<Fp> {
    let n = plan.n;
    assert!(coeffs.len() <= n, "polynomial too large for domain");
    let mut v = vec![Fp::ZERO; n];
    let mut s = Fp::ONE;
    for i in 0..coeffs.len() {
        v[i] = coeffs[i] * s;
        s *= shift;
    }
    plan.forward(&mut v);
    v
}

/// Interpolate from evaluations on coset `shift * <omega_n>` back to coefficients.
pub fn coset_interpolate(plan: &NttPlan, evals: &[Fp], shift: Fp) -> Vec<Fp> {
    let mut v = evals.to_vec();
    plan.inverse(&mut v);
    let sinv = shift.inverse();
    let mut s = Fp::ONE;
    for c in v.iter_mut() {
        *c *= s;
        s *= sinv;
    }
    v
}

/// Low-degree extension by `2^log_blowup`, computed as one size-N transform
/// per coset. Output is coset-major: element `c * n + j` is the evaluation
/// at `shift · ω_M^c · ω_N^j`, i.e. natural LDE index `i = j·2^log_blowup + c`.
pub fn lde_coset_major(plan_n: &NttPlan, coeffs: &[Fp], shift: Fp, log_blowup: u32) -> Vec<Fp> {
    let mut out = vec![Fp::ZERO; plan_n.n << log_blowup];
    lde_coset_major_into(plan_n, coeffs, shift, log_blowup, &mut out);
    out
}

/// [`lde_coset_major`] into a caller-provided buffer of length `n << log_blowup`
/// (natural-order coefficients).
pub fn lde_coset_major_into(plan_n: &NttPlan, coeffs: &[Fp], shift: Fp, log_blowup: u32, out: &mut [Fp]) {
    let n = plan_n.n;
    assert!(coeffs.len() <= n);
    let mut br = vec![Fp::ZERO; n];
    for (i, c) in coeffs.iter().enumerate() {
        br[bit_reverse_index(i, plan_n.log_n)] = *c;
    }
    let pows = CosetPowers::get(plan_n.log_n, shift, log_blowup, Fp::ONE);
    lde_from_bitrev_coeffs(plan_n, &br, &pows, out);
}

/// Precomputed `scale · (shift · ω_M^c)^rev(i)` tables (bit-reversed order),
/// one per coset, cached per (log_n, shift, blowup, scale).
pub struct CosetPowers {
    pub log_blowup: u32,
    pub tables: Vec<Vec<Fp>>,
}

impl CosetPowers {
    pub fn compute(log_n: u32, shift: Fp, log_blowup: u32, scale: Fp) -> CosetPowers {
        let n = 1usize << log_n;
        let b = 1usize << log_blowup;
        let omega_m = Fp::two_adic_root(log_n + log_blowup);
        let mut tables = Vec::with_capacity(b);
        let mut coset_shift = shift;
        for _ in 0..b {
            let mut t = vec![Fp::ZERO; n];
            let mut s = scale;
            for i in 0..n {
                t[bit_reverse_index(i, log_n)] = s;
                s *= coset_shift;
            }
            tables.push(t);
            coset_shift *= omega_m;
        }
        CosetPowers { log_blowup, tables }
    }

    /// Cached accessor (tables are shared across all columns of a commitment).
    pub fn get(log_n: u32, shift: Fp, log_blowup: u32, scale: Fp) -> std::sync::Arc<CosetPowers> {
        use std::collections::HashMap;
        use std::sync::{Arc, Mutex, OnceLock};
        static CACHE: OnceLock<Mutex<HashMap<(u32, u32, u32, u32), Arc<CosetPowers>>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let key = (log_n, shift.0, log_blowup, scale.0);
        if let Some(p) = cache.lock().unwrap().get(&key) {
            return p.clone();
        }
        let p = Arc::new(CosetPowers::compute(log_n, shift, log_blowup, scale));
        cache.lock().unwrap().insert(key, p.clone());
        p
    }
}

/// LDE from bit-reversed coefficients (length n, with the scale folded into
/// `pows`): one DIT transform per coset, no permutations.
pub fn lde_from_bitrev_coeffs(plan_n: &NttPlan, coeffs_br: &[Fp], pows: &CosetPowers, out: &mut [Fp]) {
    let n = plan_n.n;
    assert_eq!(coeffs_br.len(), n);
    let b = 1usize << pows.log_blowup;
    assert_eq!(out.len(), n * b);
    for c in 0..b {
        plan_n.forward_from_bitrev_scaled(coeffs_br, &pows.tables[c], &mut out[c * n..(c + 1) * n]);
    }
}

/// Position of natural LDE index `i` inside a coset-major LDE vector.
#[inline(always)]
pub fn coset_major_pos(i: usize, n: usize, log_blowup: u32) -> usize {
    let b = 1usize << log_blowup;
    (i & (b - 1)) * n + (i >> log_blowup)
}

pub fn coset_evaluate_ext(plan: &NttPlan, coeffs: &[Fp4], shift: Fp) -> Vec<Fp4> {
    let n = plan.n;
    assert!(coeffs.len() <= n);
    let mut v = vec![Fp4::ZERO; n];
    let mut s = Fp::ONE;
    for i in 0..coeffs.len() {
        v[i] = coeffs[i].mul_base(s);
        s *= shift;
    }
    plan.forward_ext(&mut v);
    v
}

pub fn coset_interpolate_ext(plan: &NttPlan, evals: &[Fp4], shift: Fp) -> Vec<Fp4> {
    let mut v = evals.to_vec();
    plan.inverse_ext(&mut v);
    let sinv = shift.inverse();
    let mut s = Fp::ONE;
    for c in v.iter_mut() {
        *c = c.mul_base(s);
        s *= sinv;
    }
    v
}

/// The four coefficient vectors of an extension vector.
pub fn ext_split(v: &[Fp4]) -> [Vec<Fp>; 4] {
    core::array::from_fn(|k| v.iter().map(|x| x.0[k]).collect())
}

pub fn ext_join(parts: &[Vec<Fp>; 4]) -> Vec<Fp4> {
    (0..parts[0].len()).map(|i| Fp4([parts[0][i], parts[1][i], parts[2][i], parts[3][i]])).collect()
}

/// Zero-pad bit-reversed coefficients of length n to length n << k (still
/// bit-reversed): natural index j < n lands at bit-reversed position
/// rev_n(j) << k.
pub fn pad_bitrev(coeffs_br: &[Fp], k: u32) -> Vec<Fp> {
    let n = coeffs_br.len();
    let mut out = vec![Fp::ZERO; n << k];
    for (j, c) in coeffs_br.iter().enumerate() {
        out[j << k] = *c;
    }
    out
}

/// Evaluate on `shift·<ω_m>` from bit-reversed coefficients of length m
/// (scale folded into `pows`): `pows` must come from
/// `CosetPowers::get(log_m, shift, 0, scale)`.
pub fn evaluate_from_bitrev(plan: &NttPlan, coeffs_br: &[Fp], pows: &CosetPowers) -> Vec<Fp> {
    let mut v = vec![Fp::ZERO; plan.n];
    plan.forward_from_bitrev_scaled(coeffs_br, &pows.tables[0], &mut v);
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly;

    #[test]
    fn lde_coset_major_matches_big_ntt() {
        let n = 16;
        let plan_n = NttPlan::new(n);
        let plan_m = NttPlan::new(n * 4);
        let coeffs: Vec<Fp> = (0..n).map(|i| Fp::new(i as u64 * 11 + 3)).collect();
        let shift = Fp::new(7);
        let big = coset_evaluate(&plan_m, &coeffs, shift);
        let cm = lde_coset_major(&plan_n, &coeffs, shift, 2);
        for i in 0..n * 4 {
            assert_eq!(cm[coset_major_pos(i, n, 2)], big[i], "i={i}");
        }
    }

    #[test]
    fn roundtrip() {
        for log in [0u32, 1, 2, 6, 8, 9, 10] {
            let n = 1usize << log;
            let plan = NttPlan::new(n);
            let coeffs: Vec<Fp> = (0..n).map(|i| Fp::new((i * i + 7) as u64)).collect();
            let mut v = coeffs.clone();
            plan.forward(&mut v);
            let omega = plan.omega();
            for i in 0..n {
                let x = omega.pow(i as u64);
                assert_eq!(v[i], poly::eval(&coeffs, x), "n={n} i={i}");
            }
            plan.inverse(&mut v);
            assert_eq!(v, coeffs);
        }
    }

    #[test]
    fn bitrev_cores_match_natural_api() {
        for log in [0u32, 1, 3, 7, 8, 9, 10, 11, 12] {
            let n = 1usize << log;
            let plan = NttPlan::new(n);
            let coeffs: Vec<Fp> = (0..n).map(|i| Fp::new((i as u64 * 0x9E37_79B9_7F4A_7C15).wrapping_add(5))).collect();
            // forward_from_bitrev(bitrev(c)) == forward(c)
            let mut nat = coeffs.clone();
            plan.forward(&mut nat);
            let mut br = coeffs.clone();
            bit_reverse_permute(&mut br);
            plan.forward_from_bitrev(&mut br);
            assert_eq!(br, nat, "dit n={n}");
            // inverse_to_bitrev_unscaled then bitrev + scale == inverse
            let mut back = nat.clone();
            plan.inverse_to_bitrev_unscaled(&mut back);
            bit_reverse_permute(&mut back);
            for x in back.iter_mut() {
                *x *= plan.n_inv();
            }
            assert_eq!(back, coeffs, "dif n={n}");
            let mut inv = nat.clone();
            plan.inverse(&mut inv);
            assert_eq!(inv, coeffs, "inverse n={n}");
        }
    }

    #[test]
    fn lde_from_bitrev_with_scale() {
        let n = 512;
        let plan = NttPlan::new(n);
        let evals: Vec<Fp> = (0..n).map(|i| Fp::new(i as u64 * 97 + 11)).collect();
        let shift = Fp::new(7);
        let coeffs = coset_interpolate(&plan, &evals, Fp::ONE);
        let reference = lde_coset_major(&plan, &coeffs, shift, 2);
        let mut br = evals.clone();
        plan.inverse_to_bitrev_unscaled(&mut br);
        let pows = CosetPowers::get(plan.log_n, shift, 2, plan.n_inv());
        let mut out = vec![Fp::ZERO; 4 * n];
        lde_from_bitrev_coeffs(&plan, &br, &pows, &mut out);
        assert_eq!(out, reference);
        // padding: evaluate the degree-<n polynomial on the size-4n coset directly
        let plan_m = NttPlan::new(4 * n);
        let padded = pad_bitrev(&br, 2);
        let pm = CosetPowers::get(plan_m.log_n, shift, 0, plan.n_inv());
        let ev = evaluate_from_bitrev(&plan_m, &padded, &pm);
        for i in 0..4 * n {
            assert_eq!(ev[i], reference[coset_major_pos(i, n, 2)], "i={i}");
        }
    }

    #[test]
    fn coset_roundtrip() {
        let n = 32;
        let plan = NttPlan::new(n);
        let coeffs: Vec<Fp> = (0..n).map(|i| Fp::new(i as u64 * 31 + 1)).collect();
        let shift = Fp::new(7);
        let ev = coset_evaluate(&plan, &coeffs, shift);
        let omega = plan.omega();
        for i in 0..n {
            assert_eq!(ev[i], poly::eval(&coeffs, shift * omega.pow(i as u64)));
        }
        assert_eq!(coset_interpolate(&plan, &ev, shift), coeffs);
    }
}
