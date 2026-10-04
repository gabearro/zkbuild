//! Sixteen-lane BabyBear arithmetic on AVX-512 (`Fp16`), the matching
//! extension vectors (`Fp4x16`), and the vector kernels the NTT and the
//! prover's row evaluation are built from. Without AVX-512 the same types
//! fall back to scalar loops.

use crate::{ExtLike, FieldLike, Fp, Fp4, P, W};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub const LANES: usize = 16;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
type Repr = core::arch::x86_64::__m512i;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
type Repr = [u32; LANES];

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub const HAS_AVX512: bool = true;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub const HAS_AVX512: bool = false;

/// 16 field elements (Montgomery form) in one 512-bit register.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
pub struct Fp16(pub Repr);

impl Fp16 {
    /// Raw Montgomery lanes.
    #[inline(always)]
    pub fn lanes(self) -> [u32; LANES] {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            let mut out = [0u32; LANES];
            core::arch::x86_64::_mm512_storeu_si512(out.as_mut_ptr() as *mut _, self.0);
            out
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
        {
            self.0
        }
    }
    #[inline(always)]
    pub fn from_lanes(a: [u32; LANES]) -> Fp16 {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            Fp16(core::arch::x86_64::_mm512_loadu_si512(a.as_ptr() as *const _))
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
        {
            Fp16(a)
        }
    }
    pub const ZERO: Fp16 = Fp16::splat(Fp::ZERO);
    pub const ONE: Fp16 = Fp16::splat(Fp::ONE);

    #[inline(always)]
    pub const fn splat(v: Fp) -> Fp16 {
        let a = [v.0; LANES];
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        {
            Fp16(unsafe { core::mem::transmute::<[u32; LANES], core::arch::x86_64::__m512i>(a) })
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
        {
            Fp16(a)
        }
    }
    /// Load 16 consecutive elements (unaligned).
    #[inline(always)]
    pub fn from_slice(s: &[Fp]) -> Fp16 {
        assert!(s.len() >= LANES);
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            Fp16(core::arch::x86_64::_mm512_loadu_si512(s.as_ptr() as *const _))
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
        {
            let mut a = [0u32; LANES];
            for i in 0..LANES {
                a[i] = s[i].0;
            }
            Fp16(a)
        }
    }
    /// Write the lanes to `dst[..16]`.
    #[inline(always)]
    pub fn store_to(self, dst: &mut [Fp]) {
        assert!(dst.len() >= LANES);
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            core::arch::x86_64::_mm512_storeu_si512(dst.as_mut_ptr() as *mut _, self.0);
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
        {
            for i in 0..LANES {
                dst[i] = Fp(self.0[i]);
            }
        }
    }
    #[inline(always)]
    pub fn lane(self, i: usize) -> Fp {
        Fp(self.lanes()[i])
    }
    #[inline(always)]
    pub fn to_fps(self) -> [Fp; LANES] {
        let l = self.lanes();
        core::array::from_fn(|i| Fp(l[i]))
    }
    pub fn is_zero(&self) -> bool {
        self.lanes().iter().all(|&x| x == 0)
    }
    pub fn from_u64(v: u64) -> Fp16 {
        Fp16::splat(Fp::new(v))
    }
}

impl PartialEq for Fp16 {
    fn eq(&self, o: &Fp16) -> bool {
        self.lanes() == o.lanes()
    }
}
impl Eq for Fp16 {}
impl core::fmt::Debug for Fp16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Fp16({:?})", self.to_fps())
    }
}
impl Default for Fp16 {
    fn default() -> Self {
        Fp16::ZERO
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub(crate) mod avx {
    use super::*;
    use core::arch::x86_64::*;

    #[inline(always)]
    pub fn p() -> __m512i {
        unsafe { _mm512_set1_epi32(P as i32) }
    }

    /// (a + b) mod p for canonical inputs (< p).
    #[inline(always)]
    pub unsafe fn add(a: __m512i, b: __m512i) -> __m512i {
        let s = _mm512_add_epi32(a, b);
        let t = _mm512_sub_epi32(s, p());
        _mm512_min_epu32(s, t) // t wraps to a huge value iff s < p
    }

    /// (a − b) mod p for canonical inputs.
    #[inline(always)]
    pub unsafe fn sub(a: __m512i, b: __m512i) -> __m512i {
        let d = _mm512_sub_epi32(a, b);
        let t = _mm512_add_epi32(d, p());
        _mm512_min_epu32(d, t) // d wraps (huge) iff a < b, then t is the answer
    }

    /// Montgomery product a·b·2^{-32} mod p, canonical output.
    #[inline(always)]
    pub unsafe fn mul(a: __m512i, b: __m512i) -> __m512i {
        let mu = _mm512_set1_epi32(0x77FF_FFFFu32 as i32);
        let pv = p();
        // even lanes: 64-bit products of the low 32 bits of each 64-bit lane
        let a_odd = _mm512_srli_epi64::<32>(a);
        let b_odd = _mm512_srli_epi64::<32>(b);
        let t_evn = _mm512_mul_epu32(a, b);
        let t_odd = _mm512_mul_epu32(a_odd, b_odd);
        // q = (t mod 2^32)·mu mod 2^32  (vpmuludq only reads the low 32 bits)
        let q_evn = _mm512_mul_epu32(t_evn, mu);
        let q_odd = _mm512_mul_epu32(t_odd, mu);
        // u = (t + q·p) >> 32  (< 2p, and t + q·p < 2^64)
        let u_evn = _mm512_srli_epi64::<32>(_mm512_add_epi64(t_evn, _mm512_mul_epu32(q_evn, pv)));
        let u_odd = _mm512_add_epi64(t_odd, _mm512_mul_epu32(q_odd, pv)); // keep the high half in place
        // merge: even results sit in the low 32 bits, odd results in the high 32 bits
        let u = _mm512_mask_blend_epi32(0xAAAA, u_evn, u_odd);
        // canonicalize: u < 2p
        let t = _mm512_sub_epi32(u, pv);
        _mm512_min_epu32(u, t)
    }

    /// Montgomery-reduce a pair of 64-bit accumulators (even/odd lanes), each
    /// holding the sum of at most two products of canonical elements
    /// (t < 2p² so t + q·p < 2^64 and the result is < 2p before the fold).
    #[inline(always)]
    unsafe fn reduce_pair(t_evn: __m512i, t_odd: __m512i) -> __m512i {
        let mu = _mm512_set1_epi32(0x77FF_FFFFu32 as i32);
        let pv = p();
        let q_evn = _mm512_mul_epu32(t_evn, mu);
        let q_odd = _mm512_mul_epu32(t_odd, mu);
        let u_evn = _mm512_srli_epi64::<32>(_mm512_add_epi64(t_evn, _mm512_mul_epu32(q_evn, pv)));
        let u_odd = _mm512_add_epi64(t_odd, _mm512_mul_epu32(q_odd, pv));
        let u = _mm512_mask_blend_epi32(0xAAAA, u_evn, u_odd);
        let t = _mm512_sub_epi32(u, pv);
        _mm512_min_epu32(u, t)
    }

    /// F_{p^4} product with lazy reduction: the four products feeding each
    /// output coefficient are summed two at a time in 64 bits and reduced
    /// twice (8 reductions instead of 19 full Montgomery multiplications).
    /// u^4 = W is folded into the second operand (w·b_i computed once).
    #[inline(always)]
    pub unsafe fn mul4(a: [__m512i; 4], b: [__m512i; 4]) -> [__m512i; 4] {
        let w = _mm512_set1_epi32(Fp::new(W as u64).0 as i32);
        let wb1 = mul(w, b[1]);
        let wb2 = mul(w, b[2]);
        let wb3 = mul(w, b[3]);
        let ao: [__m512i; 4] = [_mm512_srli_epi64::<32>(a[0]), _mm512_srli_epi64::<32>(a[1]), _mm512_srli_epi64::<32>(a[2]), _mm512_srli_epi64::<32>(a[3])];
        let be: [__m512i; 7] = [b[0], b[1], b[2], b[3], wb1, wb2, wb3];
        let bo: [__m512i; 7] = core::array::from_fn(|i| _mm512_srli_epi64::<32>(be[i]));
        // c_k = Σ_{i+j=k} a_i b_j + W Σ_{i+j=k+4} a_i b_j ; operand index 4+j-1 = W·b_j
        let term = |i: usize, j: usize| -> (__m512i, __m512i) { (_mm512_mul_epu32(a[i], be[j]), _mm512_mul_epu32(ao[i], bo[j])) };
        let pair = |(x0, x1): (__m512i, __m512i), (y0, y1): (__m512i, __m512i)| -> __m512i { reduce_pair(_mm512_add_epi64(x0, y0), _mm512_add_epi64(x1, y1)) };
        // c0 = a0b0 + W(a1b3 + a2b2 + a3b1)
        let c0 = add(pair(term(0, 0), term(1, 6)), pair(term(2, 5), term(3, 4)));
        // c1 = a0b1 + a1b0 + W(a2b3 + a3b2)
        let c1 = add(pair(term(0, 1), term(1, 0)), pair(term(2, 6), term(3, 5)));
        // c2 = a0b2 + a1b1 + a2b0 + W a3b3
        let c2 = add(pair(term(0, 2), term(1, 1)), pair(term(2, 0), term(3, 6)));
        // c3 = a0b3 + a1b2 + a2b1 + a3b0
        let c3 = add(pair(term(0, 3), term(1, 2)), pair(term(2, 1), term(3, 0)));
        [c0, c1, c2, c3]
    }
}

impl Add for Fp16 {
    type Output = Fp16;
    #[inline(always)]
    fn add(self, rhs: Fp16) -> Fp16 {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            Fp16(avx::add(self.0, rhs.0))
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
        {
            let mut o = [0u32; LANES];
            for i in 0..LANES {
                o[i] = (Fp(self.0[i]) + Fp(rhs.0[i])).0;
            }
            Fp16(o)
        }
    }
}
impl Sub for Fp16 {
    type Output = Fp16;
    #[inline(always)]
    fn sub(self, rhs: Fp16) -> Fp16 {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            Fp16(avx::sub(self.0, rhs.0))
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
        {
            let mut o = [0u32; LANES];
            for i in 0..LANES {
                o[i] = (Fp(self.0[i]) - Fp(rhs.0[i])).0;
            }
            Fp16(o)
        }
    }
}
impl Mul for Fp16 {
    type Output = Fp16;
    #[inline(always)]
    fn mul(self, rhs: Fp16) -> Fp16 {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            Fp16(avx::mul(self.0, rhs.0))
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
        {
            let mut o = [0u32; LANES];
            for i in 0..LANES {
                o[i] = (Fp(self.0[i]) * Fp(rhs.0[i])).0;
            }
            Fp16(o)
        }
    }
}
impl Neg for Fp16 {
    type Output = Fp16;
    #[inline(always)]
    fn neg(self) -> Fp16 {
        Fp16::ZERO - self
    }
}
impl AddAssign for Fp16 {
    #[inline(always)]
    fn add_assign(&mut self, r: Fp16) {
        *self = *self + r;
    }
}
impl SubAssign for Fp16 {
    #[inline(always)]
    fn sub_assign(&mut self, r: Fp16) {
        *self = *self - r;
    }
}
impl MulAssign for Fp16 {
    #[inline(always)]
    fn mul_assign(&mut self, r: Fp16) {
        *self = *self * r;
    }
}
impl From<Fp> for Fp16 {
    #[inline(always)]
    fn from(v: Fp) -> Fp16 {
        Fp16::splat(v)
    }
}

impl FieldLike for Fp16 {
    type Ext = Fp4x16;
    type Weight = Fp4;
    type Acc = Fp4x16;
    #[inline(always)]
    fn weight(alpha: Fp4) -> Fp4 {
        alpha
    }
    const ZERO: Fp16 = Fp16::ZERO;
    const ONE: Fp16 = Fp16::ONE;
    fn is_zero(&self) -> bool {
        Fp16::is_zero(self)
    }
    #[inline(always)]
    fn mul_ext(self, e: Fp4) -> Fp4x16 {
        Fp4x16 { c: [self * Fp16::splat(e.0[0]), self * Fp16::splat(e.0[1]), self * Fp16::splat(e.0[2]), self * Fp16::splat(e.0[3])] }
    }
    #[inline(always)]
    fn to_ext(self) -> Fp4x16 {
        Fp4x16 { c: [self, Fp16::ZERO, Fp16::ZERO, Fp16::ZERO] }
    }
    #[inline(always)]
    fn as_base(self) -> Option<Fp16> {
        Some(self)
    }
}

/// Sixteen lanes of F_{p^4}, stored as four coefficient vectors.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Fp4x16 {
    pub c: [Fp16; 4],
}

impl Fp4x16 {
    pub const ZERO: Fp4x16 = Fp4x16 { c: [Fp16::ZERO; 4] };
    pub const ONE: Fp4x16 = Fp4x16 { c: [Fp16::ONE, Fp16::ZERO, Fp16::ZERO, Fp16::ZERO] };
    #[inline(always)]
    pub fn splat(v: Fp4) -> Fp4x16 {
        Fp4x16 { c: [Fp16::splat(v.0[0]), Fp16::splat(v.0[1]), Fp16::splat(v.0[2]), Fp16::splat(v.0[3])] }
    }
    /// From four coefficient slices (each ≥ 16 elements).
    #[inline(always)]
    pub fn from_slices(c0: &[Fp], c1: &[Fp], c2: &[Fp], c3: &[Fp]) -> Fp4x16 {
        Fp4x16 { c: [Fp16::from_slice(c0), Fp16::from_slice(c1), Fp16::from_slice(c2), Fp16::from_slice(c3)] }
    }
    #[inline(always)]
    pub fn lane(&self, i: usize) -> Fp4 {
        Fp4([self.c[0].lane(i), self.c[1].lane(i), self.c[2].lane(i), self.c[3].lane(i)])
    }
    pub fn from_lanes(v: [Fp4; LANES]) -> Fp4x16 {
        let mut c = [[0u32; LANES]; 4];
        for i in 0..LANES {
            for k in 0..4 {
                c[k][i] = v[i].0[k].0;
            }
        }
        Fp4x16 { c: [Fp16::from_lanes(c[0]), Fp16::from_lanes(c[1]), Fp16::from_lanes(c[2]), Fp16::from_lanes(c[3])] }
    }
    pub fn to_lanes(&self) -> [Fp4; LANES] {
        let l: [[u32; LANES]; 4] = [self.c[0].lanes(), self.c[1].lanes(), self.c[2].lanes(), self.c[3].lanes()];
        core::array::from_fn(|i| Fp4([Fp(l[0][i]), Fp(l[1][i]), Fp(l[2][i]), Fp(l[3][i])]))
    }
    /// Lane-wise inverse (zero lanes stay zero).
    pub fn inverse(&self) -> Fp4x16 {
        let mut l = self.to_lanes();
        for x in l.iter_mut() {
            if !Fp4::is_zero(*x) {
                *x = x.inverse();
            }
        }
        Fp4x16::from_lanes(l)
    }
    /// Montgomery batch inversion, lane-wise. Requires every lane of every
    /// element to be nonzero (returns false and leaves the input untouched
    /// otherwise).
    pub fn batch_inverse(xs: &mut [Fp4x16]) -> bool {
        let n = xs.len();
        if n == 0 {
            return true;
        }
        let mut prefix = Vec::with_capacity(n);
        let mut acc = Fp4x16::ONE;
        for x in xs.iter() {
            prefix.push(acc);
            acc *= *x;
        }
        if acc.to_lanes().iter().any(|v| Fp4::is_zero(*v)) {
            return false;
        }
        let mut inv = acc.inverse();
        for i in (0..n).rev() {
            let x = xs[i];
            xs[i] = inv * prefix[i];
            inv *= x;
        }
        true
    }
}
impl Add for Fp4x16 {
    type Output = Fp4x16;
    #[inline(always)]
    fn add(self, r: Fp4x16) -> Fp4x16 {
        Fp4x16 { c: [self.c[0] + r.c[0], self.c[1] + r.c[1], self.c[2] + r.c[2], self.c[3] + r.c[3]] }
    }
}
impl Sub for Fp4x16 {
    type Output = Fp4x16;
    #[inline(always)]
    fn sub(self, r: Fp4x16) -> Fp4x16 {
        Fp4x16 { c: [self.c[0] - r.c[0], self.c[1] - r.c[1], self.c[2] - r.c[2], self.c[3] - r.c[3]] }
    }
}
impl Mul for Fp4x16 {
    type Output = Fp4x16;
    #[inline(always)]
    fn mul(self, r: Fp4x16) -> Fp4x16 {
        let a = self.c;
        let b = r.c;
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            let c = avx::mul4([a[0].0, a[1].0, a[2].0, a[3].0], [b[0].0, b[1].0, b[2].0, b[3].0]);
            return Fp4x16 { c: [Fp16(c[0]), Fp16(c[1]), Fp16(c[2]), Fp16(c[3])] };
        }
        #[allow(unreachable_code)]
        let w = Fp16::splat(Fp::new(W as u64));
        let c0 = a[0] * b[0] + w * (a[1] * b[3] + a[2] * b[2] + a[3] * b[1]);
        let c1 = a[0] * b[1] + a[1] * b[0] + w * (a[2] * b[3] + a[3] * b[2]);
        let c2 = a[0] * b[2] + a[1] * b[1] + a[2] * b[0] + w * (a[3] * b[3]);
        let c3 = a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0];
        Fp4x16 { c: [c0, c1, c2, c3] }
    }
}
impl Neg for Fp4x16 {
    type Output = Fp4x16;
    #[inline(always)]
    fn neg(self) -> Fp4x16 {
        Fp4x16::ZERO - self
    }
}
impl AddAssign for Fp4x16 {
    #[inline(always)]
    fn add_assign(&mut self, r: Fp4x16) {
        *self = *self + r;
    }
}
impl SubAssign for Fp4x16 {
    #[inline(always)]
    fn sub_assign(&mut self, r: Fp4x16) {
        *self = *self - r;
    }
}
impl MulAssign for Fp4x16 {
    #[inline(always)]
    fn mul_assign(&mut self, r: Fp4x16) {
        *self = *self * r;
    }
}
impl From<Fp4> for Fp4x16 {
    #[inline(always)]
    fn from(v: Fp4) -> Fp4x16 {
        Fp4x16::splat(v)
    }
}
impl ExtLike<Fp16> for Fp4x16 {
    const ZERO: Fp4x16 = Fp4x16::ZERO;
    const ONE: Fp4x16 = Fp4x16::ONE;
    const LANES: usize = LANES;
    #[inline(always)]
    fn mul_base(self, b: Fp16) -> Fp4x16 {
        Fp4x16 { c: [self.c[0] * b, self.c[1] * b, self.c[2] * b, self.c[3] * b] }
    }
    fn is_zero(&self) -> bool {
        self.c.iter().all(|x| x.is_zero())
    }
    #[inline(always)]
    fn lane(&self, i: usize) -> Fp4 {
        Fp4x16::lane(self, i)
    }
}
impl crate::WeightedAcc<Fp16> for Fp4x16 {
    #[inline(always)]
    fn zero() -> Fp4x16 {
        Fp4x16::ZERO
    }
    #[inline(always)]
    fn add_base(&mut self, w: &Fp4, v: Fp16) {
        *self += v.mul_ext(*w);
    }
    #[inline(always)]
    fn add_ext(&mut self, alpha: Fp4, v: Fp4x16) {
        *self += Fp4x16::splat(alpha) * v;
    }
    #[inline(always)]
    fn finish(self) -> Fp4x16 {
        self
    }
}

// ------------------------------------------------------------ slice kernels

/// In-place DIT butterflies: lo[k], hi[k] <- lo[k] ± hi[k]·tw[k].
#[inline]
pub fn butterfly_slices(lo: &mut [Fp], hi: &mut [Fp], tw: &[Fp]) {
    let n = lo.len();
    let mut k = 0;
    while k + LANES <= n {
        let a = Fp16::from_slice(&lo[k..k + LANES]);
        let b = Fp16::from_slice(&hi[k..k + LANES]) * Fp16::from_slice(&tw[k..k + LANES]);
        (a + b).store_to(&mut lo[k..]);
        (a - b).store_to(&mut hi[k..]);
        k += LANES;
    }
    while k < n {
        let a = lo[k];
        let b = hi[k] * tw[k];
        lo[k] = a + b;
        hi[k] = a - b;
        k += 1;
    }
}

/// DIF butterflies: lo[k], hi[k] <- lo[k] + hi[k], (lo[k] − hi[k])·tw[k].
#[inline]
pub fn dif_butterfly_slices(lo: &mut [Fp], hi: &mut [Fp], tw: &[Fp]) {
    let n = lo.len();
    let mut k = 0;
    while k + LANES <= n {
        let a = Fp16::from_slice(&lo[k..k + LANES]);
        let b = Fp16::from_slice(&hi[k..k + LANES]);
        (a + b).store_to(&mut lo[k..]);
        ((a - b) * Fp16::from_slice(&tw[k..k + LANES])).store_to(&mut hi[k..]);
        k += LANES;
    }
    while k < n {
        let a = lo[k];
        let b = hi[k];
        lo[k] = a + b;
        hi[k] = (a - b) * tw[k];
        k += 1;
    }
}

/// out[i] = a[i] · b[i].
#[inline]
pub fn mul_slices(out: &mut [Fp], a: &[Fp], b: &[Fp]) {
    let n = out.len();
    debug_assert!(a.len() >= n && b.len() >= n);
    let mut k = 0;
    while k + LANES <= n {
        (Fp16::from_slice(&a[k..k + LANES]) * Fp16::from_slice(&b[k..k + LANES])).store_to(&mut out[k..]);
        k += LANES;
    }
    while k < n {
        out[k] = a[k] * b[k];
        k += 1;
    }
}

/// a[i] *= b[i].
#[inline]
pub fn mul_slices_in_place(a: &mut [Fp], b: &[Fp]) {
    let n = a.len();
    let mut k = 0;
    while k + LANES <= n {
        (Fp16::from_slice(&a[k..k + LANES]) * Fp16::from_slice(&b[k..k + LANES])).store_to(&mut a[k..]);
        k += LANES;
    }
    while k < n {
        a[k] = a[k] * b[k];
        k += 1;
    }
}

/// a[i] *= s.
#[inline]
pub fn scale_slice(a: &mut [Fp], s: Fp) {
    let sv = Fp16::splat(s);
    let n = a.len();
    let mut k = 0;
    while k + LANES <= n {
        (Fp16::from_slice(&a[k..k + LANES]) * sv).store_to(&mut a[k..]);
        k += LANES;
    }
    while k < n {
        a[k] = a[k] * s;
        k += 1;
    }
}

// ------------------------------------------------------------ NTT kernels
//
// The first four DIT stages (last four DIF stages) act inside contiguous
// 16-element blocks. We load sixteen consecutive blocks (256 elements),
// transpose them 16×16 in registers so vector `j` holds element `j` of
// every block, run the four stages with splatted twiddles, and transpose
// back.

/// 16×16 transpose of 32-bit lanes.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn transpose16x16(r: &mut [Fp16; 16]) {
    use core::arch::x86_64::*;
    let mut t = [_mm512_setzero_si512(); 16];
    // round 1: 32-bit unpack within 64-bit pairs
    for i in 0..8 {
        t[2 * i] = _mm512_unpacklo_epi32(r[2 * i].0, r[2 * i + 1].0);
        t[2 * i + 1] = _mm512_unpackhi_epi32(r[2 * i].0, r[2 * i + 1].0);
    }
    // round 2: 64-bit unpack within 128-bit lanes
    let mut u = [_mm512_setzero_si512(); 16];
    for i in 0..4 {
        u[4 * i] = _mm512_unpacklo_epi64(t[4 * i], t[4 * i + 2]);
        u[4 * i + 1] = _mm512_unpackhi_epi64(t[4 * i], t[4 * i + 2]);
        u[4 * i + 2] = _mm512_unpacklo_epi64(t[4 * i + 1], t[4 * i + 3]);
        u[4 * i + 3] = _mm512_unpackhi_epi64(t[4 * i + 1], t[4 * i + 3]);
    }
    // After rounds 1–2, u[4i + j] holds, per 128-bit lane L (L = 0..3), the
    // elements (4L + j) of rows 4i..4i+3. Rounds 3–4 shuffle 128-bit lanes.
    // round 3: combine rows blocks (0,1) and (2,3) -> lanes 0/2 and 1/3
    for i in 0..2 {
        for j in 0..4 {
            let a = u[8 * i + j];
            let b = u[8 * i + 4 + j];
            t[8 * i + j] = _mm512_shuffle_i32x4::<0x88>(a, b); // lanes a0,a2,b0,b2
            t[8 * i + 4 + j] = _mm512_shuffle_i32x4::<0xDD>(a, b); // lanes a1,a3,b1,b3
        }
    }
    // round 4
    for j in 0..4 {
        let a = t[j];
        let b = t[8 + j];
        r[j] = Fp16(_mm512_shuffle_i32x4::<0x88>(a, b)); // element j: rows 0-3,8-11 | ...
        r[8 + j] = Fp16(_mm512_shuffle_i32x4::<0xDD>(a, b));
        let a = t[4 + j];
        let b = t[12 + j];
        r[4 + j] = Fp16(_mm512_shuffle_i32x4::<0x88>(a, b));
        r[12 + j] = Fp16(_mm512_shuffle_i32x4::<0xDD>(a, b));
    }
}

/// Twiddles of the first four stages, splatted: `w[s][k] = ω_{2^(s+1)}^k`.
#[derive(Clone, Copy)]
pub struct Small16Twiddles {
    pub w1: Fp16,         // ω_4
    pub w2: [Fp16; 4],    // ω_8^k
    pub w3: [Fp16; 8],    // ω_16^k
}

impl Small16Twiddles {
    /// `stage_tw[s]` are the per-stage twiddle tables of the plan.
    pub fn new(stage_tw: &[Vec<Fp>]) -> Small16Twiddles {
        Small16Twiddles {
            w1: Fp16::splat(stage_tw[1][1]),
            w2: core::array::from_fn(|k| Fp16::splat(stage_tw[2][k])),
            w3: core::array::from_fn(|k| Fp16::splat(stage_tw[3][k])),
        }
    }
}

/// Public 16×16 transpose of sixteen 16-lane vectors (`r[j].lane(i)` ↔ `r[i].lane(j)`).
pub fn transpose16(r: &mut [Fp16; 16]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    unsafe {
        transpose16x16(r)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    {
        let l: [[u32; LANES]; 16] = core::array::from_fn(|j| r[j].lanes());
        for j in 0..16 {
            r[j] = Fp16::from_lanes(core::array::from_fn(|i| l[i][j]));
        }
    }
}

/// Element permutation applied by [`transpose16x16`] (used by the tests and
/// by the scalar fallback to define the kernels' contract).
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
unsafe fn load_block(blk: &[Fp]) -> [Fp16; 16] {
    core::array::from_fn(|j| Fp16::from_slice(&blk[16 * j..16 * j + 16]))
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn dit16_core(r: &mut [Fp16; 16], tw: &Small16Twiddles) {
    unsafe { transpose16x16(r) };
    // stage 0: pairs (2i, 2i+1), twiddle 1
    for i in 0..8 {
        let (a, b) = (r[2 * i], r[2 * i + 1]);
        r[2 * i] = a + b;
        r[2 * i + 1] = a - b;
    }
    // stage 1: blocks of 4: (j, j+2) with ω_4^j
    for base in [0usize, 4, 8, 12] {
        let (a, b) = (r[base], r[base + 2]);
        r[base] = a + b;
        r[base + 2] = a - b;
        let (a, b) = (r[base + 1], r[base + 3] * tw.w1);
        r[base + 1] = a + b;
        r[base + 3] = a - b;
    }
    // stage 2: blocks of 8: (j, j+4) with ω_8^j
    for base in [0usize, 8] {
        for j in 0..4 {
            let b = if j == 0 { r[base + j + 4] } else { r[base + j + 4] * tw.w2[j] };
            let a = r[base + j];
            r[base + j] = a + b;
            r[base + j + 4] = a - b;
        }
    }
    // stage 3: (j, j+8) with ω_16^j
    for j in 0..8 {
        let b = if j == 0 { r[j + 8] } else { r[j + 8] * tw.w3[j] };
        let a = r[j];
        r[j] = a + b;
        r[j + 8] = a - b;
    }
    unsafe { transpose16x16(r) };
}

/// DIT stages 0..3 over `v` (len multiple of 256), in place.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub fn dit_small_stages(v: &mut [Fp], tw: &Small16Twiddles) {
    debug_assert_eq!(v.len() % 256, 0);
    for blk in v.chunks_exact_mut(256) {
        let mut r = unsafe { load_block(blk) };
        dit16_core(&mut r, tw);
        for j in 0..16 {
            r[j].store_to(&mut blk[16 * j..]);
        }
    }
}

/// Same as [`dit_small_stages`] but reads `src[i]·scale[i]` and writes to `dst`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub fn dit_small_stages_scaled(src: &[Fp], scale: &[Fp], dst: &mut [Fp], tw: &Small16Twiddles) {
    debug_assert_eq!(src.len() % 256, 0);
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len(), scale.len());
    for ((sb, cb), db) in src.chunks_exact(256).zip(scale.chunks_exact(256)).zip(dst.chunks_exact_mut(256)) {
        let mut r: [Fp16; 16] = core::array::from_fn(|j| Fp16::from_slice(&sb[16 * j..16 * j + 16]) * Fp16::from_slice(&cb[16 * j..16 * j + 16]));
        dit16_core(&mut r, tw);
        for j in 0..16 {
            r[j].store_to(&mut db[16 * j..]);
        }
    }
}

/// DIF stages 3,2,1,0 (Gentleman–Sande) over `v` (len multiple of 256):
/// natural input order per block, bit-reversed output order per block.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub fn dif_small_stages(v: &mut [Fp], tw: &Small16Twiddles) {
    debug_assert_eq!(v.len() % 256, 0);
    for blk in v.chunks_exact_mut(256) {
        let mut r = unsafe { load_block(blk) };
        unsafe { transpose16x16(&mut r) };
        // stage 3: (j, j+8): a+b, (a−b)·ω_16^j
        for j in 0..8 {
            let (a, b) = (r[j], r[j + 8]);
            r[j] = a + b;
            r[j + 8] = if j == 0 { a - b } else { (a - b) * tw.w3[j] };
        }
        // stage 2
        for base in [0usize, 8] {
            for j in 0..4 {
                let (a, b) = (r[base + j], r[base + j + 4]);
                r[base + j] = a + b;
                r[base + j + 4] = if j == 0 { a - b } else { (a - b) * tw.w2[j] };
            }
        }
        // stage 1
        for base in [0usize, 4, 8, 12] {
            let (a, b) = (r[base], r[base + 2]);
            r[base] = a + b;
            r[base + 2] = a - b;
            let (a, b) = (r[base + 1], r[base + 3]);
            r[base + 1] = a + b;
            r[base + 3] = (a - b) * tw.w1;
        }
        // stage 0
        for i in 0..8 {
            let (a, b) = (r[2 * i], r[2 * i + 1]);
            r[2 * i] = a + b;
            r[2 * i + 1] = a - b;
        }
        unsafe { transpose16x16(&mut r) };
        for j in 0..16 {
            r[j].store_to(&mut blk[16 * j..]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(seed: &mut u64) -> Fp {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        Fp::new(*seed)
    }

    #[test]
    fn simd_matches_scalar() {
        let mut seed = 0x1234_5678_9ABC_DEF1u64;
        let edge = [Fp::ZERO, Fp::ONE, Fp::new(P as u64 - 1), Fp::new(P as u64 - 2), Fp::new(1 << 30), Fp::new(0x7FFF_FFFF), Fp::from_mont(P - 1)];
        for _ in 0..2000 {
            let mut a = [Fp::ZERO; LANES];
            let mut b = [Fp::ZERO; LANES];
            for i in 0..LANES {
                a[i] = if (seed >> 3) % 5 == 0 { edge[(seed % 7) as usize] } else { rnd(&mut seed) };
                b[i] = if (seed >> 5) % 5 == 0 { edge[((seed >> 8) % 7) as usize] } else { rnd(&mut seed) };
            }
            let va = Fp16::from_slice(&a);
            let vb = Fp16::from_slice(&b);
            let s = (va + vb).to_fps();
            let d = (va - vb).to_fps();
            let m = (va * vb).to_fps();
            for i in 0..LANES {
                assert_eq!(s[i], a[i] + b[i], "add lane {i}: {:?} {:?}", a[i], b[i]);
                assert_eq!(d[i], a[i] - b[i], "sub lane {i}");
                assert_eq!(m[i], a[i] * b[i], "mul lane {i}: {:?} {:?}", a[i], b[i]);
                assert!(m[i].0 < P, "non-canonical output");
            }
        }
    }

    #[test]
    fn ext_vectors_match_scalar() {
        let mut seed = 0xDEAD_BEEF_1234_5678u64;
        for _ in 0..200 {
            let a: [Fp4; LANES] = core::array::from_fn(|_| Fp4([rnd(&mut seed), rnd(&mut seed), rnd(&mut seed), rnd(&mut seed)]));
            let b: [Fp4; LANES] = core::array::from_fn(|_| Fp4([rnd(&mut seed), rnd(&mut seed), rnd(&mut seed), rnd(&mut seed)]));
            let va = Fp4x16::from_lanes(a);
            let vb = Fp4x16::from_lanes(b);
            let m = (va * vb).to_lanes();
            let s = (va + vb).to_lanes();
            for i in 0..LANES {
                assert_eq!(m[i], a[i] * b[i]);
                assert_eq!(s[i], a[i] + b[i]);
            }
            let mut xs = vec![va, vb, va * vb];
            assert!(Fp4x16::batch_inverse(&mut xs));
            let l = xs[2].to_lanes();
            for i in 0..LANES {
                assert_eq!(l[i], (a[i] * b[i]).inverse());
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    #[test]
    fn transpose_roundtrip_and_layout() {
        let mut r: [Fp16; 16] = core::array::from_fn(|j| Fp16::from_lanes(core::array::from_fn(|i| (16 * j + i) as u32)));
        let orig = r;
        unsafe { transpose16x16(&mut r) };
        for j in 0..16 {
            let l = r[j].lanes();
            for i in 0..16 {
                assert_eq!(l[i], (16 * i + j) as u32, "r[{j}][{i}]");
            }
        }
        unsafe { transpose16x16(&mut r) };
        assert_eq!(r, orig);
    }
}
