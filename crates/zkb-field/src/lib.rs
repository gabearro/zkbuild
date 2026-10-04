//! The BabyBear prime field p = 2^31 − 2^27 + 1 (31 bits, 2-adicity 27)
//! and its quartic extension F_{p^4} = F_p[u]/(u^4 − 11).
//!
//! Elements are kept in Montgomery form (x·2^32 mod p) in a `u32`; a
//! multiplication is a 64-bit product plus one Montgomery reduction, which
//! vectorizes to ~1 op per lane with AVX-512. Challenges and lookup
//! fingerprints live in F_{p^4} (≈2^124 elements) for soundness.

pub mod columns;
pub mod ntt;
pub mod poly;
pub mod simd;

use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use serde::{Deserialize, Serialize};

/// The prime p = 15·2^27 + 1.
pub const P: u32 = 0x7800_0001;
pub const P64: u64 = P as u64;
/// Multiplicative generator of F_p^*.
pub const GENERATOR: u32 = 31;
/// p − 1 = 15·2^27.
pub const TWO_ADICITY: u32 = 27;
/// Extension defined by u^4 = W (11 is a quadratic non-residue, so u^4 − 11 is irreducible).
pub const W: u32 = 11;

/// −p^{-1} mod 2^32 (for Montgomery reduction).
const MU: u32 = 0x77FF_FFFF;
/// 2^32 mod p (Montgomery form of one).
const R1: u32 = 0x0FFF_FFFE;
/// 2^64 mod p (to convert into Montgomery form).
const R2: u32 = 0x45DD_DDE3;

#[inline(always)]
const fn mont_reduce(t: u64) -> u32 {
    // t < p·2^32: u = (t + m·p) / 2^32 < 2p
    let m = (t as u32).wrapping_mul(MU);
    let u = ((t + (m as u64) * P64) >> 32) as u32;
    if u >= P {
        u - P
    } else {
        u
    }
}

#[inline(always)]
const fn mont_mul(a: u32, b: u32) -> u32 {
    mont_reduce((a as u64) * (b as u64))
}

/// A field element in Montgomery form.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Fp(pub u32);

impl fmt::Debug for Fp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_u64())
    }
}
impl fmt::Display for Fp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_u64())
    }
}

impl Fp {
    pub const ZERO: Fp = Fp(0);
    pub const ONE: Fp = Fp(R1);
    pub const TWO: Fp = Fp::new(2);

    /// From an integer (reduced mod p).
    #[inline(always)]
    pub const fn new(v: u64) -> Fp {
        let r = (v % P64) as u32;
        Fp(mont_mul(r, R2))
    }
    /// From a canonical value < p.
    #[inline(always)]
    pub const fn from_canonical(v: u32) -> Fp {
        debug_assert!(v < P);
        Fp(mont_mul(v, R2))
    }
    #[inline(always)]
    pub fn from_u32(v: u32) -> Fp {
        Fp::new(v as u64)
    }
    #[inline(always)]
    pub fn from_i64(v: i64) -> Fp {
        let m = v.rem_euclid(P64 as i64) as u64;
        Fp::new(m)
    }
    /// Raw Montgomery representation.
    #[inline(always)]
    pub const fn from_mont(v: u32) -> Fp {
        Fp(v)
    }
    /// Canonical integer value in [0, p).
    #[inline(always)]
    pub const fn as_canonical(self) -> u32 {
        mont_reduce(self.0 as u64)
    }
    #[inline(always)]
    pub fn as_u64(self) -> u64 {
        self.as_canonical() as u64
    }
    #[inline(always)]
    pub fn is_zero(self) -> bool {
        self.0 == 0
    }
    #[inline(always)]
    pub fn square(self) -> Fp {
        self * self
    }
    #[inline(always)]
    pub fn double(self) -> Fp {
        self + self
    }
    pub fn pow(self, mut e: u64) -> Fp {
        let mut base = self;
        let mut acc = Fp::ONE;
        while e > 0 {
            if e & 1 == 1 {
                acc *= base;
            }
            base = base.square();
            e >>= 1;
        }
        acc
    }
    pub fn inverse(self) -> Fp {
        assert!(!self.is_zero(), "inverse of zero");
        self.pow(P64 - 2)
    }
    pub fn try_inverse(self) -> Option<Fp> {
        if self.is_zero() {
            None
        } else {
            Some(self.inverse())
        }
    }
    /// Primitive 2^k-th root of unity (k ≤ 27), all powers of one fixed
    /// 2^27-th root so that ω_{2k}² = ω_k.
    pub fn two_adic_root(k: u32) -> Fp {
        assert!(k <= TWO_ADICITY);
        let base = Fp::new(GENERATOR as u64).pow((P64 - 1) >> TWO_ADICITY);
        base.pow(1u64 << (TWO_ADICITY - k))
    }
    /// Canonical little-endian bytes.
    #[inline(always)]
    pub fn to_le_bytes(self) -> [u8; 4] {
        self.as_canonical().to_le_bytes()
    }
    /// A field element from 8 uniform bytes (bias ≤ 2^-33).
    pub fn from_le_bytes_reduce(b: &[u8; 8]) -> Fp {
        Fp::new(u64::from_le_bytes(*b))
    }
    /// Batch inversion (Montgomery trick). Zeros stay zero.
    pub fn batch_inverse(xs: &mut [Fp]) {
        let n = xs.len();
        if n == 0 {
            return;
        }
        let mut prefix = Vec::with_capacity(n);
        let mut acc = Fp::ONE;
        for x in xs.iter() {
            prefix.push(acc);
            if !x.is_zero() {
                acc *= *x;
            }
        }
        let mut inv = acc.inverse();
        for i in (0..n).rev() {
            if xs[i].is_zero() {
                continue;
            }
            let x = xs[i];
            xs[i] = inv * prefix[i];
            inv *= x;
        }
    }
}

impl Add for Fp {
    type Output = Fp;
    #[inline(always)]
    fn add(self, r: Fp) -> Fp {
        let s = self.0 + r.0; // < 2p < 2^32
        Fp(if s >= P { s - P } else { s })
    }
}
impl AddAssign for Fp {
    #[inline(always)]
    fn add_assign(&mut self, r: Fp) {
        *self = *self + r;
    }
}
impl Sub for Fp {
    type Output = Fp;
    #[inline(always)]
    fn sub(self, r: Fp) -> Fp {
        let (d, borrow) = self.0.overflowing_sub(r.0);
        Fp(if borrow { d.wrapping_add(P) } else { d })
    }
}
impl SubAssign for Fp {
    #[inline(always)]
    fn sub_assign(&mut self, r: Fp) {
        *self = *self - r;
    }
}
impl Mul for Fp {
    type Output = Fp;
    #[inline(always)]
    fn mul(self, r: Fp) -> Fp {
        Fp(mont_mul(self.0, r.0))
    }
}
impl MulAssign for Fp {
    #[inline(always)]
    fn mul_assign(&mut self, r: Fp) {
        *self = *self * r;
    }
}
impl Neg for Fp {
    type Output = Fp;
    #[inline(always)]
    fn neg(self) -> Fp {
        Fp::ZERO - self
    }
}
impl From<u64> for Fp {
    fn from(v: u64) -> Fp {
        Fp::new(v)
    }
}
impl From<u32> for Fp {
    fn from(v: u32) -> Fp {
        Fp::new(v as u64)
    }
}
impl From<bool> for Fp {
    fn from(v: bool) -> Fp {
        Fp::new(v as u64)
    }
}
impl core::iter::Sum for Fp {
    fn sum<I: Iterator<Item = Fp>>(iter: I) -> Fp {
        iter.fold(Fp::ZERO, |a, b| a + b)
    }
}
impl core::iter::Product for Fp {
    fn product<I: Iterator<Item = Fp>>(iter: I) -> Fp {
        iter.fold(Fp::ONE, |a, b| a * b)
    }
}

// ---------------------------------------------------------------- Fp4

/// F_{p^4} = F_p[u]/(u^4 − W), coefficients lowest degree first.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Fp4(pub [Fp; 4]);

impl fmt::Debug for Fp4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fp4({}, {}, {}, {})", self.0[0], self.0[1], self.0[2], self.0[3])
    }
}

impl Fp4 {
    pub const ZERO: Fp4 = Fp4([Fp::ZERO; 4]);
    pub const ONE: Fp4 = Fp4([Fp::ONE, Fp::ZERO, Fp::ZERO, Fp::ZERO]);
    /// The generator u of the extension.
    pub const U: Fp4 = Fp4([Fp::ZERO, Fp::ONE, Fp::ZERO, Fp::ZERO]);
    pub const W_FP: Fp = Fp::new(W as u64);

    #[inline(always)]
    pub const fn new(c: [Fp; 4]) -> Fp4 {
        Fp4(c)
    }
    #[inline(always)]
    pub const fn from_base(a: Fp) -> Fp4 {
        Fp4([a, Fp::ZERO, Fp::ZERO, Fp::ZERO])
    }
    #[inline(always)]
    pub fn is_zero(self) -> bool {
        self.0.iter().all(|c| c.is_zero())
    }
    /// True if the element lies in the base field.
    #[inline(always)]
    pub fn is_base(self) -> bool {
        self.0[1].is_zero() && self.0[2].is_zero() && self.0[3].is_zero()
    }
    #[inline(always)]
    pub fn square(self) -> Fp4 {
        self * self
    }
    pub fn pow(self, mut e: u64) -> Fp4 {
        let mut base = self;
        let mut acc = Fp4::ONE;
        while e > 0 {
            if e & 1 == 1 {
                acc *= base;
            }
            base = base.square();
            e >>= 1;
        }
        acc
    }
    /// Frobenius x ↦ x^p: u^p = u·W^((p−1)/4) =: u·c, with c a primitive 4th root of unity.
    fn frobenius(self) -> Fp4 {
        let c = Self::frob_const();
        let c2 = c * c;
        let c3 = c2 * c;
        Fp4([self.0[0], self.0[1] * c, self.0[2] * c2, self.0[3] * c3])
    }
    fn frob_const() -> Fp {
        static C: std::sync::OnceLock<Fp> = std::sync::OnceLock::new();
        *C.get_or_init(|| Fp::new(W as u64).pow((P64 - 1) / 4))
    }
    /// Inverse via the norm: x^{-1} = x^{p+p²+p³} / N(x), N(x) ∈ F_p.
    pub fn inverse(self) -> Fp4 {
        assert!(!self.is_zero(), "inverse of zero");
        let x1 = self.frobenius();
        let x2 = x1.frobenius();
        let x3 = x2.frobenius();
        let y = x1 * x2 * x3;
        let norm = self * y;
        debug_assert!(norm.is_base(), "norm not in the base field");
        y.mul_base(norm.0[0].inverse())
    }
    pub fn try_inverse(self) -> Option<Fp4> {
        if self.is_zero() {
            None
        } else {
            Some(self.inverse())
        }
    }
    #[inline(always)]
    pub fn mul_base(self, b: Fp) -> Fp4 {
        Fp4([self.0[0] * b, self.0[1] * b, self.0[2] * b, self.0[3] * b])
    }
    pub fn to_le_bytes(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        for i in 0..4 {
            out[4 * i..4 * i + 4].copy_from_slice(&self.0[i].to_le_bytes());
        }
        out
    }
    /// Batch inversion (Montgomery trick). Zeros stay zero.
    pub fn batch_inverse(xs: &mut [Fp4]) {
        let n = xs.len();
        if n == 0 {
            return;
        }
        let mut prefix = Vec::with_capacity(n);
        let mut acc = Fp4::ONE;
        for x in xs.iter() {
            prefix.push(acc);
            if !Fp4::is_zero(*x) {
                acc *= *x;
            }
        }
        let mut inv = acc.inverse();
        for i in (0..n).rev() {
            if Fp4::is_zero(xs[i]) {
                continue;
            }
            let x = xs[i];
            xs[i] = inv * prefix[i];
            inv *= x;
        }
    }
}

impl Add for Fp4 {
    type Output = Fp4;
    #[inline(always)]
    fn add(self, r: Fp4) -> Fp4 {
        Fp4([self.0[0] + r.0[0], self.0[1] + r.0[1], self.0[2] + r.0[2], self.0[3] + r.0[3]])
    }
}
impl AddAssign for Fp4 {
    #[inline(always)]
    fn add_assign(&mut self, r: Fp4) {
        *self = *self + r;
    }
}
impl Sub for Fp4 {
    type Output = Fp4;
    #[inline(always)]
    fn sub(self, r: Fp4) -> Fp4 {
        Fp4([self.0[0] - r.0[0], self.0[1] - r.0[1], self.0[2] - r.0[2], self.0[3] - r.0[3]])
    }
}
impl SubAssign for Fp4 {
    #[inline(always)]
    fn sub_assign(&mut self, r: Fp4) {
        *self = *self - r;
    }
}
impl Mul for Fp4 {
    type Output = Fp4;
    #[inline(always)]
    fn mul(self, r: Fp4) -> Fp4 {
        let a = self.0;
        let b = r.0;
        let w = Fp4::W_FP;
        let c0 = a[0] * b[0] + w * (a[1] * b[3] + a[2] * b[2] + a[3] * b[1]);
        let c1 = a[0] * b[1] + a[1] * b[0] + w * (a[2] * b[3] + a[3] * b[2]);
        let c2 = a[0] * b[2] + a[1] * b[1] + a[2] * b[0] + w * (a[3] * b[3]);
        let c3 = a[0] * b[3] + a[1] * b[2] + a[2] * b[1] + a[3] * b[0];
        Fp4([c0, c1, c2, c3])
    }
}
impl MulAssign for Fp4 {
    #[inline(always)]
    fn mul_assign(&mut self, r: Fp4) {
        *self = *self * r;
    }
}
impl Neg for Fp4 {
    type Output = Fp4;
    #[inline(always)]
    fn neg(self) -> Fp4 {
        Fp4::ZERO - self
    }
}
impl From<Fp> for Fp4 {
    #[inline(always)]
    fn from(v: Fp) -> Fp4 {
        Fp4::from_base(v)
    }
}
impl From<u64> for Fp4 {
    fn from(v: u64) -> Fp4 {
        Fp4::from_base(Fp::new(v))
    }
}
impl Add<Fp> for Fp4 {
    type Output = Fp4;
    #[inline(always)]
    fn add(self, r: Fp) -> Fp4 {
        let mut c = self.0;
        c[0] += r;
        Fp4(c)
    }
}
impl Sub<Fp> for Fp4 {
    type Output = Fp4;
    #[inline(always)]
    fn sub(self, r: Fp) -> Fp4 {
        let mut c = self.0;
        c[0] -= r;
        Fp4(c)
    }
}
impl Mul<Fp> for Fp4 {
    type Output = Fp4;
    #[inline(always)]
    fn mul(self, r: Fp) -> Fp4 {
        self.mul_base(r)
    }
}
impl Mul<Fp4> for Fp {
    type Output = Fp4;
    #[inline(always)]
    fn mul(self, r: Fp4) -> Fp4 {
        r.mul_base(self)
    }
}
impl core::iter::Sum for Fp4 {
    fn sum<I: Iterator<Item = Fp4>>(iter: I) -> Fp4 {
        iter.fold(Fp4::ZERO, |a, b| a + b)
    }
}

// ------------------------------------------------------------ generic traits

/// Accumulates Σ αᵢ·vᵢ where αᵢ ∈ F_{p^4} and vᵢ is either a base value of `F`
/// (`add_base`) or an extension value.
pub trait WeightedAcc<F: FieldLike>: Copy + Send + Sync {
    fn zero() -> Self;
    fn add_base(&mut self, w: &F::Weight, v: F);
    fn add_ext(&mut self, alpha: Fp4, v: F::Ext);
    fn finish(self) -> F::Ext;
}

impl WeightedAcc<Fp> for Fp4 {
    #[inline(always)]
    fn zero() -> Fp4 {
        Fp4::ZERO
    }
    #[inline(always)]
    fn add_base(&mut self, w: &Fp4, v: Fp) {
        *self += w.mul_base(v);
    }
    #[inline(always)]
    fn add_ext(&mut self, alpha: Fp4, v: Fp4) {
        *self += alpha * v;
    }
    #[inline(always)]
    fn finish(self) -> Fp4 {
        self
    }
}

impl WeightedAcc<Fp4> for Fp4 {
    #[inline(always)]
    fn zero() -> Fp4 {
        Fp4::ZERO
    }
    #[inline(always)]
    fn add_base(&mut self, w: &Fp4, v: Fp4) {
        *self += *w * v;
    }
    #[inline(always)]
    fn add_ext(&mut self, alpha: Fp4, v: Fp4) {
        *self += alpha * v;
    }
    #[inline(always)]
    fn finish(self) -> Fp4 {
        self
    }
}

pub trait FieldLike:
    Copy
    + Clone
    + PartialEq
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + From<Fp>
    + core::fmt::Debug
    + Send
    + Sync
    + 'static
{
    /// The matching extension-field type (challenges, aux values, results).
    type Ext: ExtLike<Self>;
    /// Precomputed form of an `Fp4` weight for `Acc::add_base`.
    type Weight: Copy + Send + Sync + 'static;
    /// Accumulator for Σ αᵢ·vᵢ sums (random linear combinations).
    type Acc: WeightedAcc<Self>;
    fn weight(alpha: Fp4) -> Self::Weight;
    const ZERO: Self;
    const ONE: Self;
    fn from_u64(v: u64) -> Self {
        Self::from(Fp::new(v))
    }
    fn is_zero(&self) -> bool;
    /// self · e (e broadcast if self is a vector).
    fn mul_ext(self, e: Fp4) -> Self::Ext;
    fn to_ext(self) -> Self::Ext;
    /// Some(self) when this is a base-field value (scalar or vector), None for Fp4.
    fn as_base(self) -> Option<Self>;
}

/// Extension-field-like values matching a [`FieldLike`] base type.
pub trait ExtLike<F>:
    Copy
    + Clone
    + PartialEq
    + Default
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + From<Fp4>
    + core::fmt::Debug
    + Send
    + Sync
    + 'static
{
    const ZERO: Self;
    const ONE: Self;
    const LANES: usize;
    fn mul_base(self, b: F) -> Self;
    fn is_zero(&self) -> bool;
    /// Lane `i` as a scalar extension element.
    fn lane(&self, i: usize) -> Fp4;
}

impl FieldLike for Fp {
    type Ext = Fp4;
    type Weight = Fp4;
    type Acc = Fp4;
    #[inline(always)]
    fn weight(alpha: Fp4) -> Fp4 {
        alpha
    }
    const ZERO: Fp = Fp::ZERO;
    const ONE: Fp = Fp::ONE;
    fn is_zero(&self) -> bool {
        Fp::is_zero(*self)
    }
    #[inline(always)]
    fn mul_ext(self, e: Fp4) -> Fp4 {
        e.mul_base(self)
    }
    #[inline(always)]
    fn to_ext(self) -> Fp4 {
        Fp4::from_base(self)
    }
    #[inline(always)]
    fn as_base(self) -> Option<Fp> {
        Some(self)
    }
}
impl FieldLike for Fp4 {
    type Ext = Fp4;
    type Weight = Fp4;
    type Acc = Fp4;
    #[inline(always)]
    fn weight(alpha: Fp4) -> Fp4 {
        alpha
    }
    const ZERO: Fp4 = Fp4::ZERO;
    const ONE: Fp4 = Fp4::ONE;
    fn is_zero(&self) -> bool {
        Fp4::is_zero(*self)
    }
    #[inline(always)]
    fn mul_ext(self, e: Fp4) -> Fp4 {
        self * e
    }
    #[inline(always)]
    fn to_ext(self) -> Fp4 {
        self
    }
    #[inline(always)]
    fn as_base(self) -> Option<Fp4> {
        None
    }
}
impl ExtLike<Fp> for Fp4 {
    const ZERO: Fp4 = Fp4::ZERO;
    const ONE: Fp4 = Fp4::ONE;
    const LANES: usize = 1;
    #[inline(always)]
    fn mul_base(self, b: Fp) -> Fp4 {
        Fp4::mul_base(self, b)
    }
    fn is_zero(&self) -> bool {
        Fp4::is_zero(*self)
    }
    fn lane(&self, _i: usize) -> Fp4 {
        *self
    }
}
impl ExtLike<Fp4> for Fp4 {
    const ZERO: Fp4 = Fp4::ZERO;
    const ONE: Fp4 = Fp4::ONE;
    const LANES: usize = 1;
    #[inline(always)]
    fn mul_base(self, b: Fp4) -> Fp4 {
        self * b
    }
    fn is_zero(&self) -> bool {
        Fp4::is_zero(*self)
    }
    fn lane(&self, _i: usize) -> Fp4 {
        *self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants() {
        assert_eq!(P, 15 * (1 << 27) + 1);
        assert_eq!(MU.wrapping_mul(P), u32::MAX); // −p^{-1}·p ≡ −1 (mod 2^32)
        assert_eq!(R1 as u64, (1u64 << 32) % P64);
        assert_eq!(R2 as u64, ((1u128 << 64) % (P64 as u128)) as u64);
        // W is a non-residue
        assert_eq!(Fp::new(W as u64).pow((P64 - 1) / 2), Fp::new(P64 - 1));
        // generator has full order
        let g = Fp::new(GENERATOR as u64);
        assert_eq!(g.pow(P64 - 1), Fp::ONE);
        for q in [2u64, 3, 5] {
            assert_ne!(g.pow((P64 - 1) / q), Fp::ONE);
        }
    }

    #[test]
    fn field_basic() {
        let a = Fp::new(P64 - 1);
        assert_eq!(a + Fp::ONE, Fp::ZERO);
        assert_eq!(a * a, Fp::ONE);
        assert_eq!(Fp::new(5).as_u64(), 5);
        assert_eq!((Fp::new(123456789) * Fp::new(987654321)).as_u64(), (123456789u64 * 987654321) % P64);
        let x = Fp::new(0x1234_5678);
        assert_eq!(x * x.inverse(), Fp::ONE);
        assert_eq!(Fp::new(7) - Fp::new(9), Fp::new(P64 - 2));
        let mut xs: Vec<Fp> = (1..20).map(|i| Fp::new(i * 7919)).collect();
        let expect: Vec<Fp> = xs.iter().map(|x| x.inverse()).collect();
        Fp::batch_inverse(&mut xs);
        assert_eq!(xs, expect);
        assert_eq!(Fp::from_i64(-1), Fp::new(P64 - 1));
    }

    #[test]
    fn roots_of_unity() {
        for k in 0..=TWO_ADICITY {
            let w = Fp::two_adic_root(k);
            assert_eq!(w.pow(1u64 << k), Fp::ONE, "k={k}");
            if k > 0 {
                assert_ne!(w.pow(1u64 << (k - 1)), Fp::ONE, "k={k}");
                assert_eq!(Fp::two_adic_root(k).square(), Fp::two_adic_root(k - 1));
            }
        }
    }

    #[test]
    fn ext_field() {
        let a = Fp4([Fp::new(3), Fp::new(5), Fp::new(7), Fp::new(11)]);
        let b = Fp4([Fp::new(13), Fp::new(17), Fp::new(19), Fp::new(23)]);
        assert_eq!(a * b, b * a);
        assert_eq!((a * b) * a, a * (b * a));
        assert_eq!(a * Fp4::ONE, a);
        assert_eq!(a * a.inverse(), Fp4::ONE);
        assert_eq!((a * b) * b.inverse(), a);
        assert_eq!(Fp4::U.pow(4), Fp4::from_base(Fp::new(W as u64)));
        // Frobenius is the identity on the base field and an automorphism
        assert_eq!(Fp4::from_base(Fp::new(9)).frobenius(), Fp4::from_base(Fp::new(9)));
        assert_eq!((a * b).frobenius(), a.frobenius() * b.frobenius());
        assert_eq!(a.pow(P64), a.frobenius());
        let mut xs = vec![a, b, Fp4::ZERO, a * b];
        Fp4::batch_inverse(&mut xs);
        assert_eq!(xs[0], a.inverse());
        assert_eq!(xs[2], Fp4::ZERO);
        assert_eq!(xs[3], (a * b).inverse());
        // u^4 - W irreducible: the multiplicative group has order p^4 - 1
        let order = (P64 as u128).pow(4) - 1;
        // a^(p^4 - 1) via repeated squaring on u128 exponent
        let mut e = order;
        let mut base = a;
        let mut acc = Fp4::ONE;
        while e > 0 {
            if e & 1 == 1 {
                acc *= base;
            }
            base = base.square();
            e >>= 1;
        }
        assert_eq!(acc, Fp4::ONE);
    }
}
