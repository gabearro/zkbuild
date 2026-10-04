//! Poseidon2 over BabyBear, width 24 (rate 16, 8-element digests), S-box
//! x⁷, 8 full + 21 partial rounds — RISC Zero's instance, bit-exact with
//! `risc0_zkp::core::hash::poseidon2` (its parameters were generated and
//! vetted for the published cryptanalysis; the round constants and internal
//! diagonal below are copied verbatim, Apache-2.0/MIT). This is the
//! field-friendly hash the recursion stack verifies at ≈1 row per
//! permutation; BLAKE3 stays the default commitment hash of the base prover.
//!
//! Scalar permutation plus a 16-way AVX-512 version (sixteen independent
//! permutations, one per lane) for Merkle leaves and tree levels.

use zkb_field::simd::{Fp16, LANES};
use zkb_field::Fp;

pub const CELLS: usize = 24;
pub const RATE: usize = 16;
pub const OUT: usize = 8;
pub const ROUNDS_HALF_FULL: usize = 4;
pub const ROUNDS_PARTIAL: usize = 21;
pub const ROUNDS: usize = 2 * ROUNDS_HALF_FULL + ROUNDS_PARTIAL;

/// Canonical (non-Montgomery) round constants, `ROUNDS × CELLS`.
pub const ROUND_CONSTANTS: [u32; ROUNDS * CELLS] = [
    0x0FA20C37, 0x0795BB97, 0x12C60B9C, 0x0EABD88E, 0x096485CA, 0x07093527, 0x1B1D4E50, 0x30A01ACE,
    0x3BD86F5A, 0x69AF7C28, 0x3F94775F, 0x731560E8, 0x465A0ECD, 0x574EF807, 0x62FD4870, 0x52CCFE44,
    0x14772B14, 0x4DEDF371, 0x260ACD7C, 0x1F51DC58, 0x75125532, 0x686A4D7B, 0x54BAC179, 0x31947706,
    0x29799D3B, 0x6E01AE90, 0x203A7A64, 0x4F7E25BE, 0x72503F77, 0x45BD3B69, 0x769BD6B4, 0x5A867F08,
    0x4FDBA082, 0x251C4318, 0x28F06201, 0x6788C43A, 0x4C6D6A99, 0x357784A8, 0x2ABAF051, 0x770F7DE6,
    0x1794B784, 0x4796C57A, 0x724B7A10, 0x449989A7, 0x64935CF1, 0x59E14AAC, 0x0E620BB8, 0x3AF5A33B,
    0x4465CC0E, 0x019DF68F, 0x4AF8D068, 0x08784F82, 0x0CEFDEAE, 0x6337A467, 0x32FA7A16, 0x486F62D6,
    0x386A7480, 0x20F17C4A, 0x54E50DA8, 0x2012CF03, 0x5FE52950, 0x09AFB6CD, 0x2523044E, 0x5C54D0EF,
    0x71C01F3C, 0x60B2C4FB, 0x4050B379, 0x5E6A70A5, 0x418543F5, 0x71DEBE56, 0x1AAD2994, 0x3368A483,
    0x07A86F3A, 0x5EA43FF1, 0x2443780E, 0x4CE444F7, 0x146F9882, 0x3132B089, 0x197EA856, 0x667030C3,
    0x2317D5DC, 0x0C2C48A7, 0x56B2DF66, 0x67BD81E9, 0x4FCDFB19, 0x4BAAEF32, 0x0328D30A, 0x6235760D,
    0x12432912, 0x0A49E258, 0x030E1B70, 0x48CAEB03, 0x49E4D9E9, 0x1051B5C6, 0x6A36DBBE, 0x4CFF27A5,
    0x1DA78EC2, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x730B0924, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x3EB56CF3, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x5BD93073, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x37204C97, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x51642D89, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x66E943E8, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x1A3E72DE, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x70BEB1E9, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x30FF3B3F, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x4240D1C4, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x12647B8D, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x65D86965, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x49EF4D7C, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x47785697, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x46B3969F, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x5C7B7A0E, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x7078FC60, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x4F22D482, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x482A9AEE, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x6BEB839D, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x032959AD, 0x2B18AF6A, 0x55D3DC8C, 0x43BD26C8, 0x0C41595F, 0x7048D2E2, 0x00DB8983, 0x2AF563D7,
    0x6E84758F, 0x611D64E1, 0x1F9977E2, 0x64163A0A, 0x5C5FC27B, 0x02E22561, 0x3A2D75DB, 0x1BA7B71A,
    0x34343F64, 0x7406B35D, 0x19DF8299, 0x6FF4480A, 0x514A81C8, 0x57AB52CE, 0x6AD69F52, 0x3E0C0E0D,
    0x48126114, 0x2A9D62CC, 0x17441F23, 0x485762BB, 0x2F218674, 0x06FDC64A, 0x0861B7F2, 0x3B36EEE6,
    0x70A11040, 0x04B31737, 0x3722A872, 0x2A351C63, 0x623560DC, 0x62584AB2, 0x382C7C04, 0x3BF9EDC7,
    0x0E38FE51, 0x376F3B10, 0x5381E178, 0x3AFC61C7, 0x5C1BCB4D, 0x6643CE1F, 0x2D0AF1C1, 0x08F583CC,
    0x5D6FF60F, 0x6324C1E5, 0x74412FB7, 0x70C0192E, 0x0B72F141, 0x4067A111, 0x57388C4F, 0x351009EC,
    0x0974C159, 0x539A58B3, 0x038C0CFF, 0x476C0392, 0x3F7BC15F, 0x4491DD2C, 0x4D1FEF55, 0x04936AE3,
    0x58214DD4, 0x683C6AAD, 0x1B42F16B, 0x6DC79135, 0x2D4E71EC, 0x3E2946EA, 0x59DCE8DB, 0x6CEE892A,
    0x47F07350, 0x7106CE93, 0x3BD4A7A9, 0x2BFE636A, 0x430011E9, 0x001CD66A, 0x307FAF5B, 0x0D9EF3FE,
    0x6D40043A, 0x2E8F470C, 0x1B6865E8, 0x0C0E6C01, 0x4D41981F, 0x423B9D3D, 0x410408CC, 0x263F0884,
    0x5311BBD0, 0x4DAE58D8, 0x30401CEA, 0x09AFA575, 0x4B3D5B42, 0x63AC0B37, 0x5FE5BB14, 0x5244E9D4,
];

/// Diagonal of the internal matrix `M_I = J + diag(·)` (J = all-ones), canonical.
pub const M_INT_DIAG: [u32; CELLS] = [
    0x409133f0, 0x1667a8a1, 0x06a6c7b6, 0x6f53160e, 0x273b11d1, 0x03176c5d, 0x72f9bbf9, 0x73ceba91,
    0x5cdef81d, 0x01393285, 0x46daee06, 0x065d7ba6, 0x52d72d6f, 0x05dd05e0, 0x3bab4b63, 0x6ada3842,
    0x2fc5fbec, 0x770d61b0, 0x5715aae9, 0x03ef0e90, 0x75b6c770, 0x242adf5f, 0x00d0ca4c, 0x36c0e388,
];

#[inline(always)]
fn sbox(x: Fp) -> Fp {
    let x2 = x * x;
    let x4 = x2 * x2;
    x4 * x2 * x
}

/// Multiplication by the 4×4 circulant `circ(2, 3, 1, 1)` (Poseidon2 paper, appendix B).
#[inline(always)]
fn circ4(x: [Fp; 4]) -> [Fp; 4] {
    let t0 = x[0] + x[1];
    let t1 = x[2] + x[3];
    let t2 = x[1] + x[1] + t1;
    let t3 = x[3] + x[3] + t0;
    let t4 = t1 + t1 + t1 + t1 + t3;
    let t5 = t0 + t0 + t0 + t0 + t2;
    let t6 = t3 + t5;
    let t7 = t2 + t4;
    [t6, t5, t7, t4]
}

/// External linear layer `M_E = circ(2·M4, M4, …, M4)`.
fn m_ext(c: &mut [Fp; CELLS]) {
    let mut sums = [Fp::ZERO; 4];
    for i in 0..CELLS / 4 {
        let o = circ4([c[4 * i], c[4 * i + 1], c[4 * i + 2], c[4 * i + 3]]);
        for j in 0..4 {
            c[4 * i + j] = o[j];
            sums[j] += o[j];
        }
    }
    for i in 0..CELLS {
        c[i] += sums[i % 4];
    }
}

fn m_int(c: &mut [Fp; CELLS]) {
    let sum = c.iter().fold(Fp::ZERO, |a, &x| a + x);
    for i in 0..CELLS {
        c[i] = sum + Fp::from_canonical(M_INT_DIAG[i]) * c[i];
    }
}

/// The permutation (`poseidon2_mix`).
pub fn permute(c: &mut [Fp; CELLS]) {
    m_ext(c);
    let mut round = 0;
    for _ in 0..ROUNDS_HALF_FULL {
        for i in 0..CELLS {
            c[i] = sbox(c[i] + Fp::from_canonical(ROUND_CONSTANTS[round * CELLS + i]));
        }
        m_ext(c);
        round += 1;
    }
    for _ in 0..ROUNDS_PARTIAL {
        c[0] = sbox(c[0] + Fp::from_canonical(ROUND_CONSTANTS[round * CELLS]));
        m_int(c);
        round += 1;
    }
    for _ in 0..ROUNDS_HALF_FULL {
        for i in 0..CELLS {
            c[i] = sbox(c[i] + Fp::from_canonical(ROUND_CONSTANTS[round * CELLS + i]));
        }
        m_ext(c);
        round += 1;
    }
}

/// Unpadded sponge hash of a fixed-length element sequence (RISC Zero's
/// `unpadded_hash`): collision resistant among inputs of equal length.
pub fn hash_elems(elems: &[Fp]) -> [Fp; OUT] {
    let mut st = [Fp::ZERO; CELLS];
    let mut unmixed = 0;
    for &v in elems {
        st[unmixed] = v;
        unmixed += 1;
        if unmixed == RATE {
            permute(&mut st);
            unmixed = 0;
        }
    }
    if unmixed != 0 || elems.is_empty() {
        for x in st.iter_mut().take(RATE).skip(unmixed) {
            *x = Fp::ZERO;
        }
        permute(&mut st);
    }
    st[..OUT].try_into().unwrap()
}

/// Two-to-one compression for Merkle trees: permute `a ‖ b ‖ 0…`, take the first 8.
pub fn hash_pair(a: &[Fp; OUT], b: &[Fp; OUT]) -> [Fp; OUT] {
    let mut st = [Fp::ZERO; CELLS];
    st[..OUT].copy_from_slice(a);
    st[OUT..2 * OUT].copy_from_slice(b);
    permute(&mut st);
    st[..OUT].try_into().unwrap()
}

// ---------------------------------------------------------------- 16-way

#[inline(always)]
fn sbox16(x: Fp16) -> Fp16 {
    let x2 = x * x;
    let x4 = x2 * x2;
    x4 * x2 * x
}

#[inline(always)]
fn circ4x16(x: [Fp16; 4]) -> [Fp16; 4] {
    let t0 = x[0] + x[1];
    let t1 = x[2] + x[3];
    let t2 = x[1] + x[1] + t1;
    let t3 = x[3] + x[3] + t0;
    let t4 = t1 + t1 + t1 + t1 + t3;
    let t5 = t0 + t0 + t0 + t0 + t2;
    let t6 = t3 + t5;
    let t7 = t2 + t4;
    [t6, t5, t7, t4]
}

fn m_ext16(c: &mut [Fp16; CELLS]) {
    let mut sums = [Fp16::ZERO; 4];
    for i in 0..CELLS / 4 {
        let o = circ4x16([c[4 * i], c[4 * i + 1], c[4 * i + 2], c[4 * i + 3]]);
        for j in 0..4 {
            c[4 * i + j] = o[j];
            sums[j] += o[j];
        }
    }
    for i in 0..CELLS {
        c[i] += sums[i % 4];
    }
}

/// Sixteen independent permutations, state cell `i` of lane `l` in `c[i].lane(l)`.
pub fn permute16(c: &mut [Fp16; CELLS]) {
    let rc = |r: usize, i: usize| Fp16::splat(Fp::from_canonical(ROUND_CONSTANTS[r * CELLS + i]));
    let diag: [Fp16; CELLS] = core::array::from_fn(|i| Fp16::splat(Fp::from_canonical(M_INT_DIAG[i])));
    m_ext16(c);
    let mut round = 0;
    for _ in 0..ROUNDS_HALF_FULL {
        for i in 0..CELLS {
            c[i] = sbox16(c[i] + rc(round, i));
        }
        m_ext16(c);
        round += 1;
    }
    for _ in 0..ROUNDS_PARTIAL {
        c[0] = sbox16(c[0] + rc(round, 0));
        let mut sum = c[0];
        for i in 1..CELLS {
            sum += c[i];
        }
        for i in 0..CELLS {
            c[i] = sum + diag[i] * c[i];
        }
        round += 1;
    }
    for _ in 0..ROUNDS_HALF_FULL {
        for i in 0..CELLS {
            c[i] = sbox16(c[i] + rc(round, i));
        }
        m_ext16(c);
        round += 1;
    }
}

/// Hash sixteen element sequences of equal length (`rows[l]` is lane `l`'s input), sponge-wise.
pub fn hash_elems16(rows: &[&[Fp]; LANES]) -> [[Fp; OUT]; LANES] {
    let n = rows[0].len();
    let mut st = [Fp16::ZERO; CELLS];
    let mut unmixed = 0;
    for k in 0..n {
        st[unmixed] = Fp16::from_lanes(core::array::from_fn(|l| rows[l][k].0));
        unmixed += 1;
        if unmixed == RATE {
            permute16(&mut st);
            unmixed = 0;
        }
    }
    if unmixed != 0 || n == 0 {
        for x in st.iter_mut().take(RATE).skip(unmixed) {
            *x = Fp16::ZERO;
        }
        permute16(&mut st);
    }
    core::array::from_fn(|l| core::array::from_fn(|i| st[i].lane(l)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn risc0_permutation_vector() {
        let mut st: [Fp; CELLS] = core::array::from_fn(|i| Fp::from_canonical(i as u32));
        permute(&mut st);
        let goal: [u32; CELLS] = [
            0x2ed3e23d, 0x12921fb0, 0x0e659e79, 0x61d81dc9, 0x32bae33b, 0x62486ae3, 0x1e681b60, 0x24b91325, 0x2a2ef5b9, 0x50e8593e, 0x5bc818ec, 0x10691997, 0x35a14520, 0x2ba6a3c5,
            0x279d47ec, 0x55014e81, 0x5953a67f, 0x2f403111, 0x6b8828ff, 0x1801301f, 0x2749207a, 0x3dc9cf21, 0x3c985ba2, 0x57a99864,
        ];
        for i in 0..CELLS {
            assert_eq!(st[i].as_canonical(), goal[i], "cell {i}");
        }
    }

    #[test]
    fn risc0_hash_vectors() {
        let buf: [u32; 32] = [
            943718400, 1887436800, 2013125296, 1761607679, 692060158, 1761607634, 566231037, 1509949437, 440401916, 1384120316, 314572795, 1258291195, 188743674, 1132462074, 62914553, 1006632953,
            1950351353, 880803832, 1824522232, 754974711, 1698693111, 629145590, 1572863990, 503316469, 1447034869, 377487348, 1321205748, 251658227, 1195376627, 125829106, 1069547506, 2013265906,
        ];
        let elems: Vec<Fp> = buf.iter().map(|&v| Fp::from_canonical(v)).collect();
        let out = hash_elems(&elems);
        let goal = [0x722baada, 0x5b352fed, 0x3684017b, 0x540d4a7b, 0x44ffd422, 0x48615f97, 0x1a496f45, 0x203ca999];
        assert_eq!(out.map(|x| x.as_canonical()), goal);
        let buf17: [u32; 17] = [943718400, 1887436800, 2013125296, 1761607679, 692060158, 1635778558, 566231037, 1509949437, 440401916, 1384120316, 314572795, 1258291195, 188743674, 1132462074, 62914553, 1006632953, 1950351353];
        let elems: Vec<Fp> = buf17.iter().map(|&v| Fp::from_canonical(v)).collect();
        let out = hash_elems(&elems);
        let goal = [0x622615d7, 0x1cfe9764, 0x166cb1c9, 0x76febcde, 0x6056219f, 0x326359cf, 0x5c2cca75, 0x233dc3ff];
        assert_eq!(out.map(|x| x.as_canonical()), goal);
    }

    #[test]
    fn vector_matches_scalar() {
        let rows: Vec<Vec<Fp>> = (0..LANES).map(|l| (0..37).map(|k| Fp::new((l * 1000 + k * 7 + 3) as u64)).collect()).collect();
        let refs: [&[Fp]; LANES] = core::array::from_fn(|l| rows[l].as_slice());
        let v = hash_elems16(&refs);
        for l in 0..LANES {
            assert_eq!(v[l], hash_elems(&rows[l]), "lane {l}");
        }
    }
}
