//! Fiat–Shamir transcript over blake3.

use crate::merkle::Digest;
use zkb_field::{Fp, Fp4, P};

pub struct Transcript {
    state: Digest,
    counter: u64,
}

impl Transcript {
    pub fn new(domain: &[u8]) -> Transcript {
        let mut h = blake3::Hasher::new();
        h.update(b"zkbuild-stark-v0");
        h.update(domain);
        Transcript { state: *h.finalize().as_bytes(), counter: 0 }
    }

    pub fn absorb(&mut self, label: &[u8], data: &[u8]) {
        let mut h = blake3::Hasher::new();
        h.update(&self.state);
        h.update(&(label.len() as u64).to_le_bytes());
        h.update(label);
        h.update(&(data.len() as u64).to_le_bytes());
        h.update(data);
        self.state = *h.finalize().as_bytes();
        self.counter = 0;
    }

    pub fn absorb_digest(&mut self, label: &[u8], d: &Digest) {
        self.absorb(label, d);
    }

    pub fn absorb_fp(&mut self, label: &[u8], v: &[Fp]) {
        let mut bytes = Vec::with_capacity(v.len() * 4);
        for x in v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        self.absorb(label, &bytes);
    }

    pub fn absorb_fp4(&mut self, label: &[u8], v: &[Fp4]) {
        let mut bytes = Vec::with_capacity(v.len() * 16);
        for x in v {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        self.absorb(label, &bytes);
    }

    fn squeeze_bytes(&mut self) -> Digest {
        let mut h = blake3::Hasher::new();
        h.update(&self.state);
        h.update(b"squeeze");
        h.update(&self.counter.to_le_bytes());
        self.counter += 1;
        *h.finalize().as_bytes()
    }

    pub fn challenge_fp(&mut self) -> Fp {
        loop {
            let b = self.squeeze_bytes();
            let v = u32::from_le_bytes(b[..4].try_into().unwrap());
            // rejection sampling for uniformity
            if v < P {
                return Fp::from_canonical(v);
            }
        }
    }

    /// One squeeze yields four base elements (32 bytes → four u64 words,
    /// each reduced mod p: bias ≤ 2^-33).
    pub fn challenge_fp4(&mut self) -> Fp4 {
        let b = self.squeeze_bytes();
        let mut c = [Fp::ZERO; 4];
        for i in 0..4 {
            c[i] = Fp::from_le_bytes_reduce(b[8 * i..8 * i + 8].try_into().unwrap());
        }
        Fp4(c)
    }

    pub fn challenge_fp4_vec(&mut self, n: usize) -> Vec<Fp4> {
        (0..n).map(|_| self.challenge_fp4()).collect()
    }

    /// Proof-of-work: find a nonce such that hashing the state with it yields
    /// `bits` leading zero bits; absorbs the nonce.
    pub fn grind(&mut self, bits: u32) -> u64 {
        if bits == 0 {
            return 0;
        }
        let mut nonce = 0u64;
        while !Self::grind_ok(&self.state, bits, nonce) {
            nonce += 1;
        }
        self.absorb(b"grind-nonce", &nonce.to_le_bytes());
        nonce
    }

    /// Verifier side of [`Transcript::grind`]: check the nonce and absorb it.
    pub fn check_grind(&mut self, bits: u32, nonce: u64) -> bool {
        let ok = bits == 0 || Self::grind_ok(&self.state, bits, nonce);
        if bits > 0 {
            self.absorb(b"grind-nonce", &nonce.to_le_bytes());
        }
        ok
    }

    fn grind_ok(state: &Digest, bits: u32, nonce: u64) -> bool {
        let mut h = blake3::Hasher::new();
        h.update(state);
        h.update(b"grind");
        h.update(&nonce.to_le_bytes());
        let d = h.finalize();
        u64::from_le_bytes(d.as_bytes()[..8].try_into().unwrap()).leading_zeros() >= bits
    }

    /// Uniform index in [0, bound) where bound is a power of two.
    pub fn challenge_index(&mut self, bound: usize) -> usize {
        assert!(bound.is_power_of_two());
        let b = self.squeeze_bytes();
        let v = u64::from_le_bytes(b[8..16].try_into().unwrap());
        (v as usize) & (bound - 1)
    }
}
