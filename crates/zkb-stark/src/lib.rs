//! A from-scratch STARK: Goldilocks field, blake3 Merkle commitments,
//! LogUp-friendly aux columns, DEEP-ALI composition and FRI, with
//! zero-knowledge blinding and multi-segment ("continuation") proofs that
//! share their lookup challenges.
//!
//! Protocol (prover side; the verifier mirrors it through Fiat–Shamir):
//!
//! 1. absorb verifying key; for every segment absorb its public values and
//!    its `main` commitment (LDE on coset `7·<ω_M>`; when `zk`, the last
//!    `random_rows()` rows of every column are uniformly random and Merkle
//!    leaves are salted — transition constraints are only enforced on the
//!    real rows, so the trace degree stays < N and the blowup stays 4)
//! 2. draw the shared `num_challenges` Fp4 challenges (LogUp β, γ, ...)
//! 3. for every segment, in order:
//!    a. commit `aux`; absorb aux public values (LogUp sums)
//!    b. draw α; build `C = Σ αⁱ·cᵢ / Zᵢ`, split into chunks of degree < N, commit
//!    c. draw z; send evaluations of every column at z and g·z, chunks at z
//!    d. draw DEEP weights; FRI on the DEEP quotient; grind; answer queries
//!
//! Segments are bound together by the single transcript; cross-segment
//! consistency of public values is the AIR user's job (see `zkb-air`).

pub mod air;
pub mod fri;
pub mod blake3x16;
pub mod merkle;
pub mod poseidon2;
pub mod prover;
pub mod transcript;
pub mod verifier;

use merkle::{Digest, MerklePath};
use serde::{Deserialize, Serialize};
use zkb_field::{Fp, Fp4};

pub use air::{Air, AirShape, CombinationWeights, ConstraintSink, RowAccess, RowView, SinkMode, SliceRow};
pub use fri::FriConfig;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct StarkConfig {
    /// LDE blowup: M = N · 2^log_blowup.
    pub log_blowup: u32,
    pub num_queries: usize,
    pub log_final_size: u32,
    /// Zero-knowledge: append random rows to every trace and salt Merkle leaves.
    pub zk: bool,
    /// Proof-of-work bits required on the transcript before the queries are drawn.
    pub grinding_bits: u32,
}

impl StarkConfig {
    /// Development parameters, no zero knowledge: blowup 4, 40 queries, no grinding.
    pub fn fast() -> StarkConfig {
        StarkConfig { log_blowup: 2, num_queries: 40, log_final_size: 5, zk: false, grinding_bits: 0 }
    }
    /// Zero-knowledge parameters: blowup 4, 40 queries (2 bits each) + 16 bits of grinding.
    pub fn zk() -> StarkConfig {
        StarkConfig { log_blowup: 2, num_queries: 40, log_final_size: 5, zk: true, grinding_bits: 16 }
    }
    pub fn zk_secure() -> StarkConfig {
        StarkConfig { log_blowup: 3, num_queries: 44, log_final_size: 5, zk: true, grinding_bits: 20 }
    }
    /// FRI rate exponent: log2(M / degree bound); committed polynomials have degree < N.
    pub fn log_rate(&self) -> u32 {
        self.log_blowup
    }
    pub fn fri(&self) -> FriConfig {
        FriConfig { log_rate: self.log_rate(), num_queries: self.num_queries, log_final_size: self.log_final_size }
    }
    /// Number of random rows appended to every trace (ethSTARK-style zero knowledge):
    /// one per opened point plus margin.
    pub fn random_rows(&self) -> usize {
        if self.zk {
            4 * self.num_queries + 8
        } else {
            0
        }
    }
}

/// Coset shift for the LDE domain.
pub const LDE_SHIFT: Fp = Fp::new(zkb_field::GENERATOR as u64);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifyingKey {
    pub log_n: u32,
    pub config: StarkConfig,
    pub fixed_root: Digest,
    pub main_width: usize,
    pub fixed_width: usize,
    pub aux_width: usize,
}

impl VerifyingKey {
    pub fn digest(&self) -> Digest {
        let mut h = blake3::Hasher::new();
        h.update(b"zkb-vk-v1");
        h.update(&self.log_n.to_le_bytes());
        h.update(&self.config.log_blowup.to_le_bytes());
        h.update(&(self.config.num_queries as u64).to_le_bytes());
        h.update(&self.config.log_final_size.to_le_bytes());
        h.update(&[self.config.zk as u8]);
        h.update(&self.config.grinding_bits.to_le_bytes());
        h.update(&self.fixed_root);
        h.update(&(self.main_width as u64).to_le_bytes());
        h.update(&(self.fixed_width as u64).to_le_bytes());
        h.update(&(self.aux_width as u64).to_le_bytes());
        *h.finalize().as_bytes()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OodEvals {
    pub main: Vec<Fp4>,
    pub main_next: Vec<Fp4>,
    pub fixed: Vec<Fp4>,
    pub fixed_next: Vec<Fp4>,
    pub aux: Vec<Fp4>,
    pub aux_next: Vec<Fp4>,
    pub comp: Vec<Fp4>,
}

/// Opened row of one commitment at one LDE position.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RowOpening {
    pub values: Vec<Fp>,
    pub salt: Option<[u8; 32]>,
    pub path: MerklePath,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PositionOpening {
    pub main: RowOpening,
    pub fixed: RowOpening,
    pub aux: RowOpening,
    pub comp: RowOpening,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryOpening {
    /// Openings at LDE positions l and l + M/2.
    pub lo: PositionOpening,
    pub hi: PositionOpening,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SegmentProof {
    pub public: Vec<Fp4>,
    pub grind_nonce: u64,
    pub main_root: Digest,
    pub aux_root: Digest,
    pub comp_root: Digest,
    pub aux_public: Vec<Fp4>,
    pub ood: OodEvals,
    pub fri: fri::FriProof,
    pub openings: Vec<QueryOpening>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Proof {
    pub log_n: u32,
    pub segments: Vec<SegmentProof>,
}

impl Proof {
    pub fn size_bytes(&self) -> usize {
        bincode::serialized_size(self).unwrap_or(0) as usize
    }
}
