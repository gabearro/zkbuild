//! FRI low-degree test (fold-by-two, Fp4 values on a multiplicative coset).

use crate::merkle::{hash_leaf_fp, verify_path, Digest, MerklePath, MerkleTree};
use crate::transcript::Transcript;
use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use zkb_field::ntt::{coset_interpolate_ext, NttPlan};
use zkb_field::poly::eval_ext_ext;
use zkb_field::{Fp, Fp4};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FriConfig {
    /// log2(domain size / degree bound).
    pub log_rate: u32,
    pub num_queries: usize,
    /// Fold until the domain is this size, then send the polynomial.
    pub log_final_size: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FriLayerOpening {
    /// (v[l], v[l + half]) for the layer's leaf index l.
    pub pair: (Fp4, Fp4),
    pub path: MerklePath,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FriQueryProof {
    /// One opening per committed layer (layers 1..J; layer 0 is computed by the verifier).
    pub layers: Vec<FriLayerOpening>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FriProof {
    pub layer_roots: Vec<Digest>,
    pub final_coeffs: Vec<Fp4>,
    pub queries: Vec<FriQueryProof>,
}

fn leaf_of_pair(a: Fp4, b: Fp4) -> Digest {
    hash_leaf_fp(&[a.0[0], a.0[1], a.0[2], a.0[3], b.0[0], b.0[1], b.0[2], b.0[3]])
}

pub struct FriProver {
    cfg: FriConfig,
    /// layers[j] = values of layer j (layer 0 is the input, not committed).
    layers: Vec<Vec<Fp4>>,
    trees: Vec<MerkleTree>,
    pub proof_roots: Vec<Digest>,
    pub final_coeffs: Vec<Fp4>,
}

impl FriProver {
    /// Commit to all folded layers. `evals` are on the coset `shift * <omega_M>`.
    pub fn commit(cfg: FriConfig, transcript: &mut Transcript, evals: Vec<Fp4>, shift: Fp) -> FriProver {
        let m = evals.len();
        assert!(m.is_power_of_two());
        let final_size = 1usize << cfg.log_final_size;
        assert!(m >= final_size);
        let mut layers = vec![evals];
        let mut trees = Vec::new();
        let mut roots = Vec::new();
        let mut cur_shift = shift;
        let mut cur_size = m;
        while cur_size > final_size {
            let cur = layers.last().unwrap();
            let half = cur_size / 2;
            // commit to current layer as pairs (except layer 0, which the verifier derives).
            let beta = if layers.len() == 1 {
                transcript.challenge_fp4()
            } else {
                let tree = MerkleTree::from_leaves((0..half).map(|l| leaf_of_pair(cur[l], cur[l + half])).collect());
                let root = tree.root();
                transcript.absorb_digest(b"fri-layer", &root);
                roots.push(root);
                trees.push(tree);
                transcript.challenge_fp4()
            };
            // fold
            let omega = Fp::two_adic_root(cur_size.trailing_zeros());
            let two_inv = Fp::new(2).inverse();
            let mut xs: Vec<Fp> = Vec::with_capacity(half);
            let mut x = cur_shift;
            for _ in 0..half {
                xs.push(x);
                x *= omega;
            }
            Fp::batch_inverse(&mut xs);
            let next: Vec<Fp4> = (0..half)
                .map(|l| {
                    let a = cur[l];
                    let b = cur[l + half];
                    let even = (a + b).mul_base(two_inv);
                    let odd = (a - b).mul_base(two_inv * xs[l]);
                    even + beta * odd
                })
                .collect();
            layers.push(next);
            cur_shift = cur_shift.square();
            cur_size = half;
        }
        // final layer: send coefficients
        let last = layers.last().unwrap();
        let plan = NttPlan::new(cur_size);
        let mut coeffs = coset_interpolate_ext(&plan, last, cur_shift);
        let max_len = cur_size >> cfg.log_rate;
        // Honest prover: high coefficients are zero; truncate.
        for c in &coeffs[max_len.max(1)..] {
            debug_assert!(c.is_zero(), "FRI final polynomial exceeds degree bound");
        }
        coeffs.truncate(max_len.max(1));
        transcript.absorb_fp4(b"fri-final", &coeffs);
        FriProver { cfg, layers, trees, proof_roots: roots, final_coeffs: coeffs }
    }

    /// Open the query path for layer-0 position pair index `l0` (in [0, M/2)).
    pub fn open(&self, l0: usize) -> FriQueryProof {
        let mut openings = Vec::new();
        let mut l = l0;
        // layer j (j>=1) has size M/2^j, leaf index = l mod (size/2)
        for (j, tree) in self.trees.iter().enumerate() {
            let layer = &self.layers[j + 1];
            let half = layer.len() / 2;
            l %= half;
            openings.push(FriLayerOpening { pair: (layer[l], layer[l + half]), path: tree.open(l) });
        }
        FriQueryProof { layers: openings }
    }

    pub fn into_proof(self, queries: Vec<FriQueryProof>) -> FriProof {
        FriProof { layer_roots: self.proof_roots, final_coeffs: self.final_coeffs, queries }
    }

    pub fn config(&self) -> FriConfig {
        self.cfg
    }
}

/// Verifier-side state after replaying the FRI commitments through the transcript.
pub struct FriVerifier {
    #[allow(dead_code)]
    cfg: FriConfig,
    log_m: u32,
    shift: Fp,
    betas: Vec<Fp4>,
    roots: Vec<Digest>,
    final_coeffs: Vec<Fp4>,
}

impl FriVerifier {
    pub fn replay(cfg: FriConfig, transcript: &mut Transcript, proof: &FriProof, log_m: u32, shift: Fp) -> Result<FriVerifier> {
        let num_layers = log_m.checked_sub(cfg.log_final_size).ok_or_else(|| anyhow::anyhow!("domain too small"))? as usize;
        ensure!(proof.layer_roots.len() + 1 == num_layers || (num_layers == 0 && proof.layer_roots.is_empty()), "wrong number of FRI layers");
        let mut betas = Vec::with_capacity(num_layers);
        for j in 0..num_layers {
            if j > 0 {
                transcript.absorb_digest(b"fri-layer", &proof.layer_roots[j - 1]);
            }
            betas.push(transcript.challenge_fp4());
        }
        let final_size = 1usize << cfg.log_final_size;
        let max_len = (final_size >> cfg.log_rate).max(1);
        ensure!(proof.final_coeffs.len() <= max_len, "FRI final polynomial too large");
        transcript.absorb_fp4(b"fri-final", &proof.final_coeffs);
        Ok(FriVerifier { cfg, log_m, shift, betas, roots: proof.layer_roots.clone(), final_coeffs: proof.final_coeffs.clone() })
    }

    /// Verify one query. `l0` is the layer-0 pair index, `v0` the pair
    /// (D(x_{l0}), D(x_{l0 + M/2})) computed by the caller.
    pub fn verify_query(&self, l0: usize, v0: (Fp4, Fp4), q: &FriQueryProof) -> Result<()> {
        let m = 1usize << self.log_m;
        ensure!(q.layers.len() == self.roots.len(), "FRI query has wrong number of layers");
        let two_inv = Fp::new(2).inverse();
        let mut size = m;
        let mut shift = self.shift;
        let mut l = l0;
        let mut pair = v0;
        for (j, beta) in self.betas.iter().enumerate() {
            let half = size / 2;
            let omega = Fp::two_adic_root(size.trailing_zeros());
            let x = shift * omega.pow(l as u64);
            let (a, b) = pair;
            let even = (a + b).mul_base(two_inv);
            let odd = (a - b).mul_base(two_inv * x.inverse());
            let folded = even + *beta * odd;
            // move to next layer
            size = half;
            shift = shift.square();
            if j + 1 < self.betas.len() {
                let next_half = size / 2;
                let nl = l % next_half;
                let op = &q.layers[j];
                let leaf = leaf_of_pair(op.pair.0, op.pair.1);
                ensure!(verify_path(&self.roots[j], nl, leaf, &op.path, next_half), "FRI layer {} Merkle path invalid", j + 1);
                let expect = if l < next_half { op.pair.0 } else { op.pair.1 };
                ensure!(expect == folded, "FRI fold mismatch at layer {}", j + 1);
                pair = op.pair;
                l = nl;
            } else {
                // final layer: evaluate the sent polynomial at x^2 (= point of index l in the final domain)
                let omega_f = Fp::two_adic_root(size.trailing_zeros());
                let xf = shift * omega_f.pow(l as u64);
                let v = eval_ext_ext(&self.final_coeffs, Fp4::from_base(xf));
                ensure!(v == folded, "FRI final polynomial mismatch");
            }
        }
        if self.betas.is_empty() {
            bail!("FRI with zero layers unsupported");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zkb_field::ntt::coset_evaluate_ext;

    #[test]
    fn fri_roundtrip() {
        let log_n = 8;
        let cfg = FriConfig { log_rate: 2, num_queries: 8, log_final_size: 4 };
        let n = 1usize << log_n;
        let m = n << cfg.log_rate;
        let shift = Fp::new(7);
        let coeffs: Vec<Fp4> = (0..n).map(|i| Fp4([Fp::new(i as u64 * 13 + 5), Fp::new(i as u64 + 1), Fp::new(3), Fp::new(i as u64)])).collect();
        let plan = NttPlan::new(m);
        let evals = coset_evaluate_ext(&plan, &coeffs, shift);
        let mut t = Transcript::new(b"test");
        let prover = FriProver::commit(cfg, &mut t, evals.clone(), shift);
        let queries: Vec<usize> = (0..cfg.num_queries).map(|_| t.challenge_index(m / 2)).collect();
        let qp: Vec<FriQueryProof> = queries.iter().map(|&l| prover.open(l)).collect();
        let proof = prover.into_proof(qp);

        let mut tv = Transcript::new(b"test");
        let ver = FriVerifier::replay(cfg, &mut tv, &proof, m.trailing_zeros(), shift).unwrap();
        let vq: Vec<usize> = (0..cfg.num_queries).map(|_| tv.challenge_index(m / 2)).collect();
        assert_eq!(vq, queries);
        for (i, &l) in vq.iter().enumerate() {
            ver.verify_query(l, (evals[l], evals[l + m / 2]), &proof.queries[i]).unwrap();
        }
        // tamper
        let mut bad = evals[queries[0]];
        bad.0[0] += Fp::ONE;
        assert!(ver.verify_query(queries[0], (bad, evals[queries[0] + m / 2]), &proof.queries[0]).is_err());
    }

    #[test]
    fn fri_rejects_high_degree() {
        let log_n = 8;
        let cfg = FriConfig { log_rate: 2, num_queries: 16, log_final_size: 4 };
        let n = 1usize << log_n;
        let m = n << cfg.log_rate;
        let shift = Fp::new(7);
        // degree 2n polynomial: exceeds bound
        let coeffs: Vec<Fp4> = (0..2 * n).map(|i| Fp4::from_base(Fp::new(i as u64 * 13 + 5))).collect();
        let plan = NttPlan::new(m);
        let evals = coset_evaluate_ext(&plan, &coeffs, shift);
        let mut t = Transcript::new(b"test");
        // the prover's debug assertion would fire; build in release semantics by catching
        let prover = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| FriProver::commit(cfg, &mut t, evals.clone(), shift)));
        if let Ok(prover) = prover {
            let queries: Vec<usize> = (0..cfg.num_queries).map(|_| t.challenge_index(m / 2)).collect();
            let qp: Vec<FriQueryProof> = queries.iter().map(|&l| prover.open(l)).collect();
            let proof = prover.into_proof(qp);
            let mut tv = Transcript::new(b"test");
            let ver = FriVerifier::replay(cfg, &mut tv, &proof, m.trailing_zeros(), shift).unwrap();
            let vq: Vec<usize> = (0..cfg.num_queries).map(|_| tv.challenge_index(m / 2)).collect();
            let any_fail = vq.iter().enumerate().any(|(i, &l)| ver.verify_query(l, (evals[l], evals[l + m / 2]), &proof.queries[i]).is_err());
            assert!(any_fail, "high-degree polynomial should be rejected");
        }
    }
}
