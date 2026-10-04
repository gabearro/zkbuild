//! STARK verifier (multi-segment).

use crate::air::{Air, CombinationWeights, ConstraintSink, RowView, SinkMode};
use crate::fri::FriVerifier;
use crate::merkle::{hash_leaf_salted, verify_path, Digest};
use crate::prover::{absorb_ood, alpha_powers, deep_at_point};
use crate::transcript::Transcript;
use crate::{PositionOpening, Proof, RowOpening, VerifyingKey, LDE_SHIFT};
use anyhow::{ensure, Result};
use zkb_field::{Fp, Fp4};

fn check_opening(root: &Digest, idx: usize, m: usize, o: &RowOpening, zk: bool, what: &str) -> Result<()> {
    ensure!(o.salt.is_some() == zk, "{what}: salt presence does not match zk mode");
    ensure!(verify_path(root, idx, hash_leaf_salted(&o.values, o.salt.as_ref()), &o.path, m), "{what} opening invalid at {idx}");
    Ok(())
}

fn check_position(roots: [&Digest; 4], idx: usize, m: usize, p: &PositionOpening, zk: bool) -> Result<()> {
    check_opening(roots[0], idx, m, &p.main, zk, "main")?;
    check_opening(roots[1], idx, m, &p.fixed, false, "fixed")?;
    check_opening(roots[2], idx, m, &p.aux, zk, "aux")?;
    check_opening(roots[3], idx, m, &p.comp, false, "comp")?;
    Ok(())
}

/// Verifies every segment; on success returns, for every segment k, the
/// global LogUp challenges the execution was proven with (see `prover::prove`).
pub fn verify<A: Air>(air: &A, vk: &VerifyingKey, proof: &Proof, publics: &[Vec<Fp4>]) -> Result<Vec<Fp4>> {
    let shape = air.shape();
    let cfg = vk.config;
    ensure!(proof.log_n == vk.log_n, "trace length mismatch with verifying key");
    ensure!(shape.main_width == vk.main_width && shape.fixed_width == vk.fixed_width && shape.aux_width == vk.aux_width, "AIR shape mismatch with verifying key");
    ensure!(proof.segments.len() == publics.len() && !publics.is_empty(), "segment count mismatch");
    let n = 1usize << vk.log_n;
    let m = n << cfg.log_blowup;
    let bound = n;
    let num_chunks = m / bound;
    let real_rows = n - cfg.random_rows();
    let num_segments = publics.len();

    let mut t = Transcript::new(b"zkb-stark-v3");
    t.absorb(b"vk", &vk.digest());
    t.absorb(b"num-segments", &(num_segments as u64).to_le_bytes());
    // phase 1: every main commitment, then the global challenges
    for (k, seg) in proof.segments.iter().enumerate() {
        ensure!(seg.public == publics[k], "segment {k}: public values mismatch");
        t.absorb_fp4(b"public", &seg.public);
        t.absorb_digest(b"main", &seg.main_root);
    }
    let challenges = t.challenge_fp4_vec(shape.num_challenges);

    let g = Fp::two_adic_root(vk.log_n);
    let excluded: Vec<Fp4> = (real_rows - 1..n).map(|i| Fp4::from_base(g.pow(i as u64))).collect();
    let g_last = excluded[0];
    let omega_m = Fp::two_adic_root(m.trailing_zeros());

    // Pass 1 (sequential): replay the transcript of every segment and record
    // its challenges. Pass 2 (parallel over segments): the actual checks,
    // which only depend on the recorded challenges.
    struct Replayed {
        alpha: Fp4,
        z: Fp4,
        d: Fp4,
        fri: FriVerifier,
        query_idx: Vec<usize>,
    }
    let mut replayed = Vec::with_capacity(num_segments);
    for (k, seg) in proof.segments.iter().enumerate() {
        let ood = &seg.ood;
        ensure!(ood.main.len() == shape.main_width && ood.main_next.len() == shape.main_width, "segment {k}: bad OOD main");
        ensure!(ood.fixed.len() == shape.fixed_width && ood.fixed_next.len() == shape.fixed_width, "segment {k}: bad OOD fixed");
        ensure!(ood.aux.len() == shape.aux_width && ood.aux_next.len() == shape.aux_width, "segment {k}: bad OOD aux");
        ensure!(ood.comp.len() == num_chunks, "segment {k}: bad OOD comp");
        ensure!(seg.openings.len() == cfg.num_queries && seg.fri.queries.len() == cfg.num_queries, "segment {k}: wrong number of queries");
        t.absorb_digest(b"aux", &seg.aux_root);
        t.absorb_fp4(b"aux-public", &seg.aux_public);
        let alpha = t.challenge_fp4();
        t.absorb_digest(b"comp", &seg.comp_root);
        let z = t.challenge_fp4();
        absorb_ood(&mut t, ood);
        let d = t.challenge_fp4();
        let fri = FriVerifier::replay(cfg.fri(), &mut t, &seg.fri, m.trailing_zeros(), LDE_SHIFT)?;
        ensure!(t.check_grind(cfg.grinding_bits, seg.grind_nonce), "segment {k}: proof-of-work check failed");
        let query_idx: Vec<usize> = (0..cfg.num_queries).map(|_| t.challenge_index(m / 2)).collect();
        replayed.push(Replayed { alpha, z, d, fri, query_idx });
    }

    use rayon::prelude::*;
    let check_segment = |k: usize| -> Result<()> {
        let seg = &proof.segments[k];
        let Replayed { alpha, z, d, fri, query_idx } = &replayed[k];
        let (alpha, z, d) = (*alpha, *z, *d);
        let gz = z.mul_base(g);
        let public = &seg.public;
        let ood = &seg.ood;

        // ---- out-of-domain constraint check
        let alphas = alpha_powers(alpha, shape.num_transition + shape.num_first + shape.num_last);
        let weights = std::sync::Arc::new(CombinationWeights::<Fp4>::new(alphas, shape.num_transition, shape.num_first));
        let mut sink: ConstraintSink<Fp4> = ConstraintSink::accumulating(SinkMode::All, weights);
        let row = RowView::new(&ood.main[..], &ood.main_next[..], &ood.fixed[..], &ood.fixed_next[..], &ood.aux[..], &ood.aux_next[..]);
        air.eval(&row, &challenges, public, &seg.aux_public, &mut sink);
        ensure!(sink.counts() == (shape.num_transition, shape.num_first, shape.num_last), "constraint count mismatch");
        let zn = z.pow(n as u64);
        let mut excl = Fp4::ONE;
        for e in &excluded {
            excl *= z - *e;
        }
        let inv_zt = excl * (zn - Fp4::ONE).inverse();
        let inv_first = (z - Fp4::ONE).inverse();
        let inv_last = (z - g_last).inverse();
        let (lo, hi) = sink.combined(inv_zt, inv_first, inv_last);
        let lhs = lo + hi;
        let zb = z.pow(bound as u64);
        let mut rhs = Fp4::ZERO;
        let mut zk_pow = Fp4::ONE;
        for c in &ood.comp {
            rhs += *c * zk_pow;
            zk_pow *= zb;
        }
        ensure!(lhs == rhs, "segment {k}: out-of-domain constraint check failed");

        // ---- queries
        let roots = [&seg.main_root, &vk.fixed_root, &seg.aux_root, &seg.comp_root];
        for (qi, &l) in query_idx.iter().enumerate() {
            let op = &seg.openings[qi];
            let hi_idx = l + m / 2;
            check_position(roots, l, m, &op.lo, cfg.zk)?;
            check_position(roots, hi_idx, m, &op.hi, cfg.zk)?;
            for p in [&op.lo, &op.hi] {
                ensure!(p.main.values.len() == shape.main_width, "bad main opening width");
                ensure!(p.fixed.values.len() >= shape.fixed_width, "bad fixed opening width");
                ensure!(p.aux.values.len() >= 4 * shape.aux_width, "bad aux opening width");
                ensure!(p.comp.values.len() == 4 * num_chunks, "bad comp opening width");
            }
            let x_lo = LDE_SHIFT * omega_m.pow(l as u64);
            let x_hi = -x_lo;
            let d_lo = deep_at_point(&shape, x_lo, &op.lo.main.values, &op.lo.fixed.values, &op.lo.aux.values, &op.lo.comp.values, ood, z, gz, d);
            let d_hi = deep_at_point(&shape, x_hi, &op.hi.main.values, &op.hi.fixed.values, &op.hi.aux.values, &op.hi.comp.values, ood, z, gz, d);
            fri.verify_query(l, (d_lo, d_hi), &seg.fri.queries[qi])?;
        }
        Ok(())
    };
    (0..num_segments).into_par_iter().map(check_segment).collect::<Result<Vec<()>>>()?;
    Ok(challenges)
}
