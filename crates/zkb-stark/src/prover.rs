//! STARK prover (multi-segment, optionally zero-knowledge).

use crate::air::{Air, CombinationWeights, ConstraintSink, RowView, SinkMode};
use crate::fri::{FriProver, FriQueryProof};
use crate::merkle::{Digest, MerkleTree};
use crate::transcript::Transcript;
use crate::{OodEvals, PositionOpening, Proof, QueryOpening, RowOpening, SegmentProof, StarkConfig, VerifyingKey, LDE_SHIFT};
use anyhow::{ensure, Result};
use rayon::prelude::*;
use zkb_field::columns::ColumnStore;
use zkb_field::ntt::{bit_reverse_index, coset_major_pos, evaluate_from_bitrev, ext_join, ext_split, lde_from_bitrev_coeffs, pad_bitrev, CosetPowers, NttPlan};
use zkb_field::simd::mul_slices_in_place;
use zkb_field::simd::{Fp16, Fp4x16, LANES};
use zkb_field::{ExtLike, FieldLike, Fp, Fp4};

fn timing() -> bool {
    std::env::var("ZKB_TIMING").is_ok()
}
macro_rules! tlog {
    ($t:expr, $($arg:tt)*) => {
        if timing() { eprintln!("  [{:>7.2?}] {}", $t.elapsed(), format!($($arg)*)); }
    };
}

/// Deterministic randomness for the zero-knowledge random rows and leaf
/// salts, derived from a seed.
#[derive(Clone)]
pub struct Blinder {
    seed: [u8; 32],
    /// Number of random rows per column.
    pub k: usize,
}

impl Blinder {
    pub fn new(seed: [u8; 32], k: usize) -> Blinder {
        Blinder { seed, k }
    }
    pub fn derive(&self, label: &[u8]) -> Blinder {
        let mut h = blake3::Hasher::new_keyed(&self.seed);
        h.update(label);
        Blinder { seed: *h.finalize().as_bytes(), k: self.k }
    }
    /// `k` random field elements for column `col` (its random rows).
    pub fn rows(&self, col: usize) -> Vec<Fp> {
        let mut h = blake3::Hasher::new_keyed(&self.seed);
        h.update(b"blind");
        h.update(&(col as u64).to_le_bytes());
        let mut xof = h.finalize_xof();
        let mut buf = vec![0u8; 8 * self.k];
        xof.fill(&mut buf);
        buf.chunks_exact(8).map(|c| Fp::new(u64::from_le_bytes(c.try_into().unwrap()))).collect()
    }
    pub fn salt(&self, leaf: usize) -> [u8; 32] {
        let mut h = blake3::Hasher::new_keyed(&self.seed);
        h.update(b"salt");
        h.update(&(leaf as u64).to_le_bytes());
        *h.finalize().as_bytes()
    }
}

impl crate::merkle::Salter for Blinder {
    fn salt(&self, leaf: usize) -> [u8; 32] {
        Blinder::salt(self, leaf)
    }
    fn salts16(&self, leaves: &[usize; 16]) -> [[u8; 32]; 16] {
        if !crate::blake3x16::AVAILABLE {
            return core::array::from_fn(|k| Blinder::salt(self, leaves[k]));
        }
        let mut lanes = crate::blake3x16::Lanes16::new(12);
        for lane in 0..16 {
            lanes.put_bytes(lane, 0, b"salt");
            lanes.put_u64(lane, 1, leaves[lane] as u64);
        }
        let mut out = [[0u8; 32]; 16];
        crate::blake3x16::hash16_keyed(&self.seed, &lanes.words, 12, &mut out);
        out
    }
}

pub fn os_seed() -> [u8; 32] {
    use std::io::Read;
    let mut seed = [0u8; 32];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        if f.read_exact(&mut seed).is_ok() {
            return seed;
        }
    }
    // fallback: time + pid (not for production)
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let mut h = blake3::Hasher::new();
    h.update(&t.to_le_bytes());
    h.update(&std::process::id().to_le_bytes());
    *h.finalize().as_bytes()
}

/// Sixteen consecutive rows (k..k+16) of base-field columns, read in place.
#[derive(Clone, Copy)]
struct ColsAt<'a> {
    cols: &'a [&'a [Fp]],
    k: usize,
}
impl crate::air::RowAccess<Fp16> for ColsAt<'_> {
    #[inline(always)]
    fn at(&self, i: usize) -> Fp16 {
        Fp16::from_slice(&self.cols[i][self.k..])
    }
}

/// Sixteen consecutive rows of extension columns stored as four base column each.
#[derive(Clone, Copy)]
struct ExtColsAt<'a> {
    cols: &'a [&'a [Fp]],
    k: usize,
}
impl crate::air::RowAccess<Fp4x16> for ExtColsAt<'_> {
    #[inline(always)]
    fn at(&self, i: usize) -> Fp4x16 {
        let k = self.k;
        Fp4x16::from_slices(&self.cols[4 * i][k..], &self.cols[4 * i + 1][k..], &self.cols[4 * i + 2][k..], &self.cols[4 * i + 3][k..])
    }
}

fn pooled_store(width: usize, len: usize) -> ColumnStore {
    ColumnStore::pooled(width, len)
}

/// A set of base-field columns committed via their LDE.
/// `lde` columns are stored coset-major (see `zkb_field::ntt::lde_coset_major`);
/// use [`Committed::at`] to read natural LDE index `i`.
/// `coeffs` are stored in *bit-reversed* order and unscaled: the true
/// coefficient of x^j is `coeffs[rev(j)] · scale`.
pub struct Committed {
    pub coeffs: ColumnStore,
    pub scale: Fp,
    pub lde: ColumnStore,
    pub tree: MerkleTree,
    blinder: Option<Blinder>,
    n: usize,
    log_blowup: u32,
}

impl Committed {
    /// `cols` are evaluations on the trace domain <ω_N>. With a blinder,
    /// the last `blinder.k` rows of every column are overwritten with random
    /// values and Merkle leaves are salted.
    pub fn new(cols: &ColumnStore, log_blowup: u32, blinder: Option<&Blinder>) -> Committed {
        let n = cols.len();
        let plan_n = NttPlan::new(n);
        let mut coeffs = pooled_store(cols.width(), n);
        coeffs.par_cols_mut().zip(cols.par_cols()).enumerate().for_each(|(ci, (c, src))| {
            c.copy_from_slice(src);
            if let Some(b) = blinder {
                let r = b.rows(ci);
                let start = n - r.len();
                c[start..].copy_from_slice(&r);
            }
            plan_n.inverse_to_bitrev_unscaled(c);
        });
        Self::from_bitrev_coeffs(coeffs, plan_n.n_inv(), log_blowup, blinder)
    }

    /// Commit from bit-reversed, unscaled coefficients (true coefficients are
    /// `coeffs[rev(j)] · scale`).
    pub fn from_bitrev_coeffs(coeffs: ColumnStore, scale: Fp, log_blowup: u32, blinder: Option<&Blinder>) -> Committed {
        let n = coeffs.len();
        let m = n << log_blowup;
        let plan_n = NttPlan::new(n);
        let t_lde = std::time::Instant::now();
        let pows = CosetPowers::get(plan_n.log_n, LDE_SHIFT, log_blowup, scale);
        let mut lde = pooled_store(coeffs.width(), m);
        lde.par_cols_mut().zip(coeffs.par_cols()).for_each(|(out, c)| lde_from_bitrev_coeffs(&plan_n, c, &pows, out));
        if timing() {
            eprintln!("      commit: {} cols, interpolate+lde {:?}", lde.width(), t_lde.elapsed());
        }
        let t_tree = std::time::Instant::now();
        let tree = MerkleTree::from_coset_major_columns(&lde.slices(), n, log_blowup, blinder.map(|b| b as &dyn crate::merkle::Salter));
        if timing() {
            eprintln!("      commit: merkle {:?}", t_tree.elapsed());
        }
        Committed { coeffs, scale, lde, tree, blinder: blinder.cloned(), n, log_blowup }
    }

    /// Value of column `col` at natural LDE index `i`.
    #[inline(always)]
    pub fn at(&self, col: usize, i: usize) -> Fp {
        self.lde.col(col)[coset_major_pos(i, self.n, self.log_blowup)]
    }

    /// Storage position of natural LDE index `i`.
    #[inline(always)]
    pub fn pos(&self, i: usize) -> usize {
        coset_major_pos(i, self.n, self.log_blowup)
    }

    pub fn root(&self) -> Digest {
        self.tree.root()
    }

    pub fn open(&self, i: usize) -> RowOpening {
        let p = self.pos(i);
        RowOpening { values: self.lde.cols().map(|c| c[p]).collect(), salt: self.blinder.as_ref().map(|b| b.salt(i)), path: self.tree.open(i) }
    }

    pub fn width(&self) -> usize {
        self.lde.width()
    }
}

pub struct ProvingKey {
    pub vk: VerifyingKey,
    pub fixed: Vec<Vec<Fp>>,
    pub fixed_c: Committed,
}

pub fn setup<A: Air>(air: &A, log_n: u32, config: StarkConfig) -> ProvingKey {
    let shape = air.shape();
    assert!(shape.max_degree <= (1usize << config.log_blowup), "constraint degree {} exceeds blowup {}", shape.max_degree, 1 << config.log_blowup);
    assert!(config.random_rows() * 4 < (1usize << log_n), "trace too small for the zero-knowledge random rows");
    let n = 1usize << log_n;
    let mut fixed = air.fixed_columns(n);
    assert_eq!(fixed.len(), shape.fixed_width);
    if fixed.is_empty() {
        fixed.push(vec![Fp::ZERO; n]);
    }
    // Fixed columns are public: never blinded.
    let fixed_c = Committed::new(&ColumnStore::from_columns(fixed.clone()), config.log_blowup, None);
    let vk = VerifyingKey {
        log_n,
        config,
        fixed_root: fixed_c.root(),
        main_width: shape.main_width,
        fixed_width: shape.fixed_width,
        aux_width: shape.aux_width,
    };
    ProvingKey { vk, fixed, fixed_c }
}

/// Prove one or more segments. `gen(k)` must return segment k's main trace.
///
/// Challenge schedule (global, two-phase): every segment's main trace is
/// committed first (phase 1); one set of LogUp challenges is then drawn for
/// the whole execution, so multiset arguments can span segments (the memory
/// carry cites entries of arbitrary earlier segments). In phase 2 each main
/// trace is re-derived (traces are deterministic, blinding is derived from
/// the master seed and the segment label, so the commitment is bit-identical
/// and checked to be) and the segment is proven with the global challenges.
pub fn prove<A: Air>(air: &A, pk: &ProvingKey, publics: &[Vec<Fp4>], gen: &(dyn Fn(usize) -> Result<ColumnStore> + Sync)) -> Result<Proof> {
    let shape = air.shape();
    let cfg = pk.vk.config;
    let log_n = pk.vk.log_n;
    let n = 1usize << log_n;
    let m = n << cfg.log_blowup;
    let bound = n;
    let num_chunks = m / bound;
    let k_rows = cfg.random_rows();
    let real_rows = n - k_rows;
    let num_segments = publics.len();
    let tstart = std::time::Instant::now();
    let master = Blinder::new(os_seed(), k_rows);
    let blind = |label: String| -> Option<Blinder> { if cfg.zk { Some(master.derive(label.as_bytes())) } else { None } };

    let mut t = Transcript::new(b"zkb-stark-v3");
    t.absorb(b"vk", &pk.vk.digest());
    t.absorb(b"num-segments", &(num_segments as u64).to_le_bytes());

    // commit segment k's main trace
    let commit_main = |k: usize| -> Result<(ColumnStore, Committed)> {
        let main = gen(k)?;
        tlog!(tstart, "segment {k}: trace generated");
        ensure!(main.width() == shape.main_width, "segment {k}: main width mismatch");
        ensure!(main.len() == n, "segment {k}: main column length mismatch");
        let main_c = Committed::new(&main, cfg.log_blowup, blind(format!("main-{k}")).as_ref());
        Ok((main, main_c))
    };

    // ---- phase 1: commit every main trace, draw the global challenges
    let mut roots = Vec::with_capacity(num_segments);
    for k in 0..num_segments {
        let (_, main_c) = commit_main(k)?;
        t.absorb_fp4(b"public", &publics[k]);
        t.absorb_digest(b"main", &main_c.root());
        roots.push(main_c.root());
        tlog!(tstart, "segment {k}: main committed (phase 1)");
    }
    let challenges = t.challenge_fp4_vec(shape.num_challenges);

    // ---- phase 2
    let mut segments = Vec::with_capacity(num_segments);
    for k in 0..num_segments {
        let (main, main_c) = commit_main(k)?;
        ensure!(main_c.root() == roots[k], "segment {k}: main trace is not deterministic across phases");
        tlog!(tstart, "segment {k}: main committed");
        let public = &publics[k];

        // aux (sums stop at the last real row; random rows are overwritten by Committed::new)
        let fixed_slices: Vec<&[Fp]> = pk.fixed[..shape.fixed_width].iter().map(|v| v.as_slice()).collect();
        let (aux, aux_public) = air.generate_aux(&main.slices(), &fixed_slices, &challenges, real_rows);
        drop(main);
        ensure!(aux.len() == shape.aux_width, "aux width mismatch");
        let aux_base_w = if aux.is_empty() { 4 } else { 4 * aux.len() };
        let mut aux_base = ColumnStore::zeros(aux_base_w, n);
        for (ci, c) in aux.iter().enumerate() {
            let parts = ext_split(c);
            for (q, part) in parts.iter().enumerate() {
                aux_base.col_mut(4 * ci + q).copy_from_slice(part);
            }
        }
        drop(aux);
        tlog!(tstart, "segment {k}: aux generated");
        let aux_c = Committed::new(&aux_base, cfg.log_blowup, blind(format!("aux-{k}")).as_ref());
        drop(aux_base);
        t.absorb_digest(b"aux", &aux_c.root());
        t.absorb_fp4(b"aux-public", &aux_public);
        tlog!(tstart, "segment {k}: aux committed");

        // composition
        let alpha = t.challenge_fp4();
        let comp_evals = compute_composition(air, &shape, &main_c, &pk.fixed_c, &aux_c, &challenges, public, &aux_public, alpha, n, m, real_rows);
        tlog!(tstart, "segment {k}: composition evaluated");
        let plan_m = NttPlan::new(m);
        // Interpolate on the shifted coset: bit-reversed, unscaled coefficients.
        let mut parts = ext_split(&comp_evals);
        drop(comp_evals);
        parts.par_iter_mut().for_each(|p| plan_m.inverse_to_bitrev_unscaled(p));
        // True coefficient j = raw[rev_m(j)] · m_inv · shift^{-j}.
        #[cfg(debug_assertions)]
        let log_m = plan_m.log_n;
        let log_bound = bound.trailing_zeros();
        let shift_inv = LDE_SHIFT.inverse();
        #[cfg(debug_assertions)]
        {
            for i in 0..m {
                let j = bit_reverse_index(i, log_m);
                if j >= shape.max_degree * (bound - 1) {
                    assert!(parts.iter().all(|p| p[i].is_zero()), "composition coefficient {j} nonzero: constraints violated or degree bound wrong");
                }
            }
        }
        // Chunk c holds natural coefficients [c·bound, (c+1)·bound). In the
        // bit-reversed size-m vector, natural index j = c·bound + r sits at
        // position rev_bound(r)·num_chunks + rev_nc(c).
        let log_nc = num_chunks.trailing_zeros();
        let mut shift_pow_br = vec![Fp::ZERO; bound];
        {
            let mut sp = Fp::ONE;
            for r in 0..bound {
                shift_pow_br[bit_reverse_index(r, log_bound)] = sp;
                sp *= shift_inv;
            }
        }
        let shift_bound_inv = shift_inv.pow(bound as u64);
        let mut chunk_cols = pooled_store(4 * num_chunks, bound);
        let mut kc = plan_m.n_inv();
        for c in 0..num_chunks {
            let base = bit_reverse_index(c, log_nc);
            for (part, src) in parts.iter().enumerate() {
                let v = chunk_cols.col_mut(4 * c + part);
                for (rp, x) in v.iter_mut().enumerate() {
                    *x = src[rp * num_chunks + base];
                }
                mul_slices_in_place(v, &shift_pow_br);
                zkb_field::simd::scale_slice(v, kc);
            }
            kc *= shift_bound_inv;
        }
        drop(parts);
        let comp_c = Committed::from_bitrev_coeffs(chunk_cols, Fp::ONE, (m / bound).trailing_zeros(), None);
        t.absorb_digest(b"comp", &comp_c.root());
        tlog!(tstart, "segment {k}: composition committed");

        // OOD
        let z = t.challenge_fp4();
        let g = Fp::two_adic_root(log_n);
        let gz = z.mul_base(g);
        // powers of z and gz once, in bit-reversed order to match the stored
        // coefficients, as SoA (four coefficient vectors) for the vectorized dot product
        let powers = |p: Fp4| -> [Vec<Fp>; 4] {
            let mut t: [Vec<Fp>; 4] = core::array::from_fn(|_| vec![Fp::ZERO; n]);
            let mut acc = Fp4::ONE;
            for i in 0..n {
                let j = bit_reverse_index(i, log_n);
                for q in 0..4 {
                    t[q][j] = acc.0[q];
                }
                acc *= p;
            }
            t
        };
        let pz = powers(z);
        let pgz = powers(gz);
        let dot = |co: &[Fp], pw: &[Vec<Fp>; 4], scale: Fp| -> Fp4 {
            let mut acc = Fp4x16::ZERO;
            let mut k = 0;
            while k + LANES <= co.len() {
                acc += Fp4x16::from_slices(&pw[0][k..], &pw[1][k..], &pw[2][k..], &pw[3][k..]).mul_base(Fp16::from_slice(&co[k..]));
                k += LANES;
            }
            let mut s = Fp4::ZERO;
            for l in 0..LANES {
                s += acc.lane(l);
            }
            while k < co.len() {
                s += Fp4([pw[0][k], pw[1][k], pw[2][k], pw[3][k]]).mul_base(co[k]);
                k += 1;
            }
            s.mul_base(scale)
        };
        let eval_base_cols = |c: &Committed, p: Fp4| -> Vec<Fp4> {
            let pw = if p == z { &pz } else { &pgz };
            c.coeffs.par_cols().map(|co| dot(co, pw, c.scale)).collect()
        };
        // an extension column is four base columns: value = Σ_q u^q · col_q
        let eval_ext_cols = |c: &Committed, p: Fp4| -> Vec<Fp4> {
            let pw = if p == z { &pz } else { &pgz };
            (0..c.coeffs.width() / 4)
                .into_par_iter()
                .map(|j| {
                    let mut v = Fp4::ZERO;
                    for q in 0..4 {
                        let d = dot(c.coeffs.col(4 * j + q), pw, c.scale);
                        v += Fp4::U.pow(q as u64) * d;
                    }
                    v
                })
                .collect()
        };
        let ood = OodEvals {
            main: eval_base_cols(&main_c, z),
            main_next: eval_base_cols(&main_c, gz),
            fixed: eval_base_cols(&pk.fixed_c, z)[..shape.fixed_width].to_vec(),
            fixed_next: eval_base_cols(&pk.fixed_c, gz)[..shape.fixed_width].to_vec(),
            aux: if shape.aux_width > 0 { eval_ext_cols(&aux_c, z) } else { vec![] },
            aux_next: if shape.aux_width > 0 { eval_ext_cols(&aux_c, gz) } else { vec![] },
            comp: eval_ext_cols(&comp_c, z),
        };
        absorb_ood(&mut t, &ood);
        tlog!(tstart, "segment {k}: ood evaluated");

        // DEEP + FRI
        let d = t.challenge_fp4();
        let deep = compute_deep(&shape, &main_c, &pk.fixed_c, &aux_c, &comp_c, &ood, z, gz, d, m);
        tlog!(tstart, "segment {k}: deep computed");
        let fri = FriProver::commit(cfg.fri(), &mut t, deep, LDE_SHIFT);
        let grind_nonce = t.grind(cfg.grinding_bits);
        let mut openings = Vec::with_capacity(cfg.num_queries);
        let mut fri_queries: Vec<FriQueryProof> = Vec::with_capacity(cfg.num_queries);
        for _ in 0..cfg.num_queries {
            let l = t.challenge_index(m / 2);
            let open_pos = |i: usize| PositionOpening { main: main_c.open(i), fixed: pk.fixed_c.open(i), aux: aux_c.open(i), comp: comp_c.open(i) };
            openings.push(QueryOpening { lo: open_pos(l), hi: open_pos(l + m / 2) });
            fri_queries.push(fri.open(l));
        }
        let fri = fri.into_proof(fri_queries);
        tlog!(tstart, "segment {k}: done");
        segments.push(SegmentProof { public: public.clone(), grind_nonce, main_root: main_c.root(), aux_root: aux_c.root(), comp_root: comp_c.root(), aux_public, ood, fri, openings });
    }
    Ok(Proof { log_n, segments })
}

pub(crate) fn absorb_ood(t: &mut Transcript, ood: &OodEvals) {
    t.absorb_fp4(b"ood-main", &ood.main);
    t.absorb_fp4(b"ood-main-next", &ood.main_next);
    t.absorb_fp4(b"ood-fixed", &ood.fixed);
    t.absorb_fp4(b"ood-fixed-next", &ood.fixed_next);
    t.absorb_fp4(b"ood-aux", &ood.aux);
    t.absorb_fp4(b"ood-aux-next", &ood.aux_next);
    t.absorb_fp4(b"ood-comp", &ood.comp);
}

/// Powers of α, one per constraint, in sink order (transition, first, last).
pub(crate) fn alpha_powers(alpha: Fp4, count: usize) -> Vec<Fp4> {
    let mut v = Vec::with_capacity(count);
    let mut a = Fp4::ONE;
    for _ in 0..count {
        v.push(a);
        a *= alpha;
    }
    v
}

/// Per-(n, m, real_rows) vanishing-polynomial inverse tables in storage
/// (coset-major) order, cached across segments.
struct CompositionDomain {
    inv_zt: Vec<Fp>,
    inv_first: Vec<Fp>,
    inv_last: Vec<Fp>,
}

impl CompositionDomain {
    fn compute(n: usize, m: usize, real_rows: usize) -> CompositionDomain {
        let b = m / n;
        let g = Fp::two_adic_root(n.trailing_zeros());
        let omega_m = Fp::two_adic_root(m.trailing_zeros());
        // x in *storage* order: position p = c*n + j  ->  x = s · ω_M^(j*b + c)
        let omega_n = omega_m.pow(b as u64);
        let mut xs = vec![Fp::ZERO; m];
        xs.par_chunks_mut(n).enumerate().for_each(|(c, chunk)| {
            let mut x = LDE_SHIFT * omega_m.pow(c as u64);
            for v in chunk.iter_mut() {
                *v = x;
                x *= omega_n;
            }
        });
        // Transition constraints hold on rows 0..real_rows-1 (so that row+1 is still real):
        // Z_T(x) = (x^N - 1) / Π_{i=real_rows-1}^{N-1} (x - g^i).  Last-row constraints at row real_rows-1.
        let excluded: Vec<Fp> = (real_rows - 1..n).map(|i| g.pow(i as u64)).collect();
        let g_last = excluded[0];
        let par_inv = |mut v: Vec<Fp>| -> Vec<Fp> {
            v.par_chunks_mut(8192).for_each(Fp::batch_inverse);
            v
        };
        let inv_first = par_inv(xs.par_iter().map(|&x| x - Fp::ONE).collect());
        let inv_last = par_inv(xs.par_iter().map(|&x| x - g_last).collect());
        // 1/Z_T = Π (x - e) / (x^N - 1); x^N is constant on each coset: (s·ω_M^c)^N
        let inv_xn_minus_1: Vec<Fp> = (0..b).map(|c| ((LDE_SHIFT * omega_m.pow(c as u64)).pow(n as u64) - Fp::ONE).inverse()).collect();
        let inv_zt: Vec<Fp> = (0..m)
            .into_par_iter()
            .map(|p| {
                let mut v = inv_xn_minus_1[p / n];
                for e in &excluded {
                    v *= xs[p] - *e;
                }
                v
            })
            .collect();
        CompositionDomain { inv_zt, inv_first, inv_last }
    }

    fn get(n: usize, m: usize, real_rows: usize) -> std::sync::Arc<CompositionDomain> {
        use std::collections::HashMap;
        use std::sync::{Arc, Mutex, OnceLock};
        static CACHE: OnceLock<Mutex<HashMap<(usize, usize, usize), Arc<CompositionDomain>>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(d) = cache.lock().unwrap().get(&(n, m, real_rows)) {
            return d.clone();
        }
        let d = Arc::new(CompositionDomain::compute(n, m, real_rows));
        let mut c = cache.lock().unwrap();
        if c.len() >= 4 {
            c.clear();
        }
        c.insert((n, m, real_rows), d.clone());
        d
    }
}

#[allow(clippy::too_many_arguments)]
fn compute_composition<A: Air>(
    air: &A,
    shape: &crate::AirShape,
    main_c: &Committed,
    fixed_c: &Committed,
    aux_c: &Committed,
    challenges: &[Fp4],
    public: &[Fp4],
    aux_public: &[Fp4],
    alpha: Fp4,
    n: usize,
    m: usize,
    real_rows: usize,
) -> Vec<Fp4> {
    let b = m / n;
    let dom = CompositionDomain::get(n, m, real_rows);
    let inv_zt = &dom.inv_zt;
    let inv_first = &dom.inv_first;
    let inv_last = &dom.inv_last;

    let fixed_w = shape.fixed_width;
    let alphas = alpha_powers(alpha, shape.num_transition + shape.num_first + shape.num_last);
    let weights = std::sync::Arc::new(CombinationWeights::<Fp16>::new(alphas, shape.num_transition, shape.num_first));
    // Work in storage (coset-major) order: position p = c*n + j holds LDE index i = j*b + c,
    // and its "next row" i + b sits at p + 1 (wrapping inside the coset). Each
    // block first copies its slice of every column into a contiguous buffer
    // (sequential reads, prefetch-friendly), then evaluates eight rows per
    // call with the vector field type.
    //
    // Degree-aware split: constraints of degree <= 2 have quotients of degree
    // < 2N, so they are evaluated only on the even LDE indices (cosets with even
    // c, a coset of the order-2N subgroup) and extended by one NTT; high-degree
    // constraints are evaluated everywhere.
    let chunk = 256.min(n);
    assert!(n % chunk == 0 && chunk % LANES == 0);
    let main_w = main_c.width();
    let aux_base_w = aux_c.lde.width();
    let mut out_high = vec![Fp4::ZERO; m];
    let mut out_low = vec![Fp4::ZERO; m / 2];
    let low_chunks: Vec<&mut [Fp4]> = out_low.chunks_mut(chunk).collect();
    let blocks: Vec<(usize, &mut [Fp4], Option<&mut [Fp4]>)> = {
        let mut v = Vec::new();
        let mut low_iter = low_chunks.into_iter();
        for (ci, hc) in out_high.chunks_mut(chunk).enumerate() {
            let c = ci * chunk / n;
            let low = if c % 2 == 0 { low_iter.next() } else { None };
            v.push((ci, hc, low));
        }
        v
    };
    let t_eval = std::time::Instant::now();
    blocks.into_par_iter().for_each(|(ci, high_chunk, low_chunk)| {
        let p0 = ci * chunk;
        let c = p0 / n;
        let stride = chunk + LANES;
        // The successor of the last row of a coset wraps to its first row, so
        // the last block of every coset is staged into a buffer; every other
        // block reads the LDE columns in place.
        let wraps = p0 + chunk == (c + 1) * n;
        let fill = |cols: &ColumnStore, width: usize, buf: &mut Vec<Fp>| {
            buf.clear();
            buf.resize(width * stride, Fp::ZERO);
            for (col, data) in cols.cols().take(width).enumerate() {
                let dst = &mut buf[col * stride..(col + 1) * stride];
                dst[..chunk].copy_from_slice(&data[p0..p0 + chunk]);
                dst[chunk] = data[c * n];
            }
        };
        let mut bm = Vec::new();
        let mut bf = Vec::new();
        let mut ba = Vec::new();
        let (sm, sf, sa): (Vec<&[Fp]>, Vec<&[Fp]>, Vec<&[Fp]>) = if wraps {
            fill(&main_c.lde, main_w, &mut bm);
            fill(&fixed_c.lde, fixed_w, &mut bf);
            fill(&aux_c.lde, aux_base_w, &mut ba);
            (bm.chunks(stride).collect(), bf.chunks(stride).collect(), ba.chunks(stride).collect())
        } else {
            fn direct<'a>(cols: &'a ColumnStore, width: usize, p0: usize, chunk: usize) -> Vec<&'a [Fp]> {
                cols.cols().take(width).map(|d| &d[p0..p0 + chunk + 1]).collect()
            }
            (direct(&main_c.lde, main_w, p0, chunk), direct(&fixed_c.lde, fixed_w, p0, chunk), direct(&aux_c.lde, aux_base_w, p0, chunk))
        };
        let mode = if low_chunk.is_some() { SinkMode::All } else { SinkMode::HighOnly };
        let mut sink: ConstraintSink<Fp16> = ConstraintSink::accumulating(mode, weights.clone());
        let mut low_chunk = low_chunk;
        let mut k = 0;
        let tm = timing();
        let (mut t_eval2, mut t_comb) = (0u64, 0u64);
        while k < chunk {
            let p = p0 + k;
            // Each column stream advances one cache line per iteration, and
            // ~300 streams are beyond the hardware prefetcher: prefetch the
            // line two iterations ahead.
            #[cfg(target_arch = "x86_64")]
            {
                use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                let ahead = k + 2 * LANES;
                if ahead + LANES <= chunk + 1 {
                    for s in sm.iter().chain(sf.iter()).chain(sa.iter()) {
                        unsafe { _mm_prefetch::<_MM_HINT_T0>(s.as_ptr().add(ahead) as *const i8) };
                    }
                }
            }
            let c1 = if tm { unsafe { core::arch::x86_64::_rdtsc() } } else { 0 };
            sink.clear();
            let row = RowView::new(ColsAt { cols: &sm, k }, ColsAt { cols: &sm, k: k + 1 }, ColsAt { cols: &sf, k }, ColsAt { cols: &sf, k: k + 1 }, ExtColsAt { cols: &sa, k }, ExtColsAt { cols: &sa, k: k + 1 });
            air.eval(&row, challenges, public, aux_public, &mut sink);
            debug_assert_eq!(sink.counts(), (shape.num_transition, shape.num_first, shape.num_last));
            let c2 = if tm { unsafe { core::arch::x86_64::_rdtsc() } } else { 0 };
            let (lo, hi) = sink.combined(Fp16::from_slice(&inv_zt[p..]), Fp16::from_slice(&inv_first[p..]), Fp16::from_slice(&inv_last[p..]));
            for l in 0..LANES {
                high_chunk[k + l] = hi.lane(l);
            }
            if let Some(lc) = low_chunk.as_mut() {
                for l in 0..LANES {
                    lc[k + l] = lo.lane(l);
                }
            }
            let c3 = if tm { unsafe { core::arch::x86_64::_rdtsc() } } else { 0 };
            t_eval2 += c2 - c1;
            t_comb += c3 - c2;
            k += LANES;
        }
        if tm && ci % 512 == 0 {
            eprintln!("      composition block {ci} ({:?}): per-iteration cycles eval {} combine {}", mode, t_eval2 / (chunk / LANES) as u64, t_comb / (chunk / LANES) as u64);
        }
    });
    if timing() {
        eprintln!("      composition: constraint evaluation {:?}", t_eval.elapsed());
    }
    let t_rest = std::time::Instant::now();
    // storage order -> natural order
    let mut high = vec![Fp4::ZERO; m];
    for p in 0..m {
        let c = p / n;
        let j = p % n;
        high[j * b + c] = out_high[p];
    }
    drop(out_high);
    // low part: storage positions of even cosets, in order (c=0 then c=2, ...): natural index i/2 = j*(b/2) + c/2
    let mut low_nat = vec![Fp4::ZERO; m / 2];
    let half_b = b / 2;
    for (q, v) in out_low.iter().enumerate() {
        let c = 2 * (q / n);
        let j = q % n;
        low_nat[j * half_b + c / 2] = *v;
    }
    drop(out_low);
    // Interpolate on the half coset and re-evaluate on the full one. The
    // coset shift cancels between the two, so this is a plain DIF (size m/2),
    // zero-padding in bit-reversed order, and a DIT (size m) scaled by 2/m.
    let plan_half = NttPlan::new(m / 2);
    let plan_m = NttPlan::new(m);
    let pows = CosetPowers::get(plan_m.log_n, Fp::ONE, 0, plan_half.n_inv());
    let mut parts = ext_split(&low_nat);
    drop(low_nat);
    let full: Vec<Vec<Fp>> = parts
        .par_iter_mut()
        .map(|p| {
            plan_half.inverse_to_bitrev_unscaled(p);
            evaluate_from_bitrev(&plan_m, &pad_bitrev(p, 1), &pows)
        })
        .collect();
    let low_full = ext_join(&[full[0].clone(), full[1].clone(), full[2].clone(), full[3].clone()]);
    if timing() {
        eprintln!("      composition: low-degree extension + reorders {:?}", t_rest.elapsed());
    }
    high.par_iter_mut().zip(low_full.par_iter()).for_each(|(h, l)| *h += *l);
    high
}

/// DEEP weights, in the order [main z..][fixed z..][aux z..][comp z..][main gz..][fixed gz..][aux gz..].
fn deep_weights(shape: &crate::AirShape, main_w: usize, num_comp: usize, d: Fp4) -> Vec<Fp4> {
    let total = 2 * (main_w + shape.fixed_width + shape.aux_width) + num_comp;
    let mut weights = Vec::with_capacity(total);
    let mut w = Fp4::ONE;
    for _ in 0..total {
        weights.push(w);
        w *= d;
    }
    weights
}

/// Constant parts of the DEEP numerators: Σ w·f(z) and Σ w·f(gz).
fn deep_constants(shape: &crate::AirShape, ood: &OodEvals, weights: &[Fp4]) -> (Fp4, Fp4) {
    let mut wi = 0;
    let mut kz = Fp4::ZERO;
    for v in ood.main.iter().chain(ood.fixed.iter()).chain(ood.aux.iter()).chain(ood.comp.iter()) {
        kz += weights[wi] * *v;
        wi += 1;
    }
    let mut kgz = Fp4::ZERO;
    for v in ood.main_next.iter().chain(ood.fixed_next.iter()).chain(ood.aux_next.iter()) {
        kgz += weights[wi] * *v;
        wi += 1;
    }
    debug_assert_eq!(wi, weights.len());
    let _ = shape;
    (kz, kgz)
}

#[allow(clippy::too_many_arguments)]
fn compute_deep(
    shape: &crate::AirShape,
    main_c: &Committed,
    fixed_c: &Committed,
    aux_c: &Committed,
    comp_c: &Committed,
    ood: &OodEvals,
    z: Fp4,
    gz: Fp4,
    d: Fp4,
    m: usize,
) -> Vec<Fp4> {
    // D(x) = (A(x) - K_z)/(x - z) + (B(x) - K_gz)/(x - gz) with
    // A = Σ w_c f_c, B = Σ w'_c f_c formed in coefficient space (one NTT each
    // instead of a 344-term sum at every LDE point).
    let fixed_w = shape.fixed_width;
    let aux_w = shape.aux_width;
    let num_comp = comp_c.width() / 4;
    let main_w = main_c.width();
    let weights = deep_weights(shape, main_w, num_comp, d);
    let (kz, kgz) = deep_constants(shape, ood, &weights);
    let n = main_c.coeffs.len();
    let w_main_z = &weights[..main_w];
    let w_fixed_z = &weights[main_w..main_w + fixed_w];
    let w_aux_z = &weights[main_w + fixed_w..main_w + fixed_w + aux_w];
    let w_comp = &weights[main_w + fixed_w + aux_w..main_w + fixed_w + aux_w + num_comp];
    let off = main_w + fixed_w + aux_w + num_comp;
    let w_main_gz = &weights[off..off + main_w];
    let w_fixed_gz = &weights[off + main_w..off + main_w + fixed_w];
    let w_aux_gz = &weights[off + main_w + fixed_w..];
    // Each (z-weight, gz-weight, base coefficient column) triple; an extension
    // column contributes (w·u^q, col_q) for its four base columns. Coefficients
    // are bit-reversed and unscaled, so each set's scale is folded into its
    // weights. The composition columns only enter A (the z part).
    let uq: [Fp4; 4] = core::array::from_fn(|q| Fp4::U.pow(q as u64));
    let mut terms: Vec<(Fp4, Fp4, &[Fp])> = Vec::new();
    for c in 0..main_w {
        terms.push((w_main_z[c].mul_base(main_c.scale), w_main_gz[c].mul_base(main_c.scale), main_c.coeffs.col(c)));
    }
    for c in 0..fixed_w {
        terms.push((w_fixed_z[c].mul_base(fixed_c.scale), w_fixed_gz[c].mul_base(fixed_c.scale), fixed_c.coeffs.col(c)));
    }
    for c in 0..aux_w {
        for q in 0..4 {
            terms.push(((w_aux_z[c] * uq[q]).mul_base(aux_c.scale), (w_aux_gz[c] * uq[q]).mul_base(aux_c.scale), aux_c.coeffs.col(4 * c + q)));
        }
    }
    for c in 0..num_comp {
        for q in 0..4 {
            terms.push(((w_comp[c] * uq[q]).mul_base(comp_c.scale), Fp4::ZERO, comp_c.coeffs.col(4 * c + q)));
        }
    }
    let n_shared = terms.len() - 4 * num_comp;
    // Σ w·col in (bit-reversed) coefficient space, 16 coefficients at a time,
    // both combinations in one pass over the columns.
    let chunk = 1024;
    debug_assert_eq!(n % LANES, 0);
    let mut za: [Vec<Fp>; 4] = core::array::from_fn(|_| vec![Fp::ZERO; n]);
    let mut ga: [Vec<Fp>; 4] = core::array::from_fn(|_| vec![Fp::ZERO; n]);
    {
        let [za0, za1, za2, za3] = &mut za;
        let [ga0, ga1, ga2, ga3] = &mut ga;
        za0.par_chunks_mut(chunk)
            .zip(za1.par_chunks_mut(chunk))
            .zip(za2.par_chunks_mut(chunk).zip(za3.par_chunks_mut(chunk)))
            .zip(ga0.par_chunks_mut(chunk).zip(ga1.par_chunks_mut(chunk)).zip(ga2.par_chunks_mut(chunk).zip(ga3.par_chunks_mut(chunk))))
            .enumerate()
            .for_each(|(ci, (((oz0, oz1), (oz2, oz3)), ((og0, og1), (og2, og3))))| {
                let lo = ci * chunk;
                let len = oz0.len();
                let mut k = 0;
                while k < len {
                    let mut az = Fp4x16::ZERO;
                    let mut ag = Fp4x16::ZERO;
                    for (wz, wg, col) in &terms[..n_shared] {
                        let v = Fp16::from_slice(&col[lo + k..lo + k + LANES]);
                        az += v.mul_ext(*wz);
                        ag += v.mul_ext(*wg);
                    }
                    for (wz, _, col) in &terms[n_shared..] {
                        let v = Fp16::from_slice(&col[lo + k..lo + k + LANES]);
                        az += v.mul_ext(*wz);
                    }
                    az.c[0].store_to(&mut oz0[k..]);
                    az.c[1].store_to(&mut oz1[k..]);
                    az.c[2].store_to(&mut oz2[k..]);
                    az.c[3].store_to(&mut oz3[k..]);
                    ag.c[0].store_to(&mut og0[k..]);
                    ag.c[1].store_to(&mut og1[k..]);
                    ag.c[2].store_to(&mut og2[k..]);
                    ag.c[3].store_to(&mut og3[k..]);
                    k += LANES;
                }
            });
    }
    let plan_m = NttPlan::new(m);
    let log_blowup = (m / n).trailing_zeros();
    let pows = CosetPowers::get(plan_m.log_n, LDE_SHIFT, 0, Fp::ONE);
    let eval = |parts: [Vec<Fp>; 4]| -> Vec<Fp4> {
        let ev: Vec<Vec<Fp>> = parts.par_iter().map(|p| evaluate_from_bitrev(&plan_m, &pad_bitrev(p, log_blowup), &pows)).collect();
        ext_join(&[ev[0].clone(), ev[1].clone(), ev[2].clone(), ev[3].clone()])
    };
    let a = eval(za);
    let bvals = eval(ga);
    // (A - K_z)/(x - z) + (B - K_gz)/(x - gz), 16 points per vector, with the
    // denominators batch-inverted per chunk (one Fp4 inversion per 64 vectors).
    let omega_m = Fp::two_adic_root(m.trailing_zeros());
    let mut out = vec![Fp4::ZERO; m];
    let kz16 = Fp4x16::splat(kz);
    let kgz16 = Fp4x16::splat(kgz);
    let z16 = Fp4x16::splat(z);
    let gz16 = Fp4x16::splat(gz);
    out.par_chunks_mut(chunk).enumerate().for_each(|(ci, o)| {
        let lo = ci * chunk;
        let len = o.len();
        let nv = len / LANES;
        // x values of this chunk in natural order: LDE_SHIFT·ω^(lo+i)
        let mut xs = Vec::with_capacity(len);
        let mut x = LDE_SHIFT * omega_m.pow(lo as u64);
        for _ in 0..len {
            xs.push(x);
            x *= omega_m;
        }
        let mut dz: Vec<Fp4x16> = Vec::with_capacity(nv);
        let mut dgz: Vec<Fp4x16> = Vec::with_capacity(nv);
        for v in 0..nv {
            let xv = Fp16::from_slice(&xs[v * LANES..]);
            let mut a_ = Fp4x16::ZERO;
            a_.c[0] = xv;
            dz.push(a_ - z16);
            dgz.push(a_ - gz16);
        }
        for d in [&mut dz, &mut dgz] {
            if !Fp4x16::batch_inverse(d) {
                // z or gz landed on the LDE domain (probability ~2^-93): lane-wise fallback
                for v in d.iter_mut() {
                    *v = v.inverse();
                }
            }
        }
        for v in 0..nv {
            let i = lo + v * LANES;
            let av = Fp4x16::from_lanes(core::array::from_fn(|k| a[i + k]));
            let bv = Fp4x16::from_lanes(core::array::from_fn(|k| bvals[i + k]));
            let r = (av - kz16) * dz[v] + (bv - kgz16) * dgz[v];
            let rl = r.to_lanes();
            o[v * LANES..(v + 1) * LANES].copy_from_slice(&rl);
        }
    });
    out
}

/// Evaluate the DEEP quotient at one LDE point from opened values (shared with the verifier).
#[allow(clippy::too_many_arguments)]
pub(crate) fn deep_at_point(
    shape: &crate::AirShape,
    x: Fp,
    main: &[Fp],
    fixed: &[Fp],
    aux: &[Fp],
    comp: &[Fp],
    ood: &OodEvals,
    z: Fp4,
    gz: Fp4,
    d: Fp4,
) -> Fp4 {
    let xe = Fp4::from_base(x);
    let num_comp = comp.len() / 4;
    let weights = deep_weights(shape, main.len(), num_comp, d);
    let (kz, kgz) = deep_constants(shape, ood, &weights);
    let mut wi = 0;
    let mut sz = Fp4::ZERO;
    for v in main {
        sz += weights[wi].mul_base(*v);
        wi += 1;
    }
    for c in 0..shape.fixed_width {
        sz += weights[wi].mul_base(fixed[c]);
        wi += 1;
    }
    for c in 0..shape.aux_width {
        sz += weights[wi] * Fp4([aux[4 * c], aux[4 * c + 1], aux[4 * c + 2], aux[4 * c + 3]]);
        wi += 1;
    }
    for c in 0..num_comp {
        sz += weights[wi] * Fp4([comp[4 * c], comp[4 * c + 1], comp[4 * c + 2], comp[4 * c + 3]]);
        wi += 1;
    }
    let mut sgz = Fp4::ZERO;
    for v in main {
        sgz += weights[wi].mul_base(*v);
        wi += 1;
    }
    for c in 0..shape.fixed_width {
        sgz += weights[wi].mul_base(fixed[c]);
        wi += 1;
    }
    for c in 0..shape.aux_width {
        sgz += weights[wi] * Fp4([aux[4 * c], aux[4 * c + 1], aux[4 * c + 2], aux[4 * c + 3]]);
        wi += 1;
    }
    (sz - kz) * (xe - z).inverse() + (sgz - kgz) * (xe - gz).inverse()
}
