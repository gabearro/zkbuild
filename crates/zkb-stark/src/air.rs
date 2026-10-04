//! The generic AIR interface the prover and verifier are written against.
//!
//! An AIR has three kinds of columns:
//!  * `fixed`  — preprocessed columns that depend only on the trace length
//!    (lookup tables, selectors). Their commitment is part of the
//!    verifying key.
//!  * `main`   — base-field witness columns, committed first.
//!  * `aux`    — extension-field columns computed *after* the verifier's
//!    random challenges are known (LogUp running sums etc.).
//!
//! Constraints are polynomial identities over one row and the next row.
//! `transition` constraints hold on every row except the last, `first`
//! constraints hold on row 0, `last` constraints hold on row N-1.
//!
//! `eval` is generic over the field the main/fixed values live in: the
//! prover evaluates on base-field LDE rows, the verifier at an extension
//! field point. Challenges and aux values are always `Fp4`.

use std::sync::Arc;
use zkb_field::{ExtLike, FieldLike, Fp, Fp4, WeightedAcc};

/// Read access to the values of one (vector of) row(s): `at(i)` is column i.
/// Slices implement it; the prover uses strided views straight over its
/// column storage so no per-row gather buffer is needed.
pub trait RowAccess<F>: Copy {
    fn at(&self, i: usize) -> F;
}

impl<F: Copy> RowAccess<F> for &[F] {
    #[inline(always)]
    fn at(&self, i: usize) -> F {
        self[i]
    }
}

pub struct RowView<F: FieldLike, M: RowAccess<F>, A: RowAccess<F::Ext>> {
    pub main: M,
    pub main_next: M,
    pub fixed: M,
    pub fixed_next: M,
    pub aux: A,
    pub aux_next: A,
    pub _f: core::marker::PhantomData<F>,
}

impl<F: FieldLike, M: RowAccess<F>, A: RowAccess<F::Ext>> RowView<F, M, A> {
    pub fn new(main: M, main_next: M, fixed: M, fixed_next: M, aux: A, aux_next: A) -> Self {
        RowView { main, main_next, fixed, fixed_next, aux, aux_next, _f: core::marker::PhantomData }
    }
}

/// Row view over plain slices.
pub type SliceRow<'a, F> = RowView<F, &'a [F], &'a [<F as FieldLike>::Ext]>;

/// Which constraints a sink evaluates.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SinkMode {
    /// Every constraint (verifier, debug checks, and the prover's "even" points).
    All,
    /// Only constraints marked high-degree (the prover's extra points).
    HighOnly,
}

/// Constraint values of one row. Constraints are given as closures so a
/// sink in `HighOnly` mode can skip evaluating low-degree ones entirely.
/// Values are kept as `(ext, base)` pairs so the prover (base-field rows)
/// can combine them with α powers using cheap `Fp4 × Fp` products.
///
/// A constraint is *low-degree* if its total degree is ≤ 2 (its quotient
/// then has degree < 2N and can be evaluated on half the LDE points);
/// authors mark the others with the `*_hi` variants. `check_degrees` in
/// `zkb-stark` verifies the marking.
pub struct ConstraintSink<F: FieldLike> {
    pub mode: SinkMode,
    /// Store individual constraint values (diagnostics, degree checks).
    pub collect: bool,
    /// Random-linear-combination weights (α powers), indexed by global
    /// constraint index: transition i → i, first i → nt + i, last i → nt + nf + i.
    /// When set, every constraint is folded into `acc` as it is declared.
    weights: Option<Arc<CombinationWeights<F>>>,
    /// [transition lo, transition hi, first lo, first hi, last lo, last hi]
    acc: [F::Acc; 6],
    pub transition: Vec<F::Ext>,
    pub first: Vec<F::Ext>,
    pub last: Vec<F::Ext>,
    pub transition_base: Vec<Option<F>>,
    pub first_base: Vec<Option<F>>,
    pub last_base: Vec<Option<F>>,
    /// Global constraint index (α power) of each stored value.
    pub transition_idx: Vec<u32>,
    pub first_idx: Vec<u32>,
    pub last_idx: Vec<u32>,
    /// Whether each stored value is a high-degree constraint.
    pub transition_hi: Vec<bool>,
    pub first_hi: Vec<bool>,
    pub last_hi: Vec<bool>,
    nt: u32,
    nf: u32,
    nl: u32,
}

impl<F: FieldLike> Default for ConstraintSink<F> {
    fn default() -> Self {
        ConstraintSink::new(SinkMode::All)
    }
}

/// α powers in both plain and precomputed (`F::Weight`) form.
pub struct CombinationWeights<F: FieldLike> {
    pub alphas: Vec<Fp4>,
    pub weights: Vec<F::Weight>,
    pub nt: usize,
    pub nf: usize,
}

impl<F: FieldLike> CombinationWeights<F> {
    pub fn new(alphas: Vec<Fp4>, nt: usize, nf: usize) -> CombinationWeights<F> {
        let weights = alphas.iter().map(|a| F::weight(*a)).collect();
        CombinationWeights { alphas, weights, nt, nf }
    }
}

impl<F: FieldLike> ConstraintSink<F> {
    pub fn new(mode: SinkMode) -> ConstraintSink<F> {
        ConstraintSink {
            mode,
            collect: true,
            weights: None,
            acc: [F::Acc::zero(); 6],
            transition: Vec::new(),
            first: Vec::new(),
            last: Vec::new(),
            transition_base: Vec::new(),
            first_base: Vec::new(),
            last_base: Vec::new(),
            transition_idx: Vec::new(),
            first_idx: Vec::new(),
            last_idx: Vec::new(),
            transition_hi: Vec::new(),
            first_hi: Vec::new(),
            last_hi: Vec::new(),
            nt: 0,
            nf: 0,
            nl: 0,
        }
    }
    /// Accumulating sink: constraints are combined with the given α powers
    /// as they are declared, nothing is stored.
    pub fn accumulating(mode: SinkMode, weights: Arc<CombinationWeights<F>>) -> ConstraintSink<F> {
        let mut s = ConstraintSink::new(mode);
        s.collect = false;
        s.weights = Some(weights);
        s
    }
    #[inline(always)]
    fn put<V: IntoConstraint<F>>(&mut self, cat: usize, i: u32, is_hi: bool, v: V) {
        let (e, b) = v.split();
        if let Some(w) = &self.weights {
            let off = match cat {
                0 => 0,
                1 => w.nt,
                _ => w.nt + w.nf,
            };
            let a = off + i as usize;
            let slot = 2 * cat + is_hi as usize;
            match b {
                Some(b) => self.acc[slot].add_base(&w.weights[a], b),
                None => self.acc[slot].add_ext(w.alphas[a], e),
            }
        }
        if self.collect {
            let (vals, base, idx, hi) = match cat {
                0 => (&mut self.transition, &mut self.transition_base, &mut self.transition_idx, &mut self.transition_hi),
                1 => (&mut self.first, &mut self.first_base, &mut self.first_idx, &mut self.first_hi),
                _ => (&mut self.last, &mut self.last_base, &mut self.last_idx, &mut self.last_hi),
            };
            vals.push(e);
            base.push(b);
            idx.push(i);
            hi.push(is_hi);
        }
    }
    /// Combined (low-degree, high-degree) sums of the accumulating sink,
    /// each divided by the matching vanishing polynomial.
    #[inline(never)]
    pub fn combined(&self, inv_zt: F, inv_first: F, inv_last: F) -> (F::Ext, F::Ext) {
        debug_assert!(self.weights.is_some());
        let f = |k: usize| self.acc[k].finish();
        (
            f(0).mul_base(inv_zt) + f(2).mul_base(inv_first) + f(4).mul_base(inv_last),
            f(1).mul_base(inv_zt) + f(3).mul_base(inv_first) + f(5).mul_base(inv_last),
        )
    }
    #[inline(always)]
    pub fn transition<V: IntoConstraint<F>, G: FnOnce() -> V>(&mut self, g: G) {
        let i = self.nt;
        self.nt += 1;
        if self.mode == SinkMode::All {
            self.put(0, i, false, g());
        }
    }
    #[inline(always)]
    pub fn transition_hi<V: IntoConstraint<F>, G: FnOnce() -> V>(&mut self, g: G) {
        let i = self.nt;
        self.nt += 1;
        self.put(0, i, true, g());
    }
    #[inline(always)]
    pub fn first<V: IntoConstraint<F>, G: FnOnce() -> V>(&mut self, g: G) {
        let i = self.nf;
        self.nf += 1;
        if self.mode == SinkMode::All {
            self.put(1, i, false, g());
        }
    }
    #[inline(always)]
    pub fn first_hi<V: IntoConstraint<F>, G: FnOnce() -> V>(&mut self, g: G) {
        let i = self.nf;
        self.nf += 1;
        self.put(1, i, true, g());
    }
    #[inline(always)]
    pub fn last<V: IntoConstraint<F>, G: FnOnce() -> V>(&mut self, g: G) {
        let i = self.nl;
        self.nl += 1;
        if self.mode == SinkMode::All {
            self.put(2, i, false, g());
        }
    }
    #[inline(always)]
    pub fn last_hi<V: IntoConstraint<F>, G: FnOnce() -> V>(&mut self, g: G) {
        let i = self.nl;
        self.nl += 1;
        self.put(2, i, true, g());
    }
    /// Total number of constraints declared so far (all modes).
    pub fn counts(&self) -> (usize, usize, usize) {
        (self.nt as usize, self.nf as usize, self.nl as usize)
    }
    #[inline(never)]
    pub fn clear(&mut self) {
        self.transition.clear();
        self.first.clear();
        self.last.clear();
        self.transition_base.clear();
        self.first_base.clear();
        self.last_base.clear();
        self.transition_idx.clear();
        self.first_idx.clear();
        self.last_idx.clear();
        self.transition_hi.clear();
        self.first_hi.clear();
        self.last_hi.clear();
        self.acc = [F::Acc::zero(); 6];
        self.nt = 0;
        self.nf = 0;
        self.nl = 0;
    }
}

/// A constraint value: either a base-field value (kept alongside its
/// extension image for cheap combination) or an extension-field value.
pub trait IntoConstraint<F: FieldLike> {
    fn split(self) -> (F::Ext, Option<F>);
}
impl<F: FieldLike> IntoConstraint<F> for F {
    #[inline(always)]
    fn split(self) -> (F::Ext, Option<F>) {
        (self.to_ext(), self.as_base())
    }
}
/// Marker wrapper for extension-field constraint values.
pub struct Ext<E>(pub E);
impl<F: FieldLike> IntoConstraint<F> for Ext<F::Ext> {
    #[inline(always)]
    fn split(self) -> (F::Ext, Option<F>) {
        (self.0, None)
    }
}

pub struct AirShape {
    pub main_width: usize,
    pub fixed_width: usize,
    /// Number of Fp4 aux columns.
    pub aux_width: usize,
    pub num_challenges: usize,
    pub num_transition: usize,
    pub num_first: usize,
    pub num_last: usize,
    /// Maximum total degree of any constraint (as a polynomial in the
    /// column values).
    pub max_degree: usize,
}

pub trait Air: Sync + Send {
    fn shape(&self) -> AirShape;
    /// Maximum degree of constraints *not* marked high (must be ≤ 2 for the
    /// prover's degree-aware composition to be sound). Default 2.
    fn low_degree_bound(&self) -> usize {
        2
    }
    /// Preprocessed columns for a trace of length `n`.
    fn fixed_columns(&self, n: usize) -> Vec<Vec<Fp>>;
    /// Prover-side: compute aux columns from main+fixed and challenges.
    /// Also returns "aux public values" (e.g. LogUp final sums) that are
    /// sent to the verifier in the clear and made available to `eval`.
    /// Only the first `real_rows` rows are meaningful (the rest are the
    /// prover's zero-knowledge random rows); sums must stop at `real_rows`.
    fn generate_aux(&self, main: &[&[Fp]], fixed: &[&[Fp]], challenges: &[Fp4], real_rows: usize) -> (Vec<Vec<Fp4>>, Vec<Fp4>);
    /// Evaluate every constraint at one row (or eight rows, when `F` is a lane type).
    fn eval<F: FieldLike, M: RowAccess<F>, A: RowAccess<F::Ext>>(&self, row: &RowView<F, M, A>, challenges: &[Fp4], public: &[Fp4], aux_public: &[Fp4], sink: &mut ConstraintSink<F>);
}

/// Debug helper: evaluate all constraints in the clear on a trace and report
/// the first failures. Used while developing an AIR; never in the protocol.
#[allow(clippy::too_many_arguments)]
pub fn check_trace<A: Air>(
    air: &A,
    main: &[&[Fp]],
    fixed: &[&[Fp]],
    aux: &[Vec<Fp4>],
    challenges: &[Fp4],
    public: &[Fp4],
    aux_public: &[Fp4],
    real_rows: usize,
    max_report: usize,
) -> Vec<String> {
    let n = real_rows;
    let full = main[0].len();
    let shape = air.shape();
    let mut failures = Vec::new();
    let gather = |cols: &[&[Fp]], i: usize, out: &mut Vec<Fp>| {
        out.clear();
        for c in cols {
            out.push(c[i]);
        }
    };
    let mut m = Vec::new();
    let mut mn = Vec::new();
    let mut f = Vec::new();
    let mut fnx = Vec::new();
    let mut a = Vec::new();
    let mut an = Vec::new();
    let mut sink: ConstraintSink<Fp> = ConstraintSink::default();
    for i in 0..n {
        let j = (i + 1) % full;
        gather(main, i, &mut m);
        gather(main, j, &mut mn);
        gather(fixed, i, &mut f);
        gather(fixed, j, &mut fnx);
        a.clear();
        an.clear();
        for c in aux {
            a.push(c[i]);
            an.push(c[j]);
        }
        sink.clear();
        let row = RowView::new(&m[..], &mn[..], &f[..], &fnx[..], &a[..], &an[..]);
        air.eval(&row, challenges, public, aux_public, &mut sink);
        assert_eq!(sink.counts(), (shape.num_transition, shape.num_first, shape.num_last), "constraint count mismatch");
        if i + 1 < n {
            for (k, v) in sink.transition.iter().enumerate() {
                if !Fp4::is_zero(*v) {
                    failures.push(format!("row {i}: transition constraint #{k} = {v:?}"));
                    if failures.len() >= max_report {
                        return failures;
                    }
                }
            }
        }
        if i == 0 {
            for (k, v) in sink.first.iter().enumerate() {
                if !Fp4::is_zero(*v) {
                    failures.push(format!("row 0: first-row constraint #{k} = {v:?}"));
                }
            }
        }
        if i + 1 == n {
            for (k, v) in sink.last.iter().enumerate() {
                if !Fp4::is_zero(*v) {
                    failures.push(format!("row {i}: last-row constraint #{k} = {v:?}"));
                }
            }
        }
    }
    failures
}

/// Check that every constraint not marked high-degree has total degree ≤
/// `low_degree_bound()` and every constraint has degree ≤ `max_degree`, by
/// evaluating the AIR on random *lines* (each column a random linear
/// function of x) and interpolating. Returns the first violation.
pub fn check_degrees<A: Air>(air: &A, seed: u64) -> Result<(), String> {
    let shape = air.shape();
    let md = shape.max_degree;
    let npts = md + 3;
    let mut rng = seed | 1;
    let mut rnd = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        Fp::new(rng)
    };
    let (mw, fw, aw) = (shape.main_width, shape.fixed_width, shape.aux_width);
    let lm: Vec<(Fp, Fp)> = (0..mw).map(|_| (rnd(), rnd())).collect();
    let lmn: Vec<(Fp, Fp)> = (0..mw).map(|_| (rnd(), rnd())).collect();
    let lf: Vec<(Fp, Fp)> = (0..fw).map(|_| (rnd(), rnd())).collect();
    let lfn: Vec<(Fp, Fp)> = (0..fw).map(|_| (rnd(), rnd())).collect();
    let la: Vec<(Fp4, Fp4)> = (0..aw).map(|_| (Fp4([rnd(), rnd(), rnd(), rnd()]), Fp4([rnd(), rnd(), rnd(), rnd()]))).collect();
    let lan: Vec<(Fp4, Fp4)> = (0..aw).map(|_| (Fp4([rnd(), rnd(), rnd(), rnd()]), Fp4([rnd(), rnd(), rnd(), rnd()]))).collect();
    let ch: Vec<Fp4> = (0..2 * shape.num_challenges).map(|_| Fp4([rnd(), rnd(), rnd(), rnd()])).collect();
    let public: Vec<Fp4> = (0..16).map(|_| Fp4([rnd(), rnd(), rnd(), rnd()])).collect();
    let aux_public: Vec<Fp4> = (0..aw).map(|_| Fp4([rnd(), rnd(), rnd(), rnd()])).collect();
    let xs: Vec<Fp> = (0..npts).map(|i| Fp::new(1000 + i as u64)).collect();
    let mut evals: Vec<Vec<Vec<Fp4>>> = Vec::new();
    let mut his: Vec<Vec<bool>> = Vec::new();
    for &x in &xs {
        let ev = |l: &[(Fp, Fp)]| -> Vec<Fp> { l.iter().map(|(a, b)| *a + *b * x).collect() };
        let eva = |l: &[(Fp4, Fp4)]| -> Vec<Fp4> { l.iter().map(|(a, b)| *a + b.mul_base(x)).collect() };
        let (m, mn, f, fnx, a, an) = (ev(&lm), ev(&lmn), ev(&lf), ev(&lfn), eva(&la), eva(&lan));
        let row = RowView::new(&m[..], &mn[..], &f[..], &fnx[..], &a[..], &an[..]);
        let mut sink: ConstraintSink<Fp> = ConstraintSink::new(SinkMode::All);
        air.eval(&row, &ch, &public, &aux_public, &mut sink);
        evals.push(vec![sink.transition.clone(), sink.first.clone(), sink.last.clone()]);
        his = vec![sink.transition_hi.clone(), sink.first_hi.clone(), sink.last_hi.clone()];
    }
    let names = ["transition", "first", "last"];
    for cat in 0..3 {
        for ci in 0..evals[0][cat].len() {
            let ys: Vec<Fp4> = evals.iter().map(|e| e[cat][ci]).collect();
            let mut coef = ys.clone();
            for level in 1..npts {
                for i in (level..npts).rev() {
                    coef[i] = (coef[i] - coef[i - 1]) * Fp4::from_base((xs[i] - xs[i - level]).inverse());
                }
            }
            let degree = coef.iter().rposition(|c| !Fp4::is_zero(*c)).unwrap_or(0);
            if degree > md {
                return Err(format!("{} constraint #{ci} has degree {degree} > max_degree {md}", names[cat]));
            }
            if !his[cat][ci] && degree > air.low_degree_bound() {
                return Err(format!("{} constraint #{ci} has degree {degree} but is not marked high-degree", names[cat]));
            }
        }
    }
    Ok(())
}
