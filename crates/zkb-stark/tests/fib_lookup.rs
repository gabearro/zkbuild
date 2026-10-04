//! End-to-end test of the STARK on a toy AIR: Fibonacci plus a LogUp lookup
//! of a witness column into a fixed table (row index). Exercises plain,
//! zero-knowledge and multi-segment modes.

use zkb_field::{ExtLike, FieldLike, Fp, Fp4};
use zkb_stark::air::Ext;
use zkb_stark::air::check_trace;
use zkb_stark::{prover, verifier, Air, AirShape, ConstraintSink, RowAccess, RowView, StarkConfig};

struct FibLookupAir;

// main: [a, b, v, mult]   fixed: [t]   aux: [s]   challenges: [gamma]   public: [b_last]
impl Air for FibLookupAir {
    fn shape(&self) -> AirShape {
        AirShape { main_width: 4, fixed_width: 1, aux_width: 1, num_challenges: 1, num_transition: 3, num_first: 3, num_last: 2, max_degree: 3 }
    }
    fn fixed_columns(&self, n: usize) -> Vec<Vec<Fp>> {
        vec![(0..n).map(|i| Fp::new(i as u64)).collect()]
    }
    fn generate_aux(&self, main: &[&[Fp]], fixed: &[&[Fp]], ch: &[Fp4], real_rows: usize) -> (Vec<Vec<Fp4>>, Vec<Fp4>) {
        let n = main[0].len();
        let gamma = ch[0];
        let mut s = vec![Fp4::ZERO; n];
        let mut acc = Fp4::ZERO;
        for i in 0..real_rows {
            s[i] = acc;
            let frac = (gamma - Fp4::from_base(main[2][i])).inverse() - (gamma - Fp4::from_base(fixed[0][i])).inverse().mul_base(main[3][i]);
            acc += frac;
        }
        (vec![s], vec![acc])
    }
    fn eval<F: FieldLike, M: RowAccess<F>, A: RowAccess<F::Ext>>(&self, r: &RowView<F, M, A>, ch: &[Fp4], public: &[Fp4], aux_public: &[Fp4], sink: &mut ConstraintSink<F>) {
        let (a, b, v, mult) = (r.main.at(0), r.main.at(1), r.main.at(2), r.main.at(3));
        let (an, bn) = (r.main_next.at(0), r.main_next.at(1));
        let t = r.fixed.at(0);
        let s = r.aux.at(0);
        let sn = r.aux_next.at(0);
        let gamma = F::Ext::from(ch[0]);
        sink.transition(|| an - b);
        sink.transition(|| bn - a - b);
        // (s' - s)(γ - v)(γ - t) = (γ - t) - mult (γ - v)
        let gv = gamma - v.to_ext();
        let gt = gamma - t.to_ext();
        let lookup = |s_next: F::Ext| (s_next - s) * gv * gt - gt + gv.mul_base(mult);
        sink.transition_hi(|| Ext(lookup(sn)));
        sink.first(|| a - F::ONE);
        sink.first(|| b - F::ONE);
        sink.first(|| Ext(s));
        sink.last(|| Ext(b.to_ext() - F::Ext::from(public[0])));
        sink.last_hi(|| Ext(lookup(F::Ext::from(aux_public[0]))));
    }
}

fn build_trace(n: usize, real: usize) -> (Vec<Vec<Fp>>, Fp) {
    let mut a = vec![Fp::ONE; n];
    let mut b = vec![Fp::ONE; n];
    for i in 1..n {
        a[i] = b[i - 1];
        b[i] = a[i - 1] + b[i - 1];
    }
    // lookups only on real rows; values stay below `real` so the table (row index) covers them
    let v: Vec<Fp> = (0..n).map(|i| Fp::new(((i * 7) % real) as u64)).collect();
    let mut mult = vec![Fp::ZERO; n];
    for x in &v[..real] {
        mult[x.as_u64() as usize] += Fp::ONE;
    }
    let last = b[real - 1];
    (vec![a, b, v, mult], last)
}

fn run(cfg: StarkConfig, log_n: u32, segments: usize) {
    let n = 1 << log_n;
    let air = FibLookupAir;
    let pk = prover::setup(&air, log_n, cfg);
    let real = n - cfg.random_rows();
    let (main, last) = build_trace(n, real);
    let public = vec![Fp4::from_base(last)];
    let ch = vec![Fp4([Fp::new(12345), Fp::new(678), Fp::new(9), Fp::new(10)]), Fp4([Fp::new(5), Fp::new(6), Fp::new(7), Fp::new(8)])];
    let main_s: Vec<&[Fp]> = main.iter().map(|v| v.as_slice()).collect();
    let fixed_s: Vec<&[Fp]> = pk.fixed.iter().map(|v| v.as_slice()).collect();
    let (aux, aux_pub) = air.generate_aux(&main_s, &fixed_s, &ch, real);
    assert_eq!(aux_pub[0], Fp4::ZERO, "lookup must balance");
    let fails = check_trace(&air, &main_s, &fixed_s, &aux, &ch, &public, &aux_pub, real, 5);
    assert!(fails.is_empty(), "{fails:?}");

    let publics: Vec<Vec<Fp4>> = (0..segments).map(|_| public.clone()).collect();
    let main_store = zkb_field::columns::ColumnStore::from_columns(main.clone());
    let proof = prover::prove(&air, &pk, &publics, &|_| Ok(main_store.clone())).unwrap();
    for s in &proof.segments {
        assert_eq!(s.aux_public[0], Fp4::ZERO);
    }
    verifier::verify(&air, &pk.vk, &proof, &publics).unwrap();
    println!("cfg {cfg:?} log_n={log_n} segments={segments}: proof {} bytes", proof.size_bytes());

    // wrong public input must fail
    let mut bad_publics = publics.clone();
    bad_publics[segments - 1][0] = Fp4::from_base(last + Fp::ONE);
    assert!(verifier::verify(&air, &pk.vk, &proof, &bad_publics).is_err());

    // tampered OOD value must fail
    let mut bad = proof.clone();
    bad.segments[0].ood.main[1] += Fp4::ONE;
    assert!(verifier::verify(&air, &pk.vk, &bad, &publics).is_err());

    // tampered opening must fail
    let mut bad = proof.clone();
    bad.segments[0].openings[3].lo.main.values[0] += Fp::ONE;
    assert!(verifier::verify(&air, &pk.vk, &bad, &publics).is_err());
}

#[test]
fn degrees_marked_correctly() {
    zkb_stark::air::check_degrees(&FibLookupAir, 42).unwrap();
}

#[test]
fn fib_lookup_plain() {
    run(StarkConfig { log_blowup: 2, num_queries: 20, log_final_size: 4, zk: false, grinding_bits: 0 }, 6, 1);
}

#[test]
fn fib_lookup_zk() {
    // random rows: 4·q + 8 = 88 of 512
    run(StarkConfig { log_blowup: 2, num_queries: 20, log_final_size: 4, zk: true, grinding_bits: 8 }, 9, 1);
}

#[test]
fn fib_lookup_two_segments_zk() {
    run(StarkConfig { log_blowup: 2, num_queries: 12, log_final_size: 4, zk: true, grinding_bits: 4 }, 8, 2);
}

#[test]
fn zk_openings_hide_trace() {
    // In zk mode, the opened LDE rows must differ from the un-blinded LDE rows
    // (a sanity check that blinding is applied), and two proofs of the same
    // trace must have different commitments (salting).
    let cfg = StarkConfig { log_blowup: 2, num_queries: 12, log_final_size: 4, zk: true, grinding_bits: 0 };
    let air = FibLookupAir;
    let pk = prover::setup(&air, 8, cfg);
    let (main, last) = build_trace(256, 256 - cfg.random_rows());
    let publics = vec![vec![Fp4::from_base(last)]];
    let main_store = zkb_field::columns::ColumnStore::from_columns(main.clone());
    let p1 = prover::prove(&air, &pk, &publics, &|_| Ok(main_store.clone())).unwrap();
    let p2 = prover::prove(&air, &pk, &publics, &|_| Ok(main_store.clone())).unwrap();
    assert_ne!(p1.segments[0].main_root, p2.segments[0].main_root);
    assert_ne!(p1.segments[0].openings[0].lo.main.values, p2.segments[0].openings[0].lo.main.values);
    let plain = prover::Committed::new(&main_store, 2, None);
    let l = 5usize; // any index: blinded values differ from plain LDE values
    let opened = &p1.segments[0].openings[0].lo.main.values;
    let idx = {
        // recover the query index by matching against the proof's own structure is not needed:
        // just check that no plain row equals the opened row.
        (0..1024).all(|i| plain.lde.cols().map(|c| c[i]).collect::<Vec<_>>() != *opened)
    };
    assert!(idx, "opened row equals an unblinded LDE row");
    let _ = l;
}
