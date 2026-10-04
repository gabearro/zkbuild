//! End-to-end: two translation units + link, bundled and verified; then
//! every cross-check in the certificate chain is attacked once.

use zkb_ca::{SourceFile, SourceTree};
use zkb_proof::{BuildCertificate, Prover, Toolchain};
use zkb_stark::StarkConfig;

fn guest(name: &str) -> Vec<u8> {
    let p = format!("{}/../../guest/build/{name}.elf", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&p).unwrap_or_else(|_| panic!("run guest/build.sh first ({p})"))
}

#[test]
fn two_tus_link_bundle_verify() {
    let tc = Toolchain::load(&guest("tu-guest"), &guest("link-guest")).unwrap();
    let mut prover = Prover::new(StarkConfig::fast());
    let tu1 = SourceTree { files: vec![SourceFile { name: "a.c".into(), content: b"int a(void){return 1;}\n".to_vec() }] };
    let tu2 = SourceTree { files: vec![SourceFile { name: "b.c".into(), content: b"int b(void){return 2;}\n".to_vec() }] };
    let (c1, o1) = prover.prove_tu(&tc, &tu1, b"-O2").unwrap();
    let (c2, o2) = prover.prove_tu(&tc, &tu2, b"-O2").unwrap();
    let (link, exe) = prover.prove_link(&tc, &[o1.clone(), o2.clone()], b"-static").unwrap();
    let air = zkb_air::Rv32Air::new();
    let build = BuildCertificate { objects: vec![c1.clone(), c2.clone()], link: link.clone() };
    let summary = build.verify(&air, &tc).unwrap();
    assert_eq!(summary.output_hash, zkb_ca::v0_executable_hash(&exe).unwrap());
    // the v0 identity toolchain's object/exe formats are known to the host
    assert_eq!(o1, tu1.v0_object(b"-O2"));
    assert_eq!(exe, zkb_ca::v0_executable(b"-static", &[o1.clone(), o2.clone()]));
    assert_eq!(summary.objects[0].0, tu1.root());

    // attack 1: reorder objects (link consumed o1,o2 — certificate claims o2,o1)
    let bad = BuildCertificate { objects: vec![c2.clone(), c1.clone()], link: link.clone() };
    assert!(bad.verify(&air, &tc).is_err());

    // attack 2: claim a different source commitment for the same proof
    let mut c1b = c1.clone();
    c1b.source_root[0] ^= 1;
    let bad = BuildCertificate { objects: vec![c1b, c2.clone()], link: link.clone() };
    assert!(bad.verify(&air, &tc).is_err());

    // attack 3: present the certificate against a different toolchain
    let other = Toolchain::load(&guest("echo-guest"), &guest("link-guest")).unwrap();
    assert!(build.verify(&air, &other).is_err());

    // attack 4: claim a different output hash
    let mut lb = link.clone();
    lb.output_hash[5] ^= 0x80;
    let bad = BuildCertificate { objects: vec![c1.clone(), c2.clone()], link: lb };
    assert!(bad.verify(&air, &tc).is_err());

    // serialization round trip
    let bytes = bincode::serialize(&build).unwrap();
    let back: BuildCertificate = bincode::deserialize(&bytes).unwrap();
    back.verify(&air, &tc).unwrap();
}
