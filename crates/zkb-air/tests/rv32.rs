use zkb_air::{Rv32Air, Segment};
use zkb_field::{Fp, Fp4};
use zkb_stark::air::check_trace;
use zkb_stark::{Air, StarkConfig};
use zkb_vm::{elf::load_elf, Vm};

fn guest(name: &str) -> Vec<u8> {
    let p = format!("{}/../../guest/build/{name}.elf", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&p).unwrap_or_else(|_| panic!("run guest/build.sh first ({p})"))
}

/// Build one segment trace (whole execution) and check every constraint in the clear.
fn check_single_segment(name: &str, witness: Vec<u32>) {
    let image = load_elf(&guest(name)).unwrap();
    let mut vm = Vm::new(&image, witness);
    vm.begin_segment(0);
    let run = vm.run_segment(1 << 20, true).unwrap();
    assert!(run.halted && vm.exit_code == 0);
    let init_entries = vm.init_entries();
    let closing = vm.closing_entries(&image);
    let seg = Segment { index: 0, init_entries: &init_entries, closing: &closing, records: &run.records, pc_in: run.pc_in, pc_out: run.pc_out, jcount_in: run.jcount_in, jcount_out: run.jcount_out, halted: true };
    let air = Rv32Air::new();
    let k = 168; // pretend zk random rows are reserved
    let log_n = zkb_air::trace::min_log_n(run.records.len(), init_entries.len() + closing.len(), k);
    let real = (1usize << log_n) - k;
    let t0 = std::time::Instant::now();
    let main_store = zkb_air::trace::generate_main(&air.l, &seg, log_n, real).unwrap();
    let main = main_store.slices();
    println!("{name}: {} cycles, {} init + {} closing entries, log_n={log_n}, {} main cols, built in {:?}", run.cycles, init_entries.len(), closing.len(), main.len(), t0.elapsed());
    let fixed_cols = air.fixed_columns(1 << log_n);
    let fixed: Vec<&[Fp]> = fixed_cols.iter().map(|v| v.as_slice()).collect();
    let ch = vec![Fp4([Fp::new(0x1234567), Fp::new(99), Fp::new(3), Fp::new(4)]), Fp4([Fp::new(0xabcdef), Fp::new(7), Fp::new(8), Fp::new(9)])];
    let (aux, aux_pub) = air.generate_aux(&main, &fixed, &ch, real);
    let e = zkb_air::lookups::Enc::new(ch[0], ch[1]);
    let chain: Fp4 = aux_pub[zkb_air::lookups::AUX_CHAIN].iter().cloned().sum();
    let rest: Fp4 = aux_pub[zkb_air::lookups::AUX_CHAIN.end..].iter().cloned().sum();
    assert_eq!(chain, zkb_air::image_correction(&image, &e), "memory chain balance (single segment)");
    assert_eq!(rest, zkb_air::journal_sum(&vm.journal, 0, vm.journal.len(), &e), "journal balance");
    let public = seg.public_values();
    let fails = check_trace(&air, &main, &fixed, &aux, &ch, &public, &aux_pub, real, 20);
    assert!(fails.is_empty(), "constraint failures:\n{}", fails.join("\n"));
}

#[test]
fn constraint_degrees_marked_correctly() {
    zkb_stark::air::check_degrees(&Rv32Air::new(), 7).unwrap();
}

#[test]
fn echo_trace_satisfies_constraints() {
    check_single_segment("echo-guest", vec![41]);
}

#[test]
fn mul_trace_satisfies_constraints() {
    for (a, b) in [(7u32, 3u32), (0xFFFF_FFF0, 0x7FFF_FFFF), (0x8000_0000, 0xFFFF_FFFF), (123456789, 0), (0, 5), (0xDEADBEEF, 0xCAFEBABE), (0x8000_0000, 3)] {
        let image = load_elf(&guest("mul-guest")).unwrap();
        let (_img, res) = zkb_vm::execute(&guest("mul-guest"), vec![a, b], 1 << 20, false).unwrap();
        let _ = image;
        let (sa, sb) = (a as i32, b as i32);
        let expect: Vec<u32> = vec![
            a.wrapping_mul(b),
            ((sa as i64 * sb as i64) >> 32) as u32,
            ((sa as i64 * b as i64) >> 32) as u32,
            ((a as u64 * b as u64) >> 32) as u32,
            if b == 0 { u32::MAX } else { sa.wrapping_div(sb) as u32 },
            if b == 0 { u32::MAX } else { a / b },
            if b == 0 { a } else { sa.wrapping_rem(sb) as u32 },
            if b == 0 { a } else { a % b },
            0x8000_0000,
            0,
        ];
        assert_eq!(res.journal, expect, "VM M-extension semantics for ({a:#x},{b:#x})");
        check_single_segment("mul-guest", vec![a, b]);
    }
}

#[test]
fn echo_prove_verify_single_segment() {
    let image = load_elf(&guest("echo-guest")).unwrap();
    let air = Rv32Air::new();
    let cfg = StarkConfig::fast();
    let log_n = zkb_air::choose_log_n(&image, &[41], 17, 1 << 24, &cfg).unwrap();
    let pk = zkb_air::setup(&air, log_n, cfg);
    let t0 = std::time::Instant::now();
    let (ep, out) = zkb_air::prove_execution(&air, &pk, &image, &[41], 1 << 24).unwrap();
    println!("prove {:?}: {} cycles, {} segment(s), proof {} KB", t0.elapsed(), out.cycles, out.segments, ep.proof.size_bytes() / 1024);
    assert_eq!(out.segments, 1);
    let t0 = std::time::Instant::now();
    zkb_air::verify_execution(&air, &ep, &image).unwrap();
    println!("verify {:?}", t0.elapsed());
    let mut bad = ep.clone();
    bad.claim.journal[0] += 1;
    assert!(zkb_air::verify_execution(&air, &bad, &image).is_err());
    let mut img2 = image.clone();
    img2.words[0].1 ^= 1;
    assert!(zkb_air::verify_execution(&air, &ep, &img2).is_err());
}

#[test]
fn echo_prove_verify_three_segments_zk() {
    // Force three segments by capping cycles per segment; zero-knowledge config.
    let image = load_elf(&guest("echo-guest")).unwrap();
    let air = Rv32Air::new();
    let cfg = StarkConfig::zk();
    let pk = zkb_air::setup(&air, 17, cfg);
    let t0 = std::time::Instant::now();
    let (ep, out) = zkb_air::prove_execution_capped(&air, &pk, &image, &[41], 1 << 24, Some(30000)).unwrap();
    println!("prove {:?}: {} cycles, {} segments, proof {} KB", t0.elapsed(), out.cycles, out.segments, ep.proof.size_bytes() / 1024);
    assert_eq!(out.segments, 3);
    zkb_air::verify_execution(&air, &ep, &image).unwrap();
    // drop the middle segment: chain must break
    let mut bad = ep.clone();
    bad.proof.segments.remove(1);
    assert!(zkb_air::verify_execution(&air, &bad, &image).is_err());
    // swap segments
    let mut bad = ep.clone();
    bad.proof.segments.swap(0, 1);
    assert!(zkb_air::verify_execution(&air, &bad, &image).is_err());
    // claim a different journal
    let mut bad = ep.clone();
    bad.claim.journal[3] ^= 1;
    assert!(zkb_air::verify_execution(&air, &bad, &image).is_err());
}

#[test]
fn print_constraint_counts() {
    let air = zkb_air::Rv32Air::new();
    let shape = zkb_stark::Air::shape(&air);
    eprintln!("constraints: transition={} first={} last={} main_w={} aux_w={} fixed_w={} max_degree={}", shape.num_transition, shape.num_first, shape.num_last, shape.main_width, shape.aux_width, shape.fixed_width, shape.max_degree);
}

