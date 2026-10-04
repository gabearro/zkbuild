use zkb_vm::{execute, pack_witness_bytes};

fn guest(name: &str) -> Vec<u8> {
    let p = format!("{}/../../guest/build/{name}.elf", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&p).unwrap_or_else(|_| panic!("run guest/build.sh first ({p})"))
}

fn journal_bytes32(words: &[u32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, w) in words.iter().enumerate().take(8) {
        out[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    out
}

#[test]
fn echo_guest() {
    let (_img, res) = execute(&guest("echo-guest"), vec![41], 1_000_000, true).unwrap();
    assert_eq!(res.exit_code, 0);
    assert_eq!(res.journal[0], 42);
    assert_eq!(res.journal[1], 123);
    let h = journal_bytes32(&res.journal[2..10]);
    let expect = hex::decode("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad").unwrap();
    assert_eq!(&h[..], &expect[..]);
    // blake3 test vectors against the host crate
    let pat: Vec<u8> = (0..1100u32).map(|i| (i % 251) as u8).collect();
    assert_eq!(journal_bytes32(&res.journal[10..18]), *blake3::hash(b"abc").as_bytes(), "blake3(abc)");
    assert_eq!(journal_bytes32(&res.journal[18..26]), *blake3::hash(&pat).as_bytes(), "blake3(1100-byte pattern)");
    assert_eq!(journal_bytes32(&res.journal[26..34]), *blake3::hash(&pat[..1024]).as_bytes(), "blake3(1024-byte pattern)");
    // private output stream: [5]["hell"]["o\0\0\0"] [6]["worl"]["d!\0\0"]
    let expect_out: Vec<u32> = vec![5, u32::from_le_bytes(*b"hell"), u32::from_le_bytes(*b"o\0\0\0"), 6, u32::from_le_bytes(*b"worl"), u32::from_le_bytes(*b"d!\0\0")];
    assert_eq!(res.output, expect_out, "framed output syscall");
    println!("echo guest: {} cycles", res.cycles);
}

#[test]
fn tu_guest_matches_host_hash() {
    let flags = b"-O2 -ffreestanding";
    let files: Vec<(&[u8], &[u8])> = vec![(b"a.c", b"int a(void){return 1;}\n"), (b"b.c", b"int b(void){return 2;}\n")];
    let mut w = pack_witness_bytes(flags);
    w.push(files.len() as u32);
    for (n, c) in &files {
        w.extend(pack_witness_bytes(n));
        w.extend(pack_witness_bytes(c));
    }
    let (_img, res) = execute(&guest("tu-guest"), w, 10_000_000, false).unwrap();
    assert_eq!(res.exit_code, 0, "guest exit code");
    assert_eq!(res.journal.len(), 25);
    // recompute on host
    let mut leaves = Vec::new();
    for (n, c) in &files {
        let mut h = blake3::Hasher::new();
        h.update(&[0u8]);
        h.update(&(n.len() as u32).to_le_bytes());
        h.update(n);
        h.update(&(c.len() as u32).to_le_bytes());
        h.update(c);
        leaves.extend_from_slice(h.finalize().as_bytes());
    }
    let mut h = blake3::Hasher::new();
    h.update(&[1u8]);
    h.update(&(files.len() as u32).to_le_bytes());
    h.update(&leaves);
    let root: [u8; 32] = *h.finalize().as_bytes();
    assert_eq!(journal_bytes32(&res.journal[0..8]), root);
    let mut h = blake3::Hasher::new();
    h.update(b"ZKOBJ0");
    h.update(&(flags.len() as u32).to_le_bytes());
    h.update(flags);
    for (_, c) in &files {
        h.update(c);
    }
    let ohash: [u8; 32] = *h.finalize().as_bytes();
    assert_eq!(journal_bytes32(&res.journal[16..24]), ohash);
    assert_eq!(res.journal[24], 2);
    println!("tu guest: {} cycles for {} bytes of source", res.cycles, files.iter().map(|f| f.1.len()).sum::<usize>());
}

/// The real compiler guest (chibicc) compiles a C file inside the VM.
#[test]
fn zkcc_compiles_fib() {
    let src = std::fs::read(format!("{}/../../examples/fib/fib.c", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let mut w = pack_witness_bytes(b"-cc1 -cc1-input fib.c -cc1-output - fib.c");
    w.push(1);
    w.extend(pack_witness_bytes(b"fib.c"));
    w.extend(pack_witness_bytes(&src));
    let elf = std::fs::read(format!("{}/../../guest/build/zkcc.elf", env!("CARGO_MANIFEST_DIR"))).expect("run guest/build-zkcc.sh first");
    let (_img, res) = execute(&elf, w, 1 << 28, false).unwrap();
    assert_eq!(res.exit_code, 0, "compiler exit code");
    // reassemble the private output stream
    let mut out = Vec::new();
    let mut i = 0;
    while i < res.output.len() {
        let len = res.output[i] as usize;
        i += 1;
        let nw = len.div_ceil(4);
        for (k, wv) in res.output[i..i + nw].iter().enumerate() {
            let b = wv.to_le_bytes();
            out.extend_from_slice(&b[..(len - 4 * k).min(4)]);
        }
        i += nw;
    }
    let asm = String::from_utf8(out).unwrap();
    assert!(asm.contains(".globl main") && asm.contains("fib:") && asm.contains("mov $55, %rax"), "unexpected assembly");
    // the journal's object hash must be the hash of the output
    let h: [u8; 32] = *blake3::hash(asm.as_bytes()).as_bytes();
    assert_eq!(journal_bytes32(&res.journal[16..24]), h);
    println!("zkcc compiled fib.c in {} cycles -> {} bytes of x86-64 assembly", res.cycles, asm.len());
}
