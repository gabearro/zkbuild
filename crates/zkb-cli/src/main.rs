//! `zkb` — zero-knowledge reproducible-build attestation.
//!
//! Pipeline: `commit` → `prove-tu` (per translation unit) → `link` → `bundle` → `verify`.

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use zkb_ca::{hex, SourceTree};
use zkb_proof::{load, save, BuildCertificate, LinkCertificate, ObjectCertificate, Prover, Toolchain};
use zkb_stark::StarkConfig;

// Large column vectors are allocated and freed every segment; mimalloc keeps
// that memory mapped instead of returning it to the OS (and re-faulting it).
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser)]
#[command(name = "zkb", version, about = "Zero-knowledge reproducible build attestation")]
struct Cli {
    /// Use the larger (slower, more secure) STARK parameter set.
    #[arg(long, global = true)]
    secure: bool,
    /// Disable zero-knowledge blinding (faster; proofs then leak trace rows).
    #[arg(long, global = true)]
    no_zk: bool,
    /// log2 of the maximum rows per proof segment (memory bound; 16..=18).
    #[arg(long, global = true, default_value_t = 17)]
    segment_log: u32,
    #[arg(short, long, global = true)]
    verbose: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print the source commitment (source_root) of a directory or set of files.
    Commit {
        /// Directory (all files, sorted by relative path) or individual files.
        paths: Vec<PathBuf>,
    },
    /// Execute a guest ELF in the VM (no proof) and print its journal.
    Run {
        elf: PathBuf,
        /// Witness words (decimal or 0x hex).
        #[arg(long, value_delimiter = ',')]
        witness: Vec<String>,
        /// Witness built from a toolchain-guest invocation: flags string + source dir/files.
        #[arg(long, allow_hyphen_values = true)]
        flags: Option<String>,
        #[arg(long, num_args = 1..)]
        src: Vec<PathBuf>,
        /// Save the private output stream here.
        #[arg(long)]
        output: Option<PathBuf>,
        /// Print a per-function cycle profile (needs llvm-nm on PATH).
        #[arg(long)]
        profile: bool,
        /// Witness for the link guest: these object files (with --flags as the link flags, default empty).
        #[arg(long, num_args = 1..)]
        link_objects: Vec<PathBuf>,
    },
    /// Prove one translation unit: produces an object certificate and the object bytes.
    ProveTu {
        #[arg(long)]
        tu_guest: PathBuf,
        #[arg(long)]
        link_guest: PathBuf,
        #[arg(long, default_value = "", allow_hyphen_values = true)]
        flags: String,
        /// Source directory or files.
        #[arg(long, required = true, num_args = 1..)]
        src: Vec<PathBuf>,
        #[arg(long)]
        cert: PathBuf,
        #[arg(long)]
        object: PathBuf,
    },
    /// Prove a link step over object files.
    Link {
        #[arg(long)]
        tu_guest: PathBuf,
        #[arg(long)]
        link_guest: PathBuf,
        #[arg(long, default_value = "", allow_hyphen_values = true)]
        flags: String,
        #[arg(long, required = true, num_args = 1..)]
        objects: Vec<PathBuf>,
        #[arg(long)]
        cert: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Bundle object certificates and a link certificate into a build certificate.
    Bundle {
        #[arg(long, required = true, num_args = 1..)]
        objects: Vec<PathBuf>,
        #[arg(long)]
        link: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Verify a build certificate against a trusted toolchain (and optionally the binary).
    Verify {
        cert: PathBuf,
        #[arg(long)]
        tu_guest: PathBuf,
        #[arg(long)]
        link_guest: PathBuf,
        /// If given, check that this binary's hash is the certified output.
        #[arg(long)]
        binary: Option<PathBuf>,
    },
    /// Print the public statement of a certificate.
    Info { cert: PathBuf },
    /// Print the toolchain root for a pair of guest images.
    Toolchain {
        #[arg(long)]
        tu_guest: PathBuf,
        #[arg(long)]
        link_guest: PathBuf,
    },
}

fn load_tree(paths: &[PathBuf]) -> Result<SourceTree> {
    if paths.len() == 1 && paths[0].is_dir() {
        SourceTree::from_dir(&paths[0])
    } else {
        let files: Vec<String> = paths.iter().map(|p| p.to_string_lossy().to_string()).collect();
        SourceTree::from_files(&files)
    }
}

fn load_toolchain(tu: &PathBuf, link: &PathBuf) -> Result<Toolchain> {
    Toolchain::load(&std::fs::read(tu).context("tu guest elf")?, &std::fs::read(link).context("link guest elf")?)
}

fn parse_word(s: &str) -> Result<u32> {
    if let Some(h) = s.strip_prefix("0x") {
        Ok(u32::from_str_radix(h, 16)?)
    } else {
        Ok(s.parse()?)
    }
}

fn main() -> Result<()> {
    // Never purge freed pages back to the OS: the prover recycles the same
    // large buffers every segment, and re-faulting them costs more than the
    // arithmetic on this kind of VM.
    unsafe {
        // mi_option_purge_delay (index 15 in mimalloc's option enum): -1 = never purge
        const MI_OPTION_PURGE_DELAY: libmimalloc_sys::mi_option_t = 15;
        libmimalloc_sys::mi_option_set(MI_OPTION_PURGE_DELAY, -1);
        if std::env::var("ZKB_TIMING").is_ok() {
            eprintln!("[alloc] mimalloc purge_delay = {}", libmimalloc_sys::mi_option_get(MI_OPTION_PURGE_DELAY));
        }
    }
    let cli = Cli::parse();
    let config = if cli.no_zk { StarkConfig::fast() } else if cli.secure { StarkConfig::zk_secure() } else { StarkConfig::zk() };
    match cli.cmd {
        Cmd::Commit { paths } => {
            let tree = load_tree(&paths)?;
            for f in &tree.files {
                println!("{}  {}  ({} bytes)", hex(&SourceTree::leaf(f)), f.name, f.content.len());
            }
            println!("source_root = {}", hex(&tree.root()));
        }
        Cmd::Run { elf, witness, flags, src, output, profile, link_objects } => {
            let w: Vec<u32> = if !link_objects.is_empty() {
                let objs: Result<Vec<Vec<u8>>> = link_objects.iter().map(|p| std::fs::read(p).with_context(|| format!("reading {}", p.display()))).collect();
                zkb_ca::link_witness(flags.as_deref().unwrap_or("").as_bytes(), &objs?)
            } else if let Some(fl) = flags {
                load_tree(&src)?.tu_witness(fl.as_bytes())
            } else {
                witness.iter().map(|s| parse_word(s)).collect::<Result<Vec<u32>>>()?
            };
            let t0 = std::time::Instant::now();
            let elf_bytes = std::fs::read(&elf)?;
            let (image, res) = if profile {
                let image = zkb_vm::elf::load_elf(&elf_bytes)?;
                let mut vm = zkb_vm::Vm::new(&image, w);
                let mut counts: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
                while !vm.halted {
                    let pc = vm.pc;
                    vm.step().map_err(|t| anyhow::anyhow!("trap: {t}"))?;
                    *counts.entry(pc).or_default() += 1;
                }
                // symbolize with llvm-nm
                let nm = std::process::Command::new("llvm-nm").args(["--numeric-sort", "--defined-only"]).arg(&elf).output()?;
                let mut syms: Vec<(u32, String)> = String::from_utf8_lossy(&nm.stdout)
                    .lines()
                    .filter_map(|l| {
                        let mut it = l.split_whitespace();
                        let a = u32::from_str_radix(it.next()?, 16).ok()?;
                        let t = it.next()?;
                        let n = it.next()?;
                        if t == "T" || t == "t" { Some((a, n.to_string())) } else { None }
                    })
                    .collect();
                syms.sort();
                // (cycles, entries) per function; entries = executions of the first instruction
                let mut per_fn: std::collections::HashMap<String, (u64, u64)> = std::collections::HashMap::new();
                for (pc, c) in &counts {
                    let idx = syms.partition_point(|(a, _)| *a <= *pc);
                    let (start, name) = if idx == 0 { (0, "?".to_string()) } else { (syms[idx - 1].0, syms[idx - 1].1.clone()) };
                    let e = per_fn.entry(name).or_default();
                    e.0 += c;
                    if *pc == start {
                        e.1 += c;
                    }
                }
                let mut v: Vec<(String, (u64, u64))> = per_fn.into_iter().collect();
                v.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
                println!("top functions by cycles (self):");
                println!("  {:>12}  {:>6}  {:>10}  {:>8}  function", "cycles", "%", "entries", "cyc/call");
                for (n, (c, calls)) in v.iter().take(30) {
                    let per = if *calls > 0 { format!("{:.1}", *c as f64 / *calls as f64) } else { "-".into() };
                    println!("  {:>12}  {:5.1}%  {:>10}  {:>8}  {}", c, *c as f64 * 100.0 / vm.cycles as f64, calls, per, n);
                }
                println!("final memory state: {} nonzero words ({} KB)", vm.state_nonzero().len(), vm.state_nonzero().len() * 4 / 1024);
                let res = zkb_vm::ExecutionResult { journal: vm.journal.clone(), output: vm.output.clone(), exit_code: vm.exit_code, cycles: vm.cycles, records: vec![] };
                (image, res)
            } else {
                zkb_vm::execute(&elf_bytes, w, zkb_proof::MAX_CYCLES, false)?
            };
            println!("executed in {:?}", t0.elapsed());
            println!("image digest: {}", hex(&image.digest()));
            println!("cycles: {}  exit code: {}", res.cycles, res.exit_code);
            println!("journal: {:?}", res.journal.iter().map(|w| format!("{w:#x}")).collect::<Vec<_>>());
            if !res.output.is_empty() {
                let out = zkb_ca::unpack_output(&res.output)?;
                println!("output: {} bytes", out.len());
                if let Some(p) = output {
                    std::fs::write(&p, &out)?;
                    println!("output written to {}", p.display());
                } else {
                    println!("{}", String::from_utf8_lossy(&out[..out.len().min(2000)]));
                }
            }
        }
        Cmd::ProveTu { tu_guest, link_guest, flags, src, cert, object } => {
            let tc = load_toolchain(&tu_guest, &link_guest)?;
            let tree = load_tree(&src)?;
            let mut prover = Prover::new(config);
            prover.verbose = cli.verbose;
            prover.max_log_n = cli.segment_log;
            let t0 = std::time::Instant::now();
            let (c, obj) = prover.prove_tu(&tc, &tree, flags.as_bytes())?;
            std::fs::write(&object, &obj)?;
            save(&cert.to_string_lossy(), &c)?;
            println!("object certificate written to {} ({:?})", cert.display(), t0.elapsed());
            println!("  source_root    = {}", hex(&c.source_root));
            println!("  flags_root     = {}", hex(&c.flags_root));
            println!("  toolchain_root = {}", hex(&c.toolchain_root));
            println!("  object_hash    = {}", hex(&c.object_hash));
            println!("  proof          = {} KB, {} segment(s)", c.proof.proof.size_bytes() / 1024, c.segments());
        }
        Cmd::Link { tu_guest, link_guest, flags, objects, cert, output } => {
            let tc = load_toolchain(&tu_guest, &link_guest)?;
            let objs: Result<Vec<Vec<u8>>> = objects.iter().map(|p| std::fs::read(p).with_context(|| format!("reading {}", p.display()))).collect();
            let mut prover = Prover::new(config);
            prover.verbose = cli.verbose;
            prover.max_log_n = cli.segment_log;
            let t0 = std::time::Instant::now();
            let (c, exe) = prover.prove_link(&tc, &objs?, flags.as_bytes())?;
            std::fs::write(&output, &exe)?;
            save(&cert.to_string_lossy(), &c)?;
            println!("link certificate written to {} ({:?})", cert.display(), t0.elapsed());
            println!("  inputs_root    = {}", hex(&c.inputs_root));
            println!("  flags_root     = {}", hex(&c.flags_root));
            println!("  toolchain_root = {}", hex(&c.toolchain_root));
            println!("  output_hash    = {}", hex(&c.output_hash));
        }
        Cmd::Bundle { objects, link, out } => {
            // streaming container: member certificates are copied, not held in memory together
            let paths: Vec<String> = objects.iter().map(|p| p.to_string_lossy().to_string()).collect();
            let index = zkb_proof::write_bundle(&out.to_string_lossy(), &paths, &link.to_string_lossy())?;
            println!("build certificate written to {} ({} objects + link)", out.display(), index.objects.len());
            println!("  build_root = {}", hex(&index.build_root()));
        }
        Cmd::Verify { cert, tu_guest, link_guest, binary } => {
            let tc = load_toolchain(&tu_guest, &link_guest)?;
            let air = zkb_air::Rv32Air::new();
            let t0 = std::time::Instant::now();
            let path = cert.to_string_lossy().to_string();
            let s = if zkb_proof::is_bundle(&path) {
                zkb_proof::verify_bundle(&path, &air, &tc, |msg| eprintln!("  {msg}"))?
            } else {
                let b: BuildCertificate = load(&path)?;
                b.verify(&air, &tc)?
            };
            println!("VERIFIED in {:?} ({} proof segments)", t0.elapsed(), s.segments);
            println!("  toolchain_root = {}", hex(&s.toolchain_root));
            println!("  output_hash    = {}", hex(&s.output_hash));
            println!("  build_root     = {}", hex(&s.build_root));
            for (i, (sr, fr, oh)) in s.objects.iter().enumerate() {
                println!("  object #{i}: source_root={} flags_root={} object_hash={}", hex(sr), hex(fr), hex(oh));
            }
            if let Some(bin) = binary {
                let h = zkb_ca::v0_executable_hash(&std::fs::read(&bin)?)?;
                if h == s.output_hash {
                    println!("  binary {} matches the certified output", bin.display());
                } else {
                    bail!("binary {} does NOT match the certified output ({})", bin.display(), hex(&h));
                }
            }
        }
        Cmd::Info { cert } => {
            let path = cert.to_string_lossy().to_string();
            if zkb_proof::is_bundle(&path) {
                let ix = zkb_proof::read_bundle_index(&path)?;
                println!("build certificate: {} objects, build_root = {}", ix.objects.len(), hex(&ix.build_root()));
                for (i, (sr, _fr, oh, nf)) in ix.object_statements.iter().enumerate() {
                    println!("  object #{i}: source_root={} object_hash={} n_files={}", hex(sr), hex(oh), nf);
                }
                println!("  link: inputs_root={} output_hash={}", hex(&ix.link_statement.0), hex(&ix.link_statement.2));
                println!("  toolchain_root={}", hex(&ix.toolchain_root));
            } else if let Ok(b) = load::<BuildCertificate>(&path) {
                println!("build certificate: {} objects, build_root = {}", b.objects.len(), hex(&b.root()));
                for (i, o) in b.objects.iter().enumerate() {
                    println!("  object #{i}: source_root={} object_hash={} n_files={}", hex(&o.source_root), hex(&o.object_hash), o.n_files);
                }
                println!("  link: inputs_root={} output_hash={}", hex(&b.link.inputs_root), hex(&b.link.output_hash));
                println!("  toolchain_root={}", hex(&b.link.toolchain_root));
            } else if let Ok(o) = load::<ObjectCertificate>(&path) {
                println!("object certificate: source_root={} flags_root={} object_hash={} toolchain_root={} n_files={}", hex(&o.source_root), hex(&o.flags_root), hex(&o.object_hash), hex(&o.toolchain_root), o.n_files);
            } else if let Ok(l) = load::<LinkCertificate>(&path) {
                println!("link certificate: inputs_root={} flags_root={} output_hash={} toolchain_root={} n_objects={}", hex(&l.inputs_root), hex(&l.flags_root), hex(&l.output_hash), hex(&l.toolchain_root), l.n_objects);
            } else {
                bail!("unrecognized certificate file");
            }
        }
        Cmd::Toolchain { tu_guest, link_guest } => {
            let tc = load_toolchain(&tu_guest, &link_guest)?;
            println!("tu image     = {}", hex(&tc.tu_image.digest()));
            println!("link image   = {}", hex(&tc.link_image.digest()));
            println!("toolchain_root = {}", hex(&tc.root()));
        }
    }
    Ok(())
}
