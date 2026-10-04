//! Build certificates: ZK-attested compilation units and links, and their
//! aggregation into a build certificate.
//!
//! ```text
//! ObjectCertificate   "object O was produced by toolchain T with flags F from a
//!                      source tree committed as R"            (source stays private)
//! LinkCertificate     "executable E was produced by toolchain T with flags F from
//!                      objects with hashes [O_i]"
//! BuildCertificate    objects + link, cross-checked: link inputs == object hashes
//! ```
//!
//! The toolchain *is* the guest program image: `toolchain_root` commits to
//! the exact RV32IM images of the compile and link guests. The verifier must
//! hold (or trust a digest of) those images — that is the "known compiler C"
//! of the design note.
//!
//! Guest contract (any toolchain implementing it works unchanged):
//!
//! ```text
//! TU guest    witness: flags | n | (name | content)*
//!             journal: source_root[8] | flags_root[8] | object_hash[8] | n
//!             output : the object bytes (private stream; the host checks its hash)
//! link guest  witness: flags | n | object*
//!             journal: inputs_root[8] | flags_root[8] | output_hash[8] | n
//!             output : the linked artifact bytes
//! ```

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use zkb_air::{ExecutionProof, Rv32Air};
use zkb_ca::{flags_root, inputs_root_from_hashes, journal_hash, content_hash, unpack_output, Hash, SourceTree};
use zkb_stark::prover::ProvingKey;
use zkb_stark::StarkConfig;
use zkb_vm::{elf::load_elf, Image};

pub const MAX_CYCLES: u64 = 1 << 34;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Toolchain {
    pub tu_image: Image,
    pub link_image: Image,
}

impl Toolchain {
    pub fn load(tu_elf: &[u8], link_elf: &[u8]) -> Result<Toolchain> {
        Ok(Toolchain { tu_image: load_elf(tu_elf).context("tu guest")?, link_image: load_elf(link_elf).context("link guest")? })
    }
    /// Commitment to the whole toolchain (both guest images).
    pub fn root(&self) -> Hash {
        let mut h = blake3::Hasher::new();
        h.update(b"zkb-toolchain-v2");
        h.update(&self.tu_image.digest());
        h.update(&self.link_image.digest());
        *h.finalize().as_bytes()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ObjectCertificate {
    pub source_root: Hash,
    pub flags_root: Hash,
    pub toolchain_root: Hash,
    pub object_hash: Hash,
    pub n_files: u32,
    pub proof: ExecutionProof,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinkCertificate {
    pub inputs_root: Hash,
    pub flags_root: Hash,
    pub toolchain_root: Hash,
    pub output_hash: Hash,
    pub n_objects: u32,
    pub proof: ExecutionProof,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BuildCertificate {
    pub objects: Vec<ObjectCertificate>,
    pub link: LinkCertificate,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BuildSummary {
    pub toolchain_root: Hash,
    pub output_hash: Hash,
    pub link_flags_root: Hash,
    pub objects: Vec<(Hash, Hash, Hash)>, // (source_root, flags_root, object_hash)
    pub build_root: Hash,
    pub segments: usize,
}

fn hash_fields(tag: &[u8], fields: &[&[u8]]) -> Hash {
    let mut h = blake3::Hasher::new();
    h.update(tag);
    for f in fields {
        h.update(f);
    }
    *h.finalize().as_bytes()
}

impl ObjectCertificate {
    /// Statement digest (without the proof): what a signature or a parent
    /// aggregation would commit to.
    pub fn digest(&self) -> Hash {
        hash_fields(b"zkb-objcert-v1", &[&self.source_root, &self.flags_root, &self.toolchain_root, &self.object_hash, &self.n_files.to_le_bytes()])
    }

    pub fn verify(&self, air: &Rv32Air, tc: &Toolchain) -> Result<()> {
        ensure!(self.toolchain_root == tc.root(), "object certificate: toolchain mismatch");
        zkb_air::verify_execution(air, &self.proof, &tc.tu_image).context("object certificate: execution proof")?;
        let j = &self.proof.claim.journal;
        ensure!(j.len() == 25, "object certificate: malformed journal");
        ensure!(journal_hash(&j[0..8])? == self.source_root, "object certificate: source_root mismatch");
        ensure!(journal_hash(&j[8..16])? == self.flags_root, "object certificate: flags_root mismatch");
        ensure!(journal_hash(&j[16..24])? == self.object_hash, "object certificate: object_hash mismatch");
        ensure!(j[24] == self.n_files, "object certificate: n_files mismatch");
        Ok(())
    }

    pub fn segments(&self) -> usize {
        self.proof.proof.segments.len()
    }
}

impl LinkCertificate {
    pub fn digest(&self) -> Hash {
        hash_fields(b"zkb-linkcert-v1", &[&self.inputs_root, &self.flags_root, &self.toolchain_root, &self.output_hash, &self.n_objects.to_le_bytes()])
    }

    pub fn verify(&self, air: &Rv32Air, tc: &Toolchain) -> Result<()> {
        ensure!(self.toolchain_root == tc.root(), "link certificate: toolchain mismatch");
        zkb_air::verify_execution(air, &self.proof, &tc.link_image).context("link certificate: execution proof")?;
        let j = &self.proof.claim.journal;
        ensure!(j.len() == 25, "link certificate: malformed journal");
        ensure!(journal_hash(&j[0..8])? == self.inputs_root, "link certificate: inputs_root mismatch");
        ensure!(journal_hash(&j[8..16])? == self.flags_root, "link certificate: flags_root mismatch");
        ensure!(journal_hash(&j[16..24])? == self.output_hash, "link certificate: output_hash mismatch");
        ensure!(j[24] == self.n_objects, "link certificate: n_objects mismatch");
        Ok(())
    }
}

impl BuildCertificate {
    pub fn root(&self) -> Hash {
        let mut h = blake3::Hasher::new();
        h.update(b"zkb-buildcert-v2");
        h.update(&(self.objects.len() as u32).to_le_bytes());
        for o in &self.objects {
            h.update(&o.digest());
        }
        h.update(&self.link.digest());
        *h.finalize().as_bytes()
    }

    /// Verify every proof and the cross-links between them.
    pub fn verify(&self, air: &Rv32Air, tc: &Toolchain) -> Result<BuildSummary> {
        let mut segments = 0;
        for (i, o) in self.objects.iter().enumerate() {
            o.verify(air, tc).with_context(|| format!("object #{i}"))?;
            segments += o.segments();
        }
        self.link.verify(air, tc)?;
        segments += self.link.proof.proof.segments.len();
        let hashes: Vec<Hash> = self.objects.iter().map(|o| o.object_hash).collect();
        ensure!(self.link.n_objects as usize == hashes.len(), "link consumed a different number of objects");
        ensure!(inputs_root_from_hashes(&hashes) == self.link.inputs_root, "link inputs do not match the certified objects");
        Ok(BuildSummary {
            toolchain_root: tc.root(),
            output_hash: self.link.output_hash,
            link_flags_root: self.link.flags_root,
            objects: self.objects.iter().map(|o| (o.source_root, o.flags_root, o.object_hash)).collect(),
            build_root: self.root(),
            segments,
        })
    }
}

/// Stateful prover: caches proving keys per trace size.
pub struct Prover {
    pub air: Rv32Air,
    pub config: StarkConfig,
    /// Largest trace (log2 rows) per segment; longer executions are split.
    pub max_log_n: u32,
    keys: HashMap<u32, ProvingKey>,
    pub verbose: bool,
}

pub struct Proven {
    pub proof: ExecutionProof,
    pub output: Vec<u8>,
    pub cycles: u64,
    pub segments: usize,
}

impl Prover {
    pub fn new(config: StarkConfig) -> Prover {
        Prover { air: Rv32Air::new(), config, max_log_n: 17, keys: HashMap::new(), verbose: false }
    }

    fn ensure_key(&mut self, log_n: u32) {
        if !self.keys.contains_key(&log_n) {
            let t0 = std::time::Instant::now();
            self.keys.insert(log_n, zkb_air::setup(&self.air, log_n, self.config));
            if self.verbose {
                eprintln!("[setup] proving key for 2^{log_n} rows in {:?}", t0.elapsed());
            }
        }
    }

    /// Execute `image` on `witness` and prove it.
    pub fn run_and_prove(&mut self, image: &Image, witness: Vec<u32>, what: &str) -> Result<Proven> {
        let t0 = std::time::Instant::now();
        let log_n = zkb_air::choose_log_n(image, &witness, self.max_log_n, MAX_CYCLES, &self.config).with_context(|| format!("{what}: guest execution"))?;
        self.ensure_key(log_n);
        let pk = &self.keys[&log_n];
        if self.verbose {
            eprintln!("[{what}] executed in {:?}; proving with 2^{log_n}-row segments", t0.elapsed());
        }
        let t0 = std::time::Instant::now();
        let (ep, out) = zkb_air::prove_execution(&self.air, pk, image, &witness, MAX_CYCLES)?;
        if self.verbose {
            eprintln!("[{what}] {} cycles in {} segment(s) proved in {:?} ({} KB)", out.cycles, out.segments, t0.elapsed(), ep.proof.size_bytes() / 1024);
        }
        let output = unpack_output(&out.output)?;
        Ok(Proven { proof: ep, output, cycles: out.cycles, segments: out.segments })
    }

    /// Prove one translation unit. Returns the certificate and the object
    /// bytes produced by the toolchain inside the VM.
    pub fn prove_tu(&mut self, tc: &Toolchain, tree: &SourceTree, flags: &[u8]) -> Result<(ObjectCertificate, Vec<u8>)> {
        let witness = tree.tu_witness(flags);
        let pr = self.run_and_prove(&tc.tu_image, witness, "tu")?;
        let j = &pr.proof.claim.journal;
        ensure!(j.len() == 25, "tu guest produced a malformed journal");
        let cert = ObjectCertificate {
            source_root: journal_hash(&j[0..8])?,
            flags_root: journal_hash(&j[8..16])?,
            toolchain_root: tc.root(),
            object_hash: journal_hash(&j[16..24])?,
            n_files: j[24],
            proof: pr.proof,
        };
        ensure!(cert.source_root == tree.root(), "guest/host source_root disagree");
        ensure!(cert.flags_root == flags_root(flags), "guest/host flags_root disagree");
        ensure!(cert.object_hash == content_hash(&pr.output), "guest journaled a hash that does not match its output");
        Ok((cert, pr.output))
    }

    pub fn prove_link(&mut self, tc: &Toolchain, objects: &[Vec<u8>], flags: &[u8]) -> Result<(LinkCertificate, Vec<u8>)> {
        let witness = zkb_ca::link_witness(flags, objects);
        let pr = self.run_and_prove(&tc.link_image, witness, "link")?;
        let j = &pr.proof.claim.journal;
        ensure!(j.len() == 25, "link guest produced a malformed journal");
        let cert = LinkCertificate {
            inputs_root: journal_hash(&j[0..8])?,
            flags_root: journal_hash(&j[8..16])?,
            toolchain_root: tc.root(),
            output_hash: journal_hash(&j[16..24])?,
            n_objects: j[24],
            proof: pr.proof,
        };
        ensure!(cert.inputs_root == zkb_ca::inputs_root(objects), "guest/host inputs_root disagree");
        ensure!(cert.output_hash == zkb_ca::v0_executable_hash(&pr.output)?, "guest journaled a hash that does not match its output");
        Ok((cert, pr.output))
    }
}

pub fn save<T: Serialize>(path: &str, t: &T) -> Result<()> {
    let bytes = bincode::serialize(t)?;
    std::fs::write(path, bytes).with_context(|| format!("writing {path}"))?;
    Ok(())
}

pub fn load<T: for<'de> Deserialize<'de>>(path: &str) -> Result<T> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {path}"))?;
    Ok(bincode::deserialize(&bytes)?)
}

// ---------------------------------------------------------------------------
// Streaming build-certificate container. A build with thousands of segment
// proofs is gigabytes of certificates; this container holds the member
// certificates as raw blobs behind an index so that bundling copies files
// and verification deserializes one member at a time.
//
//   "ZKBUNDL3" | u64 index_len | bincode(BundleIndex) | blob_0 | blob_1 | ...

const BUNDLE_MAGIC: &[u8; 8] = b"ZKBUNDL3";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BundleMember {
    pub offset: u64,
    pub len: u64,
    /// Statement digest of the member certificate (ObjectCertificate::digest / LinkCertificate::digest).
    pub digest: Hash,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BundleIndex {
    pub objects: Vec<BundleMember>,
    pub link: BundleMember,
    /// Statement summaries so `info` needs no proof parsing.
    pub object_statements: Vec<(Hash, Hash, Hash, u32)>, // source_root, flags_root, object_hash, n_files
    pub link_statement: (Hash, Hash, Hash, u32),         // inputs_root, flags_root, output_hash, n_objects
    pub toolchain_root: Hash,
}

impl BundleIndex {
    /// Same value as `BuildCertificate::root()` of the equivalent in-memory certificate.
    pub fn build_root(&self) -> Hash {
        let mut h = blake3::Hasher::new();
        h.update(b"zkb-buildcert-v2");
        h.update(&(self.objects.len() as u32).to_le_bytes());
        for o in &self.objects {
            h.update(&o.digest);
        }
        h.update(&self.link.digest);
        *h.finalize().as_bytes()
    }
}

/// Write a streaming build certificate from member certificate files
/// (each is deserialized once, alone, for its statement digest).
pub fn write_bundle(out: &str, object_paths: &[String], link_path: &str) -> Result<BundleIndex> {
    use std::io::Write;
    let mut object_statements = Vec::new();
    let mut toolchain_root = None;
    let mut blobs: Vec<(String, Hash)> = Vec::new();
    for p in object_paths {
        let o: ObjectCertificate = load(p)?;
        ensure!(toolchain_root.map_or(true, |t| t == o.toolchain_root), "object certificates use different toolchains");
        toolchain_root = Some(o.toolchain_root);
        object_statements.push((o.source_root, o.flags_root, o.object_hash, o.n_files));
        blobs.push((p.clone(), o.digest()));
    }
    let l: LinkCertificate = load(link_path)?;
    ensure!(toolchain_root.map_or(true, |t| t == l.toolchain_root), "link certificate uses a different toolchain");
    let link_statement = (l.inputs_root, l.flags_root, l.output_hash, l.n_objects);
    let link_digest = l.digest();
    let toolchain_root = toolchain_root.unwrap_or(l.toolchain_root);
    drop(l);
    // lay out: header + index (fixed size once computed) + blobs
    let mut lens = Vec::new();
    for (p, _) in &blobs {
        lens.push(std::fs::metadata(p)?.len());
    }
    let link_len = std::fs::metadata(link_path)?.len();
    // index size does not depend on offsets' values (fixed-width u64), so build it with zeros first
    let make_index = |offsets: &[u64], link_off: u64| BundleIndex {
        objects: blobs.iter().zip(offsets).zip(&lens).map(|(((_, d), &o), &n)| BundleMember { offset: o, len: n, digest: *d }).collect(),
        link: BundleMember { offset: link_off, len: link_len, digest: link_digest },
        object_statements: object_statements.clone(),
        link_statement,
        toolchain_root,
    };
    let probe = bincode::serialize(&make_index(&vec![0; blobs.len()], 0))?;
    let mut off = 8 + 8 + probe.len() as u64;
    let mut offsets = Vec::new();
    for &n in &lens {
        offsets.push(off);
        off += n;
    }
    let link_off = off;
    let index = make_index(&offsets, link_off);
    let index_bytes = bincode::serialize(&index)?;
    ensure!(index_bytes.len() == probe.len(), "index size changed");
    let mut f = std::io::BufWriter::new(std::fs::File::create(out).with_context(|| format!("creating {out}"))?);
    f.write_all(BUNDLE_MAGIC)?;
    f.write_all(&(index_bytes.len() as u64).to_le_bytes())?;
    f.write_all(&index_bytes)?;
    for (p, _) in &blobs {
        std::io::copy(&mut std::fs::File::open(p)?, &mut f)?;
    }
    std::io::copy(&mut std::fs::File::open(link_path)?, &mut f)?;
    f.flush()?;
    Ok(index)
}

pub fn is_bundle(path: &str) -> bool {
    use std::io::Read;
    let mut magic = [0u8; 8];
    std::fs::File::open(path).and_then(|mut f| f.read_exact(&mut magic)).is_ok() && &magic == BUNDLE_MAGIC
}

pub fn read_bundle_index(path: &str) -> Result<BundleIndex> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).with_context(|| format!("opening {path}"))?;
    let mut hdr = [0u8; 16];
    f.read_exact(&mut hdr)?;
    ensure!(&hdr[..8] == BUNDLE_MAGIC, "not a build certificate bundle");
    let n = u64::from_le_bytes(hdr[8..16].try_into().unwrap()) as usize;
    let mut buf = vec![0u8; n];
    f.read_exact(&mut buf)?;
    Ok(bincode::deserialize(&buf)?)
}

fn read_member<T: for<'de> Deserialize<'de>>(path: &str, m: &BundleMember) -> Result<T> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path)?;
    f.seek(SeekFrom::Start(m.offset))?;
    let mut buf = vec![0u8; m.len as usize];
    f.read_exact(&mut buf)?;
    Ok(bincode::deserialize(&buf)?)
}

/// Verify a streaming build certificate, one member certificate in memory at a time.
pub fn verify_bundle(path: &str, air: &Rv32Air, tc: &Toolchain, mut progress: impl FnMut(&str)) -> Result<BuildSummary> {
    let index = read_bundle_index(path)?;
    ensure!(index.toolchain_root == tc.root(), "bundle: toolchain mismatch");
    let mut segments = 0;
    let mut hashes = Vec::new();
    let mut objects = Vec::new();
    for (i, m) in index.objects.iter().enumerate() {
        let o: ObjectCertificate = read_member(path, m).with_context(|| format!("object #{i}"))?;
        ensure!(o.digest() == m.digest, "object #{i}: statement digest does not match the index");
        o.verify(air, tc).with_context(|| format!("object #{i}"))?;
        segments += o.segments();
        hashes.push(o.object_hash);
        objects.push((o.source_root, o.flags_root, o.object_hash));
        progress(&format!("object #{i}: {} segments verified", o.segments()));
    }
    let l: LinkCertificate = read_member(path, &index.link).context("link certificate")?;
    ensure!(l.digest() == index.link.digest, "link: statement digest does not match the index");
    l.verify(air, tc)?;
    segments += l.proof.proof.segments.len();
    progress(&format!("link: {} segments verified", l.proof.proof.segments.len()));
    ensure!(l.n_objects as usize == hashes.len(), "link consumed a different number of objects");
    ensure!(inputs_root_from_hashes(&hashes) == l.inputs_root, "link inputs do not match the certified objects");
    Ok(BuildSummary { toolchain_root: tc.root(), output_hash: l.output_hash, link_flags_root: l.flags_root, objects, build_root: index.build_root(), segments })
}
