//! Content addressing shared between the host and the guests.
//!
//! Hashing scheme (BLAKE3, little-endian length prefixes), identical to
//! `guest/tu-guest.c` and `guest/link-guest.c`:
//!
//! ```text
//! leaf_i       = H(0x00 || len(name) || name || len(content) || content)
//! source_root  = H(0x01 || n || leaf_0 || ... || leaf_{n-1})
//! flags_root   = H(0x02 || flags)
//! inputs_root  = H(0x03 || n || H(obj_0) || ... || H(obj_{n-1}))
//! object       = "ZKOBJ0" || len(flags) || flags || content_0 || ...   (v0 toolchain)
//! exe          = "ZKEXE1" || len(flags) || flags || n || (len || obj)*      (v0 toolchain; output_hash = v0_executable_hash)
//! ```

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use std::path::Path;

pub type Hash = [u8; 32];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceFile {
    pub name: String,
    pub content: Vec<u8>,
}

/// A canonical, ordered source manifest (what the TU guest consumes).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct SourceTree {
    pub files: Vec<SourceFile>,
}

/// The commitment hash of the whole system (guests and host alike): BLAKE3.
/// (It replaced SHA-256, which cost ~63 cycles/byte inside the RV32IM guest
/// against BLAKE3's ~22; `object_hash`/`output_hash` are `b3sum` values.)
pub fn content_hash(data: &[u8]) -> Hash {
    *blake3::hash(data).as_bytes()
}

fn h_u32(h: &mut blake3::Hasher, v: u32) {
    h.update(&v.to_le_bytes());
}

impl SourceTree {
    /// Load every regular file under `dir` (recursively), sorted by relative path.
    pub fn from_dir(dir: &Path) -> Result<SourceTree> {
        let mut paths = Vec::new();
        fn walk(base: &Path, d: &Path, out: &mut Vec<String>) -> Result<()> {
            for entry in std::fs::read_dir(d).with_context(|| format!("reading {}", d.display()))? {
                let e = entry?;
                let p = e.path();
                if p.is_dir() {
                    walk(base, &p, out)?;
                } else if p.is_file() {
                    let rel = p.strip_prefix(base)?.to_string_lossy().replace('\\', "/");
                    out.push(rel);
                }
            }
            Ok(())
        }
        walk(dir, dir, &mut paths)?;
        paths.sort();
        let mut files = Vec::new();
        for rel in paths {
            let content = std::fs::read(dir.join(&rel))?;
            files.push(SourceFile { name: rel, content });
        }
        Ok(SourceTree { files })
    }

    pub fn from_files(paths: &[String]) -> Result<SourceTree> {
        let mut files = Vec::new();
        for p in paths {
            let content = std::fs::read(p).with_context(|| format!("reading {p}"))?;
            let name = Path::new(p).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| p.clone());
            files.push(SourceFile { name, content });
        }
        files.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(SourceTree { files })
    }

    pub fn leaf(f: &SourceFile) -> Hash {
        let mut h = blake3::Hasher::new();
        h.update(&[0u8]);
        h_u32(&mut h, f.name.len() as u32);
        h.update(f.name.as_bytes());
        h_u32(&mut h, f.content.len() as u32);
        h.update(&f.content);
        *h.finalize().as_bytes()
    }

    pub fn root(&self) -> Hash {
        let mut h = blake3::Hasher::new();
        h.update(&[1u8]);
        h_u32(&mut h, self.files.len() as u32);
        for f in &self.files {
            h.update(&Self::leaf(f));
        }
        *h.finalize().as_bytes()
    }

    /// Witness stream for `tu-guest`: flags | n | (name | content)*
    pub fn tu_witness(&self, flags: &[u8]) -> Vec<u32> {
        let mut w = pack_bytes(flags);
        w.push(self.files.len() as u32);
        for f in &self.files {
            w.extend(pack_bytes(f.name.as_bytes()));
            w.extend(pack_bytes(&f.content));
        }
        w
    }

    /// What the v0 toolchain ("zkcc0") produces for this tree.
    pub fn v0_object(&self, flags: &[u8]) -> Vec<u8> {
        let mut o = b"ZKOBJ0".to_vec();
        o.extend_from_slice(&(flags.len() as u32).to_le_bytes());
        o.extend_from_slice(flags);
        for f in &self.files {
            o.extend_from_slice(&f.content);
        }
        o
    }
}

pub fn flags_root(flags: &[u8]) -> Hash {
    let mut h = blake3::Hasher::new();
    h.update(&[2u8]);
    h.update(flags);
    *h.finalize().as_bytes()
}

pub fn inputs_root(objects: &[Vec<u8>]) -> Hash {
    let mut h = blake3::Hasher::new();
    h.update(&[3u8]);
    h_u32(&mut h, objects.len() as u32);
    for o in objects {
        h.update(&content_hash(o));
    }
    *h.finalize().as_bytes()
}

pub fn inputs_root_from_hashes(hashes: &[Hash]) -> Hash {
    let mut h = blake3::Hasher::new();
    h.update(&[3u8]);
    h_u32(&mut h, hashes.len() as u32);
    for x in hashes {
        h.update(x);
    }
    *h.finalize().as_bytes()
}

pub fn link_witness(flags: &[u8], objects: &[Vec<u8>]) -> Vec<u32> {
    let mut w = pack_bytes(flags);
    w.push(objects.len() as u32);
    for o in objects {
        w.extend(pack_bytes(o));
    }
    w
}

/// The v0 executable: `"ZKEXE1" | u32 flags_len | flags | u32 n | (u32 len | object)*`.
pub fn v0_executable(flags: &[u8], objects: &[Vec<u8>]) -> Vec<u8> {
    let mut o = b"ZKEXE1".to_vec();
    o.extend_from_slice(&(flags.len() as u32).to_le_bytes());
    o.extend_from_slice(flags);
    o.extend_from_slice(&(objects.len() as u32).to_le_bytes());
    for obj in objects {
        o.extend_from_slice(&(obj.len() as u32).to_le_bytes());
        o.extend_from_slice(obj);
    }
    o
}

/// Commitment to a v0 executable: BLAKE3 over its header and the hash of
/// every object (`"ZKEXE1" | u32 flags_len | flags | u32 n | (u32 len | blake3(object))*`).
/// Binding to the executable bytes, and computable by the link guest in one
/// pass over each object (the per-object hashes are also its `inputs_root`).
pub fn v0_executable_hash_parts(flags: &[u8], objects: &[(u32, Hash)]) -> Hash {
    let mut h = blake3::Hasher::new();
    h.update(b"ZKEXE1");
    h.update(&(flags.len() as u32).to_le_bytes());
    h.update(flags);
    h.update(&(objects.len() as u32).to_le_bytes());
    for (len, oh) in objects {
        h.update(&len.to_le_bytes());
        h.update(oh);
    }
    *h.finalize().as_bytes()
}

/// [`v0_executable_hash_parts`] of an executable's bytes (parses the v0 format).
pub fn v0_executable_hash(exe: &[u8]) -> Result<Hash> {
    let rd_u32 = |p: &mut usize| -> Result<u32> {
        anyhow::ensure!(*p + 4 <= exe.len(), "truncated v0 executable");
        let v = u32::from_le_bytes(exe[*p..*p + 4].try_into().unwrap());
        *p += 4;
        Ok(v)
    };
    anyhow::ensure!(exe.len() >= 6 && &exe[..6] == b"ZKEXE1", "not a v0 executable");
    let mut p = 6;
    let flen = rd_u32(&mut p)? as usize;
    anyhow::ensure!(p + flen <= exe.len(), "truncated v0 executable");
    let flags = &exe[p..p + flen];
    p += flen;
    let n = rd_u32(&mut p)? as usize;
    let mut parts = Vec::with_capacity(n);
    for _ in 0..n {
        let len = rd_u32(&mut p)? as usize;
        anyhow::ensure!(p + len <= exe.len(), "truncated v0 executable");
        parts.push((len as u32, content_hash(&exe[p..p + len])));
        p += len;
    }
    anyhow::ensure!(p == exe.len(), "trailing bytes in v0 executable");
    Ok(v0_executable_hash_parts(flags, &parts))
}

/// Length-prefixed little-endian word packing (matches `zk_read_bytes`).
pub fn pack_bytes(data: &[u8]) -> Vec<u32> {
    let mut out = vec![data.len() as u32];
    for chunk in data.chunks(4) {
        let mut w = [0u8; 4];
        w[..chunk.len()].copy_from_slice(chunk);
        out.push(u32::from_le_bytes(w));
    }
    out
}

/// Reassemble the private output stream: a sequence of [len][words...] chunks.
pub fn unpack_output(words: &[u32]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < words.len() {
        let len = words[i] as usize;
        i += 1;
        let nw = len.div_ceil(4);
        if i + nw > words.len() {
            bail!("truncated output stream");
        }
        for (k, w) in words[i..i + nw].iter().enumerate() {
            let b = w.to_le_bytes();
            let take = (len - 4 * k).min(4);
            out.extend_from_slice(&b[..take]);
        }
        i += nw;
    }
    Ok(out)
}

/// Read 8 journal words as a 32-byte hash (matches `zk_write_bytes32`).
pub fn journal_hash(words: &[u32]) -> Result<Hash> {
    if words.len() < 8 {
        bail!("journal too short for a hash");
    }
    let mut out = [0u8; 32];
    for i in 0..8 {
        out[4 * i..4 * i + 4].copy_from_slice(&words[i].to_le_bytes());
    }
    Ok(out)
}

pub fn hex(h: &Hash) -> String {
    h.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pack_roundtrip_len() {
        let w = pack_bytes(b"hello");
        assert_eq!(w[0], 5);
        assert_eq!(w.len(), 3);
    }
}
