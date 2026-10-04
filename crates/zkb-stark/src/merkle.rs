//! Binary Merkle trees over blake3, with row-leaves of field elements.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use zkb_field::Fp;

pub type Digest = [u8; 32];

/// Per-leaf salt source for hiding commitments. `salts16` may derive sixteen
/// salts at once (the prover does, with the 16-way blake3).
pub trait Salter: Sync {
    fn salt(&self, leaf: usize) -> Digest;
    fn salts16(&self, leaves: &[usize; 16]) -> [Digest; 16] {
        core::array::from_fn(|k| self.salt(leaves[k]))
    }
}

impl<F: Fn(usize) -> Digest + Sync> Salter for F {
    fn salt(&self, leaf: usize) -> Digest {
        self(leaf)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MerklePath {
    pub siblings: Vec<Digest>,
}

pub struct MerkleTree {
    /// layers[0] = leaves, layers.last() = [root]
    layers: Vec<Vec<Digest>>,
}

pub fn hash_leaf_fp(row: &[Fp]) -> Digest {
    hash_leaf_salted(row, None)
}

/// Leaf hash of a row of field elements with an optional 32-byte salt
/// (used in zero-knowledge mode to make the commitment hiding).
pub fn hash_leaf_salted(row: &[Fp], salt: Option<&[u8; 32]>) -> Digest {
    let mut buf = Vec::with_capacity(4 + row.len() * 4 + 36);
    serialize_leaf(row, salt, &mut buf);
    *blake3::hash(&buf).as_bytes()
}

/// Byte encoding of a leaf: "leaf" ‖ values (LE u64) ‖ ["salt" ‖ salt].
#[inline]
pub fn serialize_leaf(row: &[Fp], salt: Option<&[u8; 32]>, buf: &mut Vec<u8>) {
    buf.clear();
    buf.extend_from_slice(b"leaf");
    for v in row {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    if let Some(s) = salt {
        buf.extend_from_slice(b"salt");
        buf.extend_from_slice(s);
    }
}

pub fn hash_leaf_bytes(bytes: &[u8]) -> Digest {
    let mut h = blake3::Hasher::new();
    h.update(b"leaf");
    h.update(bytes);
    *h.finalize().as_bytes()
}

#[inline]
pub fn hash_node(l: &Digest, r: &Digest) -> Digest {
    let mut h = blake3::Hasher::new();
    h.update(b"node");
    h.update(l);
    h.update(r);
    *h.finalize().as_bytes()
}


/// One Merkle level: `out[i] = hash_node(prev[2i], prev[2i+1])`, sixteen
/// nodes at a time with the 16-way blake3 when available.
fn hash_level(prev: &[Digest]) -> Vec<Digest> {
    let n = prev.len() / 2;
    let mut out = vec![[0u8; 32]; n];
    const GROUP: usize = 16 * 16;
    out.par_chunks_mut(GROUP).enumerate().for_each(|(gi, chunk)| {
        let base = gi * GROUP;
        let mut k = 0;
        if crate::blake3x16::AVAILABLE {
            let mut lanes = crate::blake3x16::Lanes16::new(4 + 64);
            while k + 16 <= chunk.len() {
                for lane in 0..16 {
                    let i = base + k + lane;
                    lanes.put_bytes(lane, 0, b"node");
                    lanes.put_bytes(lane, 4, &prev[2 * i]);
                    lanes.put_bytes(lane, 36, &prev[2 * i + 1]);
                }
                let mut o = [[0u8; 32]; 16];
                crate::blake3x16::hash16(&lanes.words, 68, &mut o);
                chunk[k..k + 16].copy_from_slice(&o);
                k += 16;
            }
        }
        while k < chunk.len() {
            let i = base + k;
            chunk[k] = hash_node(&prev[2 * i], &prev[2 * i + 1]);
            k += 1;
        }
    });
    out
}

impl MerkleTree {
    pub fn from_leaves(leaves: Vec<Digest>) -> MerkleTree {
        assert!(leaves.len().is_power_of_two() && !leaves.is_empty());
        let mut layers = vec![leaves];
        while layers.last().unwrap().len() > 1 {
            let prev = layers.last().unwrap();
            let next = hash_level(prev);
            layers.push(next);
        }
        MerkleTree { layers }
    }

    /// Build a tree whose leaf `i` is the hash of `row_fn(i)`.
    pub fn from_rows<F, S>(n: usize, row_fn: F, salt_fn: S) -> MerkleTree
    where
        F: Fn(usize, &mut Vec<Fp>) + Sync,
        S: Fn(usize) -> Option<[u8; 32]> + Sync,
    {
        let leaves: Vec<Digest> = (0..n)
            .into_par_iter()
            .map_init(Vec::new, |buf, i| {
                buf.clear();
                row_fn(i, buf);
                let salt = salt_fn(i);
                hash_leaf_salted(buf, salt.as_ref())
            })
            .collect();
        MerkleTree::from_leaves(leaves)
    }

    /// Build a tree over `m = n << log_blowup` leaves from coset-major LDE
    /// columns: leaf `i` is the row at storage position `pos(i)`. Leaves are
    /// hashed in storage order in blocks so every column is read sequentially.
    pub fn from_coset_major_columns(cols: &[&[Fp]], n: usize, log_blowup: u32, salter: Option<&dyn Salter>) -> MerkleTree {
        let salt_fn = |i: usize| salter.map(|s| s.salt(i));
        let m = n << log_blowup;
        let b = 1usize << log_blowup;
        let block = 256;
        let width = cols.len();
        let salted = salter.is_some();
        let leaf_len = 4 + width * 4 + if salted { 36 } else { 0 };
        let mut leaves = vec![[0u8; 32]; m];
        leaves.par_chunks_mut(block).enumerate().for_each(|(bi, out)| {
            // storage positions [p0, p0+len): leaf i = (j * b + c) with p = c * n + j
            let p0 = bi * block;
            let len = out.len();
            let leaf_index = |k: usize| {
                let p = p0 + k;
                let c = p / n;
                let j = p % n;
                j * b + c
            };
            let mut k = 0;
            if crate::blake3x16::AVAILABLE {
                // Serialize sixteen rows straight into the transposed SIMD
                // layout ("leaf" ‖ values ‖ ["salt" ‖ salt]) and hash them together.
                let mut lanes = crate::blake3x16::Lanes16::new(leaf_len);
                while k + 16 <= len {
                    for lane in 0..16 {
                        lanes.put_bytes(lane, 0, b"leaf");
                    }
                    for (col, data) in cols.iter().enumerate() {
                        let src = &data[p0 + k..p0 + k + 16];
                        let mut v = [0u32; 16];
                        for lane in 0..16 {
                            v[lane] = src[lane].as_canonical();
                        }
                        lanes.put_u32x16(1 + col, &v);
                    }
                    if let Some(s) = salter {
                        let idx: [usize; 16] = core::array::from_fn(|lane| leaf_index(k + lane));
                        let salts = s.salts16(&idx);
                        for lane in 0..16 {
                            lanes.put_bytes(lane, 4 + width * 4, b"salt");
                            lanes.put_bytes(lane, 8 + width * 4, &salts[lane]);
                        }
                    }
                    let mut o = [[0u8; 32]; 16];
                    crate::blake3x16::hash16(&lanes.words, leaf_len, &mut o);
                    out[k..k + 16].copy_from_slice(&o);
                    k += 16;
                }
            }
            if k < len {
                let mut row = vec![Fp::ZERO; width];
                let mut buf = Vec::with_capacity(leaf_len);
                while k < len {
                    for (col, data) in cols.iter().enumerate() {
                        row[col] = data[p0 + k];
                    }
                    let salt = salt_fn(leaf_index(k));
                    serialize_leaf(&row, salt.as_ref(), &mut buf);
                    out[k] = *blake3::hash(&buf).as_bytes();
                    k += 1;
                }
            }
        });
        // permute storage order -> leaf order
        let mut ordered = vec![[0u8; 32]; m];
        for p in 0..m {
            let c = p / n;
            let j = p % n;
            ordered[j * b + c] = leaves[p];
        }
        MerkleTree::from_leaves(ordered)
    }

    pub fn root(&self) -> Digest {
        self.layers.last().unwrap()[0]
    }

    pub fn num_leaves(&self) -> usize {
        self.layers[0].len()
    }

    pub fn open(&self, mut idx: usize) -> MerklePath {
        let mut siblings = Vec::with_capacity(self.layers.len() - 1);
        for layer in &self.layers[..self.layers.len() - 1] {
            siblings.push(layer[idx ^ 1]);
            idx >>= 1;
        }
        MerklePath { siblings }
    }
}

pub fn verify_path(root: &Digest, mut idx: usize, leaf: Digest, path: &MerklePath, num_leaves: usize) -> bool {
    if num_leaves.trailing_zeros() as usize != path.siblings.len() || idx >= num_leaves {
        return false;
    }
    let mut cur = leaf;
    for s in &path.siblings {
        cur = if idx & 1 == 0 { hash_node(&cur, s) } else { hash_node(s, &cur) };
        idx >>= 1;
    }
    cur == *root
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_verify() {
        let n = 16;
        let t = MerkleTree::from_rows(
            n,
            |i, buf| {
                buf.push(Fp::new(i as u64));
                buf.push(Fp::new((i * 3) as u64));
            },
            |_| None,
        );
        for i in 0..n {
            let p = t.open(i);
            let leaf = hash_leaf_fp(&[Fp::new(i as u64), Fp::new((i * 3) as u64)]);
            assert!(verify_path(&t.root(), i, leaf, &p, n));
            let bad = hash_leaf_fp(&[Fp::new(i as u64 + 1)]);
            assert!(!verify_path(&t.root(), i, bad, &p, n));
        }
    }

    #[test]
    fn vectorized_leaves_match_scalar() {
        // coset-major columns of width w over n rows, blowup 4, with and without salts
        for (w, log_n, salted) in [(250usize, 7u32, true), (58, 8, false), (8, 9, true), (3, 6, false)] {
            let n = 1usize << log_n;
            let m = n * 4;
            let cols: Vec<Vec<Fp>> = (0..w).map(|c| (0..m).map(|p| Fp::new((p as u64 * 0x9E37_79B9 + c as u64 * 0x1234_5678_9ABC) % (zkb_field::P64 - 1) + 1)).collect()).collect();
            let salt = |i: usize| *blake3::hash(&(i as u64).to_le_bytes()).as_bytes();
            let salter: Option<&dyn Salter> = if salted { Some(&salt) } else { None };
            let col_slices: Vec<&[Fp]> = cols.iter().map(|c| c.as_slice()).collect();
            let fast = MerkleTree::from_coset_major_columns(&col_slices, n, 2, salter);
            let slow = MerkleTree::from_rows(
                m,
                |i, buf| {
                    let p = zkb_field::ntt::coset_major_pos(i, n, 2);
                    for c in 0..w {
                        buf.push(cols[c][p]);
                    }
                },
                |i| if salted { Some(salt(i)) } else { None },
            );
            assert_eq!(fast.root(), slow.root(), "w={w} log_n={log_n} salted={salted}");
            for i in [0usize, 1, 17, m - 1] {
                assert_eq!(fast.open(i), slow.open(i));
            }
        }
    }
}
