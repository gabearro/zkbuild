//! Sixteen-way BLAKE3 with AVX-512: hashes 16 equal-length messages at once,
//! one message per 32-bit lane, producing exactly `blake3::hash(msg)` for
//! each. Messages are supplied *transposed* (`words[w * 16 + lane]` is the
//! `w`-th little-endian u32 word of message `lane`), which lets the Merkle
//! leaf serializer write straight into the SIMD layout without any shuffles.
//!
//! This is the same algorithm the `blake3` crate runs internally for
//! multi-chunk inputs, applied across independent messages instead of across
//! the chunks of one message; the single-stream path can only use the 2-way
//! parallelism of a 2 KiB leaf, which is what made leaf hashing the biggest
//! fixed cost of a commitment.

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub use avx::*;

pub const LANES: usize = 16;

const IV: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];
const MSG_SCHEDULE: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];
const CHUNK_START: u32 = 1;
const CHUNK_END: u32 = 2;
const PARENT: u32 = 4;
const ROOT: u32 = 8;
const KEYED_HASH: u32 = 16;
const CHUNK_LEN: usize = 1024;
const BLOCK_LEN: usize = 64;

/// Whether the vectorized path is compiled in.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
pub const AVAILABLE: bool = true;
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
pub const AVAILABLE: bool = false;

/// Number of u32 words per message needed in the transposed buffer for a
/// message of `len` bytes: padded to whole 64-byte blocks (a zero-length
/// message still occupies one block).
pub fn padded_words(len: usize) -> usize {
    let blocks = len.div_ceil(BLOCK_LEN).max(1);
    blocks * 16
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
mod avx {
    use super::*;
    use core::arch::x86_64::*;

    type V = __m512i;

    #[inline(always)]
    unsafe fn rotr<const N: i32>(x: V) -> V {
        _mm512_ror_epi32::<N>(x)
    }

    #[inline(always)]
    unsafe fn g(st: &mut [V; 16], a: usize, b: usize, c: usize, d: usize, x: V, y: V) {
        st[a] = _mm512_add_epi32(_mm512_add_epi32(st[a], st[b]), x);
        st[d] = rotr::<16>(_mm512_xor_si512(st[d], st[a]));
        st[c] = _mm512_add_epi32(st[c], st[d]);
        st[b] = rotr::<12>(_mm512_xor_si512(st[b], st[c]));
        st[a] = _mm512_add_epi32(_mm512_add_epi32(st[a], st[b]), y);
        st[d] = rotr::<8>(_mm512_xor_si512(st[d], st[a]));
        st[c] = _mm512_add_epi32(st[c], st[d]);
        st[b] = rotr::<7>(_mm512_xor_si512(st[b], st[c]));
    }

    #[inline(always)]
    unsafe fn round(st: &mut [V; 16], m: &[V; 16]) {
        g(st, 0, 4, 8, 12, m[0], m[1]);
        g(st, 1, 5, 9, 13, m[2], m[3]);
        g(st, 2, 6, 10, 14, m[4], m[5]);
        g(st, 3, 7, 11, 15, m[6], m[7]);
        g(st, 0, 5, 10, 15, m[8], m[9]);
        g(st, 1, 6, 11, 12, m[10], m[11]);
        g(st, 2, 7, 8, 13, m[12], m[13]);
        g(st, 3, 4, 9, 14, m[14], m[15]);
    }

    /// One compression for 16 lanes. `block` holds the 16 message words
    /// (each a vector of 16 lanes); counter / block_len / flags are shared.
    #[inline(always)]
    unsafe fn compress(cv: &[V; 8], block: &[V; 16], counter: u64, block_len: u32, flags: u32) -> [V; 16] {
        let mut st: [V; 16] = [
            cv[0],
            cv[1],
            cv[2],
            cv[3],
            cv[4],
            cv[5],
            cv[6],
            cv[7],
            _mm512_set1_epi32(IV[0] as i32),
            _mm512_set1_epi32(IV[1] as i32),
            _mm512_set1_epi32(IV[2] as i32),
            _mm512_set1_epi32(IV[3] as i32),
            _mm512_set1_epi32(counter as u32 as i32),
            _mm512_set1_epi32((counter >> 32) as u32 as i32),
            _mm512_set1_epi32(block_len as i32),
            _mm512_set1_epi32(flags as i32),
        ];
        let mut m = *block;
        for r in 0..7 {
            round(&mut st, &m);
            if r < 6 {
                let mut p = [m[0]; 16];
                for i in 0..16 {
                    p[i] = m[MSG_SCHEDULE[i]];
                }
                m = p;
            }
        }
        for i in 0..8 {
            st[i] = _mm512_xor_si512(st[i], st[i + 8]);
            st[i + 8] = _mm512_xor_si512(st[i + 8], cv[i]);
        }
        st
    }

    #[inline(always)]
    unsafe fn load_block(words: &[u32], w0: usize) -> [V; 16] {
        let mut b = [_mm512_setzero_si512(); 16];
        for i in 0..16 {
            b[i] = _mm512_loadu_si512(words.as_ptr().add((w0 + i) * LANES) as *const _);
        }
        b
    }

    unsafe fn iv_vec() -> [V; 8] {
        let mut v = [_mm512_setzero_si512(); 8];
        for i in 0..8 {
            v[i] = _mm512_set1_epi32(IV[i] as i32);
        }
        v
    }

    /// Chaining value of chunk `chunk_idx` (bytes `[chunk_idx·1024, ..)` of
    /// each message), with `root` set when the chunk is the whole message.
    unsafe fn chunk_cv(words: &[u32], len: usize, chunk_idx: usize, root: bool, key: &[V; 8], base_flags: u32) -> [V; 16] {
        let start = chunk_idx * CHUNK_LEN;
        let chunk_bytes = (len - start).min(CHUNK_LEN);
        let blocks = chunk_bytes.div_ceil(BLOCK_LEN).max(1);
        let mut cv = *key;
        let mut st = [_mm512_setzero_si512(); 16];
        for b in 0..blocks {
            let bstart = start + b * BLOCK_LEN;
            let block_len = (len - bstart).min(BLOCK_LEN) as u32;
            let mut flags = base_flags;
            if b == 0 {
                flags |= CHUNK_START;
            }
            if b + 1 == blocks {
                flags |= CHUNK_END;
                if root {
                    flags |= ROOT;
                }
            }
            let block = load_block(words, bstart / 4);
            st = compress(&cv, &block, chunk_idx as u64, block_len, flags);
            cv.copy_from_slice(&st[..8]);
        }
        st
    }

    unsafe fn parent(l: &[V; 8], r: &[V; 8], root: bool, key: &[V; 8], base_flags: u32) -> [V; 16] {
        let mut block = [_mm512_setzero_si512(); 16];
        block[..8].copy_from_slice(l);
        block[8..].copy_from_slice(r);
        compress(key, &block, 0, BLOCK_LEN as u32, base_flags | PARENT | if root { ROOT } else { 0 })
    }

    /// Non-root chaining value of the subtree covering chunks [lo, hi).
    unsafe fn subtree_cv(words: &[u32], len: usize, lo: usize, hi: usize, key: &[V; 8], base_flags: u32) -> [V; 8] {
        let t = hi - lo;
        if t == 1 {
            let st = chunk_cv(words, len, lo, false, key, base_flags);
            let mut cv = [_mm512_setzero_si512(); 8];
            cv.copy_from_slice(&st[..8]);
            return cv;
        }
        let left = largest_pow2_below(t);
        let l = subtree_cv(words, len, lo, lo + left, key, base_flags);
        let r = subtree_cv(words, len, lo + left, hi, key, base_flags);
        let st = parent(&l, &r, false, key, base_flags);
        let mut cv = [_mm512_setzero_si512(); 8];
        cv.copy_from_slice(&st[..8]);
        cv
    }

    /// Hash 16 messages of `len` bytes each, given transposed as
    /// `words[w * 16 + lane]` (zero-padded to `padded_words(len)` words).
    /// Writes `blake3::hash(message[lane])` into `out[lane]`.
    pub fn hash16(words: &[u32], len: usize, out: &mut [[u8; 32]; LANES]) {
        unsafe { hash16_inner(words, len, &iv_vec(), 0, out) }
    }

    /// Keyed variant: `blake3::keyed_hash(key, message[lane])`.
    pub fn hash16_keyed(key: &[u8; 32], words: &[u32], len: usize, out: &mut [[u8; 32]; LANES]) {
        unsafe {
            let mut kv = [_mm512_setzero_si512(); 8];
            for i in 0..8 {
                let w = u32::from_le_bytes([key[4 * i], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]]);
                kv[i] = _mm512_set1_epi32(w as i32);
            }
            hash16_inner(words, len, &kv, KEYED_HASH, out)
        }
    }

    unsafe fn hash16_inner(words: &[u32], len: usize, key: &[V; 8], base_flags: u32, out: &mut [[u8; 32]; LANES]) {
        assert!(words.len() >= padded_words(len) * LANES);
        let chunks = len.div_ceil(CHUNK_LEN).max(1);
        let st = if chunks == 1 {
            chunk_cv(words, len, 0, true, key, base_flags)
        } else {
            let left = largest_pow2_below(chunks);
            let l = subtree_cv(words, len, 0, left, key, base_flags);
            let r = subtree_cv(words, len, left, chunks, key, base_flags);
            parent(&l, &r, true, key, base_flags)
        };
        // output: first 8 state words, transposed back to per-lane bytes
        let mut tmp = [[0u32; LANES]; 8];
        for i in 0..8 {
            _mm512_storeu_si512(tmp[i].as_mut_ptr() as *mut _, st[i]);
        }
        for lane in 0..LANES {
            for i in 0..8 {
                out[lane][i * 4..i * 4 + 4].copy_from_slice(&tmp[i][lane].to_le_bytes());
            }
        }
    }

    fn largest_pow2_below(t: usize) -> usize {
        debug_assert!(t >= 2);
        let mut p = 1;
        while p * 2 < t {
            p *= 2;
        }
        p
    }
}

/// Transposed message buffer builder for 16 lanes.
pub struct Lanes16 {
    pub words: Vec<u32>,
    pub len: usize,
}

impl Lanes16 {
    pub fn new(len: usize) -> Lanes16 {
        assert_eq!(len % 4, 0, "transposed messages must be whole words");
        Lanes16 { words: vec![0u32; padded_words(len) * LANES], len }
    }
    #[inline(always)]
    pub fn put_u32(&mut self, lane: usize, word: usize, v: u32) {
        self.words[word * LANES + lane] = v;
    }
    #[inline(always)]
    pub fn put_u64(&mut self, lane: usize, word: usize, v: u64) {
        self.words[word * LANES + lane] = v as u32;
        self.words[(word + 1) * LANES + lane] = (v >> 32) as u32;
    }
    /// Store sixteen u32 values (one per lane) at word `word`.
    #[inline(always)]
    pub fn put_u32x16(&mut self, word: usize, v: &[u32; LANES]) {
        assert!((word + 1) * LANES <= self.words.len());
        self.words[word * LANES..(word + 1) * LANES].copy_from_slice(v);
    }
    /// Store sixteen u64 values (one per lane) at words `word`, `word + 1`.
    #[inline(always)]
    pub fn put_u64x16(&mut self, word: usize, v: &[u64; LANES]) {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        unsafe {
            use core::arch::x86_64::*;
            let lo = _mm512_loadu_si512(v.as_ptr() as *const _);
            let hi = _mm512_loadu_si512(v.as_ptr().add(8) as *const _);
            let dst = self.words.as_mut_ptr();
            assert!((word + 2) * LANES <= self.words.len());
            // low 32 bits of each u64 → word; high 32 bits → word + 1
            let lo32 = _mm512_cvtepi64_epi32(lo);
            let hi32 = _mm512_cvtepi64_epi32(hi);
            _mm256_storeu_si256(dst.add(word * LANES) as *mut _, lo32);
            _mm256_storeu_si256(dst.add(word * LANES + 8) as *mut _, hi32);
            let lo_h = _mm512_cvtepi64_epi32(_mm512_srli_epi64(lo, 32));
            let hi_h = _mm512_cvtepi64_epi32(_mm512_srli_epi64(hi, 32));
            _mm256_storeu_si256(dst.add((word + 1) * LANES) as *mut _, lo_h);
            _mm256_storeu_si256(dst.add((word + 1) * LANES + 8) as *mut _, hi_h);
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
        for lane in 0..LANES {
            self.put_u64(lane, word, v[lane]);
        }
    }
    pub fn put_bytes(&mut self, lane: usize, byte_off: usize, bytes: &[u8]) {
        debug_assert_eq!(byte_off % 4, 0);
        debug_assert_eq!(bytes.len() % 4, 0);
        for (k, ch) in bytes.chunks_exact(4).enumerate() {
            self.put_u32(lane, byte_off / 4 + k, u32::from_le_bytes([ch[0], ch[1], ch[2], ch[3]]));
        }
    }
    /// Message of one lane, reassembled (for tests / scalar fallback).
    pub fn lane_bytes(&self, lane: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(self.len);
        for w in 0..self.len / 4 {
            v.extend_from_slice(&self.words[w * LANES + lane].to_le_bytes());
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_u64x16_matches_scalar() {
        let mut a = Lanes16::new(64);
        let mut b = Lanes16::new(64);
        let v: [u64; LANES] = core::array::from_fn(|i| (i as u64 + 1) * 0x0123_4567_89AB_CDEF);
        a.put_u64x16(3, &v);
        for lane in 0..LANES {
            b.put_u64(lane, 3, v[lane]);
        }
        assert_eq!(a.words, b.words);
    }

    #[test]
    fn matches_blake3_crate() {
        if !AVAILABLE {
            return;
        }
        for &len in &[0usize, 4, 60, 64, 68, 128, 500, 1020, 1024, 1028, 2040, 2048, 2052, 3000, 4096, 5000, 9000] {
            let mut lanes = Lanes16::new(len);
            let mut msgs = Vec::new();
            for lane in 0..LANES {
                let msg: Vec<u8> = (0..len).map(|i| ((i * 7 + lane * 13 + len) % 251) as u8).collect();
                lanes.put_bytes(lane, 0, &msg);
                msgs.push(msg);
            }
            let mut out = [[0u8; 32]; LANES];
            hash16(&lanes.words, len, &mut out);
            let key = [7u8; 32];
            let mut outk = [[0u8; 32]; LANES];
            hash16_keyed(&key, &lanes.words, len, &mut outk);
            for lane in 0..LANES {
                assert_eq!(lanes.lane_bytes(lane), msgs[lane]);
                assert_eq!(out[lane], *blake3::hash(&msgs[lane]).as_bytes(), "len={len} lane={lane}");
                assert_eq!(outk[lane], *blake3::keyed_hash(&key, &msgs[lane]).as_bytes(), "keyed len={len} lane={lane}");
            }
        }
    }
}
