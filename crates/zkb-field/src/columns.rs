//! A set of equal-length base-field columns in one buffer, with each column
//! start staggered by one cache line.
//!
//! Row-wise access patterns ("element i of every column") are everywhere in
//! the prover: constraint evaluation, Merkle leaves, LogUp aux generation,
//! the DEEP combination. With one `Vec` per column every column starts
//! page-aligned (large allocations are), so element i of all ~300 columns
//! maps to the *same* L1/L2 cache set and the 12/16 ways thrash: every load
//! goes to L3. Staggering column `c` by `c` cache lines spreads the row over
//! distinct sets.

use crate::Fp;
use rayon::prelude::*;
use std::sync::Mutex;

/// Buffers of dropped stores, reused by `pooled`/`zeros`: the prover
/// allocates and frees the same few large column sets every segment, and
/// fresh pages (zeroing + page faults, which are expensive on virtualized
/// hosts) cost more than the arithmetic on them.
static POOL: Mutex<Vec<Vec<Fp>>> = Mutex::new(Vec::new());
const POOL_MAX_BUFFERS: usize = 48;

fn pool_take(len: usize) -> Option<Vec<Fp>> {
    let mut pool = POOL.lock().unwrap();
    let mut best: Option<(usize, usize)> = None;
    for (i, v) in pool.iter().enumerate() {
        if v.capacity() >= len && best.is_none_or(|(_, c)| v.capacity() < c) {
            best = Some((i, v.capacity()));
        }
    }
    best.map(|(i, _)| {
        let mut v = pool.swap_remove(i);
        v.resize(len, Fp::ZERO);
        v
    })
}

fn pool_give(v: Vec<Fp>) {
    if v.capacity() < 1 << 16 {
        return;
    }
    let mut pool = POOL.lock().unwrap();
    if pool.len() < POOL_MAX_BUFFERS {
        pool.push(v);
    }
}

/// Drop every pooled buffer (frees memory; buffers are re-created on demand).
pub fn clear_pool() {
    POOL.lock().unwrap().clear();
}

/// Extra elements between consecutive columns (one 64-byte line).
pub const COL_PAD: usize = 16;

#[derive(Debug, Default)]
pub struct ColumnStore {
    buf: Vec<Fp>,
    width: usize,
    len: usize,
    stride: usize,
}

impl Clone for ColumnStore {
    fn clone(&self) -> ColumnStore {
        let mut s = ColumnStore::pooled(self.width, self.len);
        let used = self.width * self.stride;
        s.buf[..used].copy_from_slice(&self.buf[..used]);
        s
    }
}

impl Drop for ColumnStore {
    fn drop(&mut self) {
        pool_give(std::mem::take(&mut self.buf));
    }
}

impl ColumnStore {
    #[inline]
    pub fn stride_for(len: usize) -> usize {
        len + COL_PAD
    }

    /// Buffer length needed for `width` columns of `len` elements.
    #[inline]
    pub fn buffer_len(width: usize, len: usize) -> usize {
        width * Self::stride_for(len)
    }

    /// A store whose contents are unspecified (recycled from the pool when
    /// possible); every element must be written before it is read.
    pub fn pooled(width: usize, len: usize) -> ColumnStore {
        let need = Self::buffer_len(width, len);
        let buf = pool_take(need).unwrap_or_else(|| vec![Fp::ZERO; need]);
        ColumnStore { buf, width, len, stride: Self::stride_for(len) }
    }

    pub fn zeros(width: usize, len: usize) -> ColumnStore {
        let mut s = Self::pooled(width, len);
        let used = width * s.stride;
        s.buf[..used].par_chunks_mut(1 << 16).for_each(|c| c.fill(Fp::ZERO));
        s
    }

    /// Wrap an existing buffer (length ≥ `buffer_len(width, len)`); contents
    /// are kept as-is.
    pub fn from_buffer(buf: Vec<Fp>, width: usize, len: usize) -> ColumnStore {
        assert!(buf.len() >= Self::buffer_len(width, len));
        ColumnStore { buf, width, len, stride: Self::stride_for(len) }
    }

    pub fn from_columns(cols: Vec<Vec<Fp>>) -> ColumnStore {
        if cols.is_empty() {
            return ColumnStore::default();
        }
        let len = cols[0].len();
        let mut s = ColumnStore::zeros(cols.len(), len);
        for (c, col) in cols.iter().enumerate() {
            assert_eq!(col.len(), len);
            s.col_mut(c).copy_from_slice(col);
        }
        s
    }

    pub fn into_buffer(mut self) -> Vec<Fp> {
        std::mem::take(&mut self.buf)
    }

    #[inline(always)]
    pub fn width(&self) -> usize {
        self.width
    }
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.width == 0
    }
    #[inline(always)]
    pub fn col(&self, c: usize) -> &[Fp] {
        let start = c * self.stride;
        &self.buf[start..start + self.len]
    }
    #[inline(always)]
    pub fn col_mut(&mut self, c: usize) -> &mut [Fp] {
        let start = c * self.stride;
        &mut self.buf[start..start + self.len]
    }
    pub fn slices(&self) -> Vec<&[Fp]> {
        (0..self.width).map(|c| self.col(c)).collect()
    }
    pub fn cols(&self) -> impl Iterator<Item = &[Fp]> + '_ {
        (0..self.width).map(move |c| self.col(c))
    }
    pub fn par_cols(&self) -> impl IndexedParallelIterator<Item = &[Fp]> + '_ {
        let len = self.len;
        self.buf[..self.width * self.stride].par_chunks(self.stride).map(move |ch| &ch[..len])
    }
    pub fn par_cols_mut(&mut self) -> impl IndexedParallelIterator<Item = &mut [Fp]> + '_ {
        let len = self.len;
        let used = self.width * self.stride;
        self.buf[..used].par_chunks_mut(self.stride).map(move |ch| &mut ch[..len])
    }
    /// Raw column start pointers (for disjoint parallel writes).
    pub fn col_ptrs(&mut self) -> Vec<*mut Fp> {
        let base = self.buf.as_mut_ptr();
        (0..self.width).map(|c| unsafe { base.add(c * self.stride) }).collect()
    }
    /// Keep only the first `width` columns.
    pub fn truncate(&mut self, width: usize) {
        assert!(width <= self.width);
        self.width = width;
    }
    pub fn to_columns(&self) -> Vec<Vec<Fp>> {
        self.cols().map(|c| c.to_vec()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_stagger() {
        let cols: Vec<Vec<Fp>> = (0..5).map(|c| (0..100).map(|i| Fp::new(c * 1000 + i)).collect()).collect();
        let s = ColumnStore::from_columns(cols.clone());
        assert_eq!(s.to_columns(), cols);
        for c in 1..5 {
            let d = s.col(c).as_ptr() as usize - s.col(c - 1).as_ptr() as usize;
            assert_eq!(d, (100 + COL_PAD) * core::mem::size_of::<Fp>());
        }
        let v: Vec<Vec<Fp>> = s.par_cols().map(|c| c.to_vec()).collect();
        assert_eq!(v, cols);
    }
}
