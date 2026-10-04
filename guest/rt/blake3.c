/* BLAKE3 (hash mode) for RV32IM guests — bit-exact with the `blake3` crate.
 *
 * Ported from the official reference implementation (chunk chaining-value
 * stack, parent nodes, ROOT finalization) with the compression function
 * fully unrolled: 7 rounds × 8 G's with the message permutation resolved at
 * compile time, so a 64-byte block costs ~1.5 k cycles (~24 cycles/byte,
 * against ~63 for SHA-256 on an ISA without rotates). */
#include "zk.h"

#define B3_OUT_LEN 32
#define B3_BLOCK_LEN 64
#define B3_CHUNK_LEN 1024
#define CHUNK_START 1
#define CHUNK_END 2
#define PARENT 4
#define ROOT 8

static const u32 IV[8] = { 0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19 };

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define G(a, b, c, d, mx, my)                  \
    do {                                       \
        a = a + b + (mx);                      \
        d = ROTR(d ^ a, 16);                   \
        c = c + d;                             \
        b = ROTR(b ^ c, 12);                   \
        a = a + b + (my);                      \
        d = ROTR(d ^ a, 8);                    \
        c = c + d;                             \
        b = ROTR(b ^ c, 7);                    \
    } while (0)
/* One round with the message words at the permuted positions p0..p15. */
#define ROUND(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15) \
    G(s0, s4, s8, s12, m[p0], m[p1]);                                                \
    G(s1, s5, s9, s13, m[p2], m[p3]);                                                \
    G(s2, s6, s10, s14, m[p4], m[p5]);                                               \
    G(s3, s7, s11, s15, m[p6], m[p7]);                                               \
    G(s0, s5, s10, s15, m[p8], m[p9]);                                               \
    G(s1, s6, s11, s12, m[p10], m[p11]);                                             \
    G(s2, s7, s8, s13, m[p12], m[p13]);                                              \
    G(s3, s4, s9, s14, m[p14], m[p15])

/* Full 16-word compression output (the caller takes the first 8 words as a
 * chaining value, or all 16 for root output). */
static void compress(const u32 cv[8], const u32 m[16], u32 counter_lo, u32 counter_hi, u32 block_len, u32 flags, u32 out[16]) {
    u32 s0 = cv[0], s1 = cv[1], s2 = cv[2], s3 = cv[3], s4 = cv[4], s5 = cv[5], s6 = cv[6], s7 = cv[7];
    u32 s8 = IV[0], s9 = IV[1], s10 = IV[2], s11 = IV[3];
    u32 s12 = counter_lo, s13 = counter_hi, s14 = block_len, s15 = flags;
    /* message schedule: round r uses MSG_PERMUTATION^r applied to the indices */
    ROUND(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    ROUND(2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8);
    ROUND(3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1);
    ROUND(10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6);
    ROUND(12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4);
    ROUND(9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7);
    ROUND(11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13);
    out[0] = s0 ^ s8; out[1] = s1 ^ s9; out[2] = s2 ^ s10; out[3] = s3 ^ s11;
    out[4] = s4 ^ s12; out[5] = s5 ^ s13; out[6] = s6 ^ s14; out[7] = s7 ^ s15;
    out[8] = s8 ^ cv[0]; out[9] = s9 ^ cv[1]; out[10] = s10 ^ cv[2]; out[11] = s11 ^ cv[3];
    out[12] = s12 ^ cv[4]; out[13] = s13 ^ cv[5]; out[14] = s14 ^ cv[6]; out[15] = s15 ^ cv[7];
}

static void words_from_bytes(const u8 *b, u32 n, u32 w[16]) {
    /* b is the 4-aligned context buffer, n <= 64; the block is zero-padded.
     * Whole words are loaded little-endian directly (RV32 is little-endian). */
    const u32 *bw = (const u32 *)b;
    u32 nw = n >> 2;
    for (u32 i = 0; i < nw; i++) w[i] = bw[i];
    for (u32 i = nw; i < 16; i++) w[i] = 0;
    for (u32 i = nw << 2; i < n; i++) w[i >> 2] |= (u32)b[i] << (8 * (i & 3));
}

/* Compress one block of the current chunk into its chaining value. */
static void chunk_block(blake3_ctx *c, const u32 m[16], u32 block_len, u32 extra_flags, u32 out[16]) {
    u32 flags = extra_flags | (c->blocks_compressed == 0 ? CHUNK_START : 0);
    compress(c->cv, m, (u32)c->chunk_counter, (u32)(c->chunk_counter >> 32), block_len, flags, out);
}

static void parent_cv(const u32 left[8], const u32 right[8], u32 out[8]) {
    u32 m[16], o[16];
    for (int i = 0; i < 8; i++) { m[i] = left[i]; m[8 + i] = right[i]; }
    compress(IV, m, 0, 0, B3_BLOCK_LEN, PARENT, o);
    for (int i = 0; i < 8; i++) out[i] = o[i];
}

static void chunk_reset(blake3_ctx *c, unsigned long long counter) {
    for (int i = 0; i < 8; i++) c->cv[i] = IV[i];
    c->chunk_counter = counter;
    c->block_len = 0;
    c->blocks_compressed = 0;
}

void blake3_init(blake3_ctx *c) {
    chunk_reset(c, 0);
    c->stack_len = 0;
}

/* Merge a finished chunk's chaining value into the stack (reference impl's add_chunk_chaining_value). */
static void push_chunk_cv(blake3_ctx *c, u32 cv[8], unsigned long long total_chunks) {
    while ((total_chunks & 1) == 0) {
        c->stack_len--;
        parent_cv(c->stack[c->stack_len], cv, cv);
        total_chunks >>= 1;
    }
    for (int i = 0; i < 8; i++) c->stack[c->stack_len][i] = cv[i];
    c->stack_len++;
}

static u32 chunk_len(const blake3_ctx *c) { return c->blocks_compressed * B3_BLOCK_LEN + c->block_len; }

void blake3_update(blake3_ctx *c, const u8 *d, u32 n) {
    while (n) {
        if (chunk_len(c) == B3_CHUNK_LEN) {
            /* finish the chunk (its last block is still buffered) and start the next */
            u32 m[16], o[16];
            words_from_bytes(c->block, c->block_len, m);
            chunk_block(c, m, c->block_len, CHUNK_END, o);
            unsigned long long total = c->chunk_counter + 1;
            chunk_reset(c, total);
            push_chunk_cv(c, o, total);
        }
        if (c->block_len == B3_BLOCK_LEN) {
            /* a full buffered block that is not the chunk's last: compress it */
            u32 m[16], o[16];
            words_from_bytes(c->block, B3_BLOCK_LEN, m);
            chunk_block(c, m, B3_BLOCK_LEN, 0, o);
            for (int i = 0; i < 8; i++) c->cv[i] = o[i];
            c->blocks_compressed++;
            c->block_len = 0;
        }
        u32 want = B3_BLOCK_LEN - c->block_len;
        u32 take = n < want ? n : want;
        /* whole aligned blocks straight from the input, as long as more input follows
         * (the last block of the input and of each chunk must stay buffered) */
        if (c->block_len == 0 && n > B3_BLOCK_LEN && chunk_len(c) + B3_BLOCK_LEN < B3_CHUNK_LEN && ((u32)d & 3) == 0) {
            u32 o[16];
            chunk_block(c, (const u32 *)d, B3_BLOCK_LEN, 0, o);
            for (int i = 0; i < 8; i++) c->cv[i] = o[i];
            c->blocks_compressed++;
            d += B3_BLOCK_LEN; n -= B3_BLOCK_LEN;
            continue;
        }
        memcpy(c->block + c->block_len, d, take);
        c->block_len += take; d += take; n -= take;
    }
}

void blake3_final(blake3_ctx *c, u8 out[32]) {
    /* output of the current (last) chunk, then fold the stack from the top */
    u32 m[16], cv[8], o[16];
    words_from_bytes(c->block, c->block_len, m);
    u32 flags = CHUNK_END | (c->blocks_compressed == 0 ? CHUNK_START : 0);
    u32 counter_lo = (u32)c->chunk_counter, counter_hi = (u32)(c->chunk_counter >> 32), block_len = c->block_len;
    for (int i = 0; i < 8; i++) cv[i] = c->cv[i];
    u32 n = c->stack_len;
    while (n > 0) {
        /* chaining value of the pending output, then make it the right child of a parent node */
        compress(cv, m, counter_lo, counter_hi, block_len, flags, o);
        n--;
        for (int i = 0; i < 8; i++) { m[i] = c->stack[n][i]; m[8 + i] = o[i]; cv[i] = IV[i]; }
        counter_lo = 0; counter_hi = 0; block_len = B3_BLOCK_LEN; flags = PARENT;
    }
    compress(cv, m, counter_lo, counter_hi, block_len, flags | ROOT, o);
    for (int i = 0; i < 8; i++) {
        out[4 * i] = (u8)o[i]; out[4 * i + 1] = (u8)(o[i] >> 8); out[4 * i + 2] = (u8)(o[i] >> 16); out[4 * i + 3] = (u8)(o[i] >> 24);
    }
}

void blake3(const u8 *d, u32 n, u8 out[32]) {
    blake3_ctx c; blake3_init(&c); blake3_update(&c, d, n); blake3_final(&c, out);
}
