#include "zk.h"

void *memcpy(void *d, const void *s, size_t n) {
    u8 *dd = d; const u8 *ss = s;
    while (n--) *dd++ = *ss++;
    return d;
}
void *memset(void *d, int c, size_t n) {
    u8 *dd = d;
    while (n--) *dd++ = (u8)c;
    return d;
}
int memcmp(const void *a, const void *b, size_t n) {
    const u8 *x = a, *y = b;
    for (size_t i = 0; i < n; i++) if (x[i] != y[i]) return x[i] < y[i] ? -1 : 1;
    return 0;
}
size_t strlen(const char *s) { size_t n = 0; while (s[n]) n++; return n; }

int zk_read_bytes(u8 *buf, u32 cap) {
    u32 len = zk_read_word();
    if (len > cap) return -1;
    u32 words = (len + 3) >> 2;
    if (((u32)buf & 3) == 0) {
        /* word-aligned destination: store whole words (the tail bytes past
         * `len` are the stream's zero padding) */
        u32 *bw = (u32 *)buf;
        for (u32 i = 0; i < words; i++) bw[i] = zk_read_word();
        return (int)len;
    }
    for (u32 i = 0; i < words; i++) {
        u32 w = zk_read_word();
        u32 base = i << 2;
        for (u32 k = 0; k < 4; k++) {
            if (base + k < len) buf[base + k] = (u8)(w >> (k << 3));
        }
    }
    return (int)len;
}
/* Length-prefixed frame [byte_len, data...] (word-aligned) out in one ecall. */
void zk_write_output_frame(u32 *frame) {
    zk_syscall(SYS_WRITE_OUTPUT_BUF, (u32)frame);
}
void zk_write_bytes32(const u8 *b) {
    for (int i = 0; i < 8; i++) {
        u32 w = (u32)b[4*i] | ((u32)b[4*i+1] << 8) | ((u32)b[4*i+2] << 16) | ((u32)b[4*i+3] << 24);
        zk_write_word(w);
    }
}

void zk_write_output(const u8 *b, u32 n) {
    if (n == 0) return;
    zk_write_output_word(n);
    u32 words = (n + 3) >> 2;
    for (u32 i = 0; i < words; i++) {
        u32 w = 0;
        for (u32 k = 0; k < 4; k++) {
            u32 idx = (i << 2) + k;
            if (idx < n) w |= (u32)b[idx] << (k << 3);
        }
        zk_write_output_word(w);
    }
}

/* ---- compiler-rt style integer helpers (RV32I has no M extension) ---- */
u32 __mulsi3(u32 a, u32 b) {
    u32 r = 0;
    while (b) { if (b & 1) r += a; a <<= 1; b >>= 1; }
    return r;
}
u32 __udivsi3(u32 n, u32 d) {
    if (d == 0) return 0xFFFFFFFFu;
    u32 q = 0, r = 0;
    for (int i = 31; i >= 0; i--) {
        r = (r << 1) | ((n >> i) & 1);
        if (r >= d) { r -= d; q |= 1u << i; }
    }
    return q;
}
u32 __umodsi3(u32 n, u32 d) {
    if (d == 0) return n;
    u32 r = 0;
    for (int i = 31; i >= 0; i--) {
        r = (r << 1) | ((n >> i) & 1);
        if (r >= d) r -= d;
    }
    return r;
}
int __divsi3(int a, int b) {
    int neg = (a < 0) ^ (b < 0);
    u32 ua = a < 0 ? (u32)(-(a + 1)) + 1u : (u32)a;
    u32 ub = b < 0 ? (u32)(-(b + 1)) + 1u : (u32)b;
    u32 q = __udivsi3(ua, ub);
    return neg ? -(int)q : (int)q;
}
int __modsi3(int a, int b) {
    u32 ua = a < 0 ? (u32)(-(a + 1)) + 1u : (u32)a;
    u32 ub = b < 0 ? (u32)(-(b + 1)) + 1u : (u32)b;
    u32 r = __umodsi3(ua, ub);
    return a < 0 ? -(int)r : (int)r;
}
