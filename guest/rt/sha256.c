#include "zk.h"

static const u32 K[64] = {
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

#define ROTR(x,n) (((x) >> (n)) | ((x) << (32 - (n))))

/* One compression step; the caller rotates the register names (8× unrolled)
 * so no cycles are spent shuffling a..h. */
#define RND(a, b, c, d, e, f, g, h, k, w)                                        \
    do {                                                                         \
        u32 t1 = h + (ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25)) + (g ^ (e & (f ^ g))) + (k) + (w); \
        u32 t2 = (ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22)) + (((a ^ b) & (b ^ c)) ^ b);        \
        d += t1;                                                                 \
        h = t1 + t2;                                                             \
    } while (0)
#define RND8(i)                                                        \
    RND(a, b, c, d, e, f, g, hh, K[i + 0], w[i + 0]);                   \
    RND(hh, a, b, c, d, e, f, g, K[i + 1], w[i + 1]);                   \
    RND(g, hh, a, b, c, d, e, f, K[i + 2], w[i + 2]);                   \
    RND(f, g, hh, a, b, c, d, e, K[i + 3], w[i + 3]);                   \
    RND(e, f, g, hh, a, b, c, d, K[i + 4], w[i + 4]);                   \
    RND(d, e, f, g, hh, a, b, c, K[i + 5], w[i + 5]);                   \
    RND(c, d, e, f, g, hh, a, b, K[i + 6], w[i + 6]);                   \
    RND(b, c, d, e, f, g, hh, a, K[i + 7], w[i + 7])

static void sha256_block(u32 h[8], const u8 *p) {
    u32 w[64];
    for (int i = 0; i < 16; i++)
        w[i] = ((u32)p[4*i] << 24) | ((u32)p[4*i+1] << 16) | ((u32)p[4*i+2] << 8) | (u32)p[4*i+3];
    for (int i = 16; i < 64; i++) {
        u32 x = w[i-15], y = w[i-2];
        u32 s0 = ROTR(x,7) ^ ROTR(x,18) ^ (x >> 3);
        u32 s1 = ROTR(y,17) ^ ROTR(y,19) ^ (y >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    u32 a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    RND8(0); RND8(8); RND8(16); RND8(24); RND8(32); RND8(40); RND8(48); RND8(56);
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
}

void sha256_init(sha256_ctx *c) {
    c->h[0]=0x6a09e667; c->h[1]=0xbb67ae85; c->h[2]=0x3c6ef372; c->h[3]=0xa54ff53a;
    c->h[4]=0x510e527f; c->h[5]=0x9b05688c; c->h[6]=0x1f83d9ab; c->h[7]=0x5be0cd19;
    c->buflen = 0; c->len_lo = 0; c->len_hi = 0;
}
void sha256_update(sha256_ctx *c, const u8 *d, u32 n) {
    u32 old = c->len_lo;
    c->len_lo += n;
    if (c->len_lo < old) c->len_hi++;
    while (n) {
        if (c->buflen == 0 && n >= 64) {
            /* whole blocks straight from the input, no staging copy */
            sha256_block(c->h, d);
            d += 64; n -= 64;
            continue;
        }
        u32 take = 64 - c->buflen;
        if (take > n) take = n;
        memcpy(c->buf + c->buflen, d, take);
        c->buflen += take; d += take; n -= take;
        if (c->buflen == 64) { sha256_block(c->h, c->buf); c->buflen = 0; }
    }
}
void sha256_final(sha256_ctx *c, u8 out[32]) {
    u32 bits_lo = c->len_lo << 3;
    u32 bits_hi = (c->len_hi << 3) | (c->len_lo >> 29);
    u8 pad = 0x80;
    sha256_update(c, &pad, 1);
    u8 z = 0;
    while (c->buflen != 56) sha256_update(c, &z, 1);
    u8 lenb[8];
    for (int i = 0; i < 4; i++) { lenb[i] = (u8)(bits_hi >> (24 - 8*i)); lenb[4+i] = (u8)(bits_lo >> (24 - 8*i)); }
    sha256_update(c, lenb, 8);
    for (int i = 0; i < 8; i++) {
        out[4*i] = (u8)(c->h[i] >> 24); out[4*i+1] = (u8)(c->h[i] >> 16);
        out[4*i+2] = (u8)(c->h[i] >> 8); out[4*i+3] = (u8)c->h[i];
    }
}
void sha256(const u8 *d, u32 n, u8 out[32]) {
    sha256_ctx c; sha256_init(&c); sha256_update(&c, d, n); sha256_final(&c, out);
}
