/* zkbuild v0 link guest ("zkld0").
 *
 * Witness (private):  flags | n_objects | object*
 * Journal (public):   inputs_root[8] | flags_root[8] | output_hash[8] | n_objects
 *
 *   inputs_root = BLAKE3(0x03 || n || BLAKE3(obj_0) || ... )
 *   output      = "ZKEXE1" || u32 flags_len || flags || u32 n || (u32 len || obj)*
 *   output_hash = BLAKE3("ZKEXE1" || u32 flags_len || flags || u32 n || (u32 len || BLAKE3(obj))*)
 *
 * output_hash commits to the executable through the per-object hashes, so
 * every object is hashed exactly once (the same hash serves inputs_root).
 */
#include "zk.h"

#define MAX_FLAGS 1024
#define MAX_OBJ   (1u << 20)

static u8 flags[MAX_FLAGS];
/* objects are read into a word-aligned [len, data...] frame and written out with one ecall */
static u32 obj_frame[1 + MAX_OBJ / 4];
static u8 *const obj = (u8 *)(obj_frame + 1);
static u8 hashes[64 * 32];

static void put_u32(blake3_ctx *c, u32 v) {
    u8 b[4] = { (u8)v, (u8)(v >> 8), (u8)(v >> 16), (u8)(v >> 24) };
    blake3_update(c, b, 4);
}
static void out_u32(u32 v) {
    u8 b[4] = { (u8)v, (u8)(v >> 8), (u8)(v >> 16), (u8)(v >> 24) };
    zk_write_output(b, 4);
}

int main(void) {
    int flen = zk_read_bytes(flags, MAX_FLAGS);
    if (flen < 0) return 2;
    u32 n = zk_read_word();
    if (n > 64) return 3;

    blake3_ctx out;
    blake3_init(&out);
    blake3_update(&out, (const u8 *)"ZKEXE1", 6);
    zk_write_output((const u8 *)"ZKEXE1", 6);
    put_u32(&out, (u32)flen);
    out_u32((u32)flen);
    blake3_update(&out, flags, (u32)flen);
    zk_write_output(flags, (u32)flen);
    put_u32(&out, n);
    out_u32(n);

    for (u32 i = 0; i < n; i++) {
        int olen = zk_read_bytes(obj, MAX_OBJ);
        if (olen < 0) return 4;
        blake3(obj, (u32)olen, hashes + 32 * i);
        put_u32(&out, (u32)olen);
        blake3_update(&out, hashes + 32 * i, 32);
        out_u32((u32)olen);
        if (olen) { obj_frame[0] = (u32)olen; zk_write_output_frame(obj_frame); }
    }

    u8 iroot[32], froot[32], ohash[32];
    blake3_ctx r;
    blake3_init(&r);
    u8 tag3 = 3;
    blake3_update(&r, &tag3, 1);
    put_u32(&r, n);
    blake3_update(&r, hashes, 32 * n);
    blake3_final(&r, iroot);

    blake3_ctx fr;
    blake3_init(&fr);
    u8 tag2 = 2;
    blake3_update(&fr, &tag2, 1);
    blake3_update(&fr, flags, (u32)flen);
    blake3_final(&fr, froot);

    blake3_final(&out, ohash);

    zk_write_bytes32(iroot);
    zk_write_bytes32(froot);
    zk_write_bytes32(ohash);
    zk_write_word(n);
    return 0;
}
