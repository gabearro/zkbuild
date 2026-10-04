/* zkbuild v0 translation-unit guest ("zkcc0").
 *
 * Witness (private):  flags | n_files | (name | content)*
 * Journal (public):   source_root[8] | flags_root[8] | object_hash[8] | n_files
 * Output (private):   the object bytes (chunked)
 *
 * The guest is the *toolchain*: swapping in a real compiler keeps the
 * certificate format unchanged. This v0 toolchain produces an "object"
 * that is a deterministic function of (flags, sources):
 *     object = "ZKOBJ0" || u32 flags_len || flags || content_0 || ... || content_{n-1}
 */
#include "zk.h"

#define MAX_FLAGS 1024
#define MAX_NAME  256
#define MAX_FILE  65536

static u8 flags[MAX_FLAGS];
static u8 name[MAX_NAME];
static u8 content[MAX_FILE];
static u8 leaves[64 * 32];

static void put_u32(blake3_ctx *c, u32 v) {
    u8 b[4] = { (u8)v, (u8)(v >> 8), (u8)(v >> 16), (u8)(v >> 24) };
    blake3_update(c, b, 4);
}

int main(void) {
    int flen = zk_read_bytes(flags, MAX_FLAGS);
    if (flen < 0) return 2;
    u32 n = zk_read_word();
    if (n > 64) return 3;

    blake3_ctx obj;
    blake3_init(&obj);
    blake3_update(&obj, (const u8 *)"ZKOBJ0", 6);
    zk_write_output((const u8 *)"ZKOBJ0", 6);
    put_u32(&obj, (u32)flen);
    { u8 lb[4] = { (u8)flen, (u8)(flen >> 8), (u8)(flen >> 16), (u8)(flen >> 24) }; zk_write_output(lb, 4); }
    blake3_update(&obj, flags, (u32)flen);
    zk_write_output(flags, (u32)flen);

    for (u32 i = 0; i < n; i++) {
        int nlen = zk_read_bytes(name, MAX_NAME);
        if (nlen < 0) return 4;
        int clen = zk_read_bytes(content, MAX_FILE);
        if (clen < 0) return 5;
        /* leaf_i = BLAKE3(0x00 || name_len || name || content_len || content) */
        blake3_ctx leaf;
        blake3_init(&leaf);
        u8 tag = 0;
        blake3_update(&leaf, &tag, 1);
        put_u32(&leaf, (u32)nlen);
        blake3_update(&leaf, name, (u32)nlen);
        put_u32(&leaf, (u32)clen);
        blake3_update(&leaf, content, (u32)clen);
        blake3_final(&leaf, leaves + 32 * i);
        /* "compile": append content to the object */
        blake3_update(&obj, content, (u32)clen);
        zk_write_output(content, (u32)clen);
    }

    /* source_root = BLAKE3(0x01 || n || leaves) */
    u8 root[32], froot[32], ohash[32];
    blake3_ctx r;
    blake3_init(&r);
    u8 tag1 = 1;
    blake3_update(&r, &tag1, 1);
    put_u32(&r, n);
    blake3_update(&r, leaves, 32 * n);
    blake3_final(&r, root);

    /* flags_root = BLAKE3(0x02 || flags) */
    blake3_ctx fr;
    blake3_init(&fr);
    u8 tag2 = 2;
    blake3_update(&fr, &tag2, 1);
    blake3_update(&fr, flags, (u32)flen);
    blake3_final(&fr, froot);

    blake3_final(&obj, ohash);

    zk_write_bytes32(root);
    zk_write_bytes32(froot);
    zk_write_bytes32(ohash);
    zk_write_word(n);
    return 0;
}
