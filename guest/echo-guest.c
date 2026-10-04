/* Tiny guest used by tests: reads one word, writes word+1 and word*3 to the
 * journal, then test vectors of the two hashes: sha256("abc"), blake3("abc"),
 * blake3 of a 1100-byte pattern (two chunks + parent, unaligned update boundaries),
 * and blake3 of a 1024-byte pattern (exactly one chunk). */
#include "zk.h"
static u8 pat[1100];
int main(void) {
    u32 w = zk_read_word();
    zk_write_word(w + 1);
    zk_write_word(w * 3);
    u8 h[32];
    sha256((const u8 *)"abc", 3, h);
    zk_write_bytes32(h);
    blake3((const u8 *)"abc", 3, h);
    zk_write_bytes32(h);
    for (u32 i = 0; i < sizeof pat; i++) pat[i] = (u8)(i % 251);
    blake3_ctx c;
    blake3_init(&c);
    blake3_update(&c, pat, 7);
    blake3_update(&c, pat + 7, 500);
    blake3_update(&c, pat + 507, 593);
    blake3_final(&c, h);
    zk_write_bytes32(h);
    blake3(pat, 1024, h);
    zk_write_bytes32(h);
    /* private output: one word-wise chunk and one framed chunk (SYS_WRITE_OUTPUT_BUF) */
    zk_write_output((const u8 *)"hello", 5);
    static u32 frame[1 + 2];
    frame[0] = 6; memcpy(frame + 1, "world!", 6);
    zk_write_output_frame(frame);
    return 0;
}
