#ifndef ZK_H
#define ZK_H
typedef unsigned int u32;
typedef unsigned char u8;
typedef unsigned long size_t;

#define SYS_HALT 0
#define SYS_READ_WITNESS 1
#define SYS_WRITE_JOURNAL 2
#define SYS_WRITE_OUTPUT 3
#define SYS_WRITE_OUTPUT_BUF 4 /* a0 = address of [byte_len, data words...] */

static inline u32 zk_syscall(u32 num, u32 arg) {
    register u32 a0 __asm__("a0") = arg;
    register u32 a7 __asm__("a7") = num;
    __asm__ volatile("ecall" : "+r"(a0) : "r"(a7) : "memory");
    return a0;
}
static inline u32 zk_read_word(void) { return zk_syscall(SYS_READ_WITNESS, 0); }
static inline void zk_write_word(u32 w) { zk_syscall(SYS_WRITE_JOURNAL, w); }
static inline void zk_write_output_word(u32 w) { zk_syscall(SYS_WRITE_OUTPUT, w); }
static inline void zk_halt(u32 code) { zk_syscall(SYS_HALT, code); for (;;) {} }

/* Read a length-prefixed byte string from the witness into buf (max cap). Returns length, or -1. */
int zk_read_bytes(u8 *buf, u32 cap);
void zk_write_bytes32(const u8 *b);
/* Private output stream, framed as [len][words...] chunks (host concatenates). */
void zk_write_output(const u8 *b, u32 n);
void zk_write_output_frame(u32 *frame);

void *memcpy(void *d, const void *s, size_t n);
void *memset(void *d, int c, size_t n);
int memcmp(const void *a, const void *b, size_t n);
size_t strlen(const char *s);

typedef struct { u32 h[8]; u8 buf[64]; u32 buflen; u32 len_lo, len_hi; } sha256_ctx;
void sha256_init(sha256_ctx *c);
void sha256_update(sha256_ctx *c, const u8 *d, u32 n);
void sha256_final(sha256_ctx *c, u8 out[32]);
void sha256(const u8 *d, u32 n, u8 out[32]);

/* BLAKE3 (hash mode), the toolchain's commitment hash: ~3× cheaper than SHA-256 in RV32IM. */
typedef struct {
    u32 cv[8]; u8 block[64]; u32 block_len; u32 blocks_compressed; unsigned long long chunk_counter;
    u32 stack[54][8]; u32 stack_len;
} blake3_ctx;
void blake3_init(blake3_ctx *c);
void blake3_update(blake3_ctx *c, const u8 *d, u32 n);
void blake3_final(blake3_ctx *c, u8 out[32]);
void blake3(const u8 *d, u32 n, u8 out[32]);
#endif
