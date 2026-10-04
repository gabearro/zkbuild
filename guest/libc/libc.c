/* zkbuild mini libc for RV32IM guests ("ZK build OS" userland).
 *
 * Deterministic by construction: no clock, no environment, no filesystem
 * except the in-memory VFS populated from the witness, no randomness.
 *
 * Startup (`__zk_start`, called from crt0):
 *   witness = flags | n_files | (name | content)*
 *   argv    = ["zkcc", flags split on spaces...]
 *   VFS     = the files; source_root computed from their leaf hashes
 *   stdout  = the toolchain's output (private output stream + BLAKE3)
 * exit(code):
 *   journal = source_root | flags_root | blake3(stdout) | n_files
 */
#include "zk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>

int errno;

/* ------------------------------------------------------------------ heap */
extern char __bss_end;
static char *heap_ptr;
#define HEAP_LIMIT ((char *)0x00F00000)

void *malloc(size_t n) {
    if (!heap_ptr) heap_ptr = (char *)(((u32)&__bss_end + 15) & ~15u);
    n = (n + 15) & ~15u;
    if (heap_ptr + n > HEAP_LIMIT) { zk_halt(90); }
    char *p = heap_ptr;
    heap_ptr += n;
    return p; /* fresh memory is zero in the VM */
}
void *calloc(size_t a, size_t b) {
    /* never-reused memory is zero-initialized by the VM; no memset needed */
    return malloc(a * b);
}
void *realloc(void *p, size_t n) {
    void *q = malloc(n);
    if (p) memcpy(q, p, n); /* over-copy within the bump heap is safe */
    return q;
}
void free(void *p) { (void)p; }

/* --------------------------------------------------------------- strings */
void *memcpy(void *d, const void *s, size_t n) {
    u8 *dd = d; const u8 *ss = s;
    if (n >= 8 && (((u32)dd ^ (u32)ss) & 3) == 0) {
        /* same alignment: byte head, word body, byte tail */
        while ((u32)dd & 3) { *dd++ = *ss++; n--; }
        u32 *dw = (u32 *)dd; const u32 *sw = (const u32 *)ss;
        size_t nw = n >> 2;
        for (size_t i = 0; i < nw; i++) dw[i] = sw[i];
        dd += nw << 2; ss += nw << 2; n &= 3;
    }
    while (n--) *dd++ = *ss++;
    return d;
}
void *memmove(void *d, const void *s, size_t n) {
    u8 *dd = d; const u8 *ss = s;
    if (dd == ss || n == 0) return d;
    if (dd < ss || dd >= ss + n) return memcpy(d, s, n); /* no forward overlap */
    dd += n; ss += n;
    if (n >= 8 && (((u32)dd ^ (u32)ss) & 3) == 0) {
        while ((u32)dd & 3) { *--dd = *--ss; n--; }
        size_t nw = n >> 2;
        u32 *dw = (u32 *)dd; const u32 *sw = (const u32 *)ss;
        while (nw--) *--dw = *--sw;
        dd = (u8 *)dw; ss = (const u8 *)sw; n &= 3;
    }
    while (n--) *--dd = *--ss;
    return d;
}
void *memset(void *d, int c, size_t n) {
    u8 *dd = d;
    if (n >= 8) {
        while ((u32)dd & 3) { *dd++ = (u8)c; n--; }
        u32 v = (u8)c * 0x01010101u;
        u32 *dw = (u32 *)dd;
        size_t nw = n >> 2;
        for (size_t i = 0; i < nw; i++) dw[i] = v;
        dd += nw << 2; n &= 3;
    }
    while (n--) *dd++ = (u8)c;
    return d;
}
int memcmp(const void *a, const void *b, size_t n) {
    /* byte loop on purpose: chibicc compares short identifier keys at arbitrary alignment */
    const u8 *x = a, *y = b;
    for (size_t i = 0; i < n; i++) if (x[i] != y[i]) return x[i] < y[i] ? -1 : 1;
    return 0;
}
void *memchr(const void *s, int c, size_t n) {
    const u8 *p = s;
    for (size_t i = 0; i < n; i++) if (p[i] == (u8)c) return (void *)(p + i);
    return NULL;
}
/* Byte loop on purpose: chibicc's strings are short (avg. 4 bytes), where
 * the word-at-a-time version's alignment prologue costs more than it saves. */
size_t strlen(const char *s) { const char *p = s; while (*p) p++; return (size_t)(p - s); }
int strcmp(const char *a, const char *b) {
    while (*a && *a == *b) { a++; b++; }
    return (u8)*a - (u8)*b;
}
int strncmp(const char *a, const char *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) return (u8)a[i] - (u8)b[i];
        if (!a[i]) return 0;
    }
    return 0;
}
char *strcpy(char *d, const char *s) { char *r = d; while ((*d++ = *s++)) {} return r; }
char *strncpy(char *d, const char *s, size_t n) {
    size_t i = 0;
    for (; i < n && s[i]; i++) d[i] = s[i];
    for (; i < n; i++) d[i] = 0;
    return d;
}
char *strcat(char *d, const char *s) { strcpy(d + strlen(d), s); return d; }
char *strchr(const char *s, int c) {
    for (;; s++) { if (*s == (char)c) return (char *)s; if (!*s) return NULL; }
}
char *strrchr(const char *s, int c) {
    const char *r = NULL;
    for (;; s++) { if (*s == (char)c) r = s; if (!*s) return (char *)r; }
}
char *strstr(const char *h, const char *n) {
    size_t ln = strlen(n);
    if (!ln) return (char *)h;
    for (; *h; h++) if (!strncmp(h, n, ln)) return (char *)h;
    return NULL;
}
char *strdup(const char *s) { size_t n = strlen(s); char *p = malloc(n + 1); memcpy(p, s, n + 1); return p; }
char *strndup(const char *s, size_t n) {
    size_t l = 0;
    while (l < n && s[l]) l++;
    char *p = malloc(l + 1);
    memcpy(p, s, l);
    p[l] = 0;
    return p;
}
int strcasecmp(const char *a, const char *b) {
    while (*a && tolower((u8)*a) == tolower((u8)*b)) { a++; b++; }
    return tolower((u8)*a) - tolower((u8)*b);
}
int strncasecmp(const char *a, const char *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        int x = tolower((u8)a[i]), y = tolower((u8)b[i]);
        if (x != y) return x - y;
        if (!a[i]) return 0;
    }
    return 0;
}
char *strerror(int e) { (void)e; return "error"; }

/* ----------------------------------------------------------------- ctype */
int isspace(int c) { return c == ' ' || (c >= '\t' && c <= '\r'); }
int isdigit(int c) { return c >= '0' && c <= '9'; }
int isupper(int c) { return c >= 'A' && c <= 'Z'; }
int islower(int c) { return c >= 'a' && c <= 'z'; }
int isalpha(int c) { return isupper(c) || islower(c); }
int isalnum(int c) { return isalpha(c) || isdigit(c); }
int isxdigit(int c) { return isdigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }
int isprint(int c) { return c >= 32 && c < 127; }
int isgraph(int c) { return c > 32 && c < 127; }
int ispunct(int c) { return isgraph(c) && !isalnum(c); }
int toupper(int c) { return islower(c) ? c - 32 : c; }
int tolower(int c) { return isupper(c) ? c + 32 : c; }

/* ---------------------------------------------------------------- stdlib */
static int digit_val(int c) {
    if (isdigit(c)) return c - '0';
    if (c >= 'a' && c <= 'z') return c - 'a' + 10;
    if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
    return 99;
}
unsigned long long strtoull(const char *s, char **end, int base) {
    while (isspace((u8)*s)) s++;
    int neg = 0;
    if (*s == '+' || *s == '-') { neg = *s == '-'; s++; }
    if ((base == 0 || base == 16) && s[0] == '0' && (s[1] == 'x' || s[1] == 'X') && isxdigit((u8)s[2])) { s += 2; base = 16; }
    else if (base == 0 && s[0] == '0') base = 8;
    else if (base == 0) base = 10;
    unsigned long long v = 0;
    const char *start = s;
    for (;;) {
        int d = digit_val((u8)*s);
        if (d >= base) break;
        v = v * (unsigned)base + (unsigned)d;
        s++;
    }
    if (end) *end = (char *)(s == start ? start : s);
    return neg ? (unsigned long long)(-(long long)v) : v;
}
unsigned long strtoul(const char *s, char **end, int base) { return (unsigned long)strtoull(s, end, base); }
long long strtoll(const char *s, char **end, int base) { return (long long)strtoull(s, end, base); }
long strtol(const char *s, char **end, int base) { return (long)strtoull(s, end, base); }
int atoi(const char *s) { return (int)strtol(s, NULL, 10); }
char *getenv(const char *name) { (void)name; return NULL; }

/* ----------------------------------------------------------------- stdio */
enum { F_NONE, F_STDOUT, F_STDERR, F_MEM, F_VFS };
struct _zk_FILE {
    int kind;
    /* memstream */
    char *buf; size_t len, cap; char **out_buf; size_t *out_len;
    /* vfs reader */
    const u8 *data; size_t size, pos;
    int err;
};
static struct _zk_FILE f_stdin = { F_NONE, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
static struct _zk_FILE f_stdout = { F_STDOUT, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
static struct _zk_FILE f_stderr = { F_STDERR, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
FILE *stdin = &f_stdin;
FILE *stdout = &f_stdout;
FILE *stderr = &f_stderr;

/* stdout: buffered, hashed, streamed to the private output */
/* The output buffer is a word-aligned frame [byte_len, data...] so a full
 * buffer leaves the VM with one SYS_WRITE_OUTPUT_BUF ecall (1 cycle) instead
 * of 1025 SYS_WRITE_OUTPUT ecalls. */
static u32 out_frame[1 + 1024];
static u8 *const out_buf = (u8 *)(out_frame + 1);
#define OUT_CAP 4096u
static u32 out_len;
static blake3_ctx out_hash;
static u32 out_total;
static void out_flush(void) {
    if (out_len) {
        blake3_update(&out_hash, out_buf, out_len);
        /* zero the tail of the last word so the frame matches zk_write_output's packing */
        for (u32 i = out_len; i & 3; i++) out_buf[i] = 0;
        out_frame[0] = out_len;
        zk_syscall(SYS_WRITE_OUTPUT_BUF, (u32)out_frame);
        out_total += out_len;
        out_len = 0;
    }
}
static void out_byte(u8 c) {
    out_buf[out_len++] = c;
    if (out_len == OUT_CAP) out_flush();
}
/* stderr: kept in memory, dumped on failure */
#define ERR_CAP 8192
static u8 err_buf[ERR_CAP];
static u32 err_len;

/* VFS: witness files first, then the toolchain's baked-in sysroot headers */
#define MAX_FILES 64
static struct { char *name; u8 *data; u32 len; } vfs[MAX_FILES];
static u32 vfs_n;
extern const struct zk_sysfile { const char *name; const u8 *data; u32 len; } zk_sysroot[];
extern const u32 zk_sysroot_n;
static const char *vfs_normalize(const char *path) {
    /* a real filesystem treats "./x", ".//x" and "x" alike */
    while (path[0] == '.' && path[1] == '/') { path += 2; while (*path == '/') path++; }
    return path;
}
static int vfs_lookup(const char *path, const u8 **data, u32 *len) {
    path = vfs_normalize(path);
    for (u32 i = 0; i < vfs_n; i++)
        if (!strcmp(vfs[i].name, path)) { *data = vfs[i].data; *len = vfs[i].len; return 1; }
    for (u32 i = 0; i < zk_sysroot_n; i++)
        if (!strcmp(vfs_normalize(zk_sysroot[i].name), path)) { *data = zk_sysroot[i].data; *len = zk_sysroot[i].len; return 1; }
    return 0;
}

static void mem_reserve(struct _zk_FILE *f, size_t extra) {
    if (f->len + extra + 1 > f->cap) {
        size_t ncap = f->cap ? f->cap : 256;
        while (ncap < f->len + extra + 1) ncap *= 2;
        char *nb = malloc(ncap);
        if (f->buf) memcpy(nb, f->buf, f->len);
        f->buf = nb; f->cap = ncap;
    }
}
/* Block write to any stream (the single output path of stdio). */
static void file_write(struct _zk_FILE *f, const u8 *p, size_t n) {
    switch (f->kind) {
    case F_STDOUT:
        if (n < 16 && out_len + n < OUT_CAP) { /* short run: no memcpy call */
            u8 *q = out_buf + out_len;
            out_len += n;
            while (n--) *q++ = *p++;
            break;
        }
        while (n) {
            u32 room = OUT_CAP - out_len;
            u32 take = n < room ? (u32)n : room;
            memcpy(out_buf + out_len, p, take);
            out_len += take; p += take; n -= take;
            if (out_len == OUT_CAP) out_flush();
        }
        break;
    case F_STDERR:
        while (n-- && err_len < ERR_CAP) err_buf[err_len++] = *p++;
        break;
    case F_MEM:
        mem_reserve(f, n);
        if (n < 16) { u8 *q = (u8 *)f->buf + f->len; for (size_t i = 0; i < n; i++) q[i] = p[i]; }
        else memcpy(f->buf + f->len, p, n);
        f->len += n;
        f->buf[f->len] = 0;
        if (f->out_buf) *f->out_buf = f->buf;
        if (f->out_len) *f->out_len = f->len;
        break;
    default: f->err = 1; break;
    }
}
int fputc(int c, FILE *f) {
    u8 b = (u8)c;
    if (f->kind == F_STDOUT) { out_byte(b); return b; }
    if (f->kind == F_NONE || f->kind == F_VFS) { f->err = 1; return EOF; }
    file_write(f, &b, 1);
    return b;
}
int putc(int c, FILE *f) { return fputc(c, f); }
int putchar(int c) { return fputc(c, stdout); }
int fputs(const char *s, FILE *f) { file_write(f, (const u8 *)s, strlen(s)); return 0; }
int puts(const char *s) { fputs(s, stdout); fputc('\n', stdout); return 0; }
size_t fwrite(const void *buf, size_t sz, size_t n, FILE *f) {
    size_t total = sz * n;
    if (f->kind == F_NONE || f->kind == F_VFS) { f->err = 1; return 0; }
    file_write(f, buf, total);
    return n;
}
int fgetc(FILE *f) {
    if (f->kind != F_VFS || f->pos >= f->size) return EOF;
    return f->data[f->pos++];
}
size_t fread(void *buf, size_t sz, size_t n, FILE *f) {
    if (f->kind != F_VFS) return 0;
    size_t want = sz * n, avail = f->size - f->pos;
    size_t take = want < avail ? want : avail;
    memcpy(buf, f->data + f->pos, take);
    f->pos += take;
    return sz ? take / sz : 0;
}
int fflush(FILE *f) { (void)f; return 0; }
int ferror(FILE *f) { return f->err; }
int feof(FILE *f) { return f->kind == F_VFS && f->pos >= f->size; }
FILE *fopen(const char *path, const char *mode) {
    if (mode[0] != 'r') { errno = EINVAL; return NULL; }
    const u8 *data; u32 len;
    if (vfs_lookup(path, &data, &len)) {
        struct _zk_FILE *f = calloc(1, sizeof *f);
        f->kind = F_VFS; f->data = data; f->size = len;
        return f;
    }
    errno = ENOENT;
    return NULL;
}
int fclose(FILE *f) {
    if (f->kind == F_MEM) {
        if (!f->buf) { f->buf = malloc(1); f->buf[0] = 0; }
        if (f->out_buf) *f->out_buf = f->buf;
        if (f->out_len) *f->out_len = f->len;
    }
    return 0;
}
FILE *open_memstream(char **buf, size_t *len) {
    struct _zk_FILE *f = calloc(1, sizeof *f);
    f->kind = F_MEM; f->out_buf = buf; f->out_len = len;
    *buf = NULL; *len = 0;
    return f;
}

/* printf family: flags - 0, width (n or *), precision (.n or .*), lengths h hh l ll z t, conv d i u x X o c s p % */
/* Fixed/scientific formatting of a long double via integer arithmetic (not glibc-exact for extreme values). */
static int fmt_float(char *out, int cap, double v, char conv, int prec) {
    int n = 0, neg = v < 0;
    if (neg) v = -v;
    if (prec > 18) prec = 18;
    int exp10 = 0;
    if (conv == 'e' || conv == 'E' || conv == 'g' || conv == 'G') {
        if (v != 0) { while (v >= 10) { v /= 10; exp10++; } while (v < 1) { v *= 10; exp10--; } }
    }
    unsigned long long scale = 1;
    for (int i = 0; i < prec; i++) scale *= 10;
    unsigned long long ip = (unsigned long long)v;
    double frac = v - (double)ip;
    unsigned long long fp = (unsigned long long)(frac * (double)scale + 0.5);
    if (fp >= scale) { ip++; fp -= scale; }
    if ((conv == 'e' || conv == 'E') && ip >= 10) { ip /= 10; exp10++; }
    char ib[32]; int in = 0;
    do { ib[in++] = '0' + (int)(ip % 10); ip /= 10; } while (ip);
    if (neg && n < cap) out[n++] = '-';
    while (in && n < cap) out[n++] = ib[--in];
    if (prec > 0 && n < cap) {
        out[n++] = '.';
        char fb[24]; int fn = 0;
        for (int i = 0; i < prec; i++) { fb[fn++] = '0' + (int)(fp % 10); fp /= 10; }
        while (fn && n < cap) out[n++] = fb[--fn];
    }
    if (conv == 'e' || conv == 'E') {
        if (n < cap) out[n++] = conv;
        if (n < cap) out[n++] = exp10 < 0 ? '-' : '+';
        int ae = exp10 < 0 ? -exp10 : exp10;
        if (n < cap) out[n++] = '0' + ae / 10 % 10;
        if (n < cap) out[n++] = '0' + ae % 10;
    }
    return n;
}

/* Formatted output goes straight into a sink buffer: for stdout that is the
 * 4 KB output buffer itself (each byte is written exactly once), for other
 * streams a 128-byte staging area flushed in one block. chibicc emits
 * ~300 k short printf's per large TU, so per-byte and per-call costs here
 * are a measurable share of the whole compilation. */
#define OUTFN_BUF 128
struct outfn {
    u8 *buf; u32 len, cap;            /* current sink window */
    void (*flush)(struct outfn *);    /* make room: hand off buf[..len] */
    void *ctx; int count;
    u8 lb[OUTFN_BUF];
};
static void emit_n(struct outfn *o, const u8 *p, size_t n) {
    o->count += (int)n;
    while (n) {
        if (o->len == o->cap) o->flush(o);
        u32 room = o->cap - o->len;
        u32 k = n < room ? (u32)n : room;
        u8 *q = o->buf + o->len;
        o->len += k; n -= k;
        while (k >= 4) { q[0] = p[0]; q[1] = p[1]; q[2] = p[2]; q[3] = p[3]; q += 4; p += 4; k -= 4; }
        while (k--) *q++ = *p++;
    }
}
static void emit(struct outfn *o, u8 c) {
    if (o->len == o->cap) o->flush(o);
    o->buf[o->len++] = c;
    o->count++;
}
static void emit_pad(struct outfn *o, int n, u8 c) { while (n-- > 0) emit(o, c); }

static int vformat(struct outfn *o, const char *fmt, va_list ap) {
    for (; *fmt; fmt++) {
        if (*fmt != '%') {
            const char *run = fmt;
            while (fmt[1] && fmt[1] != '%') fmt++;
            emit_n(o, (const u8 *)run, (size_t)(fmt + 1 - run));
            continue;
        }
        fmt++;
        /* fast paths for the unmodified conversions chibicc's codegen uses most */
        if (*fmt == '%') { emit(o, '%'); continue; }
        if (*fmt == 's') {
            const char *str = va_arg(ap, const char *);
            if (!str) str = "(null)";
            emit_n(o, (const u8 *)str, strlen(str));
            continue;
        }
        if (*fmt == 'd' || *fmt == 'i') {
            int v = va_arg(ap, int);
            u32 x = v < 0 ? (u32)(-(v + 1)) + 1u : (u32)v;
            char nb[12]; int k = 0;
            do { nb[sizeof nb - 1 - k++] = '0' + (int)(x % 10); x /= 10; } while (x);
            if (v < 0) nb[sizeof nb - 1 - k++] = '-';
            emit_n(o, (const u8 *)nb + sizeof nb - k, (size_t)k);
            continue;
        }
        if (*fmt == 'c') { emit(o, (u8)va_arg(ap, int)); continue; }
        int left = 0, zero = 0, plus = 0, space = 0, alt = 0;
        for (;; fmt++) {
            if (*fmt == '-') left = 1; else if (*fmt == '0') zero = 1; else if (*fmt == '+') plus = 1;
            else if (*fmt == ' ') space = 1; else if (*fmt == '#') alt = 1; else break;
        }
        int width = -1, prec = -1;
        if (*fmt == '*') { width = va_arg(ap, int); fmt++; if (width < 0) { left = 1; width = -width; } }
        else if (isdigit((u8)*fmt)) { width = 0; while (isdigit((u8)*fmt)) width = width * 10 + (*fmt++ - '0'); }
        if (*fmt == '.') {
            fmt++;
            if (*fmt == '*') { prec = va_arg(ap, int); fmt++; }
            else { prec = 0; while (isdigit((u8)*fmt)) prec = prec * 10 + (*fmt++ - '0'); }
        }
        int ll = 0, big = 0;
        while (*fmt == 'l' || *fmt == 'h' || *fmt == 'z' || *fmt == 't' || *fmt == 'j' || *fmt == 'L') {
            if (*fmt == 'l') ll++;
            if (*fmt == 'j') ll = 2;
            if (*fmt == 'L') big = 1;
            fmt++;
        }
        char tmp[32];
        int n = 0;
        const char *s = tmp;
        int is_num = 0, neg = 0;
        switch (*fmt) {
        case 'd': case 'i': {
            long long v = ll >= 2 ? va_arg(ap, long long) : (long long)va_arg(ap, int);
            unsigned long long u;
            if (v < 0) { neg = 1; u = (unsigned long long)(-(v + 1)) + 1u; } else u = (unsigned long long)v;
            if ((u >> 32) == 0) { /* 32-bit fast path: one-cycle div/rem */
                u32 x = (u32)u;
                do { tmp[sizeof tmp - 1 - n++] = '0' + (int)(x % 10); x /= 10; } while (x);
            } else {
                do { tmp[sizeof tmp - 1 - n++] = '0' + (int)(u % 10); u /= 10; } while (u);
            }
            s = tmp + sizeof tmp - n; is_num = 1; break;
        }
        case 'u': case 'x': case 'X': case 'o': {
            unsigned long long u = ll >= 2 ? va_arg(ap, unsigned long long) : (unsigned long long)va_arg(ap, unsigned int);
            unsigned base = *fmt == 'u' ? 10 : *fmt == 'o' ? 8 : 16;
            const char *dig = *fmt == 'X' ? "0123456789ABCDEF" : "0123456789abcdef";
            if ((u >> 32) == 0) {
                u32 x = (u32)u;
                do { tmp[sizeof tmp - 1 - n++] = dig[x % base]; x /= base; } while (x);
            } else {
                do { tmp[sizeof tmp - 1 - n++] = dig[u % base]; u /= base; } while (u);
            }
            if (alt && base == 16) { tmp[sizeof tmp - 1 - n++] = *fmt == 'X' ? 'X' : 'x'; tmp[sizeof tmp - 1 - n++] = '0'; }
            s = tmp + sizeof tmp - n; is_num = 1; break;
        }
        case 'p': {
            unsigned u = (unsigned)(uintptr_t)va_arg(ap, void *);
            do { tmp[sizeof tmp - 1 - n++] = "0123456789abcdef"[u % 16]; u /= 16; } while (u);
            tmp[sizeof tmp - 1 - n++] = 'x'; tmp[sizeof tmp - 1 - n++] = '0';
            s = tmp + sizeof tmp - n; break;
        }
        case 'f': case 'F': case 'e': case 'E': case 'g': case 'G': {
            double v = va_arg(ap, double); (void)big; /* no long double on this platform */
            n = fmt_float(tmp, sizeof tmp, v, *fmt, prec < 0 ? 6 : prec);
            s = tmp; is_num = 0; break;
        }
        case 'c': tmp[0] = (char)va_arg(ap, int); n = 1; break;
        case 's': {
            s = va_arg(ap, const char *);
            if (!s) s = "(null)";
            n = 0;
            while (s[n] && (prec < 0 || n < prec)) n++;
            break;
        }
        case '%': emit(o, '%'); continue;
        default: emit(o, '%'); emit(o, (u8)*fmt); continue;
        }
        int sign_len = is_num && (neg || plus || space) ? 1 : 0;
        int zeros = (is_num && prec > n) ? prec - n : 0;
        int body = n + zeros + sign_len;
        int pad = width > body ? width - body : 0;
        if (!left && !(zero && is_num && prec < 0)) emit_pad(o, pad, ' ');
        if (sign_len) emit(o, neg ? '-' : plus ? '+' : ' ');
        if (!left && zero && is_num && prec < 0) emit_pad(o, pad, '0');
        emit_pad(o, zeros, '0');
        emit_n(o, (const u8 *)s, (size_t)n);
        if (left) emit_pad(o, pad, ' ');
    }
    o->flush(o);
    return o->count;
}
/* stdout sink: the window is out_buf itself */
static void flush_stdout(struct outfn *o) {
    out_len = o->len;
    if (out_len == OUT_CAP) out_flush();
    o->len = out_len;
}
/* memstream sink: the window is the stream's own buffer (grown ahead of the writer) */
static void mem_window(struct outfn *o) {
    struct _zk_FILE *f = o->ctx;
    mem_reserve(f, OUTFN_BUF);
    o->buf = (u8 *)f->buf + f->len; o->len = 0; o->cap = (u32)(f->cap - f->len - 1);
}
static void flush_mem(struct outfn *o) {
    struct _zk_FILE *f = o->ctx;
    f->len += o->len;
    f->buf[f->len] = 0;
    if (f->out_buf) *f->out_buf = f->buf;
    if (f->out_len) *f->out_len = f->len;
    mem_window(o);
}
/* generic stream sink: staging buffer, block write */
static void flush_file(struct outfn *o) {
    if (o->len) file_write((FILE *)o->ctx, o->lb, o->len);
    o->len = 0;
}
struct sn { char *buf; size_t n, pos; };
static void flush_sn(struct outfn *o) {
    struct sn *s = o->ctx;
    for (u32 i = 0; i < o->len; i++) { if (s->pos + 1 < s->n) s->buf[s->pos] = o->lb[i]; s->pos++; }
    o->len = 0;
}

int vfprintf(FILE *f, const char *fmt, va_list ap) {
    struct outfn o; /* lb left uninitialized on purpose (no 128-byte clear per call) */
    o.ctx = f; o.count = 0;
    if (f->kind == F_STDOUT) { o.buf = out_buf; o.len = out_len; o.cap = OUT_CAP; o.flush = flush_stdout; }
    else if (f->kind == F_MEM) { o.flush = flush_mem; mem_window(&o); }
    else { o.buf = o.lb; o.len = 0; o.cap = OUTFN_BUF; o.flush = flush_file; }
    return vformat(&o, fmt, ap);
}
int fprintf(FILE *f, const char *fmt, ...) { va_list ap; va_start(ap, fmt); int r = vfprintf(f, fmt, ap); va_end(ap); return r; }
int printf(const char *fmt, ...) { va_list ap; va_start(ap, fmt); int r = vfprintf(stdout, fmt, ap); va_end(ap); return r; }
int vsnprintf(char *buf, size_t n, const char *fmt, va_list ap) {
    struct sn s = { buf, n, 0 };
    struct outfn o;
    o.ctx = &s; o.count = 0; o.buf = o.lb; o.len = 0; o.cap = OUTFN_BUF; o.flush = flush_sn;
    vformat(&o, fmt, ap);
    if (n) buf[s.pos < n ? s.pos : n - 1] = 0;
    return (int)s.pos;
}
int snprintf(char *buf, size_t n, const char *fmt, ...) { va_list ap; va_start(ap, fmt); int r = vsnprintf(buf, n, fmt, ap); va_end(ap); return r; }
int sprintf(char *buf, const char *fmt, ...) { va_list ap; va_start(ap, fmt); int r = vsnprintf(buf, 0x7fffffff, fmt, ap); va_end(ap); return r; }

/* -------------------------------------------------- 64-bit integer helpers
 * Fast 64/32 division (Hacker's Delight divlu2) — chibicc's hash map does a
 * 64-bit modulo per probe, and the generic compiler-rt routine costs ~300
 * cycles; this costs ~60. Divisors >= 2^32 fall back to a shift-subtract loop. */
static u32 nlz32(u32 x) { if (!x) return 32; u32 n = 0; while (!(x & 0x80000000u)) { x <<= 1; n++; } return n; }
static u32 divlu2(u32 u1, u32 u0, u32 v, u32 *r) {
    const u32 b = 65536;
    u32 un1, un0, vn1, vn0, q1, q0, un32, un21, un10, rhat;
    u32 s = nlz32(v);
    v <<= s;
    vn1 = v >> 16; vn0 = v & 0xFFFF;
    un32 = s ? (u1 << s) | (u0 >> (32 - s)) : u1;
    un10 = u0 << s;
    un1 = un10 >> 16; un0 = un10 & 0xFFFF;
    q1 = un32 / vn1; rhat = un32 - q1 * vn1;
    while (q1 >= b || q1 * vn0 > b * rhat + un1) { q1--; rhat += vn1; if (rhat >= b) break; }
    un21 = un32 * b + un1 - q1 * v;
    q0 = un21 / vn1; rhat = un21 - q0 * vn1;
    while (q0 >= b || q0 * vn0 > b * rhat + un0) { q0--; rhat += vn1; if (rhat >= b) break; }
    if (r) *r = (un21 * b + un0 - q0 * v) >> s;
    return q1 * b + q0;
}
static const u8 debruijn32[32] = { 0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9 };
static unsigned long long udivmod64(unsigned long long n, unsigned long long d, unsigned long long *rem) {
    u32 n1 = (u32)(n >> 32), n0 = (u32)n;
    if ((d >> 32) == 0) {
        u32 dv = (u32)d;
        if (dv == 0) { if (rem) *rem = n; return ~0ULL; }
        u32 m = dv - 1;
        __asm__("" : "+r"(m)); /* see __umoddi3 */
        if ((dv & m) == 0) {
            /* power of two: mask and shift (log2 via de Bruijn multiply) */
            u32 s = debruijn32[(dv * 0x077CB531u) >> 27];
            if (rem) *rem = n0 & m;
            return n >> s;
        }
        u32 qh = n1 / dv, r1 = n1 % dv, r;
        u32 ql = divlu2(r1, n0, dv, &r);
        if (rem) *rem = r;
        return ((unsigned long long)qh << 32) | ql;
    }
    unsigned long long q = 0, r = 0;
    for (int i = 63; i >= 0; i--) {
        r = (r << 1) | ((n >> i) & 1);
        if (r >= d) { r -= d; q |= 1ULL << i; }
    }
    if (rem) *rem = r;
    return q;
}
unsigned long long __udivmoddi4(unsigned long long a, unsigned long long b, unsigned long long *rem) { return udivmod64(a, b, rem); }
unsigned long long __udivdi3(unsigned long long a, unsigned long long b) { return udivmod64(a, b, 0); }
unsigned long long __umoddi3(unsigned long long a, unsigned long long b) {
    u32 b0 = (u32)b;
    u32 m = b0 - 1;
    __asm__("" : "+r"(m)); /* keep clang from turning the test into a 20-instruction popcount */
    if ((b >> 32) == 0 && b0 != 0 && (b0 & m) == 0) return (u32)a & m; /* power of two */
    unsigned long long r; udivmod64(a, b, &r); return r;
}
long long __divdi3(long long a, long long b) {
    int neg = (a < 0) ^ (b < 0);
    unsigned long long ua = a < 0 ? (unsigned long long)(-(a + 1)) + 1u : (unsigned long long)a;
    unsigned long long ub = b < 0 ? (unsigned long long)(-(b + 1)) + 1u : (unsigned long long)b;
    unsigned long long q = udivmod64(ua, ub, 0);
    return neg ? -(long long)q : (long long)q;
}
long long __moddi3(long long a, long long b) {
    unsigned long long ua = a < 0 ? (unsigned long long)(-(a + 1)) + 1u : (unsigned long long)a;
    unsigned long long ub = b < 0 ? (unsigned long long)(-(b + 1)) + 1u : (unsigned long long)b;
    unsigned long long r; udivmod64(ua, ub, &r);
    return a < 0 ? -(long long)r : (long long)r;
}

/* ------------------------------------------- POSIX surface (deterministic) */
#include <sys/stat.h>
#include <time.h>
#include <glob.h>

int stat(const char *path, struct stat *st) {
    const u8 *data; u32 len;
    if (vfs_lookup(path, &data, &len)) {
        st->st_size = (off_t)len; st->st_mtime = 0; st->st_mode = 0100644;
        return 0;
    }
    errno = ENOENT;
    return -1;
}
char *basename(char *path) {
    char *s = strrchr(path, '/');
    return s ? s + 1 : path;
}
char *dirname(char *path) {
    char *s = strrchr(path, '/');
    if (!s) return ".";
    if (s == path) return "/";
    *s = 0;
    return path;
}
int glob(const char *pattern, int flags, int (*errfunc)(const char *, int), glob_t *g) {
    (void)pattern; (void)flags; (void)errfunc;
    g->gl_pathc = 0; g->gl_pathv = NULL;
    return 0;
}
void globfree(glob_t *g) { (void)g; }
/* The ZK build OS has no clock: the epoch is the only time. */
time_t time(time_t *t) { if (t) *t = 0; return 0; }
struct tm *localtime(const time_t *t) {
    static struct tm epoch = { 0, 0, 0, 1, 0, 70, 4, 0, 0 };
    (void)t;
    return &epoch;
}
char *ctime_r(const time_t *t, char *buf) { (void)t; strcpy(buf, "Thu Jan  1 00:00:00 1970\n"); return buf; }
static void (*atexit_fns[16])(void);
static int atexit_n;
int atexit(void (*fn)(void)) { if (atexit_n < 16) atexit_fns[atexit_n++] = fn; return 0; }
void _exit(int code) { zk_halt((u32)code); }
char *strtok(char *s, const char *delim) {
    static char *save;
    if (!s) s = save;
    if (!s) return NULL;
    while (*s && strchr(delim, *s)) s++;
    if (!*s) { save = NULL; return NULL; }
    char *start = s;
    while (*s && !strchr(delim, *s)) s++;
    if (*s) { *s = 0; save = s + 1; } else save = NULL;
    return start;
}
int mkstemp(char *t) { (void)t; zk_halt(92); return -1; }
int fork(void) { zk_halt(92); return -1; }
int execvp(const char *f, char *const argv[]) { (void)f; (void)argv; zk_halt(92); return -1; }
int wait(int *st) { (void)st; zk_halt(92); return -1; }
int waitpid(int p, int *st, int o) { (void)p; (void)st; (void)o; zk_halt(92); return -1; }
int unlink(const char *p) { (void)p; return 0; }
int close(int fd) { (void)fd; return 0; }
/* single-threaded atomics */
_Bool __atomic_compare_exchange_4(volatile void *p, void *expected, u32 desired, _Bool weak, int so, int fo) {
    (void)weak; (void)so; (void)fo;
    u32 *ptr = (u32 *)p; u32 *e = expected;
    if (*ptr == *e) { *ptr = desired; return 1; }
    *e = *ptr; return 0;
}

/* floating point parsing (decimal and hex); good enough for a compiler front end */
double strtod(const char *s, char **end) {
    const char *p = s;
    while (isspace((u8)*p)) p++;
    int neg = 0;
    if (*p == '+' || *p == '-') { neg = *p == '-'; p++; }
    double v = 0;
    if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X')) {
        p += 2;
        int exp2 = 0, any = 0;
        while (isxdigit((u8)*p)) { v = v * 16 + digit_val((u8)*p); p++; any = 1; }
        if (*p == '.') { p++; while (isxdigit((u8)*p)) { v = v * 16 + digit_val((u8)*p); exp2 -= 4; p++; any = 1; } }
        if (!any) { if (end) *end = (char *)s; return 0; }
        if (*p == 'p' || *p == 'P') {
            const char *q = p + 1; int eneg = 0, e = 0;
            if (*q == '+' || *q == '-') { eneg = *q == '-'; q++; }
            if (isdigit((u8)*q)) { while (isdigit((u8)*q)) e = e * 10 + (*q++ - '0'); exp2 += eneg ? -e : e; p = q; }
        }
        while (exp2 > 0) { v *= 2; exp2--; }
        while (exp2 < 0) { v /= 2; exp2++; }
    } else {
        int exp10 = 0, any = 0;
        while (isdigit((u8)*p)) { v = v * 10 + (*p - '0'); p++; any = 1; }
        if (*p == '.') { p++; while (isdigit((u8)*p)) { v = v * 10 + (*p - '0'); exp10--; p++; any = 1; } }
        if (!any) { if (end) *end = (char *)s; return 0; }
        if (*p == 'e' || *p == 'E') {
            const char *q = p + 1; int eneg = 0, e = 0;
            if (*q == '+' || *q == '-') { eneg = *q == '-'; q++; }
            if (isdigit((u8)*q)) { while (isdigit((u8)*q)) e = e * 10 + (*q++ - '0'); exp10 += eneg ? -e : e; p = q; }
        }
        while (exp10 > 0) { v *= 10; exp10--; }
        while (exp10 < 0) { v /= 10; exp10++; }
    }
    if (end) *end = (char *)p;
    return neg ? -v : v;
}
float strtof(const char *s, char **end) { return (float)strtod(s, end); }

/* -------------------------------------------------- syscall byte helpers */
void zk_write_output(const u8 *b, u32 n) {
    if (n == 0) return;
    zk_write_output_word(n);
    u32 words = (n + 3) >> 2;
    for (u32 i = 0; i < words; i++) {
        u32 w = 0;
        for (u32 k = 0; k < 4; k++) { u32 idx = (i << 2) + k; if (idx < n) w |= (u32)b[idx] << (k << 3); }
        zk_write_output_word(w);
    }
}
void zk_write_bytes32(const u8 *b) {
    for (int i = 0; i < 8; i++) {
        u32 w = (u32)b[4*i] | ((u32)b[4*i+1] << 8) | ((u32)b[4*i+2] << 16) | ((u32)b[4*i+3] << 24);
        zk_write_word(w);
    }
}

/* --------------------------------------------------------------- startup */
static u8 source_root[32], flags_root[32];
static u32 n_files;
static char *flags_str;

static void put_u32(blake3_ctx *c, u32 v) {
    u8 b[4] = { (u8)v, (u8)(v >> 8), (u8)(v >> 16), (u8)(v >> 24) };
    blake3_update(c, b, 4);
}

static u8 *read_blob(u32 *len_out) {
    u32 len = zk_read_word();
    u8 *buf = malloc(len + 1);
    u32 words = (len + 3) >> 2;
    for (u32 i = 0; i < words; i++) {
        u32 w = zk_read_word();
        u32 base = i << 2;
        for (u32 k = 0; k < 4; k++) if (base + k < len) buf[base + k] = (u8)(w >> (k << 3));
    }
    buf[len] = 0;
    *len_out = len;
    return buf;
}

void __assert_fail(const char *expr, const char *file, int line) {
    fprintf(stderr, "%s:%d: assertion failed: %s\n", file, line, expr);
    exit(134);
}
void abort(void) { exit(134); }

void exit(int code) {
    out_flush();
    u8 ohash[32];
    blake3_final(&out_hash, ohash);
    if (code != 0) {
        /* failing run: dump stderr into the private output so the host can show it */
        zk_write_output((const u8 *)"\n--- stderr ---\n", 16);
        zk_write_output(err_buf, err_len);
    }
    zk_write_bytes32(source_root);
    zk_write_bytes32(flags_root);
    zk_write_bytes32(ohash);
    zk_write_word(n_files);
    zk_halt((u32)code);
}

int main(int argc, char **argv);

void __zk_start(void) {
    blake3_init(&out_hash);
    u32 flen;
    flags_str = (char *)read_blob(&flen);
    blake3_ctx fr; blake3_init(&fr);
    u8 tag2 = 2; blake3_update(&fr, &tag2, 1);
    blake3_update(&fr, (u8 *)flags_str, flen);
    blake3_final(&fr, flags_root);

    n_files = zk_read_word();
    if (n_files > MAX_FILES) zk_halt(91);
    blake3_ctx root; blake3_init(&root);
    u8 tag1 = 1; blake3_update(&root, &tag1, 1);
    put_u32(&root, n_files);
    for (u32 i = 0; i < n_files; i++) {
        u32 nlen, clen;
        char *name = (char *)read_blob(&nlen);
        u8 *content = read_blob(&clen);
        blake3_ctx leaf; blake3_init(&leaf);
        u8 tag0 = 0; blake3_update(&leaf, &tag0, 1);
        put_u32(&leaf, nlen); blake3_update(&leaf, (u8 *)name, nlen);
        put_u32(&leaf, clen); blake3_update(&leaf, content, clen);
        u8 lh[32]; blake3_final(&leaf, lh);
        blake3_update(&root, lh, 32);
        vfs[i].name = name; vfs[i].data = content; vfs[i].len = clen;
    }
    blake3_final(&root, source_root);
    vfs_n = n_files;

    /* argv from flags (split on spaces) */
    static char *argv[64];
    int argc = 0;
    argv[argc++] = "zkcc";
    char *p = flags_str;
    while (*p) {
        while (*p == ' ') p++;
        if (!*p) break;
        argv[argc++] = p;
        while (*p && *p != ' ') p++;
        if (*p) *p++ = 0;
        if (argc >= 63) break;
    }
    argv[argc] = NULL;
    int rc = main(argc, argv);
    exit(rc);
}
