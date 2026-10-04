#ifndef _ZK_STDLIB_H
#define _ZK_STDLIB_H
#include <stddef.h>
void *malloc(size_t n);
void *calloc(size_t a, size_t b);
void *realloc(void *p, size_t n);
void free(void *p);
_Noreturn void exit(int code);
_Noreturn void abort(void);
unsigned long strtoul(const char *s, char **end, int base);
long strtol(const char *s, char **end, int base);
unsigned long long strtoull(const char *s, char **end, int base);
long long strtoll(const char *s, char **end, int base);
int atoi(const char *s);
char *getenv(const char *name);
int atexit(void (*fn)(void));
int mkstemp(char *tmpl);
double strtod(const char *s, char **end);
float strtof(const char *s, char **end);
void *alloca(unsigned long n);
#endif
