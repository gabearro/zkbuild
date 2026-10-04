#ifndef _ZK_STDDEF_H
#define _ZK_STDDEF_H
typedef unsigned long size_t;
typedef int ptrdiff_t;
typedef int wchar_t;
#define NULL ((void *)0)
#define offsetof(t, m) __builtin_offsetof(t, m)
#endif
