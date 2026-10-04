#ifndef _ZK_ASSERT_H
#define _ZK_ASSERT_H
_Noreturn void __assert_fail(const char *expr, const char *file, int line);
#define assert(e) ((e) ? (void)0 : __assert_fail(#e, __FILE__, __LINE__))
#endif
