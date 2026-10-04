#ifndef _ZK_STDATOMIC_H
#define _ZK_STDATOMIC_H
#define atomic_compare_exchange_strong(p, e, d) __atomic_compare_exchange_n(p, e, d, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)
#endif
