/* Exercises RV32M: reads two words, journals products/quotients/remainders (signed and unsigned). */
#include "zk.h"
int main(void) {
    u32 a = zk_read_word();
    u32 b = zk_read_word();
    int sa = (int)a, sb = (int)b;
    zk_write_word(a * b);
    zk_write_word((u32)(((long long)sa * (long long)sb) >> 32));
    zk_write_word((u32)(((long long)sa * (long long)(unsigned long long)b) >> 32));
    zk_write_word((u32)(((unsigned long long)a * (unsigned long long)b) >> 32));
    zk_write_word(b ? (u32)(sa / sb) : 0xFFFFFFFFu);
    zk_write_word(b ? a / b : 0xFFFFFFFFu);
    zk_write_word(b ? (u32)(sa % sb) : a);
    zk_write_word(b ? a % b : a);
    /* a few fixed edge cases */
    volatile int m = (int)0x80000000, n1 = -1;
    zk_write_word((u32)(m / n1));
    zk_write_word((u32)(m % n1));
    return 0;
}
