    .section .text._start
    .globl _start
_start:
    li   sp, 0x00FFFFF0
    la   gp, __global_pointer$
    call main
    mv   a0, a0
    li   a7, 0          /* SYS_HALT with exit code in a0 */
    ecall
1:  j 1b
