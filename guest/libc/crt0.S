    .section .text._start
    .globl _start
_start:
    li   sp, 0x00FFFFF0
    call __zk_start
1:  j 1b
