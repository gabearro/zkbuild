#!/usr/bin/env bash
# Build the real-compiler toolchain guest: unmodified chibicc (see zkcc/chibicc/COMMIT)
# on top of the zkbuild mini libc, as a single static RV32IM program.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build/zkcc
CC=${CC:-clang}
LD=${LD:-ld.lld}
CFLAGS="--target=riscv32-unknown-elf -march=rv32im -mabi=ilp32 -O2 -ffreestanding -fno-builtin -nostdlib -nostdlibinc -fno-stack-protector -fno-pic -mno-relax -fsigned-char -Wall -Wno-unused-function -Ilibc/include -Irt"
$CC $CFLAGS -c libc/crt0.S -o build/zkcc/crt0.o
$CC $CFLAGS -c libc/libc.c -o build/zkcc/libc.o
$CC $CFLAGS -c rt/blake3.c -o build/zkcc/blake3.o
python3 gen-sysroot.py build/zkcc/sysroot.c
$CC $CFLAGS -c build/zkcc/sysroot.c -o build/zkcc/sysroot.o
OBJS="build/zkcc/crt0.o build/zkcc/libc.o build/zkcc/blake3.o build/zkcc/sysroot.o"
for f in zkcc/chibicc/*.c; do
  o="build/zkcc/$(basename "${f%.c}").o"
  $CC $CFLAGS -Wno-sign-compare -Wno-switch -Wno-return-type -Wno-format -c "$f" -o "$o"
  OBJS="$OBJS $o"
done
# compiler-rt builtins (soft-float etc.), as an archive so only what is referenced is linked
mkdir -p build/zkcc/rt
RTOBJS=""
for f in libc/builtins/*.c; do
  o="build/zkcc/rt/$(basename "${f%.c}").o"
  $CC $CFLAGS -w -Ilibc/builtins -c "$f" -o "$o"
  RTOBJS="$RTOBJS $o"
done
rm -f build/zkcc/librt.a
llvm-ar rcs build/zkcc/librt.a $RTOBJS
$LD -T rt/link.ld -nostdlib -o build/zkcc.elf $OBJS build/zkcc/librt.a
echo "built build/zkcc.elf"
