#!/usr/bin/env bash
# Build every guest program in this directory into build/<name>.elf.
# Toolchain: clang (RISC-V backend) + ld.lld, RV32I only (no M extension:
# multiplication/division go through libmini.c so the AIR stays RV32I).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
CC=${CC:-clang}
LD=${LD:-ld.lld}
CFLAGS="--target=riscv32-unknown-elf -march=rv32im -mabi=ilp32 -O2 -ffreestanding -fno-builtin -nostdlib -fno-stack-protector -fno-pic -mno-relax -Wall -Irt"
for src in *.c; do
  name="${src%.c}"
  $CC $CFLAGS -c rt/crt0.S -o build/crt0.o
  for f in rt/libmini.c rt/sha256.c rt/blake3.c "$src"; do
    $CC $CFLAGS -c "$f" -o "build/$(basename "${f%.c}").o"
  done
  $LD -T rt/link.ld -nostdlib -o "build/$name.elf" build/crt0.o build/libmini.o build/sha256.o build/blake3.o "build/$name.o"
  echo "built build/$name.elf"
done
