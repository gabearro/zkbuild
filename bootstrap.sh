#!/usr/bin/env bash
# Attested-compiler bootstrap demo.
#
#  1. build the toolchain guest: unmodified-but-ported chibicc + zkbuild libc, RV32IM
#  2. compile examples/fib/fib.c *inside the zkVM* and prove it (ObjectCertificate)
#  3. self-host check: compile every chibicc translation unit inside the zkVM,
#     compare each output with a native chibicc build using the same sysroot,
#     assemble the in-VM outputs into a stage-2 compiler, and check that it
#     reproduces stage-1 output (the classic bootstrap fixpoint)
#  4. (optional, slow) prove the compilation of one of the compiler's own TUs
#
# Usage: ./bootstrap.sh [--prove-self <tu.c>] [--no-zk]
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)
PROVE_SELF=""
ZKFLAG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --prove-self) PROVE_SELF="$2"; shift 2;;
    --no-zk) ZKFLAG="--no-zk"; shift;;
    *) echo "unknown arg $1"; exit 1;;
  esac
done
OUT=${OUT:-/tmp/zkb-bootstrap}
mkdir -p "$OUT"
guest/build.sh >/dev/null
guest/build-zkcc.sh >/dev/null
cargo build --release -q
Z=target/release/zkb
TC="--tu-guest guest/build/zkcc.elf --link-guest guest/build/link-guest.elf"
CC1="-cc1 -cc1-input"

echo "== toolchain (compiler image is the toolchain commitment)"
$Z toolchain $TC

echo "== 1. attested compilation of examples/fib/fib.c (real compiler inside the zkVM)"
$Z -v $ZKFLAG prove-tu $TC --flags "$CC1 fib.c -cc1-output - fib.c" --src examples/fib --cert "$OUT/fib.cert" --object "$OUT/fib.s"
$Z -v $ZKFLAG link $TC --flags "" --objects "$OUT/fib.s" --cert "$OUT/fib-link.cert" --output "$OUT/fib.bundle"
$Z bundle --objects "$OUT/fib.cert" --link "$OUT/fib-link.cert" --out "$OUT/fib-build.cert"
$Z verify "$OUT/fib-build.cert" $TC
echo "   assembling the certified output natively and running it:"
gcc -o "$OUT/fib-bin" "$OUT/fib.s" 2>/dev/null && "$OUT/fib-bin" && echo "   fib(10)==55 -> exit 0 OK"

echo "== 2. self-host check: chibicc compiling chibicc inside the zkVM"
NAT="$OUT/native"
rm -rf "$NAT" && cp -r guest/zkcc/chibicc "$NAT" && (cd "$NAT" && clang -O2 -w -o chibicc *.c)
SRC="$OUT/src"; rm -rf "$SRC"; mkdir -p "$SRC"; cp guest/zkcc/chibicc/*.c guest/zkcc/chibicc/chibicc.h "$SRC/"
mkdir -p "$OUT/stage2"
for tu in strings unicode hashmap type main tokenize codegen preprocess parse; do
  d="$OUT/tu-$tu"; rm -rf "$d"; mkdir -p "$d"; cp "$SRC/$tu.c" "$SRC/chibicc.h" "$d/"
  $Z run guest/build/zkcc.elf --flags "$CC1 $tu.c -cc1-output - $tu.c" --src "$d" --output "$OUT/stage2/$tu.s" > "$OUT/run-$tu.log"
  cyc=$(grep -o 'cycles: [0-9]*' "$OUT/run-$tu.log")
  (cd "$SRC" && "$NAT/chibicc" -cc1 -I"$NAT/include" -I"$ROOT/guest/libc/include" -cc1-input $tu.c -cc1-output - $tu.c \
     | sed "s#$ROOT/guest/libc/include#/usr/include#g; s#$NAT/include#./include#g" > "$OUT/native-$tu.s")
  if cmp -s "$OUT/native-$tu.s" "$OUT/stage2/$tu.s"; then echo "   $tu.c: in-VM output == native output  ($cyc)"; else echo "   $tu.c: OUTPUT DIFFERS"; exit 1; fi
  gcc -c -o "$OUT/stage2/$tu.o" "$OUT/stage2/$tu.s" 2>/dev/null
done
echo 'int errno;' > "$OUT/stage2/errno_shim.c"
gcc -o "$OUT/stage2/chibicc" "$OUT"/stage2/*.o "$OUT/stage2/errno_shim.c" 2>/dev/null
"$NAT/chibicc" -cc1 -cc1-input examples/fib/fib.c -cc1-output - examples/fib/fib.c > "$OUT/fib-stage1.s"
"$OUT/stage2/chibicc" -cc1 -cc1-input examples/fib/fib.c -cc1-output - examples/fib/fib.c > "$OUT/fib-stage2.s"
cmp "$OUT/fib-stage1.s" "$OUT/fib-stage2.s" && echo "   BOOTSTRAP FIXPOINT: stage-2 compiler (built entirely inside the zkVM) == stage-1 output"

if [ -n "$PROVE_SELF" ]; then
  tu="${PROVE_SELF%.c}"
  echo "== 3. proving the compilation of the compiler's own $tu.c (this takes hours on a laptop)"
  $Z -v $ZKFLAG prove-tu $TC --flags "$CC1 $tu.c -cc1-output - $tu.c" --src "$OUT/tu-$tu" --cert "$OUT/self-$tu.cert" --object "$OUT/self-$tu.s"
  cmp "$OUT/self-$tu.s" "$OUT/stage2/$tu.s" && echo "   certified object == stage-2 input for $tu.c"
fi
