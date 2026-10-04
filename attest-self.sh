#!/usr/bin/env bash
# The compiler's own build, attested: chibicc running inside the zkVM
# compiles every translation unit of chibicc, each compilation gets a
# zero-knowledge proof (ObjectCertificate), the link step gets one too
# (LinkCertificate), and the whole thing is bundled into one BuildCertificate
# and verified. Then the certified objects are assembled natively into a
# stage-2 compiler and checked to reproduce stage-1 output (the fixpoint).
#
# Statement of the resulting certificate (see README "The statement"):
#   with toolchain `toolchain_root` (the zkcc + link guest images), the
#   sources `source_root_i` under `flags_root_i` compile to `object_hash_i`,
#   and the objects link to `output_hash`.
#
# Runtime: ~4 h on the 2-core reference machine (537 M cycles of
# compilation in 2^17-row segments at ~3.2 s each). Resumable: finished
# certificates are skipped.
#
# Usage: [JOBS=n] [OUT=dir] ./attest-self.sh [--no-zk]
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)
ZKFLAG=""
[ "${1:-}" = "--no-zk" ] && ZKFLAG="--no-zk"
OUT=${OUT:-/tmp/zkb-self}
JOBS=${JOBS:-1}
mkdir -p "$OUT/stage2"
guest/build.sh >/dev/null
guest/build-zkcc.sh >/dev/null
cargo build --release -q
Z=target/release/zkb
TC="--tu-guest guest/build/zkcc.elf --link-guest guest/build/link-guest.elf"
CC1="-cc1 -cc1-input"
TUS="strings unicode hashmap type main tokenize codegen preprocess parse"

echo "== toolchain"
$Z toolchain $TC | tee "$OUT/toolchain.txt"

echo "== 1. proving every translation unit of chibicc compiled by chibicc inside the zkVM (JOBS=$JOBS in parallel)"
prove_one() {
  tu="$1"
  d="$OUT/tu-$tu"; mkdir -p "$d"; cp guest/zkcc/chibicc/$tu.c guest/zkcc/chibicc/chibicc.h "$d/"
  if [ -s "$OUT/$tu.cert" ] && [ -s "$OUT/$tu.s" ]; then
    echo "   $tu.c: certificate exists, skipping"
    return 0
  fi
  t0=$(date +%s)
  $Z -v $ZKFLAG prove-tu $TC --flags "$CC1 $tu.c -cc1-output - $tu.c" --src "$d" --cert "$OUT/$tu.cert.tmp" --object "$OUT/$tu.s" > "$OUT/$tu.log" 2>&1
  mv "$OUT/$tu.cert.tmp" "$OUT/$tu.cert"
  summary=$(grep -o '[0-9]* cycles in [0-9]* segment(s)' "$OUT/$tu.log")
  size=$(du -h "$OUT/$tu.cert" | cut -f1)
  t1=$(date +%s)
  echo "   $tu.c: $summary in $((t1 - t0)) s, certificate $size"
}
export -f prove_one
export OUT Z ZKFLAG TC CC1
# JOBS>1 proves several TUs at once (each prover already uses every core for its
# NTTs; on 16+ cores two or three concurrent provers fill the machine better)
printf '%s\n' $TUS | xargs -P "$JOBS" -I{} bash -c 'prove_one {}'
CERTS=""
for tu in $TUS; do CERTS="$CERTS $OUT/$tu.cert"; done

echo "== 2. proving the link of the nine certified objects"
OBJS=$(for tu in $TUS; do printf '%s ' "$OUT/$tu.s"; done)
if [ ! -s "$OUT/link.cert" ]; then
  t0=$(date +%s)
  $Z -v $ZKFLAG link $TC --flags "" --objects $OBJS --cert "$OUT/link.cert.tmp" --output "$OUT/chibicc.bundle" > "$OUT/link.log" 2>&1
  mv "$OUT/link.cert.tmp" "$OUT/link.cert"
  echo "   link: $(grep -o '[0-9]* cycles in [0-9]* segment(s)' "$OUT/link.log") in $(( $(date +%s) - t0 )) s"
fi

echo "== 3. bundling and verifying the build certificate"
$Z bundle --objects $CERTS --link "$OUT/link.cert" --out "$OUT/chibicc-build.cert"
ls -la "$OUT/chibicc-build.cert" | awk '{print "   certificate size: " $5 " bytes"}'
$Z verify "$OUT/chibicc-build.cert" $TC --binary "$OUT/chibicc.bundle" | tee "$OUT/verify.txt"

echo "== 4. the certified objects, assembled natively, are the stage-2 compiler"
for tu in $TUS; do
  gcc -c -o "$OUT/stage2/$tu.o" "$OUT/$tu.s" 2>/dev/null
done
echo 'int errno;' > "$OUT/stage2/errno_shim.c"
gcc -o "$OUT/stage2/chibicc" "$OUT"/stage2/*.o "$OUT/stage2/errno_shim.c" 2>/dev/null
NAT="$OUT/native"
rm -rf "$NAT" && cp -r guest/zkcc/chibicc "$NAT" && (cd "$NAT" && clang -O2 -w -o chibicc *.c)
"$NAT/chibicc" -cc1 -cc1-input examples/fib/fib.c -cc1-output - examples/fib/fib.c > "$OUT/fib-stage1.s"
"$OUT/stage2/chibicc" -cc1 -cc1-input examples/fib/fib.c -cc1-output - examples/fib/fib.c > "$OUT/fib-stage2.s"
cmp "$OUT/fib-stage1.s" "$OUT/fib-stage2.s" && echo "   BOOTSTRAP FIXPOINT: the certified stage-2 compiler == stage-1 output"
# the certified objects are exactly what a native chibicc build of the same sources produces
for tu in $TUS; do
  (cd "$NAT" && "$NAT/chibicc" -cc1 -I"$NAT/include" -I"$ROOT/guest/libc/include" -cc1-input $tu.c -cc1-output - $tu.c \
     | sed "s#$ROOT/guest/libc/include#/usr/include#g; s#$NAT/include#./include#g" > "$OUT/native-$tu.s")
  cmp -s "$OUT/native-$tu.s" "$OUT/$tu.s" || { echo "   $tu.c: certified object differs from the native build"; exit 1; }
done
echo "   CERTIFIED OBJECTS == NATIVE BUILD: all nine in-VM compilations match native chibicc byte-for-byte"
echo "== done: $OUT/chibicc-build.cert"
