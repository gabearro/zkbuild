#!/usr/bin/env bash
# Prove, bundle and verify the example project end to end.
set -euo pipefail
cd "$(dirname "$0")"
[ -f guest/build/tu-guest.elf ] || guest/build.sh
cargo build --release -q
Z=target/release/zkb
G=guest/build
OUT=${OUT:-/tmp/zkb-demo}
mkdir -p "$OUT"
TC="--tu-guest $G/tu-guest.elf --link-guest $G/link-guest.elf"

echo "== toolchain";      $Z toolchain $TC
echo "== source commits"; $Z commit examples/hello/tu1; $Z commit examples/hello/tu2
echo "== prove TU 1";     $Z -v prove-tu $TC --flags "-O2 -fPIC" --src examples/hello/tu1 --cert $OUT/tu1.cert --object $OUT/tu1.o
echo "== prove TU 2";     $Z -v prove-tu $TC --flags "-O2 -fPIC" --src examples/hello/tu2 --cert $OUT/tu2.cert --object $OUT/tu2.o
echo "== prove link";     $Z -v link $TC --flags "-static" --objects $OUT/tu1.o $OUT/tu2.o --cert $OUT/link.cert --output $OUT/app.bin
echo "== bundle";         $Z bundle --objects $OUT/tu1.cert $OUT/tu2.cert --link $OUT/link.cert --out $OUT/build.cert
echo "== verify";         $Z verify $OUT/build.cert $TC --binary $OUT/app.bin
