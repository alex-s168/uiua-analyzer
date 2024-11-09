#!/bin/bash

set -e 

target=$(llvm-config --host-target)
cflags="-O3 -target $target -fPIC -Wall -Wextra -Werror -Wno-unused-function -Wno-unused-parameter"
clang="clang"

have_interpreter=0
if [[ $(grep -F "#define HAVE_INTERPRETER" "internal.h") == \#* ]]; then 
    have_interpreter=1 
fi

target_archsub=$(echo $target | cut -d '-' -f 1)
target_venoder=$(echo $target | cut -d '-' -f 2)
target_sys=$(echo $target | cut -d '-' -f 3)
target_env=$(echo $target | cut -d '-' -f 4)

rs_target_archsub=$target_archsub
rs_target_vendor=$target_venoder
rs_target_sys=$target_sys
rs_target_env=$target_env

if [[ $target_venoder == "w64" ]]; then 
    rs_target_vendor="pc"
fi 

rs_target="$rs_target_archsub-$rs_target_vendor-$rs_target_sys-$rs_target_env"

echo "configuration:"
echo "target = $target  (can be changed in build.sh)"
echo "rust_target = $rs_target  (depends on target)"
echo "build_interpreter = $have_interpreter  (can be changed in internal.h)"
echo ""

rm -rf build 
mkdir build

for f in *.c; do
    $clang $cflags -c -fPIC $f -o build/$f.o
done

ar r build/rt_part0.a build/*.o 2>/dev/null

if [[ $have_interpreter -eq 1 ]]; then 
    export cc=$clang 

    rustup target add $rs_target

    cd rustrt
    cargo build --target $rs_target
    cp target/debug/rustrt.a ../build/ rt_part1.a
    cd ..
fi

echo "done!"
