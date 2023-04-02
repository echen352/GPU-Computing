#!/bin/bash

size="256 512 1024 2048 3072 4096 5120 7680 8192 10240 12800"
make clean all
echo
for i in $size; do for j in {1..5}
do ./edges ../../images/Lenna_org_${i}.pgm; echo; done
done
