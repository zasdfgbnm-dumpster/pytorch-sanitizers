#!/usr/bin/xonsh

import glob

files = glob.glob('pytorch/aten/src/ATen/native/cuda/*.cu')
for f in files:
    nvcc @(f) -dc -o /dev/null -Xptxas=-Werror -Xptxas=-warn-lmem-usage,-warn-spills --extended-lambda --expt-relaxed-constexpr -Ipytorch -Ipytorch/aten/src/ -Ipytorch/build -Ipytorch/build/aten/src -Ipytorch/build/caffe2/aten/src
