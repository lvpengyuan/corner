#!/usr/bin/env bash

CUDA_PATH=/usr/local/cuda/
cd rpsroi_pooling/src/cuda
rm rpsroi_pooling.cu.o
rm ../../_ext/rpsroi_pooling/_rpsroi_pooling.so
echo "Compiling roi pooling kernels by nvcc..."
nvcc -c -o rpsroi_pooling.cu.o rpsroi_pooling_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_52

#g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
#	roi_pooling_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64
#cd ../../
#python build.py
