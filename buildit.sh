#!/usr/bin/env bash

#CC="/opt/rh/llvm-toolset-7/root/usr/bin/clang"
#CXX="/opt/rh/llvm-toolset-7/root/usr/bin/clang++"
CC=gcc
CXX=g++

if [ ! -e build ]
then
  mkdir build
fi
cd build
CC=${CC} CXX=${CXX} cmake3 ..
make -j8 $1
