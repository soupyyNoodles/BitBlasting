#!/bin/bash

# Script to compile Apriori and FP-Growth algorithms

echo "Compiling Apriori..."
cd apriori/apriori/src
make clean 2>/dev/null
make -f makefile
if [ $? -ne 0 ]; then
    echo "Error: Apriori compilation failed"
    exit 1
fi
cd ../../..

echo "Compiling FP-Growth..."
cd fpgrowth/fpgrowth/src
make clean 2>/dev/null
make -f makefile
if [ $? -ne 0 ]; then
    echo "Error: FP-Growth compilation failed"
    exit 1
fi
cd ../../..

echo "Compilation successful!"
