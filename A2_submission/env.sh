#!/bin/bash
# Environment setup script for COL761 Assignment 2

# Compile C++ files for Q2
pip install numpy matplotlib scikit-learn

cd q2
g++ -O3 -std=c++11 forest_fire.cpp -o forest_fire
cd ..
