#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Compile if needed
if [ ! -f "$DIR/forest_fire" ] || [ "$DIR/forest_fire.cpp" -nt "$DIR/forest_fire" ]; then
    echo "Compiling forest_fire.cpp..."
    g++ -O3 -std=c++11 "$DIR/forest_fire.cpp" -o "$DIR/forest_fire"
fi


if [ ! -f "$DIR/forest_fire_advanced" ] || [ "$DIR/forest_fire_advanced.cpp" -nt "$DIR/forest_fire_advanced" ]; then
    echo "Compiling forest_fire_advanced.cpp..."
    g++ -O3 -std=c++11 "$DIR/forest_fire_advanced.cpp" -o "$DIR/forest_fire_advanced"
fi

if [ ! -f "$DIR/forest_fire_ris" ] || [ "$DIR/forest_fire_ris.cpp" -nt "$DIR/forest_fire_ris" ]; then
    echo "Compiling forest_fire_ris.cpp..."
    g++ -O3 -std=c++11 "$DIR/forest_fire_ris.cpp" -o "$DIR/forest_fire_ris"
fi


echo "========================================================"
echo "EVALUATING ON DATASET 1"
echo "========================================================"
echo "Running original strategy..."
"$DIR/forest_fire" "$DIR/dataset1/dataset_1.txt" "$DIR/dataset1/seedset_1.txt" "$DIR/output_1.txt" 50 50 -1
bash "$DIR/Eval/evaluate.sh" "$DIR/dataset1/dataset_1.txt" "$DIR/dataset1/seedset_1.txt" "$DIR/output_1.txt" 50 50 -1
echo "--------------------------------------------------------"

echo "Running AdvancedGreedy strategy..."
"$DIR/forest_fire_advanced" "$DIR/dataset1/dataset_1.txt" "$DIR/dataset1/seedset_1.txt" "$DIR/output_1_advanced.txt" 50 50 -1
bash "$DIR/Eval/evaluate.sh" "$DIR/dataset1/dataset_1.txt" "$DIR/dataset1/seedset_1.txt" "$DIR/output_1_advanced.txt" 50 50 -1
echo "--------------------------------------------------------"

echo "Running Edge-RIS strategy..."
"$DIR/forest_fire_ris" "$DIR/dataset1/dataset_1.txt" "$DIR/dataset1/seedset_1.txt" "$DIR/output_1_ris.txt" 50 50 -1
bash "$DIR/Eval/evaluate.sh" "$DIR/dataset1/dataset_1.txt" "$DIR/dataset1/seedset_1.txt" "$DIR/output_1_ris.txt" 50 50 -1


echo ""
echo "========================================================"
echo "EVALUATING ON DATASET 2"
echo "========================================================"
echo "Running original strategy..."
"$DIR/forest_fire" "$DIR/dataset2/dataset_2.txt" "$DIR/dataset2/seedset_2.txt" "$DIR/output_2.txt" 20 50 3
bash "$DIR/Eval/evaluate.sh" "$DIR/dataset2/dataset_2.txt" "$DIR/dataset2/seedset_2.txt" "$DIR/output_2.txt" 20 50 3
echo "--------------------------------------------------------"

echo "Running AdvancedGreedy strategy..."
"$DIR/forest_fire_advanced" "$DIR/dataset2/dataset_2.txt" "$DIR/dataset2/seedset_2.txt" "$DIR/output_2_advanced.txt" 20 50 3
bash "$DIR/Eval/evaluate.sh" "$DIR/dataset2/dataset_2.txt" "$DIR/dataset2/seedset_2.txt" "$DIR/output_2_advanced.txt" 20 50 3
echo "--------------------------------------------------------"

echo "Running Edge-RIS strategy..."
"$DIR/forest_fire_ris" "$DIR/dataset2/dataset_2.txt" "$DIR/dataset2/seedset_2.txt" "$DIR/output_2_ris.txt" 20 50 3
bash "$DIR/Eval/evaluate.sh" "$DIR/dataset2/dataset_2.txt" "$DIR/dataset2/seedset_2.txt" "$DIR/output_2_ris.txt" 20 50 3
