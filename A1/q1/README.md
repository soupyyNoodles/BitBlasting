# Question 1: Frequent Itemset Mining

## Compilation

To compile the Apriori and FP-Growth algorithms from source:

```bash
cd A1/q1
bash compile.sh
```

This will compile both algorithms and create executables in their respective directories.

## Task 1: Algorithm Comparison

Run the comparison on the webdocs dataset:

```bash
bash q1_1.sh <path_to_apriori> <path_to_fpgrowth> <path_to_dataset> <output_dir>
```

Example:
```bash
bash q1_1.sh \
    apriori/apriori/src/apriori \
    fpgrowth/fpgrowth/src/fpgrowth \
    /path/to/webdocs.dat \
    ./output
```

This will:
- Run both algorithms at support thresholds: 5%, 10%, 25%, 50%, 90%
- Save outputs to files: ap5, ap10, ap25, ap50, ap90, fp5, fp10, fp25, fp50, fp90
- Generate plot.png showing runtime comparison


## Files

- `compile.sh`: Compiles Apriori and FP-Growth
- `q1_1.sh`: Runs Task 1 pipeline
- `plot_results.py`: Python script for plotting
