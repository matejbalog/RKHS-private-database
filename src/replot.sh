#!/usr/bin/env bash

# Figure 1
python plot.py --alg1 ../results/D2_alg1_leak_M10000.json --path_save ../figures/leaksD2
python plot.py --alg1 ../results/D5_alg1_leak_M10000.json --path_save ../figures/leaksD5

# Figure 2
python plot.py --alg1 ../results/D2_alg1_random_M10000.json --alg2 ../results/D2_alg2_random_M10000.json --path_save ../figures/nodataD2
python plot.py --alg1 ../results/D5_alg1_random_M10000.json --alg2 ../results/D5_alg2_random_M10000.json --path_save ../figures/nodataD5
