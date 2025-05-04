#!/bin/bash

tau=(0.01 0.02 0.05 0.10 0.20 0.40 0.60 0.90)
burn_in=(50 100 200 300 400 500 700 1000 2000)
n_samples=1000

echo "" > out

for t in "${tau[@]}"; do
  for b in "${burn_in[@]}"; do
    echo "Running with tau=$t and burn_in=$b" >> out
    python3 sampling_algos.py --n_samples "$n_samples" --burn_in "$b" --epsilon "$t" >> out
  done
done
