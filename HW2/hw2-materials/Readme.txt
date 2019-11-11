Resources for CIS 419/519 Fall 2019 Homework 2.

latex:
  The Latex template for the writeup.

ner:
  The CoNLL and Enron NER data

synthetic:
  The sparse and dense synthetic data

generate_synthetic_data.py:
  The code which was used to generate the synthetic dataset. It can be run
  like the following:

    python generate_synthetic_data.py \
      <x-output-file> \
      <y-output-file> \
      <l> \
      <m> \
      <n> \
      <num-examples> \
      --feature-noise <feature-noise> \
      --label-noise <label-noise>

  If you want to do the extra credit, you will have to regenerate data using
  this script with varying amounts of noise.

generate_synthetic_data.sh:
  A script which we used to generate the training/dev/test splits for the
  sparse and dense data

hw2.ipynb:
  The Python template code.
