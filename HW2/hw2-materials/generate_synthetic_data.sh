python generate_synthetic_data.py \
  synthetic/sparse/train.X \
  synthetic/sparse/train.y \
  10 20 200 50000

python generate_synthetic_data.py \
  synthetic/sparse/dev.X \
  synthetic/sparse/dev.y \
  10 20 200 10000

python generate_synthetic_data.py \
  synthetic/sparse/test.X \
  synthetic/sparse/test.y \
  10 20 200 10000

python generate_synthetic_data.py \
  synthetic/dense/train.X \
  synthetic/dense/train.y \
  10 20 40 50000

python generate_synthetic_data.py \
  synthetic/dense/dev.X \
  synthetic/dense/dev.y \
  10 20 40 10000

python generate_synthetic_data.py \
  synthetic/dense/test.X \
  synthetic/dense/test.y \
  10 20 40 10000
