import argparse
import json
import numpy as np
import os


def add_noise(y, x, noise_y_rate, noise_x_rate):
    assert(noise_y_rate >= 0 and noise_y_rate <= 1)
    assert(noise_x_rate >= 0 and noise_x_rate <= 1)

    tmp_y = np.array([int(k) for k in (y + np.ones_like(y)) / 2])
    noise_y = np.random.random(y.shape) < noise_y_rate

    new_y = np.bitwise_xor(tmp_y, noise_y)
    new_y = (2 * new_y) - np.ones_like(new_y)

    noise_x = np.random.random(x.shape) < noise_x_rate
    new_x = np.bitwise_xor(x, noise_x)

    return (new_y, new_x)


def validate_dataset(y, x, l, m, n, number_of_instances):
    for idx in range(number_of_instances):
        current_y = y[idx]
        current_x = x[idx]

        if current_y > 0 and np.sum(current_x[:m]) < l:
            print("Invalid positive example found.")
        elif current_y <= 0 and np.sum(current_x[:m]) >= l:
            print("Invalid negative example found.")


def generate(l, m, n, number_of_instances, noise_y_rate, noise_x_rate):
    assert(l <= m and m <= n)  # precondition

    # balanced dataset
    p_number_of_instances = int(number_of_instances / 2)
    n_number_of_instances = number_of_instances - p_number_of_instances

    # positive example
    p_y = np.ones((p_number_of_instances, 1), dtype=int)
    p_x_first_part = np.zeros((p_number_of_instances, m), dtype=int)
    p_x_second_part = (np.random.random((p_number_of_instances, n - m)) < 0.5)
    p_x_second_part = p_x_second_part.astype(int)

    # columnwise append
    p_x = np.append(p_x_first_part, p_x_second_part, axis=1)

    for i in range(p_number_of_instances):
        candidates = np.random.permutation(m)
        n_nonzeros = l
        active_features = candidates[:n_nonzeros]
        p_x[i][active_features] = 1  # set non_zeros to 1

    # negative example
    n_y = -1 * np.ones((n_number_of_instances, 1), dtype=int)
    n_x_first_part = np.zeros((n_number_of_instances, m), dtype=int)
    n_x_second_part = (np.random.random((n_number_of_instances, n - m)) < 0.5)
    n_x_second_part = n_x_second_part.astype(int)

    # columnwise append
    n_x = np.append(n_x_first_part, n_x_second_part, axis=1)

    for i in range(n_number_of_instances):
        candidates = np.random.permutation(m)
        n_nonzeros = l - 1
        active_features = candidates[:n_nonzeros]
        n_x[i][active_features] = 1  # set non_zero to 1

    y = np.append(p_y, n_y)
    x = np.append(p_x, n_x, axis=0)

    shuffle_indices = np.random.permutation(number_of_instances)
    y = y[shuffle_indices]
    x = x[shuffle_indices][:]

    # sanity check
    validate_dataset(y, x, l, m, n, number_of_instances)

    y, x = add_noise(y, x, noise_y_rate, noise_x_rate)

    return x, y


def save_x(x, file_path):
    with open(file_path, 'w') as out:
        for i in range(x.shape[0]):
            out.write(json.dumps(x[i].tolist()) + '\n')


def save_y(y, file_path):
    with open(file_path, 'w') as out:
        for i in range(y.shape[0]):
            out.write(str(y[i]) + '\n')


def main(args):
    dirname = os.path.dirname(args.x_output_file)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    x, y = generate(args.l, args.m, args.n, args.num_examples, args.label_noise, args.feature_noise)
    save_x(x, args.x_output_file)
    save_y(y, args.y_output_file)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('x_output_file', help='The name of the output file with the features')
    argp.add_argument('y_output_file', help='The name of the output file with the labels')
    argp.add_argument('l', type=int)
    argp.add_argument('m', type=int)
    argp.add_argument('n', type=int)
    argp.add_argument('num_examples', type=int)
    argp.add_argument('--feature-noise', type=float, default=0.001)
    argp.add_argument('--label-noise', type=float, default=0.05)
    args = argp.parse_args()
    main(args)
