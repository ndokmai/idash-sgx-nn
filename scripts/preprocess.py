import sys
import h5py
import numpy as np
import struct

def write_elem(f, i):
    s = struct.pack('f', i)
    f.write(s)

def write_array(f, a):
    for i in a:
        write_elem(f, i)

def dense(dense_id, weights, fout):
    key = 'dense_' + str(dense_id)
    dataset = weights[key][key]
    kernels = np.transpose(dataset['kernel:0'])
    biases = dataset['bias:0']
    for (bias, kernel) in zip(biases, kernels):
        write_elem(fout, bias)
        write_array(fout, kernel)
    return dense_id + 1

def conv1d(conv1d_id, weights, fout):
    key = 'conv1d_' + str(conv1d_id)
    dataset = weights[key][key]
    kernels = np.transpose(dataset['kernel:0'])
    for kernel in kernels:
        for k_row in kernel:
            write_array(fout, k_row)
    return conv1d_id + 1

def batchnorm(batchnorm_id, weights, fout):
    key = 'batch_normalization_' + str(batchnorm_id)
    d = weights[key_1][key_1]
    write_array(fout, d['moving_mean:0'])
    write_array(fout, d['moving_variance:0'])
    write_array(fout, d['gamma:0'])
    write_array(fout, d['beta:0'])
    return batchnorm_id+1

def batchnorm_add_activate(batchnorm_id, weights, fout):
    key_1 = 'batch_normalization_' + str(batchnorm_id)
    key_2 = 'batch_normalization_' + str(batchnorm_id + 1)
    d_1 = weights[key_1][key_1]
    d_2 = weights[key_2][key_2]
    d_1_sd = np.sqrt(np.array(d_1['moving_variance:0'])+0.001)
    d_2_sd = np.sqrt(np.array(d_2['moving_variance:0'])+0.001)
    a_1 = np.divide(d_1['gamma:0'], d_1_sd)
    a_2 = np.divide(d_2['gamma:0'], d_2_sd)
    b = np.subtract(np.add(d_1['beta:0'], d_2['beta:0']), \
            np.add(
                np.divide(np.multiply(d_1['moving_mean:0'], d_1['gamma:0']), \
                          d_1_sd), \
                np.divide(np.multiply(d_2['moving_mean:0'], d_2['gamma:0']), \
                          d_2_sd)))
    write_array(fout, a_1)
    write_array(fout, a_2)
    write_array(fout, b)
    return batchnorm_id+2

def res1d(conv1d_id, batchnorm_id, weights, fout):
    conv1d_id = conv1d(conv1d_id, weights, weights_f) # left
    conv1d_id = conv1d(conv1d_id, weights, weights_f) # right
    batchnorm_id = batchnorm_add_activate(batchnorm_id, weights, weights_f)
    return (conv1d_id, batchnorm_id)

src_fname = sys.argv[1]
weights_fname = sys.argv[2]
src_f = h5py.File(src_fname, 'r')
weights = src_f['model_weights']
weights_f = open(weights_fname, 'wb')

dense_id = dense(1, weights, weights_f)
(conv1d_id, batchnorm_id) = res1d(1, 1, weights, weights_f)
for i in range(8):
    (conv1d_id, batchnorm_id) = res1d(conv1d_id, batchnorm_id, weights, weights_f)
    (conv1d_id, batchnorm_id) = res1d(conv1d_id, batchnorm_id, weights, weights_f)
dense_id = dense(dense_id, weights, weights_f)
dense_id = dense(dense_id, weights, weights_f)

src_f.close()
weights_f.close()
