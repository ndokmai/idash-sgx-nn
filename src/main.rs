use ndarray::{Array1, Array2, Array3, s};
use crate::weights_buffer::{RandomWeightsBuffer, WeightsBuffer};
use crate::layers::{conv1d::conv1d, 
    batchnorm_add_activate::batchnorm_add_activate,
    zeropad_avgpool::zeropad_avgpool,
    dense::dense, 
    dense::dense_sigmoid, 
};

mod weights_buffer;
mod layers;

fn res1d(inputs: Array3<f32>, n_kernel: usize, kernel_size: usize, strides: usize,
         weights: &mut Box<dyn WeightsBuffer>) -> Array3<f32> {
    let left = conv1d(inputs.view(), n_kernel, kernel_size, strides, weights);
    let right = {
        if strides > 1 || inputs.shape()[1] != n_kernel {
            if strides > 1 {
                conv1d(zeropad_avgpool(inputs).view(), n_kernel, 1, 1, weights)
            } else {
                conv1d(inputs.view(), n_kernel, 1, 1, weights)
            }
        } else {
            panic!("Incorrect parameters")
        }
    };
    batchnorm_add_activate(left, right.view(), weights)
}

fn concat(input1: Array2<f32>, input2: Array2<f32>) -> Array2<f32> {
    let mut outputs = Array2::zeros((input1.shape()[0], 
                                     input2.shape()[1]+input1.shape()[1]));
    outputs.slice_mut(s![.., ..input1.shape()[1]]).assign(&input1);
    outputs.slice_mut(s![.., input1.shape()[1]..]).assign(&input2);
    outputs
}

fn nn_eval(inputs: Array2<f32>, weights: &mut Box<dyn WeightsBuffer>) -> Array1<f32> {
    let v2 = dense(inputs.view(), 64, weights);
    let shape = (inputs.shape()[0], 1, inputs.shape()[1]);
    let inputs = inputs.into_shape(shape).unwrap();
    let mut v1 = res1d(inputs, 4, 3, 1, weights);
    for i in &[8usize, 16, 32, 64, 128, 256, 512, 1024] {
        v1 = res1d(v1, *i, 3, 1, weights);
        v1 = res1d(v1, *i, 3, 2, weights);
    }
    let shape = (v1.shape()[0],v1.shape()[1]*v1.shape()[2]);
    let v1 = v1.into_shape(shape).unwrap();
    let v = concat(v1, v2);
    let v = dense(v.view(), 32, weights);
    dense_sigmoid(v, weights)
}

fn main() {
    let input_len = 12634usize;
    let dummy_inputs_feed = (1..(input_len+1))
        .map(|x| (x*37%11/100) as f32)
        .collect::<Vec<_>>();
    let inputs = Array2::from_shape_vec((1, input_len), dummy_inputs_feed)
        .unwrap();
    let mut weights = Box::new(RandomWeightsBuffer::new()) as Box<dyn WeightsBuffer>;
    let outputs = nn_eval(inputs, &mut weights);
    println!("outputs {:?}", outputs);
}

