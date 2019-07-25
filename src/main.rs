use std::env;
use std::fs::File;
use std::io::{BufReader, BufRead};
use ndarray::{Array1, Array2, Array3, s, arr1};
use crate::weights_buffer::{RandomWeightsBuffer, WeightsBuffer, FileWeightsBuffer};
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
    println!("----------left cov1d----------");
    for (i, e) in left.slice(s![0, .., ..]).genrows().into_iter().enumerate() {
        println!("{:#?}", e.slice(s![..3]));
        if i==5 {
            println!("...more...");
            break;
        }
    }
    println!("");

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
    println!("----------right cov1d----------");
    for (i, e) in right.slice(s![0, .., ..]).genrows().into_iter().enumerate() {
        println!("{:#?}", e.slice(s![..3]));
        if i==5 {
            println!("...more...");
            break;
        }
    }
    println!("");
    let out = batchnorm_add_activate(left, right.view(), weights);
    println!("----------batchnorm-add-activate----------");
    for (i, e) in out.slice(s![0, .., ..]).genrows().into_iter().enumerate() {
        println!("{:#?}", e.slice(s![..3]));
        if i==5 {
            println!("...more...");
            break;
        }
    }
    println!("");
    //std::process::exit(0);
    //batchnorm_add_activate(left, right.view(), weights)
    out
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
    println!("----------dense----------");
    for (i, e) in v2.slice(s![0, ..]).genrows().into_iter().enumerate() {
        println!("{:#?}", e.slice(s![..10]));
        if i==5 {
            println!("...more...");
            break;
        }
    }
    println!("");
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
    println!("----------dense----------");
    for (i, e) in v.slice(s![0, ..]).genrows().into_iter().enumerate() {
        println!("{:#?}", e.slice(s![..]));
        if i==5 {
            println!("...more...");
            break;
        }
    }
    dense_sigmoid(v, weights)
}

fn main() {
    let weights_filename = &env::args().collect::<Vec<_>>()[1];
    let mut weights = Box::new(FileWeightsBuffer::new(weights_filename)) as Box<dyn WeightsBuffer>;

    let inputs_filename_1 = &env::args().collect::<Vec<_>>()[2];
    let inputs_file_1 = BufReader::new(
        File::open(inputs_filename_1).unwrap());
    let mut iter_1 = inputs_file_1.lines();
    iter_1.next();
    let inputs_filename_2 = &env::args().collect::<Vec<_>>()[3];
    let inputs_file_2 = BufReader::new(
        File::open(inputs_filename_2).unwrap());
    let mut iter_2 = inputs_file_2.lines();
    iter_2.next();

    let n_inputs = 1usize;
    let input_len = 12634usize;
    let mut inputs = Array2::zeros((input_len, n_inputs));
    for (i, line) in iter_1.enumerate() {
        let line = line.unwrap()
                    .split_whitespace()
                    .map(|x| x.parse::<f32>().unwrap())
                    .skip(2)
                    .take(usize::min(50, n_inputs))
                    .collect::<Vec<_>>();
        inputs.slice_mut(s![i, ..usize::min(50, n_inputs)]).assign(
            &Array1::<f32>::from_vec(line));
    }
    if i64::max(0, n_inputs as i64 - 50) > 0 {
        for (i, line) in iter_2.enumerate() {
            let line = line.unwrap()
                .split_whitespace()
                .map(|x| x.parse::<f32>().unwrap())
                .skip(2)
                .take(n_inputs-50)
                .collect::<Vec<_>>();
            inputs.slice_mut(s![i, (n_inputs-50)..]).assign(
                &Array1::<f32>::from_vec(line));
        }
    }
    let inputs = Array2::<f32>::reversed_axes(inputs);
    let outputs = nn_eval(inputs, &mut weights);
    println!("outputs {:?}", outputs);
}

