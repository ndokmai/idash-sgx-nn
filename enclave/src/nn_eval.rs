use ndarray::{Array1, Array2, Array3, ArrayView2, s};
use crate::layers::*;
use crate::params_buffer::ParamsBuffer;

#[cfg(feature = "debug")]
fn debug_print(title: &str, array: ArrayView2<f32>) {
    const THRESHOLD: usize = 1000;
    const MARGIN: i32 = 3;
    println!("----------{}----------", title);
    if array.len() < THRESHOLD {
        let string = array.genrows()
            .into_iter()
            .map(|x| format!("[ {}\t]", x.iter()
                             .map(|y| format!("{:.10}", y))
                             .collect::<Vec<_>>()
                             .join(", ")))
            .collect::<Vec<_>>()
            .join("\n");
        print!("{}", string);
    } else {
        for (i, r) in array.genrows().into_iter().enumerate() {
            if i==MARGIN as usize {
                println!("...");
            } else if i < MARGIN as usize || i > array.rows() - MARGIN as usize - 1 {
                print!("[ ");
                for (j, c) in r.iter().enumerate() {
                    if j==MARGIN as usize {
                        print!("\t...\t");
                    } else if j == r.len() - 1 {
                       print!("{:.10}", c);
                    } else if j < MARGIN as usize || j > r.len() - MARGIN as usize - 1  {
                       print!("{:.10}\t", c);
                    } 
                }
                println!("\t]");
            } 
        }
    }
    println!("\n\n");
}

#[cfg(not(feature = "debug"))]
fn debug_print(_title: &str, _array: ArrayView2<f32>) {
    unimplemented!()
}


fn concat(input1: Array2<f32>, input2: Array2<f32>) -> Array2<f32> {
    let mut outputs = Array2::zeros((input1.shape()[0], 
                                     input2.shape()[1]+input1.shape()[1]));
    outputs.slice_mut(s![.., ..input1.shape()[1]]).assign(&input1);
    outputs.slice_mut(s![.., input1.shape()[1]..]).assign(&input2);
    outputs
}

fn res1d(inputs: Array3<f32>, n_kernel: usize, kernel_size: usize, strides: usize,
         params_buf: &Box<dyn ParamsBuffer>) -> Array3<f32> {
    let left = conv1d(inputs.view(), n_kernel, kernel_size, strides, params_buf);
    if cfg!(feature = "debug") {
        debug_print(&"left cov1d", left.slice(s![0, .., ..]));
    }

    let right = {
        if strides > 1 || inputs.shape()[1] != n_kernel {
            if strides > 1 {
                let avgpool = zeropad_avgpool(inputs);
                if cfg!(feature = "debug") {
                    debug_print(&"avg. pooling", avgpool.slice(s![0, .., ..]));
                }
                conv1d(avgpool.view(), n_kernel, 1, 1, params_buf)
            } else {
                conv1d(inputs.view(), n_kernel, 1, 1, params_buf)
            }
        } else {
            panic!("Incorrect parameters")
        }
    };
    if cfg!(feature = "debug") {
        debug_print(&"right cov1d", right.slice(s![0, .., ..]));
    }
    let out = batchnorm_add_activate(left, right.view(), params_buf);
    if cfg!(feature = "debug") {
        debug_print(&"batchnorm-add-activate", out.slice(s![0, .., ..]));
    }
    out
}

pub fn nn_eval(inputs: Array2<f32>, 
               params_buf: &Box<dyn ParamsBuffer>) -> Array1<f32> {
    let v2 = dense(inputs.view(), 64, params_buf);
    if cfg!(feature = "debug") {
        debug_print(&"dense", v2.slice(s![..1, ..]));
    }
    let shape = (inputs.shape()[0], inputs.shape()[1], 1);
    let inputs = inputs.into_shape(shape).unwrap();
    let mut v1 = res1d(inputs, 4, 3, 1, params_buf);
    for i in &[8usize, 16, 32, 64, 128, 256, 512, 1024] {
        v1 = res1d(v1, *i, 3, 1, params_buf);
        v1 = res1d(v1, *i, 3, 2, params_buf);
    }
    let shape = (v1.shape()[0], v1.shape()[1]*v1.shape()[2]);
    let v1 = v1.into_shape(shape).unwrap();
    let v = concat(v1, v2);
    let v = dense(v.view(), 32, params_buf);
    if cfg!(feature = "debug") {
        debug_print(&"dense", v.slice(s![..1, ..]));
    }
    dense_sigmoid(v, params_buf)
}

