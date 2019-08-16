use std::io::{Read};
use ndarray::{Array1, Array2, ArrayView2, s};
use crate::params_buffer::ParamsBuffer;
use crate::layers::*;


#[allow(dead_code)]
fn debug_print(title: &str, array: ArrayView2<f32>) {
    const THRESHOLD: usize = 1000;
    const MARGIN: i32 = 3;
    println!("----------{} {:?}----------", title, array.shape());
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
fn res1d<R: Read>(input: Array2<f32>, n_kernel: usize, kernel_size: usize, 
                  strides: usize, params_buf: &mut ParamsBuffer<R>) -> Array2<f32> {
    let left = conv1d(input.view(), n_kernel, kernel_size, strides, params_buf);
    //debug_print(&"left cov1d", left.view());

    let right = {
        if strides > 1 || input.rows() != n_kernel {
            if strides > 1 {
                let avgpool = zeropad_avgpool(input);
                //debug_print(&"avg. pooling", avgpool.view());
                conv1d(avgpool.view(), n_kernel, 1, 1, params_buf)
            } else {
                conv1d(input.view(), n_kernel, 1, 1, params_buf)
            }
        } else {
            panic!("Incorrect parameters")
        }
    };
    //debug_print(&"right cov1d", right.view());

    let out = batchnorm_add_activate(left, right.view(), params_buf);
    //debug_print(&"batchnorm-add-activate", out.view());
    out
}

fn concat(input1: Array1<f32>, input2: Array1<f32>) -> Array1<f32> {
    let mut output = Array1::zeros(input1.len()+input2.len());
    output.slice_mut(s![..input1.len()]).assign(&input1);
    output.slice_mut(s![input1.len()..]).assign(&input2);
    output
}

pub fn nn_eval<R: Read>(input: Array1<f32>, mut params_buf: ParamsBuffer<R>) -> f32 {
    let params_buf = &mut params_buf;
    let v2 = dense(input.view(), 64, params_buf);
    //debug_print(&"dense", v2.view().into_shape((1, v2.len())).unwrap());
    let shape = (input.len(), 1);
    let input = input.into_shape(shape).unwrap();
    let mut v1 = res1d(input, 4, 3, 1, params_buf);
    for i in &[8usize, 16, 32, 64, 128, 256, 512, 1024] {
        v1 = res1d(v1, *i, 3, 1, params_buf);
        v1 = res1d(v1, *i, 3, 2, params_buf);
    }
    let shape = v1.rows()*v1.cols();
    let v1 = v1.into_shape(shape).unwrap();
    let v = concat(v1, v2);
    let v = dense(v.view(), 32, params_buf);
    //debug_print(&"dense", v.view().into_shape((1, v.len())).unwrap());
    dense_sigmoid(v, params_buf)
}
