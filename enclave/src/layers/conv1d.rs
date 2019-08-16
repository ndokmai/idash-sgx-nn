use std::io::{Read};
use ndarray::{Array2, ArrayView3, ArrayView2, s, Zip};
use crate::params_buffer::ParamsBuffer;

fn conv1d_k1s1(input: ArrayView2<f32>, params: ArrayView2<f32>) -> Array2<f32> {
    let n_kernel = params.shape()[0];
    let mut output = Array2::<f32>::zeros((input.rows(), n_kernel));
    output.assign(&input.dot(&params.t()));
    output
}

fn conv1d_k3s1(input: ArrayView2<f32>, params: ArrayView3<f32>) -> Array2<f32> {
    let n_kernel = params.shape()[0];
    let mut output = Array2::<f32>::zeros((input.rows(), n_kernel));
    Zip::from(&mut output.slice_mut(s![0, ..]))
        .and(params.outer_iter())
        .apply(|o, kernel|  {
            let shape = 2*input.cols();
            *o = kernel.slice(s![1.., ..])
                .into_shape(shape)
                .unwrap()
                .dot(
                    &input.slice(s![..2, ..])
                    .into_shape(shape)
                    .unwrap());
        });

    let iter = output.genrows_mut()
        .into_iter()
        .skip(1)
        .zip(input.windows((3, input.cols())).into_iter());
   
    for (mut output_row, input_window) in iter {
        Zip::from(&mut output_row)
            .and(params.outer_iter())
            .apply(|o, kernel|  {
                let shape = 3*input_window.cols();
                *o = kernel.into_shape(shape)
                    .unwrap()
                    .dot(
                        &input_window.into_shape(shape)
                        .unwrap());
            });
    }

    Zip::from(&mut output.slice_mut(s![-1, ..]))
        .and(params.outer_iter())
        .apply(|o, kernel|  {
            let shape = 2*input.cols();
            *o = kernel.slice(s![..2, ..])
                .into_shape(shape)
                .unwrap()
                .dot(
                    &input.slice(s![-2.., ..])
                    .into_shape(shape)
                    .unwrap());
        });
    output
}

fn conv1d_k3s2(input: ArrayView2<f32>, params: ArrayView3<f32>) -> Array2<f32> {
    let n_kernel = params.shape()[0];
    let mut output = Array2::<f32>::zeros(((input.rows()+1)/2, n_kernel));
    if input.rows() & 1 == 0 {
        let iter = output.genrows_mut()
            .into_iter()
            .zip(input.windows((3, input.cols()))
                 .into_iter()
                 .step_by(2));
        for (mut output_row, input_window) in iter {
            Zip::from(&mut output_row)
                .and(params.outer_iter())
                .apply(|o, kernel|  {
                    let shape = 3*input_window.cols();
                    *o = kernel.into_shape(shape)
                        .unwrap()
                        .dot(
                            &input_window.into_shape(shape)
                            .unwrap());
                });
        }
        Zip::from(&mut output.slice_mut(s![-1, ..]))
            .and(params.outer_iter())
            .apply(|o, kernel|  {
                let shape = 2*input.cols();
                *o = kernel.slice(s![..2, ..])
                    .into_shape(shape)
                    .unwrap()
                    .dot(
                        &input.slice(s![-2.., ..])
                        .into_shape(shape)
                        .unwrap());
            });
    } else {
        Zip::from(&mut output.slice_mut(s![0, ..]))
            .and(params.outer_iter())
            .apply(|o, kernel|  {
                let shape = 2*input.cols();
                *o = kernel.slice(s![1.., ..])
                    .into_shape(shape)
                    .unwrap()
                    .dot(
                        &input.slice(s![..2, ..])
                        .into_shape(shape)
                        .unwrap());
            });
        let s = input.slice(s![1.., ..]);
        let iter = output.genrows_mut()
            .into_iter()
            .skip(1)
            .zip(s.windows((3, input.cols()))
                 .into_iter()
                 .step_by(2));
        for (mut output_row, input_window) in iter {
            Zip::from(&mut output_row)
                .and(params.outer_iter())
                .apply(|o, kernel|  {
                    let shape = 3*input_window.cols();
                    *o = kernel.into_shape(shape)
                        .unwrap()
                        .dot(
                            &input_window.into_shape(shape)
                            .unwrap());
                });
        }
        Zip::from(&mut output.slice_mut(s![-1, ..]))
            .and(params.outer_iter())
            .apply(|o, kernel|  {
                let shape = 2*input.cols();
                *o = kernel.slice(s![..2, ..])
                    .into_shape(shape)
                    .unwrap()
                    .dot(
                        &input.slice(s![-2.., ..])
                        .into_shape(shape)
                        .unwrap());
            });
    }
    output
}

fn conv1d_internal(input: ArrayView2<f32>, strides: usize, 
               params: ArrayView3<f32>) -> Array2<f32> {
    let kernel_size = params.shape()[1];
    if kernel_size==1 {
        let params = params
            .into_shape((params.shape()[0], params.shape()[2]))
            .unwrap();
        return conv1d_k1s1(input, params);
    } else if kernel_size==3 && strides==1 {
        return conv1d_k3s1(input, params);
    } else if kernel_size==3 && strides==2 {
        return conv1d_k3s2(input, params);
    } else {
        panic!("invalid parameters");
    }
}

pub fn conv1d<R: Read>(input: ArrayView2<f32>, n_kernel: usize, kernel_size: usize, 
              strides: usize, params_buf: &mut ParamsBuffer<R>) -> Array2<f32> {
    let params_buf = params_buf.get_next(n_kernel*kernel_size*input.cols());
    let params = 
        ArrayView3::<f32>::from_shape(
            (n_kernel, kernel_size, input.cols()), 
            params_buf.as_ref()).unwrap();
    conv1d_internal(input, strides, params)
}
