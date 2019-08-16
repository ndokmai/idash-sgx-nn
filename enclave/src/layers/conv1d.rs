use ndarray::{Array3, ArrayView2, ArrayView3, s, Zip};
use ndarray_parallel::prelude::*;
use crate::weights_buffer::WeightsBuffer;

pub fn conv1d(inputs: ArrayView3<f32>, n_kernel: usize, kernel_size: usize, 
              strides: usize, weights: &Box<dyn WeightsBuffer>) -> Array3<f32> {
    if kernel_size==1 {
        let mut outputs = Array3::<f32>::zeros(
            (inputs.shape()[0], inputs.shape()[1], n_kernel));
        let weights = 
            ArrayView2::<f32>::from_shape((n_kernel, inputs.shape()[2]), 
                                          weights.getn_ref(n_kernel*inputs.shape()[2]))
            .unwrap();
        Zip::from(outputs.outer_iter_mut())
            .and(inputs.outer_iter())
            .par_apply(|mut output, input| output.assign(&input.dot(&weights.t())));
        return outputs;
    } if kernel_size==3 && strides==1 {
        let mut outputs = Array3::<f32>::zeros(
            (inputs.shape()[0], inputs.shape()[1], n_kernel));
        let weights = 
            ArrayView3::<f32>::from_shape((n_kernel, kernel_size, inputs.shape()[2]), 
                                          weights.getn_ref(n_kernel*inputs.shape()[2]*
                                                       kernel_size))
            .unwrap();
        Zip::from(outputs.outer_iter_mut())
            .and(inputs.outer_iter())
            .par_apply(|mut output, input| {
                Zip::from(&mut output.slice_mut(s![0, ..]))
                    .and(weights.outer_iter())
                    .apply(|o, kernel|  {
                        let shape = 2*input.shape()[1];
                        *o = kernel.slice(s![1.., ..])
                            .into_shape(shape)
                            .unwrap()
                            .dot(
                                &input.slice(s![..2, ..])
                                .into_shape(shape)
                                .unwrap());
                    });
                for (mut output_row, input_window) in output.genrows_mut()
                    .into_iter().skip(1)
                        .zip(input.windows((3, input.shape()[1])).into_iter()) {
                            Zip::from(&mut output_row)
                                .and(weights.outer_iter())
                                .apply(|o, kernel|  {
                                    let shape = 3*input_window.shape()[1];
                                    *o = kernel.into_shape(shape)
                                        .unwrap()
                                        .dot(
                                            &input_window.into_shape(shape)
                                            .unwrap());
                                });
                        }
                Zip::from(&mut output.slice_mut(s![-1, ..]))
                    .and(weights.outer_iter())
                    .apply(|o, kernel|  {
                        let shape = 2*input.shape()[1];
                        *o = kernel.slice(s![..2, ..])
                            .into_shape(shape)
                            .unwrap()
                            .dot(
                                &input.slice(s![-2.., ..])
                                .into_shape(shape)
                                .unwrap());
                    });

            });
        return outputs;
    } if kernel_size==3 && strides==2 {
        let mut outputs = Array3::<f32>::zeros(
            (inputs.shape()[0], (inputs.shape()[1]+1)/2, n_kernel));
        let weights = 
            ArrayView3::<f32>::from_shape((n_kernel, kernel_size, inputs.shape()[2]), 
                                          weights.getn_ref(n_kernel*inputs.shape()[2]*
                                                       kernel_size))
            .unwrap();

        Zip::from(outputs.outer_iter_mut())
            .and(inputs.outer_iter())
            .par_apply(|mut output, input| {
                if input.shape()[0] & 1 == 0 {
                    for (mut output_row, input_window) in output.genrows_mut()
                        .into_iter()
                            .zip(input.windows((3, input.shape()[1]))
                                 .into_iter()
                                 .step_by(2)) {
                                Zip::from(&mut output_row)
                                    .and(weights.outer_iter())
                                    .apply(|o, kernel|  {
                                        let shape = 3*input_window.shape()[1];
                                        *o = kernel.into_shape(shape)
                                            .unwrap()
                                            .dot(
                                                &input_window.into_shape(shape)
                                                .unwrap());
                                    });
                            }
                    Zip::from(&mut output.slice_mut(s![-1, ..]))
                        .and(weights.outer_iter())
                        .apply(|o, kernel|  {
                            let shape = 2*input.shape()[1];
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
                        .and(weights.outer_iter())
                        .apply(|o, kernel|  {
                            let shape = 2*input.shape()[1];
                            *o = kernel.slice(s![1.., ..])
                                .into_shape(shape)
                                .unwrap()
                                .dot(
                                    &input.slice(s![..2, ..])
                                    .into_shape(shape)
                                    .unwrap());
                        });
                    for (mut output_row, input_window) in output.genrows_mut()
                        .into_iter().skip(1)
                            .zip(input.slice(s![1.., ..])
                                 .windows((3, input.shape()[1]))
                                 .into_iter()
                                 .step_by(2)) {
                                Zip::from(&mut output_row)
                                    .and(weights.outer_iter())
                                    .apply(|o, kernel|  {
                                        let shape = 3*input_window.shape()[1];
                                        *o = kernel.into_shape(shape)
                                            .unwrap()
                                            .dot(
                                                &input_window.into_shape(shape)
                                                .unwrap());
                                    });
                            }
                    Zip::from(&mut output.slice_mut(s![-1, ..]))
                        .and(weights.outer_iter())
                        .apply(|o, kernel|  {
                            let shape = 2*input.shape()[1];
                            *o = kernel.slice(s![..2, ..])
                                .into_shape(shape)
                                .unwrap()
                                .dot(
                                    &input.slice(s![-2.., ..])
                                    .into_shape(shape)
                                    .unwrap());
                        });
                }
            });
        return outputs;
    } else {
        panic!("invalid parameters");
    }
}
