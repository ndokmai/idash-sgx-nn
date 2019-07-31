use ndarray::{Array2, Array3, ArrayView3, s, Zip};
use ndarray_parallel::prelude::*;
use crate::weights_buffer::WeightsBuffer;

pub fn conv1d(inputs: ArrayView3<f32>, n_kernel: usize, kernel_size: usize, 
          strides: usize, weights: &mut Box<dyn WeightsBuffer>) -> Array3<f32> {
    if kernel_size==1 {
        let mut outputs = Array3::<f32>::zeros(
            (inputs.shape()[0], inputs.shape()[1], n_kernel));
        let weights = 
            Array2::<f32>::from_shape_vec((n_kernel, inputs.shape()[2]), 
                                          weights.getn(n_kernel*inputs.shape()[2]))
            .unwrap();
        Zip::from(outputs.outer_iter_mut())
            .and(inputs.outer_iter())
            .par_apply(|mut output, input| output.assign(&input.dot(&weights.t())));
        return outputs;
    } if kernel_size==3 && strides==1 {
        let mut outputs = Array3::<f32>::zeros(
            (inputs.shape()[0], inputs.shape()[1], n_kernel));
        let weights = 
            Array3::<f32>::from_shape_vec((n_kernel, kernel_size, inputs.shape()[2]), 
                                          weights.getn(n_kernel*inputs.shape()[2]*
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

            });
        return outputs;
    } if kernel_size==3 && strides==2 {
        let mut outputs = Array3::<f32>::zeros(
            (inputs.shape()[0], (inputs.shape()[1]+1)/2, n_kernel));
        let weights = 
            Array3::<f32>::from_shape_vec((n_kernel, kernel_size, inputs.shape()[2]), 
                                          weights.getn(n_kernel*inputs.shape()[2]*
                                                       kernel_size))
            .unwrap();

        Zip::from(outputs.outer_iter_mut())
            .and(inputs.outer_iter())
            .par_apply(|mut output, input| {
                if input.shape()[0] & 1 == 0 {
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
                }
            });
        return outputs;
    } else {
        panic!("invalid parameters");
    }
}

//#[cfg(test)]
//mod tests {
    //use super::*;
    //use ndarray::arr3;
    //use crate::weights_buffer::TestWeightsBuffer;

    //#[test]
    //fn conv1d_k1s1_test() {
        //let n_kernel = 2;
        //let kernel_size = 1;
        //let strides = 1;
        //let inputs = arr3(&[[ [1.0, 2.0, 3.0], 
                              //[4.0, 5.0, 6.0], ]]);
        //let mut weights = Box::new(TestWeightsBuffer::new( vec![1., 2.,
                                                                //3., 4.] )) 
            //as Box<dyn WeightsBuffer>;
        //let outputs = conv1d(inputs.view(), n_kernel, kernel_size, strides, &mut weights);
        //assert_eq!(outputs, arr3(&[[[9., 12., 15.], [19., 26., 33.]]]));
    //}

    //#[test]
    //fn conv1d_k3s1_test() {
        //let n_kernel = 2;
        //let kernel_size = 3;
        //let strides = 1;
        //let inputs = arr3(&[[ [1.0, 2.0, 3.0], 
                              //[4.0, 5.0, 6.0], ]]);
        //let mut weights = Box::new(TestWeightsBuffer::new( vec![1., 2., 3.,
                                                                //4., 3., 6.,
                                                                //7., 8., 9.,
                                                                //10., 11., 12. ] ))
            //as Box<dyn WeightsBuffer>;
        //let outputs = conv1d(inputs.view(), n_kernel, kernel_size, strides, &mut weights);
        //assert_eq!(outputs, arr3(&[[[50., 81., 46.], [130., 217., 154.]]]));
    //}

    //#[test]
    //fn conv1d_k3s2_test() {
        //let n_kernel = 2;
        //let kernel_size = 3;
        //let strides = 2;
        //let inputs = arr3(&[[ [1.0, 2.0, 3.0, 4.0], 
                              //[4.0, 5.0, 6.0, 7.0], ]]);
        //let mut weights = Box::new(TestWeightsBuffer::new( vec![1., 2., 3.,
                                                                //4., 3., 6.,
                                                                //7., 8., 9.,
                                                                //10., 11., 12. ] ))
            //as Box<dyn WeightsBuffer>;
        //let outputs = conv1d(inputs.view(), n_kernel, kernel_size, strides, &mut weights);
        //assert_eq!(outputs, arr3(&[[[50., 56.], [130., 190.]]]));
    //}
//}
