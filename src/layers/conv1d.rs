use ndarray::{Array2, Array3, ArrayView3, s, Zip};
use crate::weights_buffer::WeightsBuffer;

pub fn conv1d(inputs: ArrayView3<f32>, n_kernel: usize, kernel_size: usize, 
          strides: usize, weights: &mut Box<dyn WeightsBuffer>) -> Array3<f32> {
    if kernel_size==1 {
        let mut outputs = Array3::<f32>::zeros(
            (inputs.shape()[0], n_kernel, (inputs.shape()[2]+strides-1)/strides));
        let weights = 
            Array2::<f32>::from_shape_vec((n_kernel, inputs.shape()[1]), 
                                          weights.getn(n_kernel*inputs.shape()[1]))
            .unwrap();
        Zip::from(outputs.outer_iter_mut())
            .and(inputs.outer_iter())
            .apply(|mut output, input| output.assign(&weights.dot(&input)) );
        return outputs;
    } if kernel_size==3 {
        let mut outputs = Array3::<f32>::zeros(
            (inputs.shape()[0], n_kernel, (inputs.shape()[2]+strides-1)/strides));
        let weights = 
            Array3::<f32>::from_shape_vec((n_kernel, inputs.shape()[1], kernel_size), 
                                          weights.getn(n_kernel*inputs.shape()[1]*
                                                       kernel_size))
            .unwrap();
        Zip::from(outputs.outer_iter_mut())
            .and(inputs.outer_iter())
            .apply(|mut output, input| {
                Zip::from(output.genrows_mut())
                    .and(weights.outer_iter())
                    .apply( |mut output_row, kernel| {
                        output_row[[0]] = 
                            (&kernel.slice(s![.., 1..]) * 
                             &input.slice(s![.., ..2]).to_owned()).sum();
                        let last = output_row.shape()[0]-1;
                        output_row[[last]] = 
                            (&kernel.slice(s![.., ..2]) * 
                             &input.slice(s![.., (input.shape()[1]-2)..])
                             .to_owned()).sum();

                        for (o, w) in output_row.slice_mut(s![1..last])
                            .iter_mut()
                            .step_by(strides)
                            .zip(input.windows((input.shape()[0], 3))
                                 .into_iter()
                                 .step_by(strides)) {
                                *o = (&w * &kernel).sum();
                            }
                    });
            } );
        return outputs;
    } else {
        panic!("invalid parameters");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr3;
    use crate::weights_buffer::TestWeightsBuffer;

    #[test]
    fn conv1d_k1s1_test() {
        let n_kernel = 2;
        let kernel_size = 1;
        let strides = 1;
        let inputs = arr3(&[[ [1.0, 2.0, 3.0], 
                              [4.0, 5.0, 6.0], ]]);
        let mut weights = Box::new(TestWeightsBuffer::new( vec![1., 2.,
                                                                3., 4.] )) 
            as Box<dyn WeightsBuffer>;
        let outputs = conv1d(inputs.view(), n_kernel, kernel_size, strides, &mut weights);
        assert_eq!(outputs, arr3(&[[[9., 12., 15.], [19., 26., 33.]]]));
    }

    #[test]
    fn conv1d_k3s1_test() {
        let n_kernel = 2;
        let kernel_size = 3;
        let strides = 1;
        let inputs = arr3(&[[ [1.0, 2.0, 3.0], 
                              [4.0, 5.0, 6.0], ]]);
        let mut weights = Box::new(TestWeightsBuffer::new( vec![1., 2., 3.,
                                                                4., 3., 6.,
                                                                7., 8., 9.,
                                                                10., 11., 12. ] ))
            as Box<dyn WeightsBuffer>;
        let outputs = conv1d(inputs.view(), n_kernel, kernel_size, strides, &mut weights);
        assert_eq!(outputs, arr3(&[[[50., 81., 46.], [130., 217., 154.]]]));
    }

    #[test]
    fn conv1d_k3s2_test() {
        let n_kernel = 2;
        let kernel_size = 3;
        let strides = 2;
        let inputs = arr3(&[[ [1.0, 2.0, 3.0, 4.0], 
                              [4.0, 5.0, 6.0, 7.0], ]]);
        let mut weights = Box::new(TestWeightsBuffer::new( vec![1., 2., 3.,
                                                                4., 3., 6.,
                                                                7., 8., 9.,
                                                                10., 11., 12. ] ))
            as Box<dyn WeightsBuffer>;
        let outputs = conv1d(inputs.view(), n_kernel, kernel_size, strides, &mut weights);
        assert_eq!(outputs, arr3(&[[[50., 56.], [130., 190.]]]));
    }
}
