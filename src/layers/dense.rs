use ndarray::{Array1, Array2, ArrayView2, Zip, s};
use ndarray_parallel::prelude::*;
use crate::weights_buffer::WeightsBuffer;

pub fn dense(inputs: ArrayView2<f32>, n_units: usize, 
         weights: &mut Box<dyn WeightsBuffer>) -> Array2<f32> {
    let mut outputs = Array2::zeros((inputs.shape()[0], n_units));
    let weights = 
        Array2::<f32>::from_shape_vec(
            (n_units, inputs.shape()[1]+1), 
            weights.getn((inputs.shape()[1]+1) * n_units)).unwrap();

    Zip::from(outputs.genrows_mut())
        .and(inputs.genrows())
        .par_apply(|mut output, input| {
            Zip::from(&mut output)
                .and(weights.genrows())
                .apply(|o, weights_row| {
                    let dot = input.dot(&weights_row.slice(s![1..]));
                    let bias = weights_row[0];
                    *o = f32::max(dot+bias, 0.)
                })
        });
    outputs
}

pub fn dense_sigmoid(inputs: Array2<f32>, 
                     weights: &mut Box<dyn WeightsBuffer>) -> Array1<f32> {
    let mut outputs = Array1::zeros(inputs.shape()[0]);
    let bias = weights.getn(1);
    let weights = Array1::<f32>::from_shape_vec(
        (inputs.shape()[1],), 
        weights.getn(inputs.shape()[1])).unwrap();
    Zip::from(&mut outputs)
        .and(inputs.genrows())
        .par_apply(|output, input| *output = 1./(1.+(-input.dot(&weights)-bias[0]).exp()));
    outputs
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, arr1};
    use crate::weights_buffer::TestWeightsBuffer;

    #[test]
    fn dense_test() {
        let n_units = 2;
        let inputs = arr2(&[[1., 2., ], [3., 4.]]);
        let mut weights = Box::new(TestWeightsBuffer::new( 
                (1..7).map(|x| x as f32).collect::<Vec<f32>>())) 
            as Box<dyn WeightsBuffer>;
        let outputs = dense(inputs.view(), n_units, &mut weights);
        assert_eq!(outputs, arr2(&[[9., 21.], [19., 43.]]));
    }

    #[test]
    fn dense_negative_test() {
        let n_units = 2;
        let inputs = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
        let mut weights = Box::new(TestWeightsBuffer::new( 
                (-8..0).map(|x| x as f32).collect::<Vec<f32>>())) 
            as Box<dyn WeightsBuffer>;
        let outputs = dense(inputs.view(), n_units, &mut weights);
        assert_eq!(outputs, arr2(&[[0., 0.], [0., 0.]]));
    }

    #[test]
    fn dense_sigmoid_test() {
        let inputs = arr2(&[[1., 2., 3.], [4., 5., 6.]]);
        let mut weights = Box::new(TestWeightsBuffer::new( 
                (-4..0).map(|x| x as f32).collect::<Vec<f32>>())) 
            as Box<dyn WeightsBuffer>;
        let outputs = dense_sigmoid(inputs, &mut weights);
        assert_eq!(outputs, arr1(&[0.00000083152804, 0.000000000000012664166]));
    }
}
