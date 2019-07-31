use ndarray::{Array1, Array3, ArrayView3, Zip};
use ndarray_parallel::prelude::*;
use crate::weights_buffer::WeightsBuffer;

pub fn batchnorm_add_activate(mut left_inputs: Array3<f32>, 
                          right_inputs: ArrayView3<f32>,
                          weights: &mut Box<dyn WeightsBuffer>) -> Array3<f32> {
    let left_batchnorm_params = 
        Array1::<f32>::from_shape_vec(
            (left_inputs.shape()[2], ),
            weights.getn(left_inputs.shape()[2],)).unwrap();
    let right_batchnorm_params = 
        Array1::<f32>::from_shape_vec(
            (left_inputs.shape()[2],), 
            weights.getn(left_inputs.shape()[2])).unwrap();
    let common_batchnorm_params = 
        Array1::<f32>::from_shape_vec(
            (left_inputs.shape()[2], ), 
            weights.getn(left_inputs.shape()[2])).unwrap();
    Zip::from(left_inputs.outer_iter_mut())
        .and(right_inputs.outer_iter())
        .par_apply(|mut l, r| {
            Zip::from(l.genrows_mut())
                .and(r.genrows())
                .apply(|mut l_c, r_c| {
                    Zip::from(&mut l_c)
                        .and(&r_c)
                        .and(&left_batchnorm_params)
                        .and(&right_batchnorm_params)
                        .and(&common_batchnorm_params)
                        .apply(|l_e, r_e, l_p, r_p, c_p| 
                               *l_e = f32::max(*l_e * *l_p + *r_e * *r_p + *c_p, 0.))
                });
        });
    left_inputs
}

#[allow(dead_code)]
pub fn batchnorm(mut inputs: Array3<f32>, 
                 weights: &mut Box<dyn WeightsBuffer>) -> Array3<f32> {
    let means = 
        Array1::<f32>::from_shape_vec(
            (inputs.shape()[1], ),
            weights.getn(inputs.shape()[1],)).unwrap();
    let variances = 
        Array1::<f32>::from_shape_vec(
            (inputs.shape()[1], ),
            weights.getn(inputs.shape()[1],)).unwrap();
    let gammas = 
        Array1::<f32>::from_shape_vec(
            (inputs.shape()[1], ),
            weights.getn(inputs.shape()[1],)).unwrap();
    let betas = 
        Array1::<f32>::from_shape_vec(
            (inputs.shape()[1], ),
            weights.getn(inputs.shape()[1],)).unwrap();
    Zip::from(inputs.outer_iter_mut())
        .apply(|mut input_col|
               Zip::from(input_col.gencolumns_mut())
               .apply(|mut input|
                      Zip::from(&mut input)
                      .and(&means)
                      .and(&variances)
                      .and(&gammas)
                      .and(&betas)
                      .apply(|i, m, v, g, b| *i = ((*i as f64-*m as f64)/f64::sqrt(*v as f64+1e-3) * *g as f64 + *b as f64) as f32)
                     )
              );
    inputs
}

#[allow(dead_code)]
pub fn batchnorm_add_activate_full(left_inputs: Array3<f32>, 
                          right_inputs: Array3<f32>,
                          weights: &mut Box<dyn WeightsBuffer>) -> Array3<f32> {
    let mut left_inputs = batchnorm(left_inputs, weights);
    let right_inputs = batchnorm(right_inputs, weights);
    Zip::from(left_inputs.genrows_mut())
        .and(right_inputs.genrows())
        .apply(|mut left_input, right_input|
               Zip::from(&mut left_input)
               .and(&right_input)
               .apply(|l, r| *l = f32::max(*l+*r, 0.))
               );
    left_inputs
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr3;
    use crate::weights_buffer::TestWeightsBuffer;
    
    #[test]
    fn batchnorm_add_activate_test() {
        let left_inputs = arr3(&[[ [1., 2.],
                                   [3., 4.] ]]);
        let right_inputs = arr3(&[[ [5., 6.],
                                    [7., 8.] ]]);
        let mut weights = Box::new(TestWeightsBuffer::new( 
                (1..7).map(|x| x as f32).collect::<Vec<f32>>())) 
            as Box<dyn WeightsBuffer>;
        let outputs = batchnorm_add_activate(left_inputs, right_inputs.view(),
                                                  &mut weights);
        assert_eq!(outputs, arr3(&[[[1.*1.+3.*5.+5., 1.*2.+3.*6.+5.], 
                                    [2.*3.+4.*7.+6., 2.*4.+4.*8.+6.]]]));

    }

    #[test]
    fn batchnorm_add_activate_negative_test() {
        let left_inputs = arr3(&[[ [1., 2.],
                                   [4., 5.] ]]);
        let right_inputs = arr3(&[[ [6., 7.],
                                    [8., 9.] ]]);
        let mut weights = Box::new(TestWeightsBuffer::new( 
                (-7..0).map(|x| x as f32).collect::<Vec<f32>>())) 
            as Box<dyn WeightsBuffer>;
        let outputs = batchnorm_add_activate(left_inputs, right_inputs.view(),
                                                  &mut weights);
        assert_eq!(outputs, arr3(&[[[0., 0.], [0., 0.]]]));

    }
}
