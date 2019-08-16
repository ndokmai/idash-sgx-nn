use ndarray::{Array3, ArrayView1, ArrayView3, Zip};
use ndarray_parallel::prelude::*;
use crate::weights_buffer::WeightsBuffer;

pub fn batchnorm_add_activate(mut left_inputs: Array3<f32>, 
                          right_inputs: ArrayView3<f32>,
                          weights: &Box<dyn WeightsBuffer>) -> Array3<f32> {
    let left_batchnorm_params = 
        ArrayView1::<f32>::from_shape(
            (left_inputs.shape()[2], ),
            weights.getn_ref(left_inputs.shape()[2],)).unwrap();
    let right_batchnorm_params = 
        ArrayView1::<f32>::from_shape(
            (left_inputs.shape()[2],), 
            weights.getn_ref(left_inputs.shape()[2])).unwrap();
    let common_batchnorm_params = 
        ArrayView1::<f32>::from_shape(
            (left_inputs.shape()[2], ), 
            weights.getn_ref(left_inputs.shape()[2])).unwrap();
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
