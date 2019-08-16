use ndarray::{Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut2, Zip};
use ndarray_parallel::prelude::*;
use crate::params_buffer::ParamsBuffer;

fn batchnorm_add_activate_single(mut left_input: ArrayViewMut2<f32>,
                                 right_intput: ArrayView2<f32>,
                                 left_batchnorm_params: ArrayView1<f32>,
                                 right_batchnorm_params: ArrayView1<f32>,
                                 common_batchnorm_params: ArrayView1<f32>) {
    Zip::from(left_input.genrows_mut())
        .and(right_intput.genrows())
        .apply(|mut l, r| {
            Zip::from(&mut l)
                .and(&r)
                .and(&left_batchnorm_params)
                .and(&right_batchnorm_params)
                .and(&common_batchnorm_params)
                .apply(|l_e, r_e, l_p, r_p, c_p| 
                       *l_e = f32::max(*l_e * *l_p + *r_e * *r_p + *c_p, 0.))
        });
}

pub fn batchnorm_add_activate(mut left_inputs: Array3<f32>, 
                          right_inputs: ArrayView3<f32>,
                          params_buf: &Box<dyn ParamsBuffer>) -> Array3<f32> {
    let left_batchnorm_params = 
        ArrayView1::<f32>::from_shape(
            (left_inputs.shape()[2], ),
            params_buf.getn_ref(left_inputs.shape()[2],)).unwrap();
    let right_batchnorm_params = 
        ArrayView1::<f32>::from_shape(
            (left_inputs.shape()[2],), 
            params_buf.getn_ref(left_inputs.shape()[2])).unwrap();
    let common_batchnorm_params = 
        ArrayView1::<f32>::from_shape(
            (left_inputs.shape()[2], ), 
            params_buf.getn_ref(left_inputs.shape()[2])).unwrap();
    Zip::from(left_inputs.outer_iter_mut())
        .and(right_inputs.outer_iter())
        .par_apply(|l, r| {
            batchnorm_add_activate_single(l, r, 
                                          left_batchnorm_params,
                                          right_batchnorm_params,
                                          common_batchnorm_params);
        });
    left_inputs
}
