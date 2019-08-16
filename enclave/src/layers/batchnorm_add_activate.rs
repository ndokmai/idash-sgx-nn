use std::io::Read;
use ndarray::{Array2, ArrayView1, ArrayView2, Zip};
use crate::params_buffer::ParamsBuffer;


fn batchnorm_add_activate_internal(mut left_input: Array2<f32>,
                              right_input: ArrayView2<f32>,
                              left_batchnorm_params: ArrayView1<f32>,
                              right_batchnorm_params: ArrayView1<f32>,
                              common_batchnorm_params: ArrayView1<f32>,
                              ) -> Array2<f32> {
    Zip::from(left_input.genrows_mut())
        .and(right_input.genrows())
        .apply(|mut l_c, r_c| {
            Zip::from(&mut l_c)
                .and(&r_c)
                .and(&left_batchnorm_params)
                .and(&right_batchnorm_params)
                .and(&common_batchnorm_params)
                .apply(|l_e, r_e, l_p, r_p, c_p| 
                       *l_e = 
                       f32::max(*l_e * *l_p + *r_e * *r_p + *c_p, 0.))
        });
    left_input
}

pub fn batchnorm_add_activate<R: Read>(left_input: Array2<f32>, 
                          right_input: ArrayView2<f32>,
                         params_buf: &mut ParamsBuffer<R>) -> Array2<f32> {
    let shape = left_input.cols();
    let left_params_buf = params_buf.get_next(shape);
    let left_batchnorm_params = 
        ArrayView1::<f32>::from_shape((shape, ), left_params_buf.as_ref()).unwrap();
    let right_params_buf = params_buf.get_next(shape);
    let right_batchnorm_params = 
        ArrayView1::<f32>::from_shape((shape, ), right_params_buf.as_ref()).unwrap();
    let common_params_buf = params_buf.get_next(shape);
    let common_batchnorm_params = 
        ArrayView1::<f32>::from_shape((shape, ), common_params_buf.as_ref()).unwrap();
    batchnorm_add_activate_internal(left_input, right_input,
                                    left_batchnorm_params, 
                                    right_batchnorm_params,
                                    common_batchnorm_params)
}
