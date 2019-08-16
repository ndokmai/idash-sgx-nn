use std::io::Read;
use ndarray::{Array1, ArrayView1, ArrayView2, Zip, s};
use crate::params_buffer::ParamsBuffer;

fn dense_internal(input: ArrayView1<f32>, params: ArrayView2<f32>) -> Array1<f32> {
    let mut output = Array1::zeros(params.rows());
    Zip::from(&mut output)
        .and(params.genrows())
        .apply(|o, params_row| {
            let dot = input.dot(&params_row.slice(s![1..]));
            let bias = params_row[0];
            *o = f32::max(dot+bias, 0.)
        });
    output
}

pub fn dense<R: Read>(input: ArrayView1<f32>, 
                      n_units: usize,
                      params_buf: &mut ParamsBuffer<R>) -> Array1<f32> {
    let params_buf = params_buf.get_next((input.len()+1)*n_units);
    let params = 
        ArrayView2::<f32>::from_shape(
            (n_units, input.len()+1), 
            params_buf.as_ref()).unwrap();
    dense_internal(input, params)
}

pub fn dense_sigmoid<R: Read>(input: Array1<f32>, 
                     params_buf: &mut ParamsBuffer<R>) -> f32 {
    let bias = params_buf.get_next(1)[0];
    let params_buf = params_buf.get_next(input.len());
    let params = 
        ArrayView1::<f32>::from_shape(
            (input.len(), ), 
            params_buf.as_ref()).unwrap();
    1./(1.+(-input.dot(&params)-bias).exp())
}
