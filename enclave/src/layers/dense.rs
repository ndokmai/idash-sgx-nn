use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, Zip, s};
use ndarray_parallel::prelude::*;
use crate::params_buffer::ParamsBuffer;

fn dense_single(input: ArrayView1<f32>, 
                mut output: ArrayViewMut1<f32>,
                params: ArrayView2<f32>) {
    Zip::from(&mut output)
        .and(params.genrows())
        .apply(|o, params_row| {
            let dot = input.dot(&params_row.slice(s![1..]));
            let bias = params_row[0];
            *o = f32::max(dot+bias, 0.)
        });
}

pub fn dense(inputs: ArrayView2<f32>, n_units: usize, 
             params_buf: &Box<dyn ParamsBuffer>) -> Array2<f32> {
    let mut outputs = Array2::zeros((inputs.shape()[0], n_units));
    let params = 
        ArrayView2::<f32>::from_shape(
            (n_units, inputs.shape()[1]+1), 
            params_buf.getn_ref((inputs.shape()[1]+1) * n_units)).unwrap();

    Zip::from(outputs.genrows_mut())
        .and(inputs.genrows())
        .par_apply(|output, input| {
            dense_single(input, output, params);
        });
    outputs
}

#[inline]
fn sigmoid(input: ArrayView1<f32>,
           output: &mut f32,
           bias: f32,
           params: ArrayView1<f32>) {
    *output = 1./(1.+(-input.dot(&params)-bias).exp());
}

pub fn dense_sigmoid(inputs: Array2<f32>, 
                     params_buf: &Box<dyn ParamsBuffer>) -> Array1<f32> {
    let mut outputs = Array1::zeros(inputs.shape()[0]);
    let bias = params_buf.getn_ref(1);
    let params = ArrayView1::<f32>::from_shape(
        (inputs.shape()[1],), 
        params_buf.getn_ref(inputs.shape()[1])).unwrap();
    Zip::from(&mut outputs)
        .and(inputs.genrows())
        .par_apply(|output, input| sigmoid(input, output, bias[0], params));
    outputs
}
