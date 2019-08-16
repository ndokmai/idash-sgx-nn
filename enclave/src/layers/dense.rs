use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip, s};
use ndarray_parallel::prelude::*;
use crate::weights_buffer::WeightsBuffer;

pub fn dense(inputs: ArrayView2<f32>, n_units: usize, 
             weights: &Box<dyn WeightsBuffer>) -> Array2<f32> {
    let mut outputs = Array2::zeros((inputs.shape()[0], n_units));
    let weights = 
        ArrayView2::<f32>::from_shape(
            (n_units, inputs.shape()[1]+1), 
            weights.getn_ref((inputs.shape()[1]+1) * n_units)).unwrap();

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
                     weights: &Box<dyn WeightsBuffer>) -> Array1<f32> {
    let mut outputs = Array1::zeros(inputs.shape()[0]);
    let bias = weights.getn(1);
    let weights = ArrayView1::<f32>::from_shape(
        (inputs.shape()[1],), 
        weights.getn_ref(inputs.shape()[1])).unwrap();
    Zip::from(&mut outputs)
        .and(inputs.genrows())
        .par_apply(|output, input| *output = 1./(1.+(-input.dot(&weights)-bias[0]).exp()));
    outputs
}
