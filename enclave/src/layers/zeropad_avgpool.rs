use ndarray::{Zip, Array3, ArrayViewMut2, s};
use ndarray_parallel::prelude::*;

fn zeropad_avgpool_single(input: ArrayViewMut2<f32>,
                          mut output: ArrayViewMut2<f32>,) {
    if input.shape()[0] & 1 == 1 {
        Zip::from(output.slice_mut(s![-1, ..]))
            .and(input.slice(s![-1, ..]))
            .apply(|o, i| *o = *i/2.);
    }
    for (i, mut o) in input.exact_chunks((2, input.shape()[1])).into_iter()
        .zip(output.genrows_mut()) {
            for (a, b) in (o.iter_mut()).zip(i.gencolumns().into_iter()) {
                *a = (b[0] + b[1])/2.;
            }
        }
}

pub fn zeropad_avgpool(mut inputs: Array3<f32>) -> Array3<f32> {
    let mut outputs = Array3::zeros(
        (inputs.shape()[0], (inputs.shape()[1]+1)/2, inputs.shape()[2]));
    Zip::from(outputs.outer_iter_mut())
        .and(inputs.outer_iter_mut())
        .par_apply(|output, input| {
            zeropad_avgpool_single(input, output);
        });
    outputs
}
