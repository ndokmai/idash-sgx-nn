use ndarray::{Zip, Array3, s};
use ndarray_parallel::prelude::*;

// TODO Optimize this
pub fn zeropad_avgpool(inputs: Array3<f32>) -> Array3<f32> {
    let mut outputs = Array3::zeros(
        (inputs.shape()[0], (inputs.shape()[1]+1)/2, inputs.shape()[2]));
    Zip::from(outputs.outer_iter_mut())
        .and(inputs.outer_iter())
        .par_apply(|mut output, input| {
            if input.shape()[0] & 1 == 1 {
                output.slice_mut(s![-1, ..]).assign(
                    &(&input.slice(s![-1, ..])/2.));
            }
            for (i, mut o) in input.exact_chunks((2, input.shape()[1])).into_iter()
                .zip(output.genrows_mut()) {
                    o.assign(&(&(&i.slice(s![0, ..])+&i.slice(s![1, ..]))/2.));
            }
        });
    outputs
}

//#[cfg(test)]
//mod tests {
    //use super::*;
    //use ndarray::arr3;

    //#[test]
    //fn zeropad_avgpool_even_test() {
        //let inputs = arr3(&[[[1., 2., 3., 4.],
                             //[5., 6., 7., 8.]]]);
        //let outputs = zeropad_avgpool(inputs);
        //assert_eq!(outputs, arr3(&[[[1.5, 3.5], [5.5, 7.5]]]));
    //}

    //#[test]
    //fn zeropad_avgpool_odd_test() {
        //let inputs = arr3(&[[[1., 2., 3., 4., 5.],
                             //[6., 7., 8., 9., 10.]]]);
        //let outputs = zeropad_avgpool(inputs);
        //assert_eq!(outputs, arr3(&[[[1.5, 3.5, 2.5], [6.5, 8.5, 5.]]]));
    //}

//}
