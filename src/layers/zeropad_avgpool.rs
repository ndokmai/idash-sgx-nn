use ndarray::*;

pub fn zeropad_avgpool(inputs: Array3<f32>) -> Array3<f32> {
    let mut outputs = Array3::zeros(
        (inputs.shape()[0], inputs.shape()[1],(inputs.shape()[2]+1)/2));
    Zip::from(outputs.outer_iter_mut())
        .and(inputs.outer_iter())
        .apply(|mut output, input| {
            if input.shape()[1] & 1 == 1 {
                output.slice_mut(s![.., output.shape()[1]-1]).assign(
                    &input.slice(s![.., input.shape()[1]-1]));
                Zip::from(&mut output.slice_mut(s![.., output.shape()[1]-1]))
                    .apply(|o| *o/=2.);
            }
            for (i, mut o) in input.exact_chunks((input.shape()[0], 2)).into_iter()
                .zip(output.gencolumns_mut()) {
                    for (o_e, i_e) in o.iter_mut()
                        .zip(i.genrows().into_iter()) {
                            *o_e = i_e.sum()/2.;
                        }
            }
        });
    outputs
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr3;

    #[test]
    fn zeropad_avgpool_even_test() {
        let inputs = arr3(&[[[1., 2., 3., 4.],
                             [5., 6., 7., 8.]]]);
        let outputs = zeropad_avgpool(inputs);
        assert_eq!(outputs, arr3(&[[[1.5, 3.5], [5.5, 7.5]]]));
    }

    #[test]
    fn zeropad_avgpool_odd_test() {
        let inputs = arr3(&[[[1., 2., 3., 4., 5.],
                             [6., 7., 8., 9., 10.]]]);
        let outputs = zeropad_avgpool(inputs);
        assert_eq!(outputs, arr3(&[[[1.5, 3.5, 2.5], [6.5, 8.5, 5.]]]));
    }

}
