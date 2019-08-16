use ndarray::{Zip, Array2, s};

pub fn zeropad_avgpool(input: Array2<f32>) -> Array2<f32> {
    let mut output = Array2::zeros(
        ((input.rows()+1)/2, input.cols()));
    if input.rows() & 1 == 1 {
        Zip::from(output.slice_mut(s![-1, ..]))
            .and(input.slice(s![-1, ..]))
            .apply(|o, i| *o = *i/2.);
    }
    for (i, mut o) in input.exact_chunks((2, input.cols())).into_iter()
        .zip(output.genrows_mut()) {
            for (a, b) in (o.iter_mut()).zip(i.gencolumns().into_iter()) {
                *a = (b[0] + b[1])/2.;
            }
        }
    output
}
