use std::mem::swap;

pub trait WeightsBuffer {
    fn getn(&mut self, n: usize) -> Vec<f32>;
}

pub struct TestWeightsBuffer {
    weights: Vec<f32>,
}

#[allow(dead_code)]
impl TestWeightsBuffer {
    pub fn new(weights: Vec<f32>) -> Self {
        Self { weights }
    }
}

impl WeightsBuffer for TestWeightsBuffer {
    fn getn(&mut self, n: usize) -> Vec<f32> {
        let mut out = self.weights.split_off(n);
        swap(&mut out, &mut self.weights);
        out
    }

}

pub struct RandomWeightsBuffer {
    index: usize,
}

impl RandomWeightsBuffer {
    pub fn new() -> Self {
        Self { index: 1 }
    }
}

impl WeightsBuffer for RandomWeightsBuffer {
    fn getn(&mut self, n: usize) -> Vec<f32> {
        let i = self.index;
        self.index += n;
        (i..(i+n)).map(|x| (x*37%29/100) as f32).collect()
    }
}
