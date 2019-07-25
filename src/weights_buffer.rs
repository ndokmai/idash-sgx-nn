use std::mem::swap;
use std::io::{BufReader, BufRead, Cursor};
use std::fs::File;
use std::mem::size_of;
use byteorder::{ReadBytesExt,NativeEndian};
 

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

pub struct FileWeightsBuffer {
    buffer: Box<dyn BufRead>,
}

impl FileWeightsBuffer {
    pub fn new(filename: &str) -> Self {
        let f = File::open(filename).unwrap();
        Self { buffer: Box::new(BufReader::new(f)) as Box<dyn BufRead> }
    }
    pub fn with_capacity(capacity: usize, filename: &str) -> Self {
        let f = File::open(filename).unwrap();
        Self { buffer: Box::new(BufReader::with_capacity(capacity, f)) 
            as Box<dyn BufRead> }
    }
}

impl WeightsBuffer for FileWeightsBuffer{
    fn getn(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::<f32>::with_capacity(n);
        let mut buf = vec![0u8; size_of::<f32>()*n];
        self.buffer.read_exact(&mut buf).unwrap();
        let mut rdr = Cursor::new(buf);
        for _ in 0..n {
            out.push(rdr.read_f32::<NativeEndian>().unwrap());
        }
        out
    }
}
