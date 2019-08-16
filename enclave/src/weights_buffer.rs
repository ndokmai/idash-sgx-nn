use std::io::{BufReader};
use std::net::TcpStream;
use std::cell::Cell;
use byteorder::{ReadBytesExt,NativeEndian};

pub trait WeightsBuffer: Send {
    fn getn(&self, n: usize) -> Vec<f32>;
    fn getn_ref(&self, n: usize) -> &[f32];
}

pub struct MemTcpWeightsBuffer {
    weights: Vec<f32>,
    cursor: Cell<usize>,
}

impl MemTcpWeightsBuffer {
    pub fn new(stream: TcpStream) -> Self {
        let mut stream = BufReader::new(stream);
        let mut weights = Vec::new();
        let mut next = stream.read_f32::<NativeEndian>();
        while next.is_ok() {
            weights.push(next.unwrap());
            next = stream.read_f32::<NativeEndian>();
        }
        Self { weights, cursor: Cell::new(0) }
    }
}

impl WeightsBuffer for MemTcpWeightsBuffer {
    fn getn(&self, n: usize) -> Vec<f32> {
        Vec::from(self.getn_ref(n))
    }

    fn getn_ref(&self, n: usize) -> &[f32] {
        let cursor = self.cursor.get();
        let out = &self.weights[cursor..(cursor+n)];
        self.cursor.set(cursor+n);
        if self.cursor.get() == self.weights.len() {
            self.cursor.set(0);
        }
        out

    }
}
