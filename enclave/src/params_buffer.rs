use std::io::Read;
use std::cell::Cell;
use byteorder::{ReadBytesExt, NativeEndian};

pub trait ParamsBuffer: Send {
    fn getn_ref(&self, n: usize) -> &[f32];
}

pub struct MemParamsBuffer {
    params: Vec<f32>,
    cursor: Cell<usize>,
}

impl MemParamsBuffer {
    pub fn new<R: Read>(mut stream: R) -> Self {
        let mut params = Vec::new();
        let mut next = stream.read_f32::<NativeEndian>();
        while next.is_ok() {
            params.push(next.unwrap());
            next = stream.read_f32::<NativeEndian>();
        }
        Self { params, cursor: Cell::new(0) }
    }
}

impl ParamsBuffer for MemParamsBuffer {
    fn getn_ref(&self, n: usize) -> &[f32] {
        let cursor = self.cursor.get();
        let out = &self.params[cursor..(cursor+n)];
        self.cursor.set(cursor+n);
        if self.cursor.get() == self.params.len() {
            self.cursor.set(0);
        }
        out

    }
}
