use std::io::BufReader;
use std::net::TcpStream;
use std::cell::Cell;
use byteorder::{ReadBytesExt, NativeEndian};

pub trait ParamsBuffer: Send {
    fn getn_ref(&self, n: usize) -> &[f32];
}

pub struct MemTcpParamsBuffer {
    params: Vec<f32>,
    cursor: Cell<usize>,
}

impl MemTcpParamsBuffer {
    pub fn new(stream: TcpStream) -> Self {
        let mut stream = BufReader::new(stream);
        let mut params = Vec::new();
        let mut next = stream.read_f32::<NativeEndian>();
        while next.is_ok() {
            params.push(next.unwrap());
            next = stream.read_f32::<NativeEndian>();
        }
        Self { params, cursor: Cell::new(0) }
    }
}

impl ParamsBuffer for MemTcpParamsBuffer {
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
