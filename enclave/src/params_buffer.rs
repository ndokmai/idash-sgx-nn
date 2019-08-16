use std::io::Read;
use std::sync::{Mutex, Arc, RwLock, atomic::{AtomicBool, Ordering, spin_loop_hint}};
use byteorder::{ReadBytesExt, NativeEndian};

const N_NODES: usize = 150;

pub type BufUnit = Arc<Vec<f32>>;

pub struct ParamsBuffer<R: Read> {
    inner: Arc<Mutex<R>>,
    global_buf: Arc<Vec<RwLock<Option<BufUnit>>>>,
    is_filled: Arc<Vec<AtomicBool>>,
    cursor: usize,
}

impl<R: Read> ParamsBuffer<R> {
    pub fn new(inner: R) -> Self {
        Self {
            inner: Arc::new(Mutex::new(inner)),
            global_buf: Arc::new((0..N_NODES)
                                 .map(|_| RwLock::new(None))
                                 .collect()),
            is_filled: Arc::new((0..N_NODES)
                                .map(|_| AtomicBool::new(false))
                                .collect()),
            cursor: 0,
        }
    }

    pub fn get_next(&mut self, n: usize) -> BufUnit {
        let is_filled = &self.is_filled[self.cursor];
        let out = &self.global_buf[self.cursor];
        if !is_filled.load(Ordering::Relaxed) {
            match out.try_write() {
                Ok(mut r) => {
                    let buf = {
                        let mut inner = self.inner.lock().unwrap();
                        let mut buf = Vec::with_capacity(n);
                        for _ in 0..n {
                            buf.push(
                                inner.read_f32::<NativeEndian>().unwrap());
                        }
                        buf
                    };
                    let buf = Arc::new(buf);
                    *r = Some(buf.clone());
                    is_filled.store(true, Ordering::Relaxed);
                    self.cursor += 1;
                    return buf;
                }
                Err(_) => {
                    while !is_filled.load(Ordering::Relaxed) {
                        spin_loop_hint();
                    }
                }
            }
        }
        let out = out.read()
            .unwrap()
            .as_ref()
            .expect("This shouldn't happen!")
            .clone();
        self.cursor += 1;
        out
    }
}

impl<R: Read> Clone for ParamsBuffer<R> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            global_buf: self.global_buf.clone(),
            is_filled: self.is_filled.clone(),
            cursor: 0,
        }
    }
}
