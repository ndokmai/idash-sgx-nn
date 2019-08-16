use std::io::BufReader;
use std::net::{TcpListener, TcpStream};
use byteorder::{NetworkEndian, NativeEndian, ReadBytesExt};
use rayon::{scope, ThreadPoolBuilder};
use ndarray::{Array1, Zip};
use crossbeam::bounded;
use crate::params_buffer::ParamsBuffer;
use crate::nn_eval::nn_eval;

mod params_buffer;
mod layers;
mod nn_eval;

const N_THREAD: usize = 7;
const INPUT_LEN: usize = 12634;
const LAUNCHER_BUF_CAP: usize = 0x100000;
const CLIENT_BUF_CAP: usize = 0x100000;
const INPUT_BUF: usize = 10;

fn main() {
    // connect to launcher
    let launcher_stream = TcpStream::connect("localhost:1234")
        .expect("Unable to connect.");
    let mut launcher_stream = 
        BufReader::with_capacity(LAUNCHER_BUF_CAP, launcher_stream);
    let port = launcher_stream.read_u16::<NativeEndian>().unwrap();
    let params_buf = ParamsBuffer::new(launcher_stream);

    // connect to client
    let listener = TcpListener::bind(("localhost", port)).unwrap();
    let mut client_stream = BufReader::with_capacity(CLIENT_BUF_CAP,
                                                     listener.accept().unwrap().0);
    let n_inputs = client_stream.read_u32::<NetworkEndian>().unwrap();

    let (tx, rx) = bounded(INPUT_BUF);
    for _ in 0..INPUT_BUF {
        tx.send(()).unwrap();
    }

    ThreadPoolBuilder::new().num_threads(N_THREAD).build_global().unwrap();
    scope( |s| {
        for i in 0..n_inputs {
            rx.recv().unwrap();
            let tx = tx.clone();
            let mut input = Array1::<f32>::zeros(INPUT_LEN);
            let params_buf = params_buf.clone();
            Zip::from(&mut input)
                .apply(|i| *i = client_stream.read_f32::<NetworkEndian>().unwrap());
            s.spawn(move |_| {
                let out = nn_eval(input, params_buf);
                println!("out #{}: {}", i, out);
                tx.send(()).unwrap();
            });
        }
    });
}

