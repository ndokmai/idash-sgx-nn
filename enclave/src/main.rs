use std::net::{TcpListener, TcpStream};
use ndarray::{Array2, Zip};
use byteorder::{NetworkEndian, ReadBytesExt};
use crate::params_buffer::{ParamsBuffer, MemParamsBuffer};
use crate::nn_eval::nn_eval;
use crate::decryption::EncryptedReader;

mod params_buffer;
mod layers;
mod nn_eval;
mod decryption;

const N_THREAD: usize = 8;
const INPUT_LEN: usize = 12634;
const TCP_BUF: usize = 0x100000;
const DUMMY_INPUT_KEY: [u8; 16] = [0u8; 16];
const DUMMY_FILE_KEY: [u8; 16] = [1u8; 16];

fn main() {
    let (tx_port, rx_port) = std::sync::mpsc::channel();
    let (tx_params, rx_params) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut stream = TcpStream::connect("localhost:1234")
            .expect("Unable to connect.");
        let port = stream.read_u16::<NetworkEndian>().unwrap();
        tx_port.send(port).unwrap();
        let stream = EncryptedReader::with_capacity(TCP_BUF, stream, &DUMMY_FILE_KEY);
        let params_buf = Box::new(MemParamsBuffer::new(stream)) 
            as Box<dyn ParamsBuffer>;
        tx_params.send(params_buf).unwrap();
    });

    let port = rx_port.recv().unwrap();
    let listener = TcpListener::bind(("localhost", port)).unwrap();
    let mut stream = EncryptedReader::with_capacity(
        TCP_BUF,
        listener.accept().unwrap().0,
        &DUMMY_INPUT_KEY);
    let mut n_inputs = stream.read_u32::<NetworkEndian>().unwrap() as usize;

    rayon::ThreadPoolBuilder::new().num_threads(N_THREAD).build_global().unwrap();

    let params_buf = rx_params.recv().unwrap();
    while n_inputs !=0 {
        let batch_size = usize::min(N_THREAD, n_inputs);
        n_inputs -= batch_size;
        let mut inputs = Array2::<f32>::zeros((batch_size, INPUT_LEN));
        Zip::from(&mut inputs)
            .apply(|i| *i = stream.read_f32::<NetworkEndian>().unwrap());
        let outputs = nn_eval(inputs, &params_buf);
        for o in &outputs {
            print!("{}\t", o);
        }
        println!("");
    }
}

