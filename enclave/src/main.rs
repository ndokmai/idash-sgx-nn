use std::net::{TcpListener, TcpStream};
use ndarray::Array2;
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
    eprintln!("Server: evaluating...");
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

    let params_buf = rx_params.recv().unwrap();
    rayon::ThreadPoolBuilder::new().num_threads(N_THREAD).build_global().unwrap();
    let mut done = false;
    let mut all_outputs = Vec::new();
    loop {
        let mut buf = Vec::with_capacity(N_THREAD*INPUT_LEN);
        for _ in 0..N_THREAD {
            let next = stream.read_f32::<NetworkEndian>();
            match next {
                Ok(next) => buf.push(next),
                Err(_) => {
                    done = true;
                    break;
                }
            }
            for _ in 0..(INPUT_LEN-1) {
                buf.push(stream.read_f32::<NetworkEndian>().unwrap());
            }
        }
        let inputs = Array2::<f32>::from_shape_vec(
            (buf.len()/INPUT_LEN, INPUT_LEN),
            buf
            ).unwrap();
        let outputs = nn_eval(inputs, &params_buf);
        all_outputs.push(
            outputs.into_raw_vec()
            .into_iter()
            .map(|x| match x>0.5 {
                true => 1,
                false => 0,
            })
            .collect::<Vec<u8>>());
        if done {
            break;
        }
    }
    let mut out_str = String::new();
    for i in all_outputs.into_iter().flatten() {
        out_str += &format!("{}\n", i);
    }
    eprintln!("Server: results:");
    print!("{}", out_str);
    eprintln!("Server: finished");
}

