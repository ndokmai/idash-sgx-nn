use std::io::BufReader;
use std::net::{TcpListener, TcpStream};
use ndarray::{Array2, Zip};
use byteorder::{NetworkEndian, ReadBytesExt};
use crate::params_buffer::{ParamsBuffer, MemTcpParamsBuffer};
use crate::nn_eval::nn_eval;

mod params_buffer;
mod layers;
mod nn_eval;

const N_THREAD: usize = 8;
const INPUT_LEN: usize = 12634;

fn main() {
    let (tx_port, rx_port) = std::sync::mpsc::channel();
    let (tx_params, rx_params) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut stream = TcpStream::connect("localhost:1234")
            .expect("Unable to connect.");

        let port = stream.read_u16::<NetworkEndian>().unwrap();
        tx_port.send(port).unwrap();

        let params_buf = Box::new(MemTcpParamsBuffer::new(stream)) 
            as Box<dyn ParamsBuffer>;
        tx_params.send(params_buf).unwrap();
    });

    let port = rx_port.recv().unwrap();
    let listener = TcpListener::bind(("localhost", port)).unwrap();
    let mut stream = BufReader::new(listener.accept().unwrap().0);
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

