use std::net::{TcpStream, TcpListener, Shutdown};
use std::env::{args};
use std::fs::{File};
use std::io::{Read, Write, BufReader, BufRead, BufWriter};
use std::{thread::sleep, time::Duration};
use std::process::Command;
use byteorder::{NetworkEndian, WriteBytesExt};
use ndarray::{Array1, Array2, s, Zip};

const N_INPUTS: usize = 100; 
const INPUT_LEN: usize = 12634;

fn launcher(client_port: u16, fname: &str, laucher_cmd: &str, enclave_file: &str) {
    let mut child = None;
    match laucher_cmd {
        "" => (),
        _ => match enclave_file {
            "" => {
                child = Some(
                    Command::new(laucher_cmd)
                    .spawn()
                    .expect("Failed to execute child"));
            },
            _ => {
                child = Some(
                    Command::new(laucher_cmd)
                    .arg(enclave_file)
                    .spawn()
                    .expect("Failed to execute child"));
            },
        }
    }
    let listener = TcpListener::bind("localhost:1234").unwrap();
    let mut stream = listener.accept().unwrap().0;

    // send client port
    stream.write_u16::<NetworkEndian>(client_port).unwrap();

    // send weights file
    let mut f = File::open(fname).expect("Error opening file.");
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).expect("Error reading file");
    stream.write(&buf).expect("Error sending weights.");
    stream.shutdown(Shutdown::Both).unwrap();

    if child.is_some() {
        let ecode = child.unwrap().wait()
            .expect("failed to wait on child");
        assert!(ecode.success());

    }
}

fn client(host: &str, fname_1: &str, fname_2: &str) {

    // send inputs files
    let inputs_file_1 = BufReader::new(
        File::open(fname_1).unwrap());
    let mut iter_1 = inputs_file_1.lines();
    iter_1.next();
    let inputs_file_2 = BufReader::new(
        File::open(fname_2).unwrap());
    let mut iter_2 = inputs_file_2.lines();
    iter_2.next();

    let mut inputs = Array2::zeros((INPUT_LEN, N_INPUTS));
    for (i, line) in iter_1.enumerate() {
        let line = line.unwrap()
            .split_whitespace()
            .map(|x| x.parse::<f32>().unwrap())
            .skip(2)
            .take(usize::min(50, N_INPUTS))
            .collect::<Vec<_>>();
        inputs.slice_mut(s![i, ..usize::min(50, N_INPUTS)]).assign(
            &Array1::<f32>::from_vec(line));
    }
    if i64::max(0, N_INPUTS as i64 - 50) > 0 {
        for (i, line) in iter_2.enumerate() {
            let line = line.unwrap()
                .split_whitespace()
                .map(|x| x.parse::<f32>().unwrap())
                .skip(2)
                .take(N_INPUTS-50)
                .collect::<Vec<_>>();
            inputs.slice_mut(s![i, (N_INPUTS-50)..]).assign(
                &Array1::<f32>::from_vec(line));
        }
    }
    let inputs = Array2::<f32>::reversed_axes(inputs);

    let mut stream = TcpStream::connect(host);
    while stream.is_err() {
        sleep(Duration::from_millis(50));
        stream = TcpStream::connect(host);
    }
    let mut stream = BufWriter::new(stream.unwrap());
    stream.write_u32::<NetworkEndian>(N_INPUTS as u32).unwrap();
    Zip::from(inputs.genrows())
        .apply(|input| {
            Zip::from(&input)
                .apply(|i| 
                       stream.write_f32::<NetworkEndian>(*i)
                       .expect("Error sending inputs."));

        });
}

fn main() {
    let mode = &args().nth(1).unwrap()[..];
    match mode {
        "launcher" => {
            let port = args().nth(2).unwrap().parse::<u16>().unwrap();
            launcher(port, 
                     &args().nth(3).unwrap()[..],
                     match &args().nth(4) {
                         Some(s) => &s[..],
                         None => &"",
                     },
                     match &args().nth(5) {
                         Some(s) => &s[..],
                         None => &"",
                     });
        },
        "client" => {
            client(
                &args().nth(2).unwrap()[..],
                &args().nth(3).unwrap()[..],
                &args().nth(4).unwrap()[..],
                );
        }
        _ => panic!("Shouldn't happen"),
    } 
}
