use std::net::{TcpStream, TcpListener, Shutdown};
use std::env::args;
use std::fs::{read_dir, File};
use std::io::{Read, Write, BufReader, BufRead, Result};
use std::{thread::sleep, time::Duration};
use std::process::Command;
use std::thread;
use std::path::Path;
use std::sync::mpsc::channel;
use byteorder::{NetworkEndian, WriteBytesExt};
use ndarray::{Array1, Array2, s, Zip};
use crate::encryption::EncryptedWriter;

mod encryption;

const INPUT_LEN: usize = 12634;
const TCP_BUF: usize = 0x100000;
const DUMMY_INPUT_KEY: [u8; 16] = [0u8; 16];
const DUMMY_FILE_KEY: [u8; 16] = [1u8; 16];

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

fn client_read_single_file(f: impl Read) -> Result<Array2<f32>> {
    let inputs_file = BufReader::new(f);
    let mut iter = inputs_file.lines();
    let n_inputs = iter.next().unwrap()?.split_whitespace().skip(3).count();
    let mut inputs = Array2::zeros((INPUT_LEN, n_inputs));
    for (i, line) in iter.enumerate() {
        let line = line?
            .split_whitespace()
            .skip(2)
            .map(|x| x.parse::<f32>().unwrap())
            .collect::<Vec<_>>();
        inputs.slice_mut(s![i, ..]).assign(
            &Array1::<f32>::from_vec(line));
    }
    Ok(Array2::<f32>::reversed_axes(inputs))
}

fn client(host: &str, fdir: &str) {
    let fdir = fdir.to_owned();
    let (tx, rx) =  channel();
    thread::spawn(move || {
        let dir = Path::new(&fdir);
        if dir.is_dir() {
            eprintln!("Client: sending files...");
            for (_i, entry) in read_dir(dir).unwrap().enumerate() {
                let path = entry.unwrap().path();
                eprintln!("Client: file {:?} completed", path.clone());
                let f = File::open(path).unwrap();
                let inputs = client_read_single_file(f).unwrap();
                tx.send(inputs).unwrap();
            }
        } else {
            panic!("Empty directory");
        }
        eprintln!("Client: finished")
    });

    let mut stream = TcpStream::connect(host);
    while stream.is_err() {
        sleep(Duration::from_millis(50));
        stream = TcpStream::connect(host);
    }
    let mut stream = EncryptedWriter::with_capacity(TCP_BUF, 
                                                    stream.unwrap(),
                                                    &DUMMY_INPUT_KEY);
    loop {
        match rx.recv() {
            Ok(inputs) => {
                Zip::from(inputs.genrows())
                    .apply(|input| {
                        Zip::from(&input)
                            .apply(|i| 
                                   stream.write_f32::<NetworkEndian>(*i)
                                   .expect("Error sending inputs."));

                    });
            },
            Err(_) => break,
        }
    }
}

fn file_encryptor(in_file_name: &str, out_file_name: &str) {
    let mut in_file = BufReader::new(
        File::open(in_file_name).unwrap());
    let mut out_file = EncryptedWriter::with_capacity(
        TCP_BUF,
        File::create(out_file_name).unwrap(),
        &DUMMY_FILE_KEY);
    std::io::copy(&mut in_file, &mut out_file).unwrap();
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
                );
        }
        "encrypt" => {
            file_encryptor(
                &args().nth(2).unwrap()[..],
                &args().nth(3).unwrap()[..],
                );
        }
        _ => panic!("Shouldn't happen"),
    } 
}
