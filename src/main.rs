use std::io::BufReader;
use std::net::{TcpListener, TcpStream};
use ndarray::{Array1, Array2, Array3, ArrayView2, s, Zip};
use byteorder::{NetworkEndian, ReadBytesExt};
use crate::weights_buffer::{WeightsBuffer, MemTcpWeightsBuffer};
use crate::layers::{conv1d::conv1d,
    batchnorm_add_activate::batchnorm_add_activate,
    zeropad_avgpool::zeropad_avgpool,
    dense::dense, 
    dense::dense_sigmoid, 
};

mod weights_buffer;
mod layers;

#[allow(dead_code)]
fn debug_print(title: &str, array: ArrayView2<f32>) {
    const THRESHOLD: usize = 1000;
    const MARGIN: i32 = 3;
    println!("----------{}----------", title);
    if array.len() < THRESHOLD {
        let string = array.genrows()
            .into_iter()
            .map(|x| format!("[ {}\t]", x.iter()
                             .map(|y| format!("{:.10}", y))
                             .collect::<Vec<_>>()
                             .join(", ")))
            .collect::<Vec<_>>()
            .join("\n");
        print!("{}", string);
    } else {
        for (i, r) in array.genrows().into_iter().enumerate() {
            if i==MARGIN as usize {
                println!("...");
            } else if i < MARGIN as usize || i > array.rows() - MARGIN as usize - 1 {
                print!("[ ");
                for (j, c) in r.iter().enumerate() {
                    if j==MARGIN as usize {
                        print!("\t...\t");
                    } else if j == r.len() - 1 {
                       print!("{:.10}", c);
                    } else if j < MARGIN as usize || j > r.len() - MARGIN as usize - 1  {
                       print!("{:.10}\t", c);
                    } 
                }
                println!("\t]");
            } 
        }
    }
    println!("\n\n");

}
fn concat(input1: Array2<f32>, input2: Array2<f32>) -> Array2<f32> {
    let mut outputs = Array2::zeros((input1.shape()[0], 
                                     input2.shape()[1]+input1.shape()[1]));
    outputs.slice_mut(s![.., ..input1.shape()[1]]).assign(&input1);
    outputs.slice_mut(s![.., input1.shape()[1]..]).assign(&input2);
    outputs
}

fn res1d(inputs: Array3<f32>, n_kernel: usize, kernel_size: usize, strides: usize,
         weights: &mut Box<dyn WeightsBuffer>) -> Array3<f32> {
    let left = conv1d(inputs.view(), n_kernel, kernel_size, strides, weights);
    //debug_print(&"left cov1d", left.slice(s![0, .., ..]));

    let right = {
        if strides > 1 || inputs.shape()[1] != n_kernel {
            if strides > 1 {
                let avgpool = zeropad_avgpool(inputs);
                //debug_print(&"avg. pooling", avgpool.slice(s![0, .., ..]));
                conv1d(avgpool.view(), n_kernel, 1, 1, weights)
            } else {
                conv1d(inputs.view(), n_kernel, 1, 1, weights)
            }
        } else {
            panic!("Incorrect parameters")
        }
    };
    //debug_print(&"right cov1d", right.slice(s![0, .., ..]));
    let out = batchnorm_add_activate(left, right.view(), weights);
    //debug_print(&"batchnorm-add-activate", out.slice(s![0, .., ..]));
    out
}

fn nn_eval(inputs: Array2<f32>, weights: &mut Box<dyn WeightsBuffer>) -> Array1<f32> {
    let v2 = dense(inputs.view(), 64, weights);
    //debug_print(&"dense", v2.slice(s![..1, ..]));
    let shape = (inputs.shape()[0], inputs.shape()[1], 1);
    let inputs = inputs.into_shape(shape).unwrap();
    let mut v1 = res1d(inputs, 4, 3, 1, weights);
    for i in &[8usize, 16, 32, 64, 128, 256, 512, 1024] {
        v1 = res1d(v1, *i, 3, 1, weights);
        v1 = res1d(v1, *i, 3, 2, weights);
    }
    let shape = (v1.shape()[0], v1.shape()[1]*v1.shape()[2]);
    let v1 = v1.into_shape(shape).unwrap();
    let v = concat(v1, v2);
    let v = dense(v.view(), 32, weights);
    //debug_print(&"dense", v.slice(s![..1, ..]));
    dense_sigmoid(v, weights)
}


fn main() {
    const N_THREAD: usize = 4;
    const N_MAX: usize = 20;
    const INPUT_LEN: usize = 12634;
    rayon::ThreadPoolBuilder::new().num_threads(N_THREAD).build_global().unwrap();
    let mut stream = TcpStream::connect("localhost:1234")
                                    .expect("Unable to connect.");

    let port = stream.read_u16::<NetworkEndian>().unwrap();
    let mut weights = Box::new(MemTcpWeightsBuffer::new(stream)) 
        as Box<dyn WeightsBuffer>;

    // lister to client
    let listener = TcpListener::bind(("localhost", port)).unwrap();
    let mut stream = BufReader::new(listener.accept().unwrap().0);
    loop {
        let n_inputs = stream.read_u32::<NetworkEndian>().unwrap() as usize;
        if n_inputs==0 {
            break;
        }
        if n_inputs > N_MAX {
            panic!("Too many inputs");
        }
        let mut inputs = Array2::<f32>::zeros((n_inputs, INPUT_LEN));
        Zip::from(&mut inputs)
            .apply(|i| *i = stream.read_f32::<NetworkEndian>().unwrap());
        let outputs = nn_eval(inputs, &mut weights);
        for o in &outputs {
            print!("{}\t", o);
        }
        println!("");
    }
    
}

