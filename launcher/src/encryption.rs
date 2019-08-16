use std::io::{Write, Result, Error, ErrorKind};
use ring::aead::{SealingKey, Nonce, Aad, seal_in_place, AES_128_GCM};
use ring::rand::{SystemRandom, SecureRandom};
use byteorder::{WriteBytesExt, NetworkEndian};

fn encrypt(key: &SealingKey, rand: &SystemRandom, nonce: &mut [u8; 12],
               in_out: &mut [u8]) -> Result<usize> {
    rand.fill(nonce).unwrap();
    let nonce = Nonce::assume_unique_for_key(*nonce);
    let len = seal_in_place(key, nonce, Aad::empty(), in_out, 
                            key.algorithm().tag_len()).unwrap();
    Ok(len)
}

pub struct EncryptedWriter<W: Write> {
    inner: Option<W>,
    buf: Vec<u8>,
    key: SealingKey,
    rand: SystemRandom,
    tag_len: usize,
    capacity: usize,
    // If the inner writer panics in a call to write, we don't want to
    // write the buffered data a second time in BufWriter's destructor. This
    // flag tells the Drop impl if it should skip the flush.
    panicked: bool,
}

impl<W: Write> EncryptedWriter<W> {
    pub fn with_capacity(capacity: usize, inner: W, key_bytes: &[u8; 16]) -> Self {
        Self {
            inner: Some(inner),
            buf: Vec::with_capacity(capacity + AES_128_GCM.tag_len()),
            key: SealingKey::new(&AES_128_GCM, &key_bytes[..]).unwrap(),
            rand: SystemRandom::new(),
            tag_len: AES_128_GCM.tag_len(),
            capacity,
            panicked: false,
        }
    }

    fn flush_buf(&mut self) -> Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        //println!("Launcher body(unencrypted): {:?}", self.buf);
        self.buf.resize(self.buf.len()+self.tag_len, 0);
        let mut nonce = [0u8; 12];
        let len = encrypt(&self.key, &self.rand, &mut nonce,
                &mut self.buf[..]).unwrap();

        self.panicked = true;
        let r = self.inner.as_mut().unwrap()
            .write_u32::<NetworkEndian>(len as u32);
        self.panicked = false;
        match r {
            Ok(_) => {}
            Err(e) => { return Err(e); }
        }

        //println!("Launcher packet len: {}", len);

        let mut written = 0;
        while written < nonce.len() {
            self.panicked = true;
            let r = self.inner.as_mut().unwrap().write(&nonce[written..]);
            self.panicked = false;

            match r {
                Ok(0) => {
                    return Err(Error::new(ErrorKind::WriteZero,
                                         "Failed to write the buffered data"));
                }
                Ok(n) => written += n,
                Err(e) => { return Err(e); }
            }
        }

        //println!("Launcher nonce: {:?}", nonce);

        let mut written = 0;
        while written < len {
            self.panicked = true;
            let r = self.inner.as_mut().unwrap().write(&self.buf[written..]);
            self.panicked = false;

            match r {
                Ok(0) => {
                    return Err(Error::new(ErrorKind::WriteZero,
                                         "Failed to write the buffered data"));
                }
                Ok(n) => written += n,
                Err(e) => { return Err(e); }
            }
        }
        //println!("Launcher body: {:?}", self.buf);
        self.buf.clear();
        Ok(())
    }

    pub fn get_mut(&mut self) -> &mut W { self.inner.as_mut().unwrap() }
}

impl<W: Write> Write for EncryptedWriter<W> {
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let mut written = 0;
        let len = buf.len();
        while written < len {
            let to_write = usize::min(self.capacity - self.buf.len(), 
                                      buf.len() - written);
            written += self.buf.write(&buf[written..(written+to_write)])?;
            if self.buf.len() == self.capacity {
                self.flush_buf()?;
            }
        }
        Ok(written)
    }

    fn flush(&mut self) -> Result<()> {
        self.flush_buf().and_then(|()| self.get_mut().flush())
    }
}

impl<W: Write> Drop for EncryptedWriter<W> {
    fn drop(&mut self) {
        if self.inner.is_some() && !self.panicked {
            // dtors should not panic, so we ignore a failed flush
            let _r = self.flush_buf();
        }
    }
}
