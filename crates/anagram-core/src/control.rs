//! Per-run cooperative control, shared explicitly with workers and readers.
use std::{
    io::{self, BufRead, Read},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

#[derive(Clone, Default)]
pub struct Control {
    cancelled: Arc<AtomicBool>,
    deadline: Option<Instant>,
}
impl Control {
    pub fn with_deadline(deadline: Instant) -> Self {
        Self {
            deadline: Some(deadline),
            ..Self::default()
        }
    }
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
    pub fn flag(&self) -> &AtomicBool {
        &self.cancelled
    }
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
    pub fn check(&self) -> Result<(), &'static str> {
        if self.cancelled.load(Ordering::Relaxed) {
            Err("cancelled")
        } else if self.deadline.is_some_and(|d| Instant::now() >= d) {
            Err("timed_out")
        } else {
            Ok(())
        }
    }
    pub fn check_io(&self) -> io::Result<()> {
        // ErrorKind::Interrupted would be retried by read_to_end/read_line.
        self.check().map_err(io::Error::other)
    }
    pub fn reader<R>(&self, inner: R) -> Reader<'_, R> {
        Reader {
            inner,
            control: self,
        }
    }
}
pub struct Reader<'a, R> {
    inner: R,
    control: &'a Control,
}
impl<R: Read> Read for Reader<'_, R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.control.check_io()?;
        self.inner.read(buffer)
    }
}
impl<R: BufRead> BufRead for Reader<'_, R> {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        self.control.check_io()?;
        self.inner.fill_buf()
    }
    fn consume(&mut self, amount: usize) {
        self.inner.consume(amount);
    }
}
