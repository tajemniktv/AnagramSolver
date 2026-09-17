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
    /// Preserve the shared cancellation flag and never extend an existing limit.
    pub fn with_earlier_deadline(&self, deadline: Instant) -> Self {
        Self {
            cancelled: self.cancelled.clone(),
            deadline: Some(self.deadline.map_or(deadline, |old| old.min(deadline))),
        }
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
    /// Stable, cancellable sorting without cloning payloads or unwinding a comparator.
    /// Uses two index buffers; interruption always leaves every payload intact.
    pub fn sort_by<T>(
        &self,
        values: &mut [T],
        compare: impl FnMut(&T, &T) -> std::cmp::Ordering,
    ) -> Result<(), &'static str> {
        Self::sort_checked(values, compare, || self.check())
    }
    pub(crate) fn sort_checked<T>(
        values: &mut [T],
        mut compare: impl FnMut(&T, &T) -> std::cmp::Ordering,
        mut check: impl FnMut() -> Result<(), &'static str>,
    ) -> Result<(), &'static str> {
        check()?;
        let n = values.len();
        let mut order = Vec::with_capacity(n);
        let mut scratch = Vec::with_capacity(n);
        for i in 0..n {
            if i % 256 == 0 {
                check()?;
            }
            order.push(i);
            scratch.push(0);
        }
        let mut width = 1usize;
        while width < n {
            for start in (0..n).step_by(width.saturating_mul(2)) {
                let middle = start.saturating_add(width).min(n);
                let end = middle.saturating_add(width).min(n);
                let (mut left, mut right) = (start, middle);
                for slot in &mut scratch[start..end] {
                    check()?;
                    if left < middle
                        && (right == end
                            || compare(&values[order[left]], &values[order[right]])
                                != std::cmp::Ordering::Greater)
                    {
                        *slot = order[left];
                        left += 1;
                    } else {
                        *slot = order[right];
                        right += 1;
                    }
                }
            }
            std::mem::swap(&mut order, &mut scratch);
            width = width.saturating_mul(2);
        }
        for (destination, &source) in order.iter().enumerate() {
            check()?;
            scratch[source] = destination;
        }
        // Commit the permutation without fallible checkpoints: returning Err
        // must leave the original ordering intact, not a partial permutation.
        check()?;
        for i in 0..n {
            while scratch[i] != i {
                let j = scratch[i];
                values.swap(i, j);
                scratch.swap(i, j);
            }
        }
        Ok(())
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
