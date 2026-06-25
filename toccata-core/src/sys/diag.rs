// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! Allocation-free diagnostics.
//!
//! Code paths that run *inside* the allocator — the global allocator's OOM
//! backpressure handler, the [`Reservation`](crate) degradation notice — must not
//! allocate (the formatting machinery and `tracing` both do, and would recurse
//! into the very allocator that is failing). These helpers write fixed byte
//! strings and pre-formatted integers straight to stderr with raw `write(2)`, so
//! they touch no heap and take no lock.

/// Write raw bytes to stderr (fd 2). Returns the byte count written (best
/// effort; a short or failed write is ignored — this is diagnostics, not data).
#[cfg(unix)]
#[inline]
pub fn write_stderr(bytes: &[u8]) -> isize {
    // SAFETY: writing a borrowed byte slice to fd 2; no allocation, no unwind.
    unsafe { libc::write(2, bytes.as_ptr() as *const libc::c_void, bytes.len()) as isize }
}

#[cfg(not(unix))]
#[inline]
pub fn write_stderr(bytes: &[u8]) -> isize {
    use std::io::Write;
    let _ = std::io::stderr().write_all(bytes);
    bytes.len() as isize
}

/// Format a `usize` as decimal into `buf`, returning the filled tail slice. No
/// allocation. `buf` must be at least 20 bytes (the widest `u64`/`usize`).
#[inline]
pub fn usize_to_dec(mut n: usize, buf: &mut [u8; 20]) -> &[u8] {
    if n == 0 {
        buf[0] = b'0';
        return &buf[..1];
    }
    let mut i = buf.len();
    while n > 0 {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    &buf[i..]
}

/// Write a fixed message followed by a decimal integer and a trailing newline,
/// all via raw `write(2)`. The single entry point degradation/OOM notices use so
/// the no-alloc discipline lives in one place.
#[inline]
pub fn write_msg_num(prefix: &[u8], n: usize) {
    let _ = write_stderr(prefix);
    let mut buf = [0u8; 20];
    let _ = write_stderr(usize_to_dec(n, &mut buf));
    let _ = write_stderr(b"\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dec_formats() {
        let mut buf = [0u8; 20];
        assert_eq!(usize_to_dec(0, &mut buf), b"0");
        assert_eq!(usize_to_dec(7, &mut buf), b"7");
        assert_eq!(usize_to_dec(12345, &mut buf), b"12345");
        assert_eq!(
            usize_to_dec(usize::MAX, &mut buf),
            usize::MAX.to_string().as_bytes()
        );
    }
}
