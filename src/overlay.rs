//! A block device that keeps its writes in memory.
//!
//! Two reasons use it, and they are not the same reason.
//!
//! Tests need it because the fixtures are shared: `xfsdata-default.img` is
//! read by half a dozen suites, so a test that logged a checkpoint into it
//! would leave a dirty log behind and every one of those suites would then
//! refuse to mount it. Copying the fixture answers that, at the cost of a
//! 500 MiB copy per test.
//!
//! Writing more than one transaction per mount needs it for a different
//! reason. A journalled operation here writes a log record and applies
//! nothing to disk (see [`crate::fs`] and `log_write`), so a second
//! operation on the same mount reads the state the first one started from —
//! and hands out an inode the first one already took. Nothing exists yet
//! that applies a transaction's changes for the next one to read; this is
//! where they will go, and it is the storage half of that, not the whole of
//! it.
//!
//! The unit is 512 bytes: the smallest sector XFS supports, so a page is
//! never split across two of the device's own units.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

use fs_core::{BlockDevice, BlockRead, Result};

/// The unit writes are remembered in — see the note above.
const PAGE: usize = 512;

/// A device whose reads fall through and whose writes do not.
pub struct OverlayDevice {
    below: Arc<dyn BlockDevice>,
    written: Mutex<HashMap<u64, [u8; PAGE]>>,
}

impl OverlayDevice {
    pub fn new(below: Arc<dyn BlockDevice>) -> Self {
        Self {
            below,
            written: Mutex::new(HashMap::new()),
        }
    }
}

/// The pages a request covers, as `(page index, the span within the
/// request, the span within that page)`.
///
/// One iterator rather than two loops because the read and the write need
/// exactly the same arithmetic in opposite directions, and an off-by-one in
/// either would surface as metadata that is almost right — the hardest kind
/// of wrong to recognise in a filesystem.
fn pages(
    offset: u64,
    len: usize,
) -> impl Iterator<Item = (u64, std::ops::Range<usize>, std::ops::Range<usize>)> {
    let page = PAGE as u64;
    let end = offset + len as u64;
    (offset / page..end.div_ceil(page)).map(move |index| {
        let base = index * page;
        let from = offset.max(base);
        let to = end.min(base + page);
        (
            index,
            (from - offset) as usize..(to - offset) as usize,
            (from - base) as usize..(to - base) as usize,
        )
    })
}

impl BlockRead for OverlayDevice {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.below.read_at(offset, buf)?;
        let written = self.written.lock().expect("the overlay is not poisoned");
        for (index, in_request, in_page) in pages(offset, buf.len()) {
            if let Some(page) = written.get(&index) {
                buf[in_request].copy_from_slice(&page[in_page]);
            }
        }
        Ok(())
    }

    fn size_bytes(&self) -> u64 {
        self.below.size_bytes()
    }
}

impl BlockDevice for OverlayDevice {
    fn write_at(&self, offset: u64, buf: &[u8]) -> Result<()> {
        let mut written = self.written.lock().expect("the overlay is not poisoned");
        for (index, in_request, in_page) in pages(offset, buf.len()) {
            // A partially written page still has to read back whole, so a
            // page that has not been written yet is filled from below before
            // it is patched.
            let page = match written.entry(index) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                std::collections::hash_map::Entry::Vacant(e) => {
                    let mut fresh = [0u8; PAGE];
                    self.below.read_at(index * PAGE as u64, &mut fresh)?;
                    e.insert(fresh)
                }
            };
            page[in_page].copy_from_slice(&buf[in_request]);
        }
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        // Nothing of ours has reached the device, so there is nothing of
        // ours to flush. Flushing what is below is still the device's own
        // business and is passed on.
        self.below.flush()
    }

    /// `true`, whatever is below.
    ///
    /// Two things depend on this and they pull the same way. `mount_rw`
    /// refuses a device that does not claim to be writable
    /// ([`crate::Filesystem::mount_rw`]), and the write-refusal tests need
    /// to mount a fixture they are allowed to change without changing it —
    /// so the overlay has to say yes while the file underneath is opened
    /// read-only.
    ///
    /// The cost is stated rather than hidden: an overlay whose base cannot
    /// take the pages reports itself writable, and what it holds never
    /// arrives. Making that honest belongs to the step that pushes dirty
    /// pages down, not to this one — a `flush` that wrote the map out would
    /// be the place to fail.
    fn is_writable(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A device over a fixed image in memory, so the page arithmetic can be
    /// checked without a filesystem or a file.
    struct ConstDevice(u64);

    impl BlockRead for ConstDevice {
        fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
            for (i, slot) in buf.iter_mut().enumerate() {
                *slot = ((offset + i as u64) % 251) as u8;
            }
            Ok(())
        }
        fn size_bytes(&self) -> u64 {
            self.0
        }
    }

    impl BlockDevice for ConstDevice {}

    fn overlay() -> Arc<OverlayDevice> {
        Arc::new(OverlayDevice::new(
            Arc::new(ConstDevice(1 << 20)) as Arc<dyn BlockDevice>
        ))
    }

    #[test]
    fn an_unwritten_range_reads_through() {
        let dev = overlay();
        let mut buf = [0u8; 16];
        dev.read_at(4096, &mut buf).unwrap();
        let mut expect = [0u8; 16];
        dev.below.read_at(4096, &mut expect).unwrap();
        assert_eq!(buf, expect);
    }

    /// The whole reason the page arithmetic is shared: a write that starts
    /// and ends inside a page must leave the rest of that page as it was,
    /// and must not bleed into the neighbours.
    #[test]
    fn a_partial_page_write_keeps_its_edges() {
        let dev = overlay();
        // 300 bytes starting 100 into the first page: spans pages 0 and 1.
        let payload = vec![0xA5u8; 300];
        dev.write_at(100, &payload).unwrap();

        let mut whole = [0u8; 1024];
        dev.read_at(0, &mut whole).unwrap();
        let mut base = [0u8; 1024];
        dev.below.read_at(0, &mut base).unwrap();

        assert_eq!(&whole[..100], &base[..100], "leading bytes of page 0");
        assert!(
            whole[100..400].iter().all(|&b| b == 0xA5),
            "the written span should read back as written"
        );
        assert_eq!(
            &whole[400..512],
            &base[400..512],
            "trailing bytes of page 0"
        );
        assert_eq!(
            &whole[512..],
            &base[512..],
            "page 1 was not touched by the tail"
        );
    }

    /// Writing a whole page and reading a subrange of it must agree with the
    /// write, not with what was underneath.
    #[test]
    fn a_written_page_wins_for_any_subrange() {
        let dev = overlay();
        dev.write_at(2048, &vec![0x5Au8; 512]).unwrap();
        let mut buf = [0u8; 40];
        dev.read_at(2200, &mut buf).unwrap();
        assert!(
            buf.iter().all(|&b| b == 0x5A),
            "a subrange of a written page"
        );
    }

    /// The claim the whole type exists for: writable to the mount, and the
    /// device below never sees a byte of it. Asserting both halves together
    /// is the point — `mount_rw` refuses a device that does not say it can
    /// write, so an overlay that answered honestly about its base could not
    /// be mounted at all, and an overlay that leaked would silently dirty a
    /// shared fixture.
    #[test]
    fn writes_are_accepted_and_do_not_reach_the_device_below() {
        let plain: Arc<dyn BlockDevice> = Arc::new(ConstDevice(1 << 16));
        assert!(!plain.is_writable(), "fs_core's default models read-only");

        let ov = Arc::new(OverlayDevice::new(plain.clone()));
        assert!(ov.is_writable(), "the overlay accepts writes into memory");

        let mut buf = [0u8; 8];
        ov.write_at(512, &[0x11u8; 8]).unwrap();
        ov.read_at(512, &mut buf).unwrap();
        assert!(
            buf.iter().all(|&b| b == 0x11),
            "the overlay serves its own write"
        );

        let mut untouched = [0u8; 8];
        plain.read_at(512, &mut untouched).unwrap();
        let mut original = [0u8; 8];
        ov.below.read_at(512, &mut original).unwrap();
        assert_eq!(untouched, original, "the device below was not written to");
        assert_ne!(untouched, buf, "and it does not agree with the overlay");
    }
}
