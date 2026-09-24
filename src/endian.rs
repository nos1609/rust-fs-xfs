//! On-disk integer decoding.
//!
//! Every module that reads XFS structures goes through these helpers
//! rather than calling `from_be_bytes` directly. That is deliberate: the
//! correctness of this entire driver rests on one rule, and a rule
//! enforced in one place cannot drift out of sync with itself.
//!
//! # The rule
//!
//! **XFS stores every multi-byte on-disk field in big-endian order, on
//! every host.** It is the only big-endian format among its sibling
//! drivers — ext4, Btrfs and NTFS are all little-endian — so the habits
//! carried over from those crates are actively wrong here.
//!
//! **The single exception is checksum fields**, which are little-endian.
//! The kernel's `xfs_end_cksum()` returns `~cpu_to_le32(crc)`, so a CRC
//! read big-endian like everything around it makes every real filesystem
//! look corrupt. [`le32`] exists for exactly that case and for nothing
//! else; if you are reaching for it and the field is not a checksum, the
//! field is big-endian and you want [`be32`].
//!
//! # Why this module exists
//!
//! These four functions previously existed as private copies in three
//! separate modules — eight identical bodies. Nothing forced them to
//! agree. A single copy quietly switched to `from_le_bytes` would parse
//! its own hand-built test fixtures perfectly while disagreeing with
//! every real filesystem, which is the exact failure mode that has
//! already cost this crate three bugs. One definition removes the
//! opportunity.

/// Read a big-endian `u16` at `off`.
///
/// # Panics
///
/// Panics if `off + 2` exceeds `b`. Callers bounds-check the whole
/// structure once on entry, so a panic here means a caller skipped that
/// check rather than that the input was short.
#[inline]
pub fn be16(b: &[u8], off: usize) -> u16 {
    u16::from_be_bytes([b[off], b[off + 1]])
}

/// Read a big-endian `u32` at `off`.
///
/// # Panics
///
/// Panics if `off + 4` exceeds `b`. See [`be16`].
#[inline]
pub fn be32(b: &[u8], off: usize) -> u32 {
    u32::from_be_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// Read a big-endian `u64` at `off`.
///
/// # Panics
///
/// Panics if `off + 8` exceeds `b`. See [`be16`].
#[inline]
pub fn be64(b: &[u8], off: usize) -> u64 {
    u64::from_be_bytes([
        b[off],
        b[off + 1],
        b[off + 2],
        b[off + 3],
        b[off + 4],
        b[off + 5],
        b[off + 6],
        b[off + 7],
    ])
}

/// Read a **little-endian** `u32` at `off`.
///
/// Use this for checksum fields and nothing else — see the module
/// documentation for why XFS stores those in the opposite order from
/// every other field it has.
///
/// # Panics
///
/// Panics if `off + 4` exceeds `b`. See [`be16`].
#[inline]
pub fn le32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// Read the little-endian `u64` at `off`.
///
/// The log's item structures are the ones that use this: an on-disk
/// superblock or btree header is big-endian, while a log item is copied
/// out of native memory, so its 64-bit fields — a buffer item's device
/// address above all — are little-endian.
///
/// # Panics
///
/// Panics if `off + 8` exceeds `b`. See [`be16`].
#[inline]
pub fn le64(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes([
        b[off],
        b[off + 1],
        b[off + 2],
        b[off + 3],
        b[off + 4],
        b[off + 5],
        b[off + 6],
        b[off + 7],
    ])
}

/// Copy the 16-byte UUID at `off`.
///
/// # Panics
///
/// Panics if `off + 16` exceeds `b`. See [`be16`].
#[inline]
pub fn uuid_at(b: &[u8], off: usize) -> [u8; 16] {
    b[off..off + 16].try_into().expect("16 bytes")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decisive test for this module: the same four bytes must
    /// decode differently through the big-endian and little-endian
    /// readers. If these ever agree, one of them is wrong.
    #[test]
    fn big_and_little_endian_readers_disagree() {
        let b = [0x12u8, 0x34, 0x56, 0x78];
        assert_eq!(be32(&b, 0), 0x1234_5678);
        assert_eq!(le32(&b, 0), 0x7856_3412);
        assert_ne!(be32(&b, 0), le32(&b, 0));
    }

    #[test]
    fn reads_at_an_offset() {
        let b = [0xFFu8, 0xFF, 0x12, 0x34, 0x56, 0x78];
        assert_eq!(be16(&b, 2), 0x1234);
        assert_eq!(be32(&b, 2), 0x1234_5678);
    }

    #[test]
    fn reads_a_full_width_u64() {
        let b = [0x01u8, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF];
        assert_eq!(be64(&b, 0), 0x0123_4567_89AB_CDEF);
    }

    /// A byte pattern that is a palindrome decodes identically either
    /// way, which is why round-trip tests over symmetric fixtures cannot
    /// catch a byte-order mistake.
    #[test]
    fn palindromic_bytes_hide_byte_order_mistakes() {
        let b = [0x12u8, 0x34, 0x34, 0x12];
        assert_eq!(be32(&b, 0), le32(&b, 0));
    }

    #[test]
    fn copies_a_uuid() {
        let mut b = [0u8; 20];
        for (i, slot) in b[4..20].iter_mut().enumerate() {
            *slot = i as u8;
        }
        let u = uuid_at(&b, 4);
        assert_eq!(u[0], 0);
        assert_eq!(u[15], 15);
    }
}
