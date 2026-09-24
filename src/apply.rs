//! Apply a transaction's buffer items to the blocks they describe.
//!
//! A journalled operation in this crate writes a log record and changes
//! nothing else — that is what makes every one of them checkable, because a
//! filesystem that came out different is one something replayed. It also
//! means a second operation on the same mount reads the state the first one
//! started from. Applying the record the way a replayer would is what lets
//! the second operation see the first, and this is that step.
//!
//! # Checksums are verified before they are rewritten
//!
//! A v5 block's CRC covers the block with the CRC field zeroed (and, per
//! [`crate::ag`], over a whole sector for AG headers), so restamping is only
//! correct if the offset and span are the ones the writer used. Rather than
//! assume, each candidate is tried: if recomputing the stored CRC reproduces
//! what the block already carries, the candidate is the block's own format
//! and the restamp is safe; if none reproduce, the write is refused. A wrong
//! guess here produces a block whose checksum no reader will accept, which is
//! worse than not applying anything, and the caller cannot tell the two
//! apart without a second tool.
//!
//! # Inode items are refused, not skipped
//!
//! An inode item logs the same change a buffer item does but addressed
//! through the cluster and its offset (see [`crate::log_write::InodeBuffer`]),
//! and the logged chunk carries a stale inode image in some workloads — the
//! module docs of [`crate::format::log_items`] record that applying it
//! verbatim corrupts the inode. Until that case is handled deliberately,
//! refusing names the missing piece instead of quietly leaving a block
//! half-applied.

use std::sync::Arc;

use fs_core::BlockDevice;

use crate::ag::offsets as ag_offsets;
use crate::ag::{XFS_AGF_MAGIC, XFS_AGI_MAGIC};
use crate::alloc_btree::{XFS_ABTB_CRC_MAGIC, XFS_ABTB_MAGIC, XFS_ABTC_CRC_MAGIC, XFS_ABTC_MAGIC};
use crate::bmbt::{XFS_BMAP_CRC_MAGIC, XFS_BMAP_MAGIC};
use crate::endian::{le32, le64};
use crate::error::{Error, Result};
use crate::format::log_items::{
    buf_log_format::{offsets as blf, BLF_CHUNK, BLF_HEADER_SIZE},
    item_types::{XFS_LI_BUF, XFS_LI_INODE},
    BBSIZE,
};
use crate::log_write::Op;
use crate::superblock::{crc32c_with_zeroed_crc, Superblock};

/// A block whose CRC can be restamped: the magic that identifies it, the
/// field's offset, and how much of the block the checksum covers.
struct CrcKind {
    magic: u32,
    crc_off: usize,
    /// `Sector` for AG headers, `FsBlock` for B+tree blocks.
    span: Span,
}

enum Span {
    Sector,
    FsBlock,
}

/// The kinds this step knows the checksum layout of. The offsets are the
/// ones the parsers already verify with, taken from them rather than
/// repeated here.
const CRC_KINDS: &[CrcKind] = &[
    CrcKind {
        magic: XFS_AGF_MAGIC,
        crc_off: ag_offsets::agf::CRC,
        span: Span::Sector,
    },
    CrcKind {
        magic: XFS_AGI_MAGIC,
        crc_off: ag_offsets::agi::CRC,
        span: Span::Sector,
    },
    CrcKind {
        magic: XFS_ABTB_CRC_MAGIC,
        crc_off: crate::alloc_btree::offsets::CRC,
        span: Span::FsBlock,
    },
    CrcKind {
        magic: XFS_ABTB_MAGIC,
        crc_off: crate::alloc_btree::offsets::CRC,
        span: Span::FsBlock,
    },
    CrcKind {
        magic: XFS_ABTC_CRC_MAGIC,
        crc_off: crate::alloc_btree::offsets::CRC,
        span: Span::FsBlock,
    },
    CrcKind {
        magic: XFS_ABTC_MAGIC,
        crc_off: crate::alloc_btree::offsets::CRC,
        span: Span::FsBlock,
    },
    CrcKind {
        magic: XFS_BMAP_CRC_MAGIC,
        crc_off: crate::bmbt::offsets::CRC,
        span: Span::FsBlock,
    },
    CrcKind {
        magic: XFS_BMAP_MAGIC,
        crc_off: crate::bmbt::offsets::CRC,
        span: Span::FsBlock,
    },
];

/// A parsed buffer item: where its bytes go, and the runs themselves.
struct BufItem {
    /// Absolute device address, in 512-byte basic blocks.
    blkno: u64,
    /// `(byte offset within the buffer, length)` per data operation.
    runs: Vec<(usize, usize)>,
}

/// Where a maximal run of set bits starts and how long it is, in chunks.
fn bitmap_runs(map: &[u8]) -> Vec<(usize, usize)> {
    let chunks = map.len() * 8;
    let bit = |c: usize| map[c / 8] & (1 << (c % 8)) != 0;
    let mut runs = Vec::new();
    let mut c = 0;
    while c < chunks {
        if bit(c) {
            let start = c;
            while c < chunks && bit(c) {
                c += 1;
            }
            runs.push((start * BLF_CHUNK, (c - start) * BLF_CHUNK));
        } else {
            c += 1;
        }
    }
    runs
}

/// Parse one buffer item's format operation and claim its data operations.
///
/// Returns the item and how many following operations it consumed.
fn parse_buf(ops: &[Op], at: usize) -> Result<(BufItem, usize)> {
    let fmt = &ops[at].data;
    if fmt.len() < BLF_HEADER_SIZE {
        return Err(Error::CorruptLog(format!(
            "a buffer item's format operation is {} bytes, shorter than the {BLF_HEADER_SIZE}-byte header",
            fmt.len()
        )));
    }
    let size = u16::from_le_bytes([fmt[blf::SIZE], fmt[blf::SIZE + 1]]) as usize;
    let map_words = le32(fmt, blf::MAP_SIZE) as usize;
    if fmt.len() < BLF_HEADER_SIZE + 4 * map_words {
        return Err(Error::CorruptLog(format!(
            "a buffer item with a {map_words}-word bitmap needs {} bytes and has {}",
            BLF_HEADER_SIZE + 4 * map_words,
            fmt.len()
        )));
    }
    // The address is signed in the on-disk struct; every observed value is
    // positive, and a negative one is not a place to write.
    let blkno = le64(fmt, blf::BLKNO);
    if blkno > i64::MAX as u64 {
        return Err(Error::CorruptLog(format!(
            "a buffer item names device address {blkno}, which is not a positive block number"
        )));
    }

    let map = &fmt[BLF_HEADER_SIZE..BLF_HEADER_SIZE + 4 * map_words];
    let runs = bitmap_runs(map);
    if runs.len() != size - 1 {
        return Err(Error::CorruptLog(format!(
            "a buffer item claims {size} operations, which is {} data operations, and its bitmap has {} runs",
            size - 1,
            runs.len()
        )));
    }
    if at + size > ops.len() {
        return Err(Error::CorruptLog(format!(
            "a buffer item needs {size} operations and only {} remain",
            ops.len() - at
        )));
    }
    // The invariant the corpus held on all 434 items: the data operations
    // carry exactly the bytes the bitmap says they must.
    let declared: usize = runs.iter().map(|(_, len)| *len).sum();
    let actual: usize = (1..size).map(|k| ops[at + k].data.len()).sum();
    if declared != actual {
        return Err(Error::CorruptLog(format!(
            "a buffer item's bitmap covers {declared} bytes and its data operations carry {actual}"
        )));
    }
    for (k, &(_, len)) in runs.iter().enumerate() {
        if ops[at + 1 + k].data.len() != len {
            return Err(Error::CorruptLog(format!(
                "data operation {k} is {} bytes for a run of {len}",
                ops[at + 1 + k].data.len()
            )));
        }
    }
    Ok((BufItem { blkno, runs }, size))
}

impl CrcKind {
    /// Whether this is the format the block is actually in: the stored
    /// checksum has to come out of the block's own bytes.
    fn matches(&self, buf: &[u8], sb: &Superblock) -> bool {
        if buf.len() < 4 || le32(buf, 0) != self.magic {
            return false;
        }
        let span = match self.span {
            Span::Sector => usize::from(sb.sectsize),
            Span::FsBlock => sb.blocksize as usize,
        };
        let span = span.min(buf.len());
        if span <= self.crc_off + 4 {
            return false;
        }
        le32(buf, self.crc_off) == crc32c_with_zeroed_crc(&buf[..span], self.crc_off)
    }

    fn restamp(&self, buf: &mut [u8], sb: &Superblock) {
        let span = match self.span {
            Span::Sector => usize::from(sb.sectsize),
            Span::FsBlock => sb.blocksize as usize,
        }
        .min(buf.len());
        let crc = crc32c_with_zeroed_crc(&buf[..span], self.crc_off);
        buf[self.crc_off..self.crc_off + 4].copy_from_slice(&crc.to_le_bytes());
    }
}

/// Apply every buffer item in `ops` to `device`.
///
/// Returns how many byte runs were written. Operations of a kind this cannot
/// apply are refused with the missing case named, so a caller never ends up
/// with half a transaction on disk and a success to show for it.
///
/// # Errors
///
/// [`Error::CorruptLog`] for a malformed or truncated item, [`Error::BadSuperblock`]
/// for a block whose checksum format is not recognised, [`Error::UnsupportedFeature`]
/// for an item kind that is not applied here, and whatever the device returns.
pub fn apply_buf_items(
    device: &Arc<dyn BlockDevice>,
    sb: &Superblock,
    ops: &[Op],
) -> Result<usize> {
    let mut written = 0usize;
    let mut at = 0usize;
    while at < ops.len() {
        let data = &ops[at].data;
        // A transaction is bracketed by START and COMMIT operations, which
        // carry no item; their type byte is not one of the item magics.
        if data.len() < 2 {
            at += 1;
            continue;
        }
        let kind = u16::from_le_bytes([data[0], data[1]]);
        match kind {
            XFS_LI_BUF => {
                let (item, consumed) = parse_buf(ops, at)?;
                apply_one(device, sb, &item, &ops[at + 1..at + consumed])?;
                written += item.runs.len();
                at += consumed;
            }
            XFS_LI_INODE => {
                return Err(Error::UnsupportedFeature(format!(
                    "inode items are not applied yet (operation {at} of this transaction); \
                     the logged chunk needs the cluster/offset addressing and must not be \
                     copied verbatim"
                )));
            }
            other => {
                return Err(Error::UnsupportedFeature(format!(
                    "unrecognised log item type {other:#06x} at operation {at}"
                )));
            }
        }
    }
    Ok(written)
}

/// Patch one buffer item's runs into its block and restamp the checksum.
fn apply_one(
    device: &Arc<dyn BlockDevice>,
    sb: &Superblock,
    item: &BufItem,
    payloads: &[Op],
) -> Result<()> {
    let daddr = item.blkno * BBSIZE as u64;
    let mut buf = vec![0u8; sb.blocksize as usize];
    device.read_at(daddr, &mut buf)?;

    let kind = CRC_KINDS
        .iter()
        .find(|k| k.matches(&buf, sb))
        .ok_or_else(|| {
            Error::BadSuperblock(format!(
                "block at device address {} (magic {:#010x}) is not a kind whose checksum \
                 this step knows; refusing rather than leave it with a stale CRC",
                item.blkno,
                le32(&buf, 0)
            ))
        })?;

    for (run, op) in item.runs.iter().zip(payloads) {
        let (offset, len) = run;
        if offset + len > buf.len() {
            return Err(Error::CorruptLog(format!(
                "a buffer item wants bytes {}..{} of a {}-byte block",
                offset,
                offset + len,
                buf.len()
            )));
        }
        buf[*offset..*offset + len].copy_from_slice(&op.data);
    }

    kind.restamp(&mut buf, sb);
    device.write_at(daddr, &buf)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ag::offsets as ag_off;
    use crate::ag::XFS_AGF_MAGIC;
    use crate::format::log_items::buf_log_format::offsets as blf_off;
    use std::sync::Mutex;

    /// A device of `blocks` filesystem blocks, addressable by byte offset.
    struct Mem {
        bytes: Mutex<Vec<u8>>,
    }

    impl fs_core::BlockRead for Mem {
        fn read_at(&self, offset: u64, buf: &mut [u8]) -> fs_core::Result<()> {
            let b = self.bytes.lock().unwrap();
            let start = offset as usize;
            if start + buf.len() > b.len() {
                return Err(fs_core::Error::ShortRead {
                    offset,
                    want: buf.len(),
                    got: b.len().saturating_sub(start),
                });
            }
            buf.copy_from_slice(&b[start..start + buf.len()]);
            Ok(())
        }
        fn size_bytes(&self) -> u64 {
            self.bytes.lock().unwrap().len() as u64
        }
    }

    impl fs_core::BlockDevice for Mem {}

    /// A v5 superblock: 4 KiB blocks, 512-byte sectors, one AG.
    fn sb_v5() -> Superblock {
        let mut b = vec![0u8; 512];
        b[0..4].copy_from_slice(&crate::superblock::XFS_SB_MAGIC.to_be_bytes());
        b[4..8].copy_from_slice(&4096u32.to_be_bytes());
        b[8..16].copy_from_slice(&4000u64.to_be_bytes()); // dblocks
        b[84..88].copy_from_slice(&1024u32.to_be_bytes()); // agblocks
        b[88..92].copy_from_slice(&1u32.to_be_bytes()); // agcount
        b[100..102].copy_from_slice(&5u16.to_be_bytes()); // v5
        b[102..104].copy_from_slice(&512u16.to_be_bytes()); // sectsize
        b[104..106].copy_from_slice(&512u16.to_be_bytes()); // inodesize
        b[124] = 10; // agblklog
        Superblock::parse(&b).expect("a parseable v5 superblock")
    }

    /// An AGF-shaped block with a checksum that is really its own.
    fn agf_block(sb: &Superblock) -> Vec<u8> {
        let mut b = vec![0u8; 4096];
        b[0..4].copy_from_slice(&XFS_AGF_MAGIC.to_be_bytes());
        let crc = crc32c_with_zeroed_crc(&b[..512], ag_off::agf::CRC);
        b[ag_off::agf::CRC..ag_off::agf::CRC + 4].copy_from_slice(&crc.to_le_bytes());
        assert!(sb.is_v5());
        b
    }

    /// One buffer item: format operation plus one data operation covering
    /// the chunk at `chunk`, exactly as `payload` frames them.
    fn buf_item(blkno_bb: u64, chunk: usize, data: Vec<u8>) -> Vec<Op> {
        let mut fmt = vec![0u8; BLF_HEADER_SIZE + 4];
        fmt[blf_off::TYPE..blf_off::TYPE + 2].copy_from_slice(&XFS_LI_BUF.to_le_bytes());
        fmt[blf_off::SIZE..blf_off::SIZE + 2].copy_from_slice(&2u16.to_le_bytes());
        fmt[blf_off::LEN..blf_off::LEN + 2].copy_from_slice(&8u16.to_le_bytes());
        fmt[blf_off::BLKNO..blf_off::BLKNO + 8].copy_from_slice(&blkno_bb.to_le_bytes());
        fmt[blf_off::MAP_SIZE..blf_off::MAP_SIZE + 4].copy_from_slice(&1u32.to_le_bytes());
        fmt[BLF_HEADER_SIZE + chunk / 32] |= 1 << (chunk % 32);
        vec![
            Op {
                flags: 0,
                data: fmt,
            },
            Op { flags: 0, data },
        ]
    }

    fn device_with(block: &[u8]) -> Arc<dyn BlockDevice> {
        Arc::new(Mem {
            bytes: Mutex::new(block.to_vec()),
        }) as Arc<dyn BlockDevice>
    }

    #[test]
    fn a_buffer_item_lands_at_the_address_it_names_and_keeps_the_checksum_valid() {
        let sb = sb_v5();
        let dev = device_with(&agf_block(&sb));
        let payload = vec![0xC3u8; BLF_CHUNK];
        let ops = buf_item(0, 1, payload.clone());

        let runs = apply_buf_items(&dev, &sb, &ops).expect("apply one buffer item");
        assert_eq!(runs, 1);

        let mut after = vec![0u8; 4096];
        dev.read_at(0, &mut after).unwrap();
        assert!(
            after[BLF_CHUNK..2 * BLF_CHUNK].iter().all(|&x| x == 0xC3),
            "chunk 1 of the block should hold the logged bytes"
        );
        assert_eq!(
            after[BLF_CHUNK / 2],
            0,
            "untouched chunks must stay untouched"
        );
        assert_eq!(
            le32(&after, ag_off::agf::CRC),
            crc32c_with_zeroed_crc(&after[..512], ag_off::agf::CRC),
            "the restamped checksum has to verify against the block's own bytes"
        );
    }

    /// The restamp is only safe if the offset and span are the writer's, so
    /// an unrecognised block is refused rather than quietly left with a
    /// checksum that no reader will accept.
    #[test]
    fn a_block_of_an_unknown_checksum_kind_is_refused() {
        let sb = sb_v5();
        let mut block = vec![0u8; 4096];
        block[0..4].copy_from_slice(&0x58534441u32.to_be_bytes()); // "XSDA": not a kind here
        let dev = device_with(&block);
        let ops = buf_item(0, 0, vec![0x11u8; BLF_CHUNK]);
        let err = apply_buf_items(&dev, &sb, &ops)
            .expect_err("an unknown block kind must not be applied");
        assert!(
            err.to_string().contains("checksum"),
            "the refusal should say why: {err}"
        );
        let mut same = vec![0u8; 4096];
        dev.read_at(0, &mut same).unwrap();
        assert_eq!(same, block, "a refusal must not leave a half-written block");
    }

    #[test]
    fn an_inode_item_is_refused_by_name_rather_than_skipped() {
        let sb = sb_v5();
        let dev = device_with(&agf_block(&sb));
        let mut data = vec![0u8; 4];
        data[0..2].copy_from_slice(&XFS_LI_INODE.to_le_bytes());
        let err = apply_buf_items(&dev, &sb, &[Op { flags: 0, data }])
            .expect_err("inode items are not applied yet");
        assert!(
            matches!(err, Error::UnsupportedFeature(_)),
            "the refusal should name the missing kind: {err}"
        );
    }

    /// The bitmap and the data operations are two accounts of the same
    /// change; the parser is where a disagreement between them surfaces.
    #[test]
    fn a_bitmap_and_data_disagreement_is_caught() {
        let sb = sb_v5();
        let dev = device_with(&agf_block(&sb));
        let mut ops = buf_item(0, 0, vec![0x22u8; BLF_CHUNK]);
        ops[1].data.push(0); // one byte too many for the single bit set
        let err = apply_buf_items(&dev, &sb, &ops).expect_err("a short item must not apply");
        assert!(
            err.to_string().contains("covers"),
            "expected the count mismatch: {err}"
        );
    }
}
