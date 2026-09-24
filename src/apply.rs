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
//! The table below is that measurement rather than a recollection. It was
//! taken on a real image with `tmp/ai/scripts/xfs-dir3-probe.py`, which also
//! settled two things a scan alone gets wrong: a directory or attribute
//! block's magic is a `u16` at offset 8, not the `u32` at 0 that every other
//! kind leads with, and its checksum is little-endian like the rest, despite
//! being declared `__be32` upstream.
//!
//! # The sequence number is stamped, not copied
//!
//! Upstream's write verifier sets the metadata's `lsn` from the log item and
//! only then recomputes the checksum, and neither field is written to the log
//! at all — so a replayer has to supply the transaction's own sequence number
//! and cannot read one out of the record. That is why applying takes an
//! `lsn`; [`crate::format::log_items::log_dinode`] goes further and forbids
//! recovering a logged `di_lsn` into the inode even when it is present.
//!
//! # Inode clusters carry no block checksum
//!
//! A logged inode is addressed through its cluster, and a cluster is inodes
//! laid end to end: per-object CRCs and self-identifiers, with the buffer
//! verifier only checking that the magics are in the expected slots. So an
//! inode item is applied to one inode's bytes and restamped as one inode,
//! which is the opposite of what a block item needs. What is still refused is
//! a fork logged alongside the core, whose bytes belong at `di_forkoff`.

use std::sync::Arc;

use fs_core::BlockDevice;

use crate::ag::offsets as ag_offsets;
use crate::ag::{XFS_AGF_MAGIC, XFS_AGI_MAGIC};
use crate::alloc_btree::{XFS_ABTB_CRC_MAGIC, XFS_ABTB_MAGIC, XFS_ABTC_CRC_MAGIC, XFS_ABTC_MAGIC};
use crate::bmbt::{XFS_BMAP_CRC_MAGIC, XFS_BMAP_MAGIC};
use crate::endian::{be16, be32, be64, le32, le64};
use crate::error::{Error, Result};
use crate::format::dir::offsets::da_blk;
use crate::format::dir::offsets::dir3_blk;
use crate::format::dir::{
    XFS_DA3_NODE_MAGIC, XFS_DIR3_BLOCK_MAGIC, XFS_DIR3_DATA_MAGIC, XFS_DIR3_FREE_MAGIC,
    XFS_DIR3_LEAF1_MAGIC, XFS_DIR3_LEAFN_MAGIC,
};
use crate::format::log_items::{
    buf_log_format::{offsets as blf, BLF_CHUNK, BLF_HEADER_SIZE},
    inode_log_format::{
        offsets as ilf, INODE_LOG_FORMAT_SIZE, XFS_ILOG_ADATA, XFS_ILOG_AEXT, XFS_ILOG_CORE,
        XFS_ILOG_DDATA, XFS_ILOG_DEXT,
    },
    item_types::{XFS_LI_BUF, XFS_LI_INODE},
    log_dinode::{offsets as dinode, DI_VERSION_3, LOG_DINODE_SIZE, XFS_DINODE_MAGIC},
    BBSIZE,
};
use crate::log_write::{log_dinode_to_disk, Op};
use crate::superblock::{crc32c_with_zeroed_crc, Superblock};

/// A block's type marker: where it sits, and how wide it is.
///
/// The width matters because directory and attribute blocks lead with two
/// sibling pointers, so their magic is a `u16` further into the header,
/// while every other kind here opens with a `u32` magic.
enum Magic {
    Be32 { off: usize, value: u32 },
    Be16 { off: usize, value: u16 },
}

impl Magic {
    fn is_at(&self, buf: &[u8]) -> bool {
        match self {
            Magic::Be32 { off, value } => buf.len() >= off + 4 && be32(buf, *off) == *value,
            Magic::Be16 { off, value } => buf.len() >= off + 2 && be16(buf, *off) == *value,
        }
    }

    /// How far into a block this marker reaches, so a kind can demand that
    /// much buffer before reading its own fields out of it.
    fn reach(&self) -> usize {
        match self {
            Magic::Be32 { off, .. } => *off + 4,
            Magic::Be16 { off, .. } => *off + 2,
        }
    }
}

/// A block whose CRC can be restamped: the magic that identifies it, the
/// field's offset, and how much of the block the checksum covers.
struct CrcKind {
    magic: Magic,
    crc_off: usize,
    lsn_off: usize,
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
        magic: Magic::Be32 {
            off: 0,
            value: XFS_AGF_MAGIC,
        },
        crc_off: ag_offsets::agf::CRC,
        lsn_off: ag_offsets::agf::LSN,
        span: Span::Sector,
    },
    CrcKind {
        magic: Magic::Be32 {
            off: 0,
            value: XFS_AGI_MAGIC,
        },
        crc_off: ag_offsets::agi::CRC,
        lsn_off: ag_offsets::agi::LSN,
        span: Span::Sector,
    },
    CrcKind {
        magic: Magic::Be32 {
            off: 0,
            value: XFS_ABTB_CRC_MAGIC,
        },
        crc_off: crate::alloc_btree::offsets::CRC,
        lsn_off: crate::alloc_btree::offsets::LSN,
        span: Span::FsBlock,
    },
    CrcKind {
        magic: Magic::Be32 {
            off: 0,
            value: XFS_ABTB_MAGIC,
        },
        crc_off: crate::alloc_btree::offsets::CRC,
        lsn_off: crate::alloc_btree::offsets::LSN,
        span: Span::FsBlock,
    },
    CrcKind {
        magic: Magic::Be32 {
            off: 0,
            value: XFS_ABTC_CRC_MAGIC,
        },
        crc_off: crate::alloc_btree::offsets::CRC,
        lsn_off: crate::alloc_btree::offsets::LSN,
        span: Span::FsBlock,
    },
    CrcKind {
        magic: Magic::Be32 {
            off: 0,
            value: XFS_ABTC_MAGIC,
        },
        crc_off: crate::alloc_btree::offsets::CRC,
        lsn_off: crate::alloc_btree::offsets::LSN,
        span: Span::FsBlock,
    },
    CrcKind {
        magic: Magic::Be32 {
            off: 0,
            value: XFS_BMAP_CRC_MAGIC,
        },
        crc_off: crate::bmbt::offsets::CRC,
        lsn_off: crate::bmbt::offsets::LSN,
        span: Span::FsBlock,
    },
    CrcKind {
        magic: Magic::Be32 {
            off: 0,
            value: XFS_BMAP_MAGIC,
        },
        crc_off: crate::bmbt::offsets::CRC,
        lsn_off: crate::bmbt::offsets::LSN,
        span: Span::FsBlock,
    },
    // Directory blocks, all little-endian-checksummed over one filesystem
    // block. The three that lead with their magic share `dir3_blk`; the
    // index blocks that lead with sibling pointers share `da_blk`, whose
    // magic, checksum and sequence number all sit 8 bytes later.
    CrcKind {
        magic: Magic::Be32 {
            off: dir3_blk::MAGIC,
            value: XFS_DIR3_DATA_MAGIC,
        },
        crc_off: dir3_blk::CRC,
        lsn_off: dir3_blk::LSN,
        span: Span::FsBlock,
    },
    CrcKind {
        magic: Magic::Be32 {
            off: dir3_blk::MAGIC,
            value: XFS_DIR3_BLOCK_MAGIC,
        },
        crc_off: dir3_blk::CRC,
        lsn_off: dir3_blk::LSN,
        span: Span::FsBlock,
    },
    CrcKind {
        magic: Magic::Be32 {
            off: dir3_blk::MAGIC,
            value: XFS_DIR3_FREE_MAGIC,
        },
        crc_off: dir3_blk::CRC,
        lsn_off: dir3_blk::LSN,
        span: Span::FsBlock,
    },
    CrcKind {
        magic: Magic::Be16 {
            off: da_blk::MAGIC,
            value: XFS_DIR3_LEAF1_MAGIC,
        },
        crc_off: da_blk::CRC,
        lsn_off: da_blk::LSN,
        span: Span::FsBlock,
    },
    CrcKind {
        magic: Magic::Be16 {
            off: da_blk::MAGIC,
            value: XFS_DIR3_LEAFN_MAGIC,
        },
        crc_off: da_blk::CRC,
        lsn_off: da_blk::LSN,
        span: Span::FsBlock,
    },
    CrcKind {
        magic: Magic::Be16 {
            off: da_blk::MAGIC,
            value: XFS_DA3_NODE_MAGIC,
        },
        crc_off: da_blk::CRC,
        lsn_off: da_blk::LSN,
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
    /// The bytes the checksum of this kind covers, clamped to what was read.
    fn span(&self, buf: &[u8], sb: &Superblock) -> usize {
        let full = match self.span {
            Span::Sector => usize::from(sb.sectsize),
            Span::FsBlock => sb.blocksize as usize,
        };
        full.min(buf.len())
    }

    /// Whether this is the format the block is actually in: the stored
    /// checksum has to come out of the block's own bytes.
    fn matches(&self, buf: &[u8], sb: &Superblock) -> bool {
        // The magic decides the kind; the checksum then decides that this
        // kind's offsets are the ones the writer used. Both are needed,
        // because a truncated or wrong-kind buffer can carry a magic and no
        // usable checksum field.
        if buf.len() < self.reach().max(self.crc_off + 4) {
            return false;
        }
        if !self.magic.is_at(buf) {
            return false;
        }
        let span = self.span(buf, sb);
        span > self.crc_off + 4
            && le32(buf, self.crc_off) == crc32c_with_zeroed_crc(&buf[..span], self.crc_off)
    }

    /// Width of the header this kind needs intact, so a short buffer cannot
    /// be read past its own fields.
    fn reach(&self) -> usize {
        self.magic
            .reach()
            .max(self.crc_off + 4)
            .max(self.lsn_off + 8)
    }

    /// Stamp this transaction's sequence number in, then recompute the
    /// checksum over what the block now says.
    ///
    /// That order is upstream's: the write verifier assigns the LSN and only
    /// then calls `xfs_update_cksum`, so stamping afterwards would leave a
    /// checksum over bytes that no longer match it.
    fn restamp(&self, buf: &mut [u8], sb: &Superblock, lsn: u64) {
        if buf.len() >= self.lsn_off + 8 {
            buf[self.lsn_off..self.lsn_off + 8].copy_from_slice(&lsn.to_be_bytes());
        }
        let span = self.span(buf, sb);
        let crc = crc32c_with_zeroed_crc(&buf[..span], self.crc_off);
        buf[self.crc_off..self.crc_off + 4].copy_from_slice(&crc.to_le_bytes());
    }
}

/// A parsed inode item: which inode, where its cluster is, and the core.
struct InodeItem {
    ino: u64,
    /// Basic-block address of the cluster, not of the inode.
    blkno: u64,
    /// Cluster length in basic blocks.
    len_bb: u32,
    /// The inode's byte offset within the cluster.
    boffset: u32,
    /// `xfs_log_dinode`, as logged.
    core: Vec<u8>,
}

/// Parse one inode item's format operation and take the core from the next.
fn parse_inode(ops: &[Op], at: usize) -> Result<(InodeItem, usize)> {
    let fmt = &ops[at].data;
    if fmt.len() < INODE_LOG_FORMAT_SIZE {
        return Err(Error::CorruptLog(format!(
            "an inode item's format operation is {} bytes, shorter than the \
             {INODE_LOG_FORMAT_SIZE}-byte header",
            fmt.len()
        )));
    }
    let size = u16::from_le_bytes([fmt[ilf::SIZE], fmt[ilf::SIZE + 1]]) as usize;
    if size < 2 {
        return Err(Error::CorruptLog(format!(
            "an inode item claims {size} operations, which cannot include its core"
        )));
    }
    if at + size > ops.len() {
        return Err(Error::CorruptLog(format!(
            "an inode item needs {size} operations and only {} remain",
            ops.len() - at
        )));
    }
    let fields = le32(fmt, ilf::FIELDS);
    if fields & XFS_ILOG_CORE == 0 {
        return Err(Error::CorruptLog(format!(
            "an inode item logs fields {fields:#010x} without XFS_ILOG_CORE, so there is no \
             core to apply"
        )));
    }
    let fork = fields & (XFS_ILOG_DDATA | XFS_ILOG_DEXT | XFS_ILOG_ADATA | XFS_ILOG_AEXT);
    if fork != 0 {
        // The fork's bytes belong at `di_forkoff` inside the inode, or in
        // blocks the core's own counts point at, so applying them is a
        // second step rather than part of this one.
        return Err(Error::UnsupportedFeature(format!(
            "an inode item logging a data or attribute fork ({fork:#010x}) is not applied; \
             only the core is, and the fork's bytes would have to land at di_forkoff"
        )));
    }
    Ok((
        InodeItem {
            ino: le64(fmt, ilf::INO),
            blkno: le64(fmt, ilf::BLKNO),
            len_bb: le32(fmt, ilf::LEN) as u32,
            boffset: le32(fmt, ilf::BOFFSET) as u32,
            core: ops[at + 1].data.clone(),
        },
        size,
    ))
}

/// Patch one inode's core into its cluster and restamp the inode's own CRC.
///
/// Returns whether the inode was written: a cluster already carrying a newer
/// sequence number is left alone, which is a correct outcome, not a failure.
fn apply_inode(
    device: &Arc<dyn BlockDevice>,
    sb: &Superblock,
    item: &InodeItem,
    lsn: u64,
) -> Result<bool> {
    let inodesize = usize::from(sb.inodesize);
    let cluster = item.len_bb as usize * BBSIZE;
    // The item's own buffer length is trusted only within the bounds this
    // geometry allows; a huge one is a damaged field, not a large read.
    if !(inodesize..=sb.inode_cluster_bytes() as usize).contains(&cluster) {
        return Err(Error::CorruptLog(format!(
            "an inode item for inode {} names a {}-byte buffer, outside this geometry's \
             inodesize {inodesize}..cluster {}",
            item.ino,
            cluster,
            sb.inode_cluster_bytes(),
        )));
    }
    let boff = item.boffset as usize;
    if boff % inodesize != 0 || boff + inodesize > cluster {
        return Err(Error::CorruptLog(format!(
            "an inode item puts inode {} at offset {boff} of a {cluster}-byte buffer of \
             {inodesize}-byte inodes",
            item.ino
        )));
    }
    // A logged core too short to hold a v3 layout, or belonging to a version
    // this step cannot lay out on disk: v2 inodes have no checksum field to
    // restamp, and a v5 filesystem does not use them, so reaching one means
    // the addressing is wrong rather than merely old.
    if item.core.len() < LOG_DINODE_SIZE {
        return Err(Error::CorruptLog(format!(
            "an inode item for inode {} carries a {}-byte core, shorter than a v3 core's {LOG_DINODE_SIZE}",
            item.ino,
            item.core.len()
        )));
    }
    if item.core[dinode::VERSION] != DI_VERSION_3 {
        return Err(Error::UnsupportedFeature(format!(
            "an inode item for inode {} logs version {}, which this step cannot write back",
            item.ino,
            item.core[dinode::VERSION]
        )));
    }
    let core = log_dinode_to_disk(&item.core)
        .map_err(|why| Error::CorruptLog(format!("inode item for inode {}: {why}", item.ino)))?;
    if core.len() > inodesize {
        return Err(Error::CorruptLog(format!(
            "a {}-byte logged core does not fit a {inodesize}-byte inode",
            core.len()
        )));
    }

    let daddr = item.blkno * BBSIZE as u64;
    let mut buf = vec![0u8; cluster];
    device.read_at(daddr, &mut buf)?;
    let mut inode = buf[boff..boff + inodesize].to_vec();

    // The cluster addresses the inode, so nothing else has checked that it
    // is the *same* inode. An allocated one states its own number; a free
    // slot says nothing and is this item's to fill.
    if be16(&inode, dinode::MAGIC) == XFS_DINODE_MAGIC {
        let here = be64(&inode, dinode::INO);
        if here != item.ino {
            return Err(Error::CorruptLog(format!(
                "an inode item for inode {} lands on inode {here} at byte {boff} of buffer \
                 block {}",
                item.ino, item.blkno
            )));
        }
        // Upstream compares the on-disk sequence number against the
        // transaction's and skips a strictly newer inode, because the logged
        // `di_lsn` cannot be trusted for that decision and the disk's own can.
        let stamped = be64(&inode, dinode::LSN);
        if stamped != 0 && stamped != u64::MAX && stamped > lsn {
            return Ok(false);
        }
    }

    inode[..core.len()].copy_from_slice(&core);
    inode[dinode::LSN..dinode::LSN + 8].copy_from_slice(&lsn.to_be_bytes());
    // The inode's checksum covers the whole inode, not the core, so it has to
    // be recomputed over `inodesize` bytes after the fork area below the core
    // has been read back in.
    let crc = crc32c_with_zeroed_crc(&inode, dinode::CRC);
    inode[dinode::CRC..dinode::CRC + 4].copy_from_slice(&crc.to_le_bytes());
    buf[boff..boff + inodesize].copy_from_slice(&inode);
    // The whole cluster goes back, as the kernel writes the whole buffer:
    // an inode is not necessarily a whole sector, and a sub-sector write is
    // not something every device can honour.
    device.write_at(daddr, &buf)?;
    Ok(true)
}

/// Apply one transaction's items to the blocks and inodes they describe.
///
/// Returns how many byte runs were written — one per buffer item's run and
/// one per inode applied. Operations of a kind this cannot apply are refused
/// with the missing case named, so a caller never ends up with half a
/// transaction on disk and a success to show for it.
///
/// `lsn` is this transaction's log sequence number, `(cycle << 32 | block)`,
/// and is written into every object touched: a replayer has to supply it,
/// because neither the checksum nor the sequence number is logged.
///
/// # Errors
///
/// [`Error::CorruptLog`] for a malformed or truncated item, [`Error::BadSuperblock`]
/// for a block whose checksum format is not recognised, [`Error::UnsupportedFeature`]
/// for an item kind that is not applied here, and whatever the device returns.
pub fn apply_transaction(
    device: &Arc<dyn BlockDevice>,
    sb: &Superblock,
    ops: &[Op],
    lsn: u64,
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
                apply_one(device, sb, &item, &ops[at + 1..at + consumed], lsn)?;
                written += item.runs.len();
                at += consumed;
            }
            XFS_LI_INODE => {
                let (item, consumed) = parse_inode(ops, at)?;
                if apply_inode(device, sb, &item, lsn)? {
                    written += 1;
                }
                at += consumed;
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
    lsn: u64,
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
                be32(&buf, 0)
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

    kind.restamp(&mut buf, sb, lsn);
    device.write_at(daddr, &buf)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ag::offsets as ag_off;
    use crate::ag::XFS_AGF_MAGIC;
    use crate::format::log_items::buf_log_format::offsets as blf_off;
    use crate::format::log_items::inode_log_format::offsets as ilf_off;
    use crate::format::log_items::log_dinode::offsets as di;
    use crate::log_write::{inode_log_format, log_dinode_from_disk, InodeBuffer};
    use std::sync::Mutex;

    /// A transaction's log sequence number: cycle 1, log block 40.
    const LSN: u64 = (1 << 32) | 40;

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

    impl fs_core::BlockDevice for Mem {
        // Without this the trait's read-only default answers every apply
        // with "device is read-only", which reads like a production bug
        // and is only a fixture that forgot to let anyone write.
        fn write_at(&self, offset: u64, buf: &[u8]) -> fs_core::Result<()> {
            let mut b = self.bytes.lock().unwrap();
            let start = offset as usize;
            if start + buf.len() > b.len() {
                return Err(fs_core::Error::ShortRead {
                    offset,
                    want: buf.len(),
                    got: 0,
                });
            }
            b[start..start + buf.len()].copy_from_slice(buf);
            Ok(())
        }

        fn is_writable(&self) -> bool {
            true
        }
    }

    /// A parseable v5 superblock: 4 KiB blocks, 512-byte sectors and
    /// inodes, 4 AGs of 1000 blocks.
    ///
    /// A local copy of the builder in `dir.rs`'s tests, which follows the
    /// choice already made in `bmbt.rs`: two fixture builders that can
    /// diverge are better than one that silently changes another module's
    /// ground. It is also the reason this test exists — an incomplete one
    /// parses to a `Superblock` only if its own CRC is stamped, and the
    /// first version of this helper skipped the geometry logs and the
    /// checksum and died in `parse`.
    fn sb_v5() -> Superblock {
        let mut b = vec![0u8; 512];
        b[0..4].copy_from_slice(&crate::superblock::XFS_SB_MAGIC.to_be_bytes());
        b[4..8].copy_from_slice(&4096u32.to_be_bytes()); // blocksize
        b[8..16].copy_from_slice(&4000u64.to_be_bytes()); // dblocks
        b[48..56].copy_from_slice(&100u64.to_be_bytes()); // logstart
        b[56..64].copy_from_slice(&128u64.to_be_bytes()); // rootino
        b[84..88].copy_from_slice(&1000u32.to_be_bytes()); // agblocks
        b[88..92].copy_from_slice(&4u32.to_be_bytes()); // agcount
        b[100..102]
            .copy_from_slice(&(5u16 | crate::superblock::version_flags::MOREBITSBIT).to_be_bytes());
        b[102..104].copy_from_slice(&512u16.to_be_bytes()); // sectsize
        b[104..106].copy_from_slice(&512u16.to_be_bytes()); // inodesize
        b[106..108].copy_from_slice(&8u16.to_be_bytes()); // inopblock
        b[120] = 12; // blocklog
        b[121] = 9; // sectlog
        b[122] = 9; // inodelog
        b[123] = 3; // inopblog
        b[124] = 10; // agblklog
        b[192] = 0; // dirblklog: directory blocks are one fs block
        for (i, slot) in b[32..48].iter_mut().enumerate() {
            *slot = i as u8; // distinctive UUID
        }
        let crc = crc32c_with_zeroed_crc(&b, 224);
        b[224..228].copy_from_slice(&crc.to_le_bytes());
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
        // Chunk 3 (bytes 384..512) rather than a low one: the AGF's own
        // checksum sits at 216, inside chunk 1, and the restamp rewrites
        // those four bytes — which is the right order for a replayer
        // (patch, then stamp) and the wrong place to look for the patch.
        let ops = buf_item(0, 3, payload.clone());

        let runs = apply_transaction(&dev, &sb, &ops, LSN).expect("apply one buffer item");
        assert_eq!(runs, 1);

        let mut after = vec![0u8; 4096];
        dev.read_at(0, &mut after).unwrap();
        let run = 3 * BLF_CHUNK;
        assert!(
            after[run..run + BLF_CHUNK].iter().all(|&x| x == 0xC3),
            "chunk 3 of the block should hold the logged bytes"
        );
        assert_eq!(
            after[BLF_CHUNK / 2],
            0,
            "untouched chunks must stay untouched"
        );
        assert_eq!(
            be64(&after, ag_off::agf::LSN),
            LSN,
            "the block's sequence number has to become this transaction's, since neither it \
             nor the checksum is logged"
        );
        assert_eq!(
            le32(&after, ag_off::agf::CRC),
            crc32c_with_zeroed_crc(&after[..512], ag_off::agf::CRC),
            "the restamped checksum has to verify against the block's own bytes, sequence \
             number included"
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
        let err = apply_transaction(&dev, &sb, &ops, LSN)
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
        let err = apply_transaction(&dev, &sb, &[Op { flags: 0, data }], LSN)
            .expect_err("an inode item without a core is not applicable");
        assert!(
            matches!(err, Error::CorruptLog(_)),
            "a one-operation inode item should be reported as malformed: {err}"
        );
    }

    /// An allocated v3 inode of `ino`, with a checksum that is its own.
    ///
    /// `flags2` stays zero, so the core uses the base layout: `bigtime` and
    /// `nrext64` move fields, and how the logged form tracks them is
    /// `log_dinode_from_disk`'s own concern, tested there.
    fn disk_inode(sb: &Superblock, ino: u64) -> Vec<u8> {
        let mut b = vec![0u8; usize::from(sb.inodesize)];
        b[di::MAGIC..di::MAGIC + 2].copy_from_slice(&XFS_DINODE_MAGIC.to_be_bytes());
        b[di::VERSION] = DI_VERSION_3;
        b[di::MODE..di::MODE + 2].copy_from_slice(&0o100644u16.to_be_bytes());
        b[di::NLINK..di::NLINK + 4].copy_from_slice(&1u32.to_be_bytes());
        b[di::INO..di::INO + 8].copy_from_slice(&ino.to_be_bytes());
        b[di::NEXT_UNLINKED..di::NEXT_UNLINKED + 4].copy_from_slice(&u32::MAX.to_be_bytes());
        let crc = crc32c_with_zeroed_crc(&b, di::CRC);
        b[di::CRC..di::CRC + 4].copy_from_slice(&crc.to_le_bytes());
        b
    }

    /// One inode item: the format operation naming `buffer`, then the core.
    fn inode_item(ino: u64, fields: u32, buffer: &InodeBuffer, core: &[u8]) -> Vec<Op> {
        let mut fmt = inode_log_format(ino, fields, buffer);
        fmt[ilf_off::TYPE..ilf_off::TYPE + 2].copy_from_slice(&XFS_LI_INODE.to_le_bytes());
        vec![
            Op {
                flags: 0,
                data: fmt,
            },
            Op {
                flags: 0,
                data: core.to_vec(),
            },
        ]
    }

    /// The round trip the replayer depends on: log an inode, apply it back,
    /// and the inode on disk is the one that went in — with the two fields a
    /// log cannot carry (`di_lsn`, `di_crc`) replaced by this transaction's.
    #[test]
    fn a_logged_inode_core_round_trips_into_the_cluster_slot_it_was_addressed_to() {
        let sb = sb_v5();
        let cluster = sb.inode_cluster_bytes() as usize;
        let boff = usize::from(sb.inodesize); // the cluster's second inode
        let before = disk_inode(&sb, 64);

        let mut image = vec![0u8; cluster];
        image[boff..boff + before.len()].copy_from_slice(&before);
        let dev = Arc::new(Mem {
            bytes: Mutex::new(image.clone()),
        }) as Arc<dyn BlockDevice>;

        let buffer = InodeBuffer::containing(boff as u64, sb.inode_cluster_bytes());
        assert_eq!(
            (buffer.blkno, buffer.len, buffer.boffset),
            (0, cluster as u32 / BBSIZE as u32, boff as u32),
            "the cluster is addressed absolutely, so its first inode sits at offset 0"
        );
        let core = log_dinode_from_disk(&before).expect("log the inode");
        let ops = inode_item(64, XFS_ILOG_CORE, &buffer, &core);

        let runs = apply_transaction(&dev, &sb, &ops, LSN).expect("apply the inode item");
        assert_eq!(runs, 1);

        let mut after = vec![0u8; cluster];
        dev.read_at(0, &mut after).unwrap();
        let got = &after[boff..boff + usize::from(sb.inodesize)];

        // Everything the log does carry has to come back identical.
        let strip = |b: &[u8]| {
            let mut v = b.to_vec();
            v[di::CRC..di::CRC + 4].fill(0);
            v[di::LSN..di::LSN + 8].fill(0);
            v
        };
        assert_eq!(
            strip(got),
            strip(&before),
            "the applied inode differs from the logged one only in the fields a log cannot carry"
        );
        assert_eq!(
            be64(got, di::LSN),
            LSN,
            "the inode's sequence number is the transaction's, never the logged core's"
        );
        assert_eq!(
            le32(got, di::CRC),
            crc32c_with_zeroed_crc(got, di::CRC),
            "an inode's checksum covers the whole inode, so it is recomputed, not copied"
        );
        // The rest of the cluster is somebody else's inode.
        assert_eq!(
            &after[0..boff],
            &image[0..boff],
            "applying one inode must not touch the neighbours packed into its cluster"
        );
        assert_eq!(
            &after[boff + usize::from(sb.inodesize)..],
            &image[boff + usize::from(sb.inodesize)..],
            "applying one inode must not touch the neighbours packed into its cluster"
        );
    }

    /// The cluster and offset address an inode; nothing else has checked that
    /// they address the one the item names. Getting this wrong while the
    /// checksum still works is how a replayer would silently relocate files.
    #[test]
    fn an_inode_item_naming_a_different_inode_than_occupies_the_slot_is_refused() {
        let sb = sb_v5();
        let cluster = sb.inode_cluster_bytes() as usize;
        let boff = usize::from(sb.inodesize);
        let before = disk_inode(&sb, 64);
        let mut image = vec![0u8; cluster];
        image[boff..boff + before.len()].copy_from_slice(&before);
        let dev = Arc::new(Mem {
            bytes: Mutex::new(image.clone()),
        }) as Arc<dyn BlockDevice>;

        let buffer = InodeBuffer::containing(boff as u64, sb.inode_cluster_bytes());
        let core = log_dinode_from_disk(&before).expect("log the inode");
        let ops = inode_item(99, XFS_ILOG_CORE, &buffer, &core);

        let err = apply_transaction(&dev, &sb, &ops, LSN)
            .expect_err("an item cannot claim a slot that holds another inode");
        let why = err.to_string();
        assert!(
            why.contains("99") && why.contains("64"),
            "the refusal should name both inode numbers: {why}"
        );
        let mut same = vec![0u8; cluster];
        dev.read_at(0, &mut same).unwrap();
        assert_eq!(same, image, "a refusal must not leave the cluster edited");
    }

    /// Upstream compares the disk inode's own sequence number against the
    /// transaction's and leaves a strictly newer one alone, which is what
    /// makes replaying an older checkpoint harmless.
    #[test]
    fn an_inode_already_stamped_by_a_newer_transaction_is_left_alone() {
        let sb = sb_v5();
        let cluster = sb.inode_cluster_bytes() as usize;
        let boff = usize::from(sb.inodesize);
        let mut before = disk_inode(&sb, 64);
        before[di::LSN..di::LSN + 8].copy_from_slice(&(LSN + 1).to_be_bytes());
        let crc = crc32c_with_zeroed_crc(&before, di::CRC);
        before[di::CRC..di::CRC + 4].copy_from_slice(&crc.to_le_bytes());

        let mut image = vec![0u8; cluster];
        image[boff..boff + before.len()].copy_from_slice(&before);
        let dev = Arc::new(Mem {
            bytes: Mutex::new(image.clone()),
        }) as Arc<dyn BlockDevice>;

        let buffer = InodeBuffer::containing(boff as u64, sb.inode_cluster_bytes());
        let core = log_dinode_from_disk(&before).expect("log the inode");
        let ops = inode_item(64, XFS_ILOG_CORE, &buffer, &core);

        let runs = apply_transaction(&dev, &sb, &ops, LSN).expect("skipping is not failing");
        assert_eq!(
            runs, 0,
            "the newer inode on disk wins, so nothing is written"
        );
        let mut same = vec![0u8; cluster];
        dev.read_at(0, &mut same).unwrap();
        assert_eq!(same, image);
    }

    /// A core can be logged with fork bytes after it. Those belong at
    /// `di_forkoff`, which is a different step, so the whole item is refused
    /// rather than half-applied.
    #[test]
    fn an_inode_item_logging_a_fork_with_its_core_is_refused_by_name() {
        let sb = sb_v5();
        let before = disk_inode(&sb, 64);
        let mut image = vec![0u8; sb.inode_cluster_bytes() as usize];
        image[0..before.len()].copy_from_slice(&before);
        let dev = Arc::new(Mem {
            bytes: Mutex::new(image.clone()),
        }) as Arc<dyn BlockDevice>;

        let buffer = InodeBuffer::containing(0, sb.inode_cluster_bytes());
        let core = log_dinode_from_disk(&before).expect("log the inode");
        let ops = inode_item(64, XFS_ILOG_CORE | XFS_ILOG_DDATA, &buffer, &core);

        let err = apply_transaction(&dev, &sb, &ops, LSN)
            .expect_err("fork bytes are not applied with the core");
        assert!(
            matches!(err, Error::UnsupportedFeature(_)) && err.to_string().contains("fork"),
            "the refusal should name the missing case: {err}"
        );
    }

    /// A directory index block keeps its magic a `u16` eight bytes in, behind
    /// the two sibling pointers — the shape that made a scan of offset 0
    /// conclude the checksum was nowhere. Recognising it is what lets a
    /// renamed entry's block be restamped instead of refused.
    #[test]
    fn a_directory_index_block_is_recognised_by_the_magic_eight_bytes_in() {
        let sb = sb_v5();
        let mut block = vec![0u8; 4096];
        block[da_blk::MAGIC..da_blk::MAGIC + 2]
            .copy_from_slice(&XFS_DIR3_LEAFN_MAGIC.to_be_bytes());
        let crc = crc32c_with_zeroed_crc(&block, da_blk::CRC);
        block[da_blk::CRC..da_blk::CRC + 4].copy_from_slice(&crc.to_le_bytes());
        let dev = device_with(&block);

        let ops = buf_item(0, 5, vec![0x77u8; BLF_CHUNK]);
        let runs = apply_transaction(&dev, &sb, &ops, LSN).expect("apply to a leaf block");
        assert_eq!(runs, 1);

        let mut after = vec![0u8; 4096];
        dev.read_at(0, &mut after).unwrap();
        assert!(after[5 * BLF_CHUNK..6 * BLF_CHUNK]
            .iter()
            .all(|&x| x == 0x77));
        assert_eq!(be64(&after, da_blk::LSN), LSN);
        assert_eq!(
            le32(&after, da_blk::CRC),
            crc32c_with_zeroed_crc(&after, da_blk::CRC),
            "the leaf's checksum covers the whole block, as measured"
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
        let err = apply_transaction(&dev, &sb, &ops, LSN).expect_err("a short item must not apply");
        assert!(
            err.to_string().contains("covers"),
            "expected the count mismatch: {err}"
        );
    }
}
