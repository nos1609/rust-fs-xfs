//! The block-map B+tree (`bmbt`).
//!
//! A file's data fork holds its extents inline as a plain array for as
//! long as they fit in the inode. Once they stop fitting, XFS moves them
//! into a B+tree and the fork holds that tree's root instead. Everything
//! else about the extents is unchanged — the leaves store exactly the
//! same 16-byte packed records [`crate::extent`] already decodes, so the
//! only new work is finding them.
//!
//! # Two shapes of the same node
//!
//! The root lives *inside the inode*, where there is no room for the
//! self-describing header an on-disk block carries, so it is stored in a
//! shortened form (`xfs_bmdr_block`) with a 4-byte header and no magic,
//! no siblings, no checksum. Every other node is a full block
//! (`xfs_bmbt_block`) with a 24-byte header on v4 and 72 on v5.
//!
//! The bodies are laid out identically, and the layout is the part worth
//! stating plainly because it is not the obvious one:
//!
//! ```text
//! leaf   (level 0):  [ header ][ rec rec rec ... ]
//! node   (level >0): [ header ][ key key key ... ][ ptr ptr ptr ... ]
//!                              |<-- maxrecs -->|
//! ```
//!
//! The pointers do **not** begin after the *used* keys — they begin
//! after the space reserved for the *maximum* number of keys the node
//! could hold. Sizing that space from `numrecs` reads every pointer from
//! the wrong offset in any node that is not completely full, which is
//! most of them. `maxrecs` therefore has to be derived from how much
//! room the node has, and it differs between the root (limited by the
//! fork) and the rest (limited by the block size).
//!
//! # Why a depth-first walk
//!
//! The leaves are also threaded together by sibling pointers, and
//! following that thread from the leftmost leaf is the cheaper way to
//! collect every record. This walks the tree instead, because a
//! descent can check each step against the parent that led to it —
//! level below parent's, magic as expected, record count within bounds —
//! while a sibling chain can only be followed and trusted. On a
//! filesystem this driver did not create and may not have been shut down
//! cleanly, that difference is worth more than the saved reads.

use crate::endian::{be16, be32, be64, le32, uuid_at};
use crate::error::{Error, Result};
use crate::extent::{self, Extent, EXTENT_RECORD_SIZE};
use crate::superblock::{crc32c_with_zeroed_crc, Superblock};

/// `XFS_BMAP_MAGIC` — "BMAP", the v4 on-disk node.
pub const XFS_BMAP_MAGIC: u32 = 0x424d_4150;
/// `XFS_BMAP_CRC_MAGIC` — "BMA3", the v5 on-disk node.
pub const XFS_BMAP_CRC_MAGIC: u32 = 0x424d_4133;

/// `XFS_BMDR_BLOCK_LEN` — the in-inode root's header: level and numrecs.
const ROOT_HEADER_LEN: usize = 4;
/// `XFS_BMBT_BLOCK_LEN` for v4: magic, level, numrecs, two siblings.
const V4_HEADER_LEN: usize = 24;
/// `XFS_BMBT_BLOCK_LEN` for v5: the v4 fields plus block number, LSN,
/// UUID, owner, CRC and padding.
const V5_HEADER_LEN: usize = 72;

/// Both a key (`xfs_bmbt_key`, one `startoff`) and a pointer
/// (`xfs_bmbt_ptr`, one filesystem block number) are 8 bytes, and a leaf
/// record is 16. So a node spends 16 bytes per slot whichever kind it
/// is, which is why one `maxrecs` serves for leaves and nodes alike.
const KEY_LEN: usize = 8;
const PTR_LEN: usize = 8;

/// Byte offsets within an on-disk node header.
pub(crate) mod offsets {
    pub const MAGIC: usize = 0;
    pub const LEVEL: usize = 4;
    pub const NUMRECS: usize = 6;
    /// v5 only, from here down.
    ///
    /// `bb_blkno` — the block's own address, in 512-byte basic blocks.
    /// It sits at 24 rather than the short-form tree's 16 by the same
    /// +8 shift that puts `UUID` at 40 instead of 32: the long form
    /// carries 64-bit sibling pointers.
    pub const BLKNO: usize = 24;
    pub const UUID: usize = 40;
    pub const OWNER: usize = 56;
    pub const CRC: usize = 64;
}

/// Offsets within the in-inode root, which has no magic to precede them.
mod root_offsets {
    pub const LEVEL: usize = 0;
    pub const NUMRECS: usize = 2;
}

/// A tree deeper than this is rejected rather than descended. XFS's own
/// limit is `XFS_BM_MAXLEVELS`, which is 9 even for the largest
/// supported geometry; anything claiming more is corrupt, and bounding
/// it means a damaged `level` cannot drive an unbounded descent.
const MAX_LEVELS: u16 = 9;

/// How many slots a node of `usable` body bytes can hold.
///
/// Mirrors `xfs_bmbt_maxrecs` / `xfs_bmdr_maxrecs`. Both a key+pointer
/// pair and a leaf record occupy 16 bytes, so the two cases coincide.
fn maxrecs(usable: usize) -> usize {
    usable / (KEY_LEN + PTR_LEN)
}

/// The parsed header of a node, whichever form it came in.
struct Node {
    level: u16,
    numrecs: u16,
    /// Where the body starts — past the header.
    body: usize,
    /// Slots this node could hold, which is what positions the pointers.
    maxrecs: usize,
}

/// Read the in-inode root's header out of the data fork.
///
/// The root is never a leaf: a level-0 tree is precisely the case XFS
/// keeps as an inline extent array, so reaching here with `level == 0`
/// means the inode's format and its fork disagree.
fn parse_root(fork: &[u8], ino: u64) -> Result<Node> {
    if fork.len() < ROOT_HEADER_LEN {
        return Err(Error::BadSuperblock(format!(
            "inode {ino}: data fork holds {} bytes, too few for a bmbt root",
            fork.len()
        )));
    }
    let level = be16(fork, root_offsets::LEVEL);
    let numrecs = be16(fork, root_offsets::NUMRECS);
    if level == 0 {
        return Err(Error::BadSuperblock(format!(
            "inode {ino}: bmbt root claims level 0, which would be an inline extent list"
        )));
    }
    if level > MAX_LEVELS {
        return Err(Error::BadSuperblock(format!(
            "inode {ino}: bmbt root claims level {level}, beyond the {MAX_LEVELS}-level maximum"
        )));
    }
    let maxrecs = maxrecs(fork.len() - ROOT_HEADER_LEN);
    if usize::from(numrecs) > maxrecs {
        return Err(Error::BadSuperblock(format!(
            "inode {ino}: bmbt root claims {numrecs} records but has room for {maxrecs}"
        )));
    }
    Ok(Node {
        level,
        numrecs,
        body: ROOT_HEADER_LEN,
        maxrecs,
    })
}

/// Read and check an on-disk node's header.
///
/// `expect_level` is the level the parent said this child sits at, and
/// `ino` the inode the tree belongs to. On v5 the block's own recorded
/// address is checked too. Together they make the descent
/// self-verifying: a block that is not part of this tree, not at the
/// depth the parent believed, or not the one that was asked for, is
/// rejected before its contents are read as extents.
fn parse_block(
    buf: &[u8],
    sb: &Superblock,
    ino: u64,
    fsblock: u64,
    expect_level: u16,
) -> Result<Node> {
    let header = if sb.is_v5() {
        V5_HEADER_LEN
    } else {
        V4_HEADER_LEN
    };
    if buf.len() < header {
        return Err(Error::BadSuperblock(format!(
            "inode {ino}: bmbt block {fsblock} is {} bytes, shorter than its {header}-byte header",
            buf.len()
        )));
    }

    let want_magic = if sb.is_v5() {
        XFS_BMAP_CRC_MAGIC
    } else {
        XFS_BMAP_MAGIC
    };
    let magic = be32(buf, offsets::MAGIC);
    if magic != want_magic {
        return Err(Error::BadSuperblock(format!(
            "inode {ino}: bmbt block {fsblock} has magic {magic:#010x}, expected {want_magic:#010x}"
        )));
    }

    if sb.is_v5() {
        // The owner check is the one that matters: a stale block from
        // another file would otherwise decode into plausible extents and
        // hand back another inode's data.
        let stored = le32(buf, offsets::CRC);
        let computed = crc32c_with_zeroed_crc(buf, offsets::CRC);
        if stored != computed {
            return Err(Error::ChecksumMismatch {
                what: "bmbt block",
                block: fsblock,
            });
        }
        if uuid_at(buf, offsets::UUID) != sb.meta_uuid {
            return Err(Error::BlockIdentityMismatch {
                what: "bmbt block",
                expected: fsblock,
                found: u64::MAX, // UUID mismatch: the address is not meaningful
            });
        }
        let owner = be64(buf, offsets::OWNER);
        if owner != ino {
            return Err(Error::BlockIdentityMismatch {
                what: "bmbt block owner",
                expected: ino,
                found: owner,
            });
        }
        // The block records its own address, so one read from the wrong
        // place says so rather than being believed.
        //
        // Both short-form trees check this and this one did not, which
        // made it the weakest of the three parsers against exactly the
        // failure the check exists for: a pointer that has been
        // corrupted into another valid block of the SAME inode passes
        // the CRC (it is a real block) and passes the owner check (same
        // file), and the level check only catches it when the depths
        // differ. Its address is what tells the two apart.
        let stated = be64(buf, offsets::BLKNO);
        let expected = crate::alloc_btree::blkno_of_fsblock(sb, fsblock);
        if stated != expected {
            return Err(Error::BlockIdentityMismatch {
                what: "bmbt block address",
                expected,
                found: stated,
            });
        }
    }

    let level = be16(buf, offsets::LEVEL);
    if level != expect_level {
        return Err(Error::BadSuperblock(format!(
            "inode {ino}: bmbt block {fsblock} is at level {level}, but its parent points to it \
             as level {expect_level}"
        )));
    }

    let numrecs = be16(buf, offsets::NUMRECS);
    let maxrecs = maxrecs(buf.len() - header);
    if usize::from(numrecs) > maxrecs {
        return Err(Error::BadSuperblock(format!(
            "inode {ino}: bmbt block {fsblock} claims {numrecs} records but has room for {maxrecs}"
        )));
    }

    Ok(Node {
        level,
        numrecs,
        body: header,
        maxrecs,
    })
}

/// The child pointers of an internal node, in order.
///
/// Reads them from `body + maxrecs * KEY_LEN`, not from after the keys
/// actually in use — see the layout note at the top of this module.
fn pointers(buf: &[u8], node: &Node, ino: u64, source: u64) -> Result<Vec<u64>> {
    let first = node.body + node.maxrecs * KEY_LEN;
    let end = first + usize::from(node.numrecs) * PTR_LEN;
    if end > buf.len() {
        return Err(Error::BadSuperblock(format!(
            "inode {ino}: bmbt block {source} needs {end} bytes for its pointer array \
             but is only {} long",
            buf.len()
        )));
    }
    Ok((0..usize::from(node.numrecs))
        .map(|i| be64(buf, first + i * PTR_LEN))
        .collect())
}

/// The extent records of a leaf, in order.
fn records(buf: &[u8], node: &Node, ino: u64, source: u64) -> Result<Vec<Extent>> {
    let end = node.body + usize::from(node.numrecs) * EXTENT_RECORD_SIZE;
    if end > buf.len() {
        return Err(Error::BadSuperblock(format!(
            "inode {ino}: bmbt leaf {source} needs {end} bytes for its {} records \
             but is only {} long",
            node.numrecs,
            buf.len()
        )));
    }
    extent::parse_list(&buf[node.body..end], u64::from(node.numrecs))
}

/// Collect every extent in an inode's block-map B+tree, in file order.
///
/// `fork` is the inode's data fork — the in-inode root. `read_fsblock`
/// fetches one filesystem block by number; the walker never assumes a
/// block is where the caller's cache would put it.
///
/// `nextents` is what the inode claims, and is checked against what the
/// tree actually yields. A mismatch means one of the two is wrong and
/// there is no way to tell which, so it is an error rather than a
/// preference for either.
pub fn walk<F>(
    fork: &[u8],
    nextents: u64,
    sb: &Superblock,
    ino: u64,
    mut read_fsblock: F,
) -> Result<Vec<Extent>>
where
    F: FnMut(u64) -> Result<Vec<u8>>,
{
    let root = parse_root(fork, ino)?;
    let mut extents = Vec::with_capacity(nextents as usize);

    // Depth-first, left to right, so records arrive in file order and
    // the check below is a plain comparison rather than a sort.
    //
    // Each entry carries the level its parent said it is at, which is
    // what `parse_block` verifies. Blocks are pushed in reverse so the
    // leftmost is visited first.
    let mut stack: Vec<(u64, u16)> = pointers(fork, &root, ino, 0)?
        .into_iter()
        .rev()
        .map(|p| (p, root.level - 1))
        .collect();

    // A corrupt or malicious tree can point a node at one of its own
    // ancestors. Bounding the total visited by what the tree could
    // legitimately contain stops that becoming an unbounded walk without
    // needing to remember every block seen.
    let mut budget = nextents.saturating_add(1).saturating_mul(2).max(64);

    while let Some((fsblock, expect_level)) = stack.pop() {
        budget = budget.checked_sub(1).ok_or_else(|| {
            Error::BadSuperblock(format!(
                "inode {ino}: bmbt walk visited more blocks than {nextents} extents can justify; \
                 the tree is cyclic or its extent count is wrong"
            ))
        })?;

        let buf = read_fsblock(fsblock)?;
        let node = parse_block(&buf, sb, ino, fsblock, expect_level)?;

        if node.level == 0 {
            extents.extend(records(&buf, &node, ino, fsblock)?);
        } else {
            let children = pointers(&buf, &node, ino, fsblock)?;
            stack.extend(children.into_iter().rev().map(|p| (p, node.level - 1)));
        }
    }

    if extents.len() as u64 != nextents {
        return Err(Error::BadSuperblock(format!(
            "inode {ino}: bmbt holds {} extents but the inode records {nextents}",
            extents.len()
        )));
    }
    Ok(extents)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extent::Extent;

    const BLOCK_SIZE: usize = 4096;

    /// Encode one extent into the packed 16-byte record.
    ///
    /// The field widths are the decoder's, imported rather than repeated,
    /// so a fixture cannot drift into agreeing with a mis-shifted reader.
    /// `parse_list` round-trips these in the tests below, which is what
    /// makes them worth anything.
    fn record(e: &Extent) -> [u8; EXTENT_RECORD_SIZE] {
        // bit 127 flag | 73..=126 startoff | 21..=72 startblock | 0..=20 blockcount
        let value: u128 = (u128::from(e.unwritten) << 127)
            | (u128::from(e.startoff) << 73)
            | (u128::from(e.startblock) << 21)
            | u128::from(e.blockcount);
        value.to_be_bytes()
    }

    /// A v5 superblock: 4 KiB blocks, 512-byte sectors and inodes, 4 AGs.
    /// Same shape as `dir.rs`'s test builder, kept local so the two can
    /// diverge without one silently changing the other's fixtures.
    fn v5_superblock() -> Superblock {
        let (agcount, agblocks, agblklog) = (4u32, 1024u32, 10u8);
        let mut b = vec![0u8; 512];
        b[0..4].copy_from_slice(&crate::superblock::XFS_SB_MAGIC.to_be_bytes());
        b[4..8].copy_from_slice(&4096u32.to_be_bytes()); // blocksize
        let dblocks = u64::from(agcount) * u64::from(agblocks);
        b[8..16].copy_from_slice(&dblocks.to_be_bytes());
        b[48..56].copy_from_slice(&100u64.to_be_bytes()); // logstart
        b[56..64].copy_from_slice(&128u64.to_be_bytes()); // rootino
        b[84..88].copy_from_slice(&agblocks.to_be_bytes());
        b[88..92].copy_from_slice(&agcount.to_be_bytes());
        let versionnum = 5u16 | crate::superblock::version_flags::MOREBITSBIT;
        b[100..102].copy_from_slice(&versionnum.to_be_bytes());
        b[102..104].copy_from_slice(&512u16.to_be_bytes()); // sectsize
        b[104..106].copy_from_slice(&512u16.to_be_bytes()); // inodesize
        b[106..108].copy_from_slice(&8u16.to_be_bytes()); // inopblock
        b[120] = 12; // blocklog
        b[121] = 9; // sectlog
        b[122] = 9; // inodelog
        b[123] = 3; // inopblog
        b[124] = agblklog;
        for (i, slot) in b[32..48].iter_mut().enumerate() {
            *slot = i as u8;
        }
        let crc = crc32c_with_zeroed_crc(&b, 224);
        b[224..228].copy_from_slice(&crc.to_le_bytes());
        Superblock::parse(&b).expect("v5 superblock")
    }

    /// A leaf block holding `recs`, checksummed if `sb` is v5.
    ///
    /// `at` is the filesystem block the test will hand this back for.
    /// A v5 block records its own address, so a fixture that does not
    /// set one is not a block the reader should accept — which is
    /// exactly what these fixtures were before the check existed.
    fn leaf(sb: &Superblock, ino: u64, at: u64, recs: &[Extent]) -> Vec<u8> {
        let header = if sb.is_v5() {
            V5_HEADER_LEN
        } else {
            V4_HEADER_LEN
        };
        let mut b = vec![0u8; BLOCK_SIZE];
        let magic = if sb.is_v5() {
            XFS_BMAP_CRC_MAGIC
        } else {
            XFS_BMAP_MAGIC
        };
        b[0..4].copy_from_slice(&magic.to_be_bytes());
        b[4..6].copy_from_slice(&0u16.to_be_bytes()); // level 0
        b[6..8].copy_from_slice(&(recs.len() as u16).to_be_bytes());
        for (i, e) in recs.iter().enumerate() {
            let at = header + i * EXTENT_RECORD_SIZE;
            b[at..at + EXTENT_RECORD_SIZE].copy_from_slice(&record(e));
        }
        if sb.is_v5() {
            b[offsets::UUID..offsets::UUID + 16].copy_from_slice(&sb.meta_uuid);
            b[offsets::OWNER..offsets::OWNER + 8].copy_from_slice(&ino.to_be_bytes());
            let blkno = crate::alloc_btree::blkno_of_fsblock(sb, at);
            b[offsets::BLKNO..offsets::BLKNO + 8].copy_from_slice(&blkno.to_be_bytes());
            let crc = crc32c_with_zeroed_crc(&b, offsets::CRC);
            b[offsets::CRC..offsets::CRC + 4].copy_from_slice(&crc.to_le_bytes());
        }
        b
    }

    /// An internal block at `level` pointing at `children`, to be read
    /// at filesystem block `at`.
    fn node(sb: &Superblock, ino: u64, at: u64, level: u16, children: &[(u64, u64)]) -> Vec<u8> {
        let header = if sb.is_v5() {
            V5_HEADER_LEN
        } else {
            V4_HEADER_LEN
        };
        let mut b = vec![0u8; BLOCK_SIZE];
        let magic = if sb.is_v5() {
            XFS_BMAP_CRC_MAGIC
        } else {
            XFS_BMAP_MAGIC
        };
        b[0..4].copy_from_slice(&magic.to_be_bytes());
        b[4..6].copy_from_slice(&level.to_be_bytes());
        b[6..8].copy_from_slice(&(children.len() as u16).to_be_bytes());
        let mx = maxrecs(BLOCK_SIZE - header);
        for (i, (startoff, ptr)) in children.iter().enumerate() {
            let k = header + i * KEY_LEN;
            b[k..k + 8].copy_from_slice(&startoff.to_be_bytes());
            let p = header + mx * KEY_LEN + i * PTR_LEN;
            b[p..p + 8].copy_from_slice(&ptr.to_be_bytes());
        }
        if sb.is_v5() {
            b[offsets::UUID..offsets::UUID + 16].copy_from_slice(&sb.meta_uuid);
            b[offsets::OWNER..offsets::OWNER + 8].copy_from_slice(&ino.to_be_bytes());
            let blkno = crate::alloc_btree::blkno_of_fsblock(sb, at);
            b[offsets::BLKNO..offsets::BLKNO + 8].copy_from_slice(&blkno.to_be_bytes());
            let crc = crc32c_with_zeroed_crc(&b, offsets::CRC);
            b[offsets::CRC..offsets::CRC + 4].copy_from_slice(&crc.to_le_bytes());
        }
        b
    }

    /// An in-inode root of `fork_len` bytes pointing at `children`.
    fn root(level: u16, fork_len: usize, children: &[(u64, u64)]) -> Vec<u8> {
        let mut f = vec![0u8; fork_len];
        f[0..2].copy_from_slice(&level.to_be_bytes());
        f[2..4].copy_from_slice(&(children.len() as u16).to_be_bytes());
        let mx = maxrecs(fork_len - ROOT_HEADER_LEN);
        for (i, (startoff, ptr)) in children.iter().enumerate() {
            let k = ROOT_HEADER_LEN + i * KEY_LEN;
            f[k..k + 8].copy_from_slice(&startoff.to_be_bytes());
            let p = ROOT_HEADER_LEN + mx * KEY_LEN + i * PTR_LEN;
            f[p..p + 8].copy_from_slice(&ptr.to_be_bytes());
        }
        f
    }

    fn ext(startoff: u64, startblock: u64, blockcount: u64) -> Extent {
        Extent {
            startoff,
            startblock,
            blockcount,
            unwritten: false,
        }
    }

    #[test]
    fn single_leaf_under_the_root() {
        let sb = v5_superblock();
        let ino = 1234;
        let recs = [ext(0, 100, 1), ext(1, 200, 2), ext(3, 300, 1)];
        let block = leaf(&sb, ino, 42, &recs);
        let fork = root(1, 96, &[(0, 42)]);

        let got = walk(&fork, 3, &sb, ino, |b| {
            assert_eq!(b, 42);
            Ok(block.clone())
        })
        .expect("walk");
        assert_eq!(got, recs);
    }

    /// The layout trap: pointers sit after room for `maxrecs` keys, not
    /// after the keys in use. A root with two children in a fork with
    /// space for many is the case that catches getting this wrong.
    #[test]
    fn pointers_are_placed_past_the_full_key_array() {
        let sb = v5_superblock();
        let ino = 7;
        let left = [ext(0, 10, 1)];
        let right = [ext(1, 20, 1)];
        let lb = leaf(&sb, ino, 11, &left);
        let rb = leaf(&sb, ino, 22, &right);
        let fork = root(1, 200, &[(0, 11), (1, 22)]);

        let got = walk(&fork, 2, &sb, ino, |b| match b {
            11 => Ok(lb.clone()),
            22 => Ok(rb.clone()),
            other => panic!("unexpected block {other}"),
        })
        .expect("walk");
        assert_eq!(got, [left[0], right[0]]);
    }

    #[test]
    fn three_level_tree_is_walked_in_file_order() {
        let sb = v5_superblock();
        let ino = 99;
        let a = [ext(0, 10, 1)];
        let b = [ext(1, 20, 1)];
        let c = [ext(2, 30, 1)];
        let la = leaf(&sb, ino, 101, &a);
        let lb = leaf(&sb, ino, 102, &b);
        let lc = leaf(&sb, ino, 103, &c);
        let n1 = node(&sb, ino, 201, 1, &[(0, 101), (1, 102)]);
        let n2 = node(&sb, ino, 202, 1, &[(2, 103)]);
        let fork = root(2, 96, &[(0, 201), (2, 202)]);

        let got = walk(&fork, 3, &sb, ino, |blk| match blk {
            201 => Ok(n1.clone()),
            202 => Ok(n2.clone()),
            101 => Ok(la.clone()),
            102 => Ok(lb.clone()),
            103 => Ok(lc.clone()),
            other => panic!("unexpected block {other}"),
        })
        .expect("walk");
        assert_eq!(got, [a[0], b[0], c[0]]);
    }

    #[test]
    fn a_block_owned_by_another_inode_is_refused() {
        let sb = v5_superblock();
        let block = leaf(&sb, 4321, 42, &[ext(0, 10, 1)]);
        let fork = root(1, 96, &[(0, 42)]);
        let err = walk(&fork, 1, &sb, 1234, |_| Ok(block.clone())).unwrap_err();
        assert!(
            format!("{err}").contains("bmbt block owner") && format!("{err}").contains("4321"),
            "got {err}"
        );
    }

    /// A block read from somewhere other than where it says it lives is
    /// refused.
    ///
    /// This was the one identity check the block-map tree did not make,
    /// while both short-form trees did — which left it weakest against
    /// exactly what the check is for. A pointer corrupted into another
    /// valid block **of the same file** passes the CRC (it is a real
    /// block), passes the owner check (same inode), and passes the level
    /// check whenever the two sit at the same depth. Its recorded
    /// address is the only field that separates them.
    ///
    /// So the fixture here is not damaged in any way: it is a
    /// well-formed leaf of the right inode at the right level, read at
    /// the wrong block.
    #[test]
    fn a_block_read_at_an_address_it_was_not_written_for_is_refused() {
        let sb = v5_superblock();
        let ino = 5;
        let block = leaf(&sb, ino, 77, &[ext(0, 10, 1)]);
        let fork = root(1, 96, &[(0, 42)]);

        let err = walk(&fork, 1, &sb, ino, |_| Ok(block.clone())).unwrap_err();
        let text = format!("{err}");
        assert!(
            text.contains("bmbt block address"),
            "the refusal should name the address: {err}"
        );

        // And it is not passing for some other reason: the same block
        // read where it belongs is accepted.
        let fork_ok = root(1, 96, &[(0, 77)]);
        walk(&fork_ok, 1, &sb, ino, |b| {
            assert_eq!(b, 77);
            Ok(block.clone())
        })
        .expect("the block is fine where it lives");
    }

    /// `bb_blkno` counts 512-byte basic blocks, not filesystem blocks.
    ///
    /// Comparing it against a filesystem block number directly is right
    /// only when the block size is 512, so the bug would hide on the one
    /// geometry nobody uses and fire on every real one.
    #[test]
    fn the_recorded_address_is_in_basic_blocks_not_filesystem_blocks() {
        let sb = v5_superblock(); // 4 KiB blocks
        assert_eq!(sb.blocksize, 4096);
        assert_eq!(
            crate::alloc_btree::blkno_of_fsblock(&sb, 1),
            8,
            "one 4 KiB block is eight 512-byte basic blocks"
        );
        assert_eq!(crate::alloc_btree::blkno_of_fsblock(&sb, 0), 0);

        // Written the wrong way round — the fsblock stamped as-is — the
        // block would be refused at its own address.
        let ino = 5;
        let mut block = leaf(&sb, ino, 42, &[ext(0, 10, 1)]);
        block[offsets::BLKNO..offsets::BLKNO + 8].copy_from_slice(&42u64.to_be_bytes());
        let crc = crc32c_with_zeroed_crc(&block, offsets::CRC);
        block[offsets::CRC..offsets::CRC + 4].copy_from_slice(&crc.to_le_bytes());

        let fork = root(1, 96, &[(0, 42)]);
        let err = walk(&fork, 1, &sb, ino, |_| Ok(block.clone())).unwrap_err();
        assert!(format!("{err}").contains("bmbt block address"), "got {err}");
    }

    #[test]
    fn a_corrupted_block_fails_its_checksum() {
        let sb = v5_superblock();
        let ino = 5;
        let mut block = leaf(&sb, ino, 42, &[ext(0, 10, 1)]);
        block[V5_HEADER_LEN + 4] ^= 0xFF;
        let fork = root(1, 96, &[(0, 42)]);
        let err = walk(&fork, 1, &sb, ino, |_| Ok(block.clone())).unwrap_err();
        assert!(format!("{err}").contains("CRC32C"), "got {err}");
    }

    #[test]
    fn a_child_at_the_wrong_level_is_refused() {
        let sb = v5_superblock();
        let ino = 5;
        // The root says level 2, so its children must be level 1; hand it
        // a leaf instead.
        let block = leaf(&sb, ino, 42, &[ext(0, 10, 1)]);
        let fork = root(2, 96, &[(0, 42)]);
        let err = walk(&fork, 1, &sb, ino, |_| Ok(block.clone())).unwrap_err();
        assert!(format!("{err}").contains("level"), "got {err}");
    }

    /// The level check already bounds *depth*: every descent expects one
    /// level below its parent, so a pointer back to an ancestor fails
    /// before it is followed and a chain must bottom out at a leaf. What
    /// it does not bound is *breadth* — a node may legitimately name
    /// hundreds of children, and nothing stops a corrupt one naming the
    /// same block hundreds of times at every level. Two levels of that is
    /// already thousands of reads for a file claiming one extent, so the
    /// walk is also budgeted against what the extent count can justify.
    #[test]
    fn a_node_fanning_out_to_one_block_repeatedly_is_bounded() {
        let sb = v5_superblock();
        let ino = 5;
        let mx = maxrecs(BLOCK_SIZE - V5_HEADER_LEN);
        let fanout: Vec<(u64, u64)> = (0..mx as u64).map(|i| (i, 300)).collect();
        let wide = node(&sb, ino, 200, 1, &fanout);
        let leaf_block = leaf(&sb, ino, 300, &[ext(0, 10, 1)]);
        let fork = root(2, 96, &[(0, 200)]);

        let err = walk(&fork, 1, &sb, ino, |b| match b {
            200 => Ok(wide.clone()),
            300 => Ok(leaf_block.clone()),
            other => panic!("unexpected block {other}"),
        })
        .unwrap_err();
        assert!(format!("{err}").contains("cyclic"), "got {err}");
    }

    #[test]
    fn a_root_claiming_level_zero_is_refused() {
        let sb = v5_superblock();
        let fork = root(0, 96, &[]);
        let err = walk(&fork, 0, &sb, 5, |_| unreachable!()).unwrap_err();
        assert!(format!("{err}").contains("level 0"), "got {err}");
    }

    #[test]
    fn an_extent_count_that_disagrees_with_the_tree_is_refused() {
        let sb = v5_superblock();
        let ino = 5;
        let block = leaf(&sb, ino, 42, &[ext(0, 10, 1)]);
        let fork = root(1, 96, &[(0, 42)]);
        let err = walk(&fork, 9, &sb, ino, |_| Ok(block.clone())).unwrap_err();
        assert!(format!("{err}").contains("records 9"), "got {err}");
    }

    #[test]
    fn maxrecs_matches_the_kernels_arithmetic() {
        // 4 KiB block, v5: (4096 - 72) / 16.
        assert_eq!(maxrecs(BLOCK_SIZE - V5_HEADER_LEN), 251);
        // 4 KiB block, v4: (4096 - 24) / 16.
        assert_eq!(maxrecs(BLOCK_SIZE - V4_HEADER_LEN), 254);
    }
}
