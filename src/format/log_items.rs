//! The on-disk shapes of the XFS journalling log.
//!
//! The published XFS format document has a chapter called "Journaling
//! Log". Its entire body is the word `TODO:`. So unlike every other
//! structure this driver reads, none of what follows could be looked
//! up: each offset, magic and flag below was established by differential
//! analysis against filesystems the kernel itself wrote — build two
//! images differing in exactly one property, dump both logs, and see
//! which bytes moved.
//!
//! This module is a reservoir, not a parser. It holds what that work
//! established, as named constants and offset tables, so that the
//! encoder in [`crate::log_write`], the state check in [`crate::log`]
//! and whatever eventually replays a log all agree on the same numbers
//! and on the same account of where those numbers came from. Where a
//! constant is already named elsewhere in the crate, the name and value
//! here are identical on purpose; this must not become a second,
//! divergent copy of the format.
//!
//! # The one rule about byte order
//!
//! XFS is big-endian on disk everywhere — except here. The format
//! document is explicit about it in the one sentence it does offer:
//! *"All on-disk values are in big-endian format except the journaling
//! log which is in native endian format."*
//!
//! In practice the split runs along the framing/payload seam:
//!
//! - The **framing** — the record header and every operation header —
//!   is big-endian, with the single exception of the record's CRC, which
//!   is little-endian like every other XFS checksum.
//! - Everything **inside** an operation — the transaction header, the
//!   item format structures, the logged inode core — is native-endian.
//!
//! With one exception that the seam does not predict: a fork logged
//! after an inode core stays big-endian. See [`log_dinode`].
//!
//! "Native" is the honest word, and the distinction matters to an
//! encoder. Every observation behind this module was made on arm64, so
//! what was actually seen was little-endian; nothing available here
//! distinguishes native-endian from fixed little-endian, and settling it
//! would need a big-endian host writing a log. The encoder therefore
//! swaps conditionally rather than unconditionally.
//!
//! # Two traps that cost real time
//!
//! **The record checksum covers the header struct, not the header
//! block.** The header is 328 bytes and lives alone in a 512-byte basic
//! block; the remaining 184 bytes are padding, and the checksum does not
//! include them. Ten plausible spans were tried against real records
//! before this one was found, and it is invisible in every other use of
//! the header — the field offsets all work, the record parses, only the
//! checksum is wrong. See [`rec_header`].
//!
//! **A logged inode chunk must not be replayed verbatim.** Unlink writes
//! four bytes at `di_next_unlinked`, which dirties the whole 128-byte
//! buffer chunk containing them — and that chunk carries a *stale*
//! inode image: `di_format` zero, timestamps zero, generation zero,
//! while the disk holds the live inode. 76 of 256 bytes differed, at
//! exactly the fields that would be uninitialised. Copying the chunk
//! across as-is corrupts the inode. Only the unlinked pointer should be
//! applied. See [`buf_log_format`].
//!
//! # Structures held here
//!
//! - [`rec_header`] — `xlog_rec_header`, the per-record frame
//! - [`op_header`] — `xlog_op_header`, the 12 bytes before each operation
//! - [`trans_header`] — `xfs_trans_header`, which opens a transaction
//! - [`item_types`] — the log item type codes
//! - [`inode_log_format`] — `xfs_inode_log_format`, the inode item
//! - [`log_dinode`] — the inode core as the log stores it
//! - [`buf_log_format`] — `xfs_buf_log_format`, the buffer item

/// A log basic block. The log is addressed in these regardless of the
/// filesystem's block size, and so are several fields below.
pub const BBSIZE: usize = 512;

// ---------------------------------------------------------------------
// xlog_rec_header
// ---------------------------------------------------------------------

/// `xlog_rec_header` — the header on every log record.
///
/// A record is one header block (occasionally more) followed by
/// `h_len` bytes of operations, padded up to a whole number of basic
/// blocks. The header carries the magic, the filesystem UUID, the
/// sequence number that orders records around the ring, and the cycle
/// data needed to undo the stamping described below.
///
/// **Byte order: big-endian**, except `h_crc`, which is little-endian
/// like every other XFS checksum, and `h_cycle_data`, which holds raw
/// payload bytes carried across verbatim.
///
/// # The cycle stamp
///
/// Before a record is written, the first four bytes of each 512-byte
/// payload block are replaced by the cycle number, and the word thus
/// displaced is kept in `h_cycle_data[k]` for block `k`. That is what
/// lets a scan tell written log from unwritten log at any offset. A
/// reader undoes it; an encoder must do it *before* checksumming,
/// because the checksum covers the stamped form.
///
/// # The checksum, and why it was expensive
///
/// [`crate::log::record_checksum`] computes CRC32C over
/// `header[..XLOG_REC_HEADER_SIZE]` with `h_crc` treated as zero,
/// followed by the `h_len` stamped payload bytes.
///
/// Ten candidate spans were tried against real records and all ten were
/// wrong. The answer came from two filesystems built identically but for
/// one byte of file data: CRC32C is affine, so for equal-length inputs
/// `crc(A) ^ crc(B)` depends only on where they differ and any unknown
/// seed or final xor cancels. One record in that pair differed only in
/// `h_cycle_data[0]`, and the single span length reproducing its
/// checksum difference was 840 — which is 328 + `h_len`. 328 is the
/// header *struct*; the 512-byte block it sits in is not what is
/// covered. Verified against all 24 checksummed records across four
/// kernel-written filesystems.
///
/// Nothing outside the crate consults these yet — this is a reference
/// module, and the constants are here to be read as much as called.
#[allow(dead_code)]
pub mod rec_header {
    /// `XLOG_HEADER_MAGIC_NUM`, at offset 0 of every record header.
    pub const XLOG_HEADER_MAGIC: u32 = 0xFEED_BABE;

    /// `sizeof(xlog_rec_header)` — the fields, padded to the 8-byte
    /// alignment its `u64` members impose. The last named field ends at
    /// 324; 328 is that rounded up.
    ///
    /// This is **not** the 512 bytes the header occupies on disk. The
    /// header sits alone in a basic block and the remaining 184 bytes
    /// are padding. Only these 328 are checksummed.
    pub const XLOG_REC_HEADER_SIZE: usize = 328;

    /// Bytes of padding between the end of the struct and the end of the
    /// basic block it occupies: `512 - 328`. Named so that the sum is
    /// visible rather than implied.
    pub const XLOG_REC_HEADER_PADDING: usize = super::BBSIZE - XLOG_REC_HEADER_SIZE;

    /// `XLOG_VERSION_1`, in `h_version`. Named by position against
    /// version 2: a header whose `0x2` bit is clear always occupies
    /// exactly one basic block, however large `h_size` claims to be.
    pub const XLOG_VERSION_1: u32 = 1;
    /// `XLOG_VERSION_2`, in `h_version` — the version every observed
    /// record carries. Version 2 records may have headers spanning more
    /// than one basic block.
    pub const XLOG_VERSION_2: u32 = 2;

    /// `XLOG_FMT_LINUX_LE`, in `h_fmt` — the value every observed
    /// record carries. `h_fmt` records which platform and byte order
    /// wrote the log, which is exactly the field a native-endian payload
    /// makes necessary. Other values must exist; none was seen here.
    pub const XLOG_FMT_LINUX_LE: u32 = 1;

    /// `XLOG_HEADER_CYCLE_SIZE` — how much log a single header block can
    /// carry cycle data for, and so the divisor for a multi-block
    /// header: a record whose `h_size` exceeds this spills its cycle
    /// data into further blocks.
    pub const XLOG_HEADER_CYCLE_SIZE: u32 = 32 * 1024;

    /// Entries in `h_cycle_data`: `XLOG_HEADER_CYCLE_SIZE / BBSIZE`,
    /// one per payload block a single header block can describe. The
    /// array runs from offset 44 to offset 300, which is what makes the
    /// count checkable rather than assumed.
    pub const XLOG_CYCLE_DATA_ENTRIES: usize = XLOG_HEADER_CYCLE_SIZE as usize / super::BBSIZE;

    /// Byte offsets within `xlog_rec_header`.
    ///
    /// Fields nothing currently reads are named anyway: an offset can
    /// only be checked against its neighbours if the neighbours are
    /// there to be counted off against, and `h_lsn` at 16 is only
    /// obviously right with `h_cycle`, `h_version` and `h_len` visible
    /// above it.
    pub mod offsets {
        /// `h_magicno`, `u32` — [`super::XLOG_HEADER_MAGIC`].
        pub const MAGICNO: usize = 0;
        /// `h_cycle`, `u32` — the ring's lap counter, also stamped over
        /// the first word of each payload block.
        pub const CYCLE: usize = 4;
        /// `h_version`, `u32` — [`super::XLOG_VERSION_2`].
        pub const VERSION: usize = 8;
        /// `h_len`, `u32` — payload bytes following the header blocks,
        /// before padding to a whole basic block.
        pub const LEN: usize = 12;
        /// `h_lsn`, `u64` — `cycle << 32 | basic block`. Comparing the
        /// whole `u64` orders by cycle first, which is the ordering
        /// wanted, so records can be ranked without unpacking it.
        pub const LSN: usize = 16;
        /// `h_tail_lsn`, `u64` — the oldest record still needed, in the
        /// same packed form. Everything before it has been applied.
        pub const TAIL_LSN: usize = 24;
        /// `h_crc`, `u32` — CRC32C over the header struct and the
        /// record's stamped data, stored **little-endian** like every
        /// other XFS checksum.
        pub const CRC: usize = 32;
        /// `h_prev_block`, `u32` — start block of the previous record,
        /// or `u32::MAX` when there is none.
        pub const PREV_BLOCK: usize = 36;
        /// `h_num_logops`, `u32` — operations in this record. A clean
        /// unmount record holds exactly one.
        pub const NUM_LOGOPS: usize = 40;
        /// `h_cycle_data[64]`, `u32` each — the words displaced by the
        /// cycle stamp, block `k` at `CYCLE_DATA + k * 4`. Runs to 300.
        pub const CYCLE_DATA: usize = 44;
        /// `h_fmt`, `u32` — [`super::XLOG_FMT_LINUX_LE`].
        pub const FMT: usize = 300;
        /// `h_fs_uuid`, 16 bytes — the filesystem's UUID, carried
        /// verbatim. This is what stops stale bytes carrying the magic
        /// from being read as a record.
        pub const FS_UUID: usize = 304;
        /// `h_size`, `u32` — the log's iclog size. Also the divisor
        /// question for multi-block headers.
        pub const SIZE: usize = 320;
    }
}

// ---------------------------------------------------------------------
// xlog_op_header
// ---------------------------------------------------------------------

/// `xlog_op_header` — the 12 bytes in front of every operation.
///
/// A record's payload is nothing but a sequence of these, each followed
/// immediately by `oh_len` bytes of data. There is **no alignment and no
/// padding** between operations: a 2156-byte operation was observed, and
/// the next header began at the next byte.
///
/// **Byte order: big-endian.** `oh_tid` and `oh_len` are the only
/// multi-byte fields; the two flag bytes are order-free, which is why
/// [`crate::log`] can read `oh_flags` without decoding anything.
///
/// Every operation of one transaction carries the same `oh_tid`. That is
/// what ties them together when a transaction spans a record boundary.
///
/// Marked `#[allow(dead_code)]`: a reference module, held to be read.
#[allow(dead_code)]
pub mod op_header {
    /// `sizeof(xlog_op_header)`.
    pub const OP_HEADER_SIZE: usize = 12;

    /// `XFS_TRANSACTION` — the client id in `oh_clientid`. Every
    /// filesystem transaction uses it; no other value was observed.
    pub const XFS_TRANSACTION: u8 = 0x69;

    /// Byte offsets within `xlog_op_header`.
    pub mod offsets {
        /// `oh_tid`, `u32` — the transaction this operation belongs to.
        pub const TID: usize = 0;
        /// `oh_len`, `u32` — payload bytes after this header, excluding
        /// the header itself. Zero for START and COMMIT.
        pub const LEN: usize = 4;
        /// `oh_clientid`, `u8` — [`super::XFS_TRANSACTION`].
        pub const CLIENTID: usize = 8;
        /// `oh_flags`, `u8` — see the flag constants.
        pub const FLAGS: usize = 9;
        /// `oh_res2`, `u16` — reserved, constant zero.
        pub const RES2: usize = 10;
    }

    /// `XLOG_START_TRANS` — an empty operation opening a transaction.
    /// Observed, and named in [`crate::log_write`].
    pub const XLOG_START_TRANS: u8 = 0x01;

    /// `XLOG_COMMIT_TRANS` — an empty operation closing a transaction.
    /// Observed, and named in [`crate::log_write`].
    pub const XLOG_COMMIT_TRANS: u8 = 0x02;

    /// The tail of an operation that did not fit in the record it
    /// started in: the leading part is written with this flag and the
    /// remainder becomes operation 0 of the next record, flagged
    /// [`XLOG_OP_CONTINUATION`].
    ///
    /// Observed: a 3328-byte data operation split as 2156 + 1172, the
    /// two halves concatenating back to `runlen * 128`. This was the one
    /// apparent invariant violation in the buffer-item audit, and it is
    /// not one — it is this.
    ///
    /// The name is descriptive, taken from the behaviour rather than
    /// from any authority.
    pub const XLOG_OP_TRUNCATED: u8 = 0x04;

    /// The remainder of a truncated operation, carried into the next
    /// record as its operation 0.
    ///
    /// Observed as this exact value, which is **two bits**: `0x08` and
    /// `0x10`. Neither is separately named here, because neither was
    /// ever seen alone and guessing which carries which meaning would be
    /// invention. Test with equality against this value, or mask both
    /// bits together, rather than assuming either bit stands alone.
    pub const XLOG_OP_CONTINUATION: u8 = 0x18;

    /// `XLOG_UNMOUNT_TRANS` — the operation flag on the record a clean
    /// unmount writes as its last act. [`crate::log`] keys the entire
    /// clean/dirty decision off this, together with `h_num_logops == 1`:
    /// a record holding more operations is ordinary work whatever its
    /// first operation is flagged as.
    pub const XLOG_UNMOUNT_TRANS: u8 = 0x20;

    /// Bits `0x40` and `0x80` were never set in any observed operation.
    pub const XLOG_OP_FLAGS_UNOBSERVED: u8 = 0xC0;
}

// ---------------------------------------------------------------------
// xfs_trans_header
// ---------------------------------------------------------------------

/// `xfs_trans_header` — the 16-byte operation that opens a transaction's
/// items, sitting between the START operation and the first item.
///
/// **Byte order: native** (observed little-endian). Note what that does
/// to the magic: the `u32` `0x5452414e` written in little-endian order
/// puts the bytes `4e 41 52 54` on disk, so a hex dump of the log reads
/// `NART`, not `TRAN`. Reading the field as a little-endian `u32` and
/// comparing against the constant is the way to test it; matching four
/// ASCII bytes in order is not.
///
/// # The framing it opens
///
/// ```text
/// op  flags 0x01, len 0                      START
/// op  len 16    "TRAN" magic, type, tid, n   n item operations follow
/// op  ...                                    the items
/// op  flags 0x02, len 0                      COMMIT
/// ```
///
/// An inode-core change comes out as five operations: START, the
/// transaction header, a 56-byte `xfs_inode_log_format`, a 176-byte
/// core, COMMIT.
///
/// Delayed logging batches many transactions into one checkpoint, so
/// that shape is only visible when the operation is alone in its
/// checkpoint. A setup phase of create, chown, chmod and touch arrived
/// as a *single 14-operation record* carrying each inode's final state
/// once. Isolating one operation means: setup, `sync`, `umount`,
/// remount, the one operation, `sync`, `umount`.
///
/// Marked `#[allow(dead_code)]`: a reference module, held to be read.
#[allow(dead_code)]
pub mod trans_header {
    /// `XFS_TRANS_HEADER_MAGIC` — the `u32` value "TRAN", stored in
    /// native byte order like the rest of an operation's payload.
    pub const XFS_TRANS_HEADER_MAGIC: u32 = 0x5452_414e;

    /// `sizeof(xfs_trans_header)`, and the `oh_len` of the operation
    /// carrying it: four `u32` fields, no padding.
    pub const TRANS_HEADER_SIZE: usize = 16;

    /// Byte offsets within `xfs_trans_header`.
    pub mod offsets {
        /// `th_magic`, `u32` — [`super::XFS_TRANS_HEADER_MAGIC`].
        pub const MAGIC: usize = 0;
        /// `th_type`, `u32` — which kind of transaction this is. The
        /// value space was not enumerated; it is named by position.
        pub const TYPE: usize = 4;
        /// `th_tid`, `u32` — the same transaction id every operation
        /// header in the transaction carries in `oh_tid`.
        pub const TID: usize = 8;
        /// `th_num_items`, `u32` — and the one place the sources behind
        /// this module disagree.
        ///
        /// The measurement says it counts **item operations**, not
        /// items: a record whose items occupied 3 + 2 + 2 + 2 + 2
        /// operations — five items — carried `11`, verified directly
        /// against the framing of kernel-written transactions.
        ///
        /// [`crate::log_write::trans_header`] documents the opposite,
        /// that it "counts log items, not operations". Only one can be
        /// right, and the observation is the side with evidence under
        /// it.
        ///
        /// The two readings coincide for a single-item transaction only
        /// when that item occupies one operation, which an inode item
        /// does not — it is `ilf_size == 2`, a format operation plus the
        /// core. So a lone inode-core transaction should carry `2` here
        /// rather than `1`, and that is the cheapest experiment that
        /// would settle it. Nothing in this module settles it.
        pub const NUM_ITEMS: usize = 12;
    }
}

// ---------------------------------------------------------------------
// log item types
// ---------------------------------------------------------------------

/// The `xfs_log_item` type codes seen in the first `u16` of an item's
/// format operation.
///
/// Only these two were mapped. Both were observed in quantity — the
/// buffer item across 434 parsed items, the inode item across 527 logged
/// cores — and both are the first field of their format structure, so
/// the type is what tells a replayer which of the layouts below to use.
///
/// Marked `#[allow(dead_code)]`: a reference module, held to be read.
#[allow(dead_code)]
pub mod item_types {
    /// `XFS_LI_INODE` — an inode item. Its format structure is
    /// [`super::inode_log_format`].
    pub const XFS_LI_INODE: u16 = 0x123b;

    /// `XFS_LI_BUF` — a buffer item: some bytes of some block changed.
    /// Its format structure is [`super::buf_log_format`]. Allocation,
    /// create, unlink and rename all rest on this one.
    pub const XFS_LI_BUF: u16 = 0x123c;
}

// ---------------------------------------------------------------------
// xfs_inode_log_format
// ---------------------------------------------------------------------

/// `xfs_inode_log_format` — the first operation of an inode item, saying
/// which inode is being logged and which parts of it follow.
///
/// **Byte order: native** (observed little-endian).
///
/// The structure is 56 bytes, of which this crate establishes four
/// fields. The spans at 8..16 and 24..56 are written as zero by the
/// encoder and were not pinned by measurement; they are deliberately
/// left unnamed here rather than named from memory, because a name
/// invented at this point would be indistinguishable from a name that
/// had been earned.
///
/// Marked `#[allow(dead_code)]`: a reference module, held to be read.
#[allow(dead_code)]
pub mod inode_log_format {
    /// `sizeof(xfs_inode_log_format)`, and the `oh_len` of the operation
    /// carrying it.
    pub const INODE_LOG_FORMAT_SIZE: usize = 56;

    /// Byte offsets within `xfs_inode_log_format`.
    pub mod offsets {
        /// `ilf_type`, `u16` — [`super::super::item_types::XFS_LI_INODE`].
        pub const TYPE: usize = 0;
        /// `ilf_size`, `u16` — how many log operations this item
        /// occupies, this format operation included. An inode item
        /// logging its core is `2`: the format, then the core.
        pub const SIZE: usize = 2;
        /// `ilf_fields`, `u32` — a bitmask of which parts of the inode
        /// follow. See [`super::XFS_ILOG_CORE`] and
        /// [`super::XFS_ILOG_DDATA`].
        pub const FIELDS: usize = 4;
        /// `ilf_asize`, `u16` — bytes of attribute fork logged.
        pub const ASIZE: usize = 8;
        /// `ilf_dsize`, `u16` — bytes of data fork logged, unpadded.
        pub const DSIZE: usize = 10;
        /// `ilf_ino`, `u64` — the inode number, which matches `di_ino`
        /// at offset 152 of the core operation that follows it. That
        /// agreement is how the core's own `di_ino` was pinned.
        pub const INO: usize = 16;
        /// `ilf_blkno`, `i64` — the **inode cluster buffer's** address in
        /// 512-byte basic blocks, not the inode's own.
        pub const BLKNO: usize = 40;
        /// `ilf_len`, `i32` — that buffer's length, in basic blocks.
        pub const LEN: usize = 48;
        /// `ilf_boffset`, `i32` — the inode's byte offset inside it.
        pub const BOFFSET: usize = 52;
    }

    /// # Addressing the inode, which is the part that is not obvious
    ///
    /// A logged inode never names its own disk address. It names the
    /// cluster buffer that holds it, and its offset within that buffer:
    /// [`offsets::BLKNO`], [`offsets::LEN`] and [`offsets::BOFFSET`].
    /// The cluster's size comes from the geometry, not from the record —
    /// see `Superblock::inode_cluster_bytes`.
    ///
    /// Leaving the three at zero costs more than it looks like it should.
    /// The record still checksums, the kernel still finds it, trusts it
    /// and begins recovery — and then fails reading block 0 for 0 bytes,
    /// refusing the mount with an I/O error that names no inode and no
    /// record. It is worth recognising that error for what it is.
    ///
    /// Measured over 7,260 inode items the kernel wrote, across four
    /// allocation groups and four geometries:
    ///
    /// - `ilf_blkno * 512` is the inode's device offset truncated to a
    ///   whole cluster — an absolute alignment, not one relative to the
    ///   allocation group;
    /// - `ilf_len * 512` is the cluster size, the same for every inode
    ///   on a filesystem;
    /// - `ilf_boffset` is exactly the difference between the two, and
    ///   always less than the cluster.
    pub mod addressing {}

    /// `XFS_ILOG_CORE` in `ilf_fields` — the item logs the inode core,
    /// which follows as the next operation.
    pub const XFS_ILOG_CORE: u32 = 0x01;

    /// `XFS_ILOG_DDATA` in `ilf_fields` — the item logs the data fork's
    /// **inline** contents, which follow as a further operation after
    /// the core.
    ///
    /// This is what a short-form directory's entries are logged as.
    /// Isolated by renaming one file inside a two-entry directory and
    /// reading the record: `ilf_fields` was `0x0003`, the item occupied
    /// three operations rather than two, and the third held the
    /// directory's entries.
    pub const XFS_ILOG_DDATA: u32 = 0x02;

    /// `XFS_ILOG_DEXT` in `ilf_fields` — the item logs the data fork's
    /// **extent list**, which follows as a further operation after the
    /// core.
    ///
    /// This is what a file that has just gained or lost blocks logs, and
    /// it is the commonest fork flag by some margin: 1,518 of the inode
    /// items in one stress fixture carried `0x0005`, this bit together
    /// with [`XFS_ILOG_CORE`].
    ///
    /// Isolated the same way [`XFS_ILOG_DDATA`] was — by what the third
    /// operation turned out to hold. Every one of those items occupied
    /// three operations and reported a `ilf_dsize` that is a multiple of
    /// sixteen: 16, 32, 48, 64 and up, which is one, two, three and four
    /// extent records. An inline fork has no such rule, and a
    /// short-form directory's 30 bytes is the proof of it.
    pub const XFS_ILOG_DEXT: u32 = 0x04;

    /// `XFS_ILOG_ADATA` in `ilf_fields` — the item logs the **attribute**
    /// fork's inline contents, after the core, sized by [`offsets::ASIZE`].
    /// The mirror of [`XFS_ILOG_DDATA`] on the other fork.
    pub const XFS_ILOG_ADATA: u32 = 0x08;

    /// `XFS_ILOG_AEXT` in `ilf_fields` — the attribute fork's extent list,
    /// the mirror of [`XFS_ILOG_DEXT`].
    pub const XFS_ILOG_AEXT: u32 = 0x10;

    /// # What the sizes mean, and how the fork operation is framed
    ///
    /// Measured on a rename inside a short-form directory — 8 operations,
    /// 2 items, the smallest shape that logs a fork at all.
    ///
    /// - [`offsets::SIZE`] counts this item's **operations**, the format
    ///   itself included. Two for a bare core; three when a fork follows.
    /// - [`offsets::DSIZE`] is the data fork's length in bytes, and it is
    ///   the *unpadded* length: a 30-byte short-form directory reads 30
    ///   here while its operation is 32 bytes long. Operations are padded
    ///   to a multiple of four, and the padding is not part of the fork.
    /// - [`offsets::ASIZE`] is the same for the attribute fork.
    ///
    /// # The fork stays big-endian
    ///
    /// The core beside it is native-endian; the fork is not. In the
    /// measured record the short-form header read `02 00 00 00 00 80` —
    /// a count of 2 and a parent inode of 128 stored big-endian —
    /// surrounded by a little-endian core. A writer converts the core
    /// and copies the fork.
    ///
    /// # A rename appends, it does not replace
    ///
    /// Worth knowing before writing one. Each short-form entry carries an
    /// offset, and those increase through the list. Renaming `aaaa` to
    /// `cccc` in a directory holding `aaaa` at `0x60` and `bbbb` at
    /// `0x70` left `bbbb` at `0x70` and `cccc` at `0x80` — the old entry
    /// removed and a new one appended with a fresh higher offset, even
    /// though the replacement name was the same length and would have fit
    /// exactly where the old one was.
    pub mod sizes {}
}

// ---------------------------------------------------------------------
// the logged inode core
// ---------------------------------------------------------------------

/// The inode core as the log stores it — `xfs_log_dinode`.
///
/// **Byte order: native** (observed little-endian), against a disk copy
/// that is big-endian throughout — but **the core only**. A fork logged
/// alongside it, a shortform directory or an extent list, stays
/// big-endian inside a native-endian record. That is not what anyone
/// would guess from "the log is native-endian", and it was observed
/// directly: a shortform directory's parent inode read `00 00 00 80` for
/// 128 in a record whose surrounding core was little-endian. So the
/// swap ends at [`log_dinode::LOG_DINODE_SIZE`], not at the end of the operation.
///
/// # The shape, in one sentence
///
/// The 176-byte log core is the v3 on-disk dinode, field for field at
/// identical offsets, differing only in endianness and in `di_crc` being
/// zeroed. That was confirmed byte for byte by dumping an on-disk inode
/// alongside its log copy, which is why the encoder is a byte-swap of a
/// structure this driver already parses rather than a second layout to
/// maintain.
///
/// Established by 24 controlled A/B pairs — fixed UUID, one variable
/// changed — plus a byte census over 527 logged cores.
///
/// # Three things that bite a writer
///
/// **A v4 filesystem logs 96 bytes, not 176.** `di_version` reads 2 and
/// the structure simply stops at [`log_dinode::V2_LOG_DINODE_SIZE`]. Assuming 176
/// overruns into the next operation.
///
/// **`nrext64` moves two counters.** The data-extent count goes to
/// offset 24 as a `u64` and the attribute count to 76 as a `u32`,
/// leaving 80 as padding. Gate on [`log_dinode::flags2::DI_FLAGS2_NREXT64`] read from the
/// inode's own `di_flags2`, not from the superblock — it is the inode's
/// encoding that matters.
///
/// **`di_crc` is zero in the log and real on disk.** A replayer must
/// recompute it; an encoder must not copy the disk value across.
///
/// # Fields that are blank, lag, or never moved
///
/// - `di_crc` (100): always zero in the log, a real checksum on disk.
/// - `di_lsn` (112): the `cycle << 32 | basic block` of that inode's
///   *previous* log record, so it lags this record by one.
/// - `di_next_unlinked` (96): constant `0xffffffff`, and not for want of
///   trying — 400 files opened, unlinked while open and synced produced
///   797 cores with `nlink == 0`, every one holding the sentinel. The
///   kernel appears to maintain the unlinked list through the inode
///   *buffer* item instead. The field's position is certain; its
///   behaviour here is not.
/// - `di_flushiter` (30, v2 only): only ever zero.
/// - Offsets 6..8, 84..90, 100..104 and 132..144 are proven constant
///   zero, which is all a writer needs. Their names are assigned by
///   position against the on-disk inode rather than by watching them
///   vary.
///
/// Marked `#[allow(dead_code)]`: a reference module, held to be read.
#[allow(dead_code)]
pub mod log_dinode {
    /// Size of the core a version-3 inode logs.
    pub const LOG_DINODE_SIZE: usize = 176;

    /// Size of the core a version-1 or -2 inode logs. The structure
    /// stops here; there is no padding out to 176.
    pub const V2_LOG_DINODE_SIZE: usize = 96;

    /// `di_magic` — `0x494e`, "IN". Constant across all 527 cores; the
    /// disk copy holds the same value in the other byte order, which is
    /// itself a check on the endianness claim.
    pub const XFS_DINODE_MAGIC: u16 = 0x494e;

    /// `di_version` on a v5 filesystem: a 176-byte core.
    pub const DI_VERSION_3: u8 = 3;
    /// `di_version` on a v4 filesystem: a 96-byte core.
    pub const DI_VERSION_2: u8 = 2;

    /// The value `di_next_unlinked` always held, `nlink == 0` included.
    pub const DI_NEXT_UNLINKED_NULL: u32 = 0xffff_ffff;

    /// Byte offsets within the logged core, which are also the offsets
    /// within the v3 on-disk dinode.
    ///
    /// Widths are given in each field's documentation because they are
    /// what a byte-swapping encoder consumes; the canonical swap table
    /// lives in [`crate::log_write`] and is deliberately not duplicated
    /// here.
    pub mod offsets {
        /// `di_magic`, `u16` — [`super::XFS_DINODE_MAGIC`].
        pub const MAGIC: usize = 0;
        /// `di_mode`, `u16`. Pinned by chmod 0644 against 0600.
        pub const MODE: usize = 2;
        /// `di_version`, `u8` — decides the core's length.
        pub const VERSION: usize = 4;
        /// `di_format`, `u8`. Pinned by a 1-entry against a 400-entry
        /// directory, and a short against a long symlink.
        pub const FORMAT: usize = 5;
        /// Unused, `u16` — constant zero, v4 included. Named by
        /// position.
        pub const UNUSED: usize = 6;
        /// `di_uid`, `u32`. Pinned by chown 4321 against 8765.
        pub const UID: usize = 8;
        /// `di_gid`, `u32`. Pinned by chown :4321 against :8765.
        pub const GID: usize = 12;
        /// `di_nlink`, `u32`. Pinned by 1 against 2 hard links.
        pub const NLINK: usize = 16;
        /// `di_projid_lo`, `u16`. Pinned by chproj 100000 vs 200000.
        pub const PROJID_LO: usize = 20;
        /// `di_projid_hi`, `u16` — the high halves of that same pair.
        pub const PROJID_HI: usize = 22;
        /// 8 bytes of padding — **or `di_big_nextents`, `u64`, when
        /// [`crate::format::log_items::log_dinode::flags2::DI_FLAGS2_NREXT64`] is set**. Zero in all 527
        /// cores, none of which had the feature on.
        pub const BIG_NEXTENTS: usize = 24;
        /// `di_flushiter`, `u16`, on a v2 inode: the last two bytes of
        /// the padding at 24. Only ever zero.
        pub const FLUSHITER: usize = 30;
        /// `di_atime`, 8 bytes. Two controlled `touch -a` values decoded
        /// exact to the nanosecond.
        pub const ATIME: usize = 32;
        /// `di_mtime`, 8 bytes — likewise, sub-second parts included.
        pub const MTIME: usize = 40;
        /// `di_ctime`, 8 bytes. Moves in *every* experiment: any
        /// metadata change bumps it, which makes it useless as a control
        /// and unmistakable as a field.
        pub const CTIME: usize = 48;
        /// `di_size`, `u64`. Pinned by truncate 100000 vs 200000.
        pub const SIZE: usize = 56;
        /// `di_nblocks`, `u64`. Pinned by 1 against 4 blocks written.
        pub const NBLOCKS: usize = 64;
        /// `di_extsize`, `u32`, in filesystem blocks. Pinned by extsize
        /// 65536 against 131072.
        pub const EXTSIZE: usize = 72;
        /// `di_nextents`, `u32` — **or `di_big_anextents` under
        /// `nrext64`**. Pinned by 3 against 1 extents.
        pub const NEXTENTS: usize = 76;
        /// `di_anextents`, `u16` — becomes padding under `nrext64`.
        /// Pinned by shortform against 7 attribute extents.
        pub const ANEXTENTS: usize = 80;
        /// `di_forkoff`, `u8`, in units of 8 bytes. Pinned by no xattr,
        /// one xattr, and eighty.
        pub const FORKOFF: usize = 82;
        /// `di_aformat`, `u8`. Pinned by none / shortform / extents.
        pub const AFORMAT: usize = 83;
        /// `di_dmevmask`, `u32` — constant zero; named by position.
        pub const DMEVMASK: usize = 84;
        /// `di_dmstate`, `u16` — constant zero; named by position.
        pub const DMSTATE: usize = 88;
        /// `di_flags`, `u16` — see the `di_flags` constants.
        pub const FLAGS: usize = 90;
        /// `di_gen`, `u32` — differs per inode, stable across records
        /// for one inode.
        pub const GEN: usize = 92;
        // --- a v2 core ends here, at 96 ---
        /// `di_next_unlinked`, `u32` — always
        /// [`super::DI_NEXT_UNLINKED_NULL`] in the log.
        pub const NEXT_UNLINKED: usize = 96;
        /// `di_crc`, `u32` — **always zero in the log**, while the disk
        /// copy holds a real checksum.
        pub const CRC: usize = 100;
        /// `di_changecount`, `u64`. Pinned by 1 chmod against 3, and by
        /// 3 against 402 after 400 creations.
        pub const CHANGECOUNT: usize = 104;
        /// `di_lsn`, `u64` — `cycle << 32 | basic block` of this
        /// inode's *previous* log record.
        pub const LSN: usize = 112;
        /// `di_flags2`, `u64` — see the `di_flags2` constants. Two of
        /// its bits change this very structure's layout.
        pub const FLAGS2: usize = 120;
        /// `di_cowextsize`, `u32`. Pinned by cowextsize 65536 against
        /// 131072.
        pub const COWEXTSIZE: usize = 128;
        /// `di_pad2`, 12 bytes — constant zero.
        pub const PAD2: usize = 132;
        /// `di_crtime`, 8 bytes. Equals the parent directory's mtime at
        /// the moment the child was created, which is how it was told
        /// apart from the other timestamps.
        pub const CRTIME: usize = 144;
        /// `di_ino`, `u64` — matches `ilf_ino` in the format operation
        /// ahead of it.
        pub const INO: usize = 152;
        /// `di_uuid`, 16 bytes, in RFC byte order — carried across
        /// without swapping. Two mkfs UUIDs came back exactly.
        pub const UUID: usize = 160;
    }

    /// `di_flags2` bits, all three pinned by building a filesystem with
    /// the feature on and one with it off.
    pub mod flags2 {
        /// `XFS_DIFLAG2_COWEXTSIZE` — `di_cowextsize` at 128 is live.
        pub const DI_FLAGS2_COWEXTSIZE: u64 = 0x04;
        /// `XFS_DIFLAG2_BIGTIME` — timestamps are the 64-bit encoding.
        /// The current mkfs default.
        pub const DI_FLAGS2_BIGTIME: u64 = 0x08;
        /// `XFS_DIFLAG2_NREXT64` — the extent counters move: data count
        /// to offset 24 as a `u64`, attribute count to 76 as a `u32`,
        /// offset 80 becoming padding.
        pub const DI_FLAGS2_NREXT64: u64 = 0x10;
    }

    /// `di_flags` bits, each named for the attribute observed to set it.
    /// These three were pinned; the rest of the mask was not exercised.
    pub mod flags {
        /// Set by `chattr +S` (synchronous updates).
        pub const DI_FLAG_SYNC: u16 = 0x0020;
        /// Set by `chattr +d` (no dump).
        pub const DI_FLAG_NODUMP: u16 = 0x0080;
        /// Set by giving the inode an extent size hint.
        pub const DI_FLAG_EXTSIZE: u16 = 0x0800;
    }

    /// The two timestamp encodings, both proven.
    ///
    /// **bigtime** ([`flags2::DI_FLAGS2_BIGTIME`], the current default):
    /// one `u64` of nanoseconds since 1901-12-13 20:45:52 UTC, i.e.
    /// unix minus 2³¹ seconds. Six controlled values decoded exactly,
    /// including root's `mkfs` atime of
    /// [`timestamps::BIGTIME_UNIX_EPOCH`] — which is unix zero, and
    /// whose arithmetic closes: `0x1dcd6500 << 32` is
    /// 2 147 483 648 × 10⁹.
    ///
    /// **legacy** (`mkfs -m bigtime=0`): an `i32` of seconds followed by
    /// a `u32` of nanoseconds. The halves swap separately, which is the
    /// practical difference for an encoder.
    pub mod timestamps {
        /// Unix seconds at the bigtime epoch: `-2^31`.
        pub const BIGTIME_EPOCH_UNIX_SECS: i64 = -2_147_483_648;
        /// The bigtime value for unix zero, observed as root's `mkfs`
        /// atime. A useful self-check on any decoder.
        pub const BIGTIME_UNIX_EPOCH: u64 = 0x1dcd_6500_0000_0000;
        /// Width of one bigtime timestamp.
        pub const BIGTIME_SIZE: usize = 8;
        /// Width of one legacy timestamp: `i32` seconds, `u32` nanoseconds.
        pub const LEGACY_TIMESTAMP_SIZE: usize = 8;
        /// Offset of the nanosecond half within a legacy timestamp.
        pub const LEGACY_NSEC_OFFSET: usize = 4;
    }

    pub use flags2::{DI_FLAGS2_BIGTIME, DI_FLAGS2_COWEXTSIZE, DI_FLAGS2_NREXT64};
}

// ---------------------------------------------------------------------
// xfs_buf_log_format
// ---------------------------------------------------------------------

/// `xfs_buf_log_format` — the buffer item, which is how XFS logs a
/// change to an allocation-group header, a B+tree block or a directory
/// block: by recording *which bytes of which buffer changed*.
///
/// **Byte order: native** (observed little-endian across the whole
/// corpus, on arm64 only).
///
/// Established by differential analysis over 12 filesystems across 4
/// geometries, 434 buffer items parsed. Every structural invariant below
/// held on all 434.
///
/// # Layout
///
/// ```text
/// off  w    field
///   0  u16  blf_type = XFS_LI_BUF
///   2  u16  blf_size      log operations this item occupies
///   4  u16  blf_flags     low 11 bits flags, high 5 a buffer type
///   6  u16  blf_len       buffer length, in 512-byte basic blocks
///   8  i64  blf_blkno     absolute device address, in basic blocks
///  16  u32  blf_map_size  dirty bitmap length, in 32-bit words
///  20  u32× blf_data_map  one bit per 128-byte chunk of the buffer
/// ```
///
/// Total size is `BLF_HEADER_SIZE + 4 * map_size`, with no padding and
/// no alignment: the next operation header begins immediately.
///
/// # The data operations
///
/// Exactly `blf_size - 1` of them, immediately after the format
/// operation: **one per maximal run of consecutive set bits**, in
/// ascending chunk order. Data operation *k* carries buffer bytes
/// `[start_k * 128, (start_k + len_k) * 128)` and its length is exactly
/// `len_k * 128`. Chunk *c* is buffer offset `c * 128`; bit *c* lives in
/// word `c / 32` at bit `c % 32`, LSB-first within each word.
///
/// Two invariants held on all 434 items and are worth asserting in a
/// parser:
///
/// ```text
/// op_len              == 20 + 4 * map_size
/// popcount(map) * 128 == sum of the data-operation lengths
/// ```
///
/// # What a replayer must do beyond copying bytes
///
/// **Recompute the checksum.** For the last item logging a block, the
/// logged chunks equal the on-disk bytes *except* the block's own CRC
/// and LSN, which are stamped at write-out, after logging. Observed on
/// btree blocks, the superblock, directory blocks and the AGI.
///
/// **Do not replay an inode chunk verbatim.** See the module
/// documentation: an unlink dirties a 128-byte chunk that carries a
/// stale inode image, and applying it as-is corrupts the inode. Only the
/// four bytes at `di_next_unlinked` should be taken from it. That the
/// mismatch exists is observed; that only the unlinked pointer should be
/// applied is the inference it forces.
///
/// # Left open
///
/// Recorded so they are not silently re-opened:
///
/// - **Discontiguous buffers with more than one map.** Never seen.
///   Producing one means fragmenting free space hard, then creating a
///   directory with `-n size=8192` whose blocks land non-contiguously.
/// - **Whether `blf_blkno` is signed.** Always positive here; the width
///   and position are certain, the signedness is not.
/// - **The AGFL never appeared as a buffer item**, in any workload,
///   including heavy allocate-and-free churn — the AGF alone carried the
///   freelist head, tail and count. A negative result, but a firm one.
///
/// Marked `#[allow(dead_code)]`: a reference module, held to be read.
#[allow(dead_code)]
pub mod buf_log_format {
    /// Bytes before `blf_data_map` — the fixed part of the structure.
    pub const BLF_HEADER_SIZE: usize = 20;

    /// The buffer region one bitmap bit covers. Everything about the
    /// data operations follows from this number.
    ///
    /// It was not guessed: appending one name at a time to a
    /// single-block directory produced bitmaps whose runs marched
    /// forward for the name area and grew backwards from chunk 31 as the
    /// leaf-entry array grew down from the end of the block — exactly
    /// what a 128-byte-chunk bitmap predicts, and predicted before it was
    /// measured.
    pub const BLF_CHUNK: usize = 128;

    /// The unit of `blf_blkno` and `blf_len`: 512-byte basic blocks.
    ///
    /// Not what one would guess, so the two plausible alternatives were
    /// varied independently:
    ///
    /// - At `bs=4096 sect=512`: AGI at blkno 2, bnobt 8, cntbt 16,
    ///   inobt 24, finobt 32, SB 0.
    /// - At `bs=1024`: the AGI is **still** blkno 2 — so not filesystem
    ///   blocks.
    /// - At `sect=4096`: the AGF is blkno **8** and the AGI **16** — so
    ///   not sectors either.
    ///
    /// And the address is absolute rather than AG-relative: headers in
    /// AG1, AG2 and AG3 came out at `agno * agblocks * bs / 512 + {1,2}`.
    ///
    /// Each was cross-checked by reading the block at that address and
    /// confirming its magic — `XAGI`, `AB3B`, `IAB3`, `XFSB` and so on.
    /// That is the strongest evidence available, and it agreed every
    /// time.
    pub const BLF_BLKNO_UNIT: usize = super::BBSIZE;

    /// Byte offsets within `xfs_buf_log_format`.
    pub mod offsets {
        /// `blf_type`, `u16` — [`super::super::item_types::XFS_LI_BUF`].
        pub const TYPE: usize = 0;
        /// `blf_size`, `u16` — log operations this item occupies: this
        /// format operation plus one per bitmap run.
        pub const SIZE: usize = 2;
        /// `blf_flags`, `u16` — low 11 bits are flags, bits 11..16 a
        /// buffer-type code.
        pub const FLAGS: usize = 4;
        /// `blf_len`, `u16` — buffer length in 512-byte basic blocks.
        pub const LEN: usize = 6;
        /// `blf_blkno`, `i64` — **absolute device address, in 512-byte
        /// basic blocks**. See [`super::BLF_BLKNO_UNIT`] for how that
        /// was established; signedness is the one thing about the field
        /// that remains unproven.
        pub const BLKNO: usize = 8;
        /// `blf_map_size`, `u32` — bitmap length in 32-bit words.
        pub const MAP_SIZE: usize = 16;
        /// `blf_data_map`, `u32` × `blf_map_size` — one bit per
        /// [`super::BLF_CHUNK`]-byte chunk of the buffer.
        pub const DATA_MAP: usize = 20;
    }

    /// Why `blf_blkno` is basic blocks and not something more obvious.
    ///
    /// The two plausible alternatives were varied independently:
    ///
    /// - At `bs=4096 sect=512`: AGI at blkno 2, bnobt 8, cntbt 16,
    ///   inobt 24, finobt 32, SB 0.
    /// - At `bs=1024`: the AGI is **still** blkno 2 — so not filesystem
    ///   blocks.
    /// - At `sect=4096`: the AGF is blkno **8** and the AGI **16** — so
    ///   not sectors either.
    ///
    /// And it is absolute rather than AG-relative: headers in AG1, AG2
    /// and AG3 came out at `agno * agblocks * bs / 512 + {1,2}`.
    ///
    /// Each was cross-checked by reading the block at that address and
    /// confirming its magic — `XAGI`, `AB3B`, `IAB3`, `XFSB` and so on.
    /// That is the strongest evidence available, and it agreed every
    /// time.
    pub const BLKNO_IS_BASIC_BLOCKS: bool = true;

    /// Low `blf_flags` bits. Only `0x000`, `0x001` and `0x002` were seen.
    pub mod flags {
        /// Set only on inode-cluster buffers, and on the cancel case
        /// below.
        pub const BLF_INODE_BUF: u16 = 0x001;

        /// The cancel record: always `blf_size == 1`, an empty map and
        /// no data operations, at the address of a block being freed or
        /// reused. Replay must let it suppress earlier items for the
        /// same block.
        pub const BLF_CANCEL: u16 = 0x002;

        /// Never observed. Presumed to be the three dquot flavours —
        /// mounting with all three quota types produced no dquot buffer
        /// item at all, so which bit is which flavour is not merely
        /// unnamed but unknown.
        pub const BLF_PRESUMED_DQUOT_A: u16 = 0x004;
        /// Never observed. See [`BLF_PRESUMED_DQUOT_A`].
        pub const BLF_PRESUMED_DQUOT_B: u16 = 0x008;
        /// Never observed. See [`BLF_PRESUMED_DQUOT_A`].
        pub const BLF_PRESUMED_DQUOT_C: u16 = 0x010;

        /// Mask of the flag bits, below the buffer-type field.
        pub const BLF_FLAG_MASK: u16 = 0x07ff;
    }

    /// How far to shift `blf_flags` to get the buffer-type code.
    pub const BLF_TYPE_SHIFT: u32 = 11;

    /// Buffer-type codes, taken from `blf_flags >> BLF_TYPE_SHIFT`.
    ///
    /// Every code below correlated 1:1 with the magic actually found at
    /// `blf_blkno` across all twelve images — that is, the type was not
    /// inferred from what XFS ought to be logging but confirmed against
    /// the block itself.
    pub mod buf_type {
        /// Any btree block: `AB3B`, `AB3C`, `IAB3`, `FIB3` and `BMA3`
        /// all share this one code.
        pub const BLFT_BTREE: u16 = 4;
        /// `XAGF`.
        pub const BLFT_AGF: u16 = 5;
        /// `XAGI`.
        pub const BLFT_AGI: u16 = 7;
        /// `IN` — an inode cluster. Always seen together with
        /// [`super::flags::BLF_INODE_BUF`].
        pub const BLFT_DINO: u16 = 8;
        /// `XSLM` — a remote symlink block.
        pub const BLFT_SYMLINK: u16 = 9;
        /// `XDB3` — a single-block directory.
        pub const BLFT_DIR_BLOCK: u16 = 10;
        /// `XDD3` — a directory data block.
        pub const BLFT_DIR_DATA: u16 = 11;
        /// `XDF3` — a directory free block.
        pub const BLFT_DIR_FREE: u16 = 12;
        /// A leaf1 directory block.
        pub const BLFT_DIR_LEAF1: u16 = 13;
        /// A leafN directory block.
        pub const BLFT_DIR_LEAFN: u16 = 14;
        /// A directory/attribute btree node block.
        pub const BLFT_DA_NODE: u16 = 15;
        /// An attribute leaf block.
        pub const BLFT_ATTR_LEAF: u16 = 16;
        /// `XFSB` — the superblock.
        pub const BLFT_SB: u16 = 18;
        /// Seen only in company with [`super::flags::BLF_CANCEL`], where
        /// there is no live block to describe.
        pub const BLFT_NONE: u16 = 0;

        /// Codes never exercised by any workload in the corpus: 1, 2, 3,
        /// 6, 17 and everything from 19 up. Realtime bitmap and summary,
        /// the AGFL, dquots, and the rmap and refcount btrees are the
        /// obvious candidates, in some order — but that ordering is a
        /// guess and is not recorded as one of the constants above.
        pub const UNEXERCISED: &[u16] = &[1, 2, 3, 6, 17];

        /// Code to the block magic observed at that address, as a table
        /// for anything that wants to check one against the other.
        pub const MAGICS: &[(u16, &str)] = &[
            (BLFT_NONE, "(cancel only)"),
            (BLFT_BTREE, "AB3B/AB3C/IAB3/FIB3/BMA3"),
            (BLFT_AGF, "XAGF"),
            (BLFT_AGI, "XAGI"),
            (BLFT_DINO, "IN"),
            (BLFT_SYMLINK, "XSLM"),
            (BLFT_DIR_BLOCK, "XDB3"),
            (BLFT_DIR_DATA, "XDD3"),
            (BLFT_DIR_FREE, "XDF3"),
            (BLFT_DIR_LEAF1, "(dir leaf1)"),
            (BLFT_DIR_LEAFN, "(dir leafN)"),
            (BLFT_DA_NODE, "(da node)"),
            (BLFT_ATTR_LEAF, "(attr leaf)"),
            (BLFT_SB, "XFSB"),
        ];
    }
}
