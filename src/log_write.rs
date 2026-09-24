//! Building log records.
//!
//! XFS records a metadata change in the log before applying it, so that
//! a change spanning several structures either happens completely or not
//! at all. Everything this driver cannot yet do — allocation, directory
//! changes, create, unlink, rename — is blocked on being able to write
//! one of these correctly.
//!
//! # What a transaction looks like
//!
//! A record's payload is a sequence of operations, each with a 12-byte
//! header. A transaction opens with an empty operation flagged
//! `XLOG_START_TRANS`, carries a transaction header and then one or more
//! items, and closes with an empty operation flagged
//! `XLOG_COMMIT_TRANS`. An inode-core change is:
//!
//! ```text
//! op  len   what
//!  0    0   START
//!  1   16   transaction header
//!  2   56   xfs_inode_log_format — which inode, which fields
//!  3  176   the inode core, as the log stores it
//!  4    0   COMMIT
//! ```
//!
//! # Two things that are easy to get wrong
//!
//! **The logged inode is native-endian.** The on-disk inode is
//! big-endian throughout; the copy in the log is not. It is the same
//! 176-byte shape with the same field offsets, byte-swapped. Writing the
//! on-disk form into a record produces something that parses and is
//! entirely wrong.
//!
//! **The cycle stamp is applied last.** Each 512-byte block of the
//! payload has its first four bytes replaced by the cycle number, and
//! the displaced word kept in the header's `h_cycle_data`. The checksum
//! is computed over the stamped form, so the order is: build, stamp,
//! then checksum.
//!
//! # Why a mistake here is survivable
//!
//! A record whose checksum does not verify is treated by the kernel as a
//! torn write: it truncates the log head back and discards that record
//! and everything after it. So a wrong encoding does not corrupt a
//! filesystem — the transaction simply never happened. That is what
//! makes this safe to develop incrementally, and it is why the encoder
//! is checked against records the kernel wrote rather than against its
//! own output.

use crate::error::{Error, Result};
use crate::fs::Filesystem;
use crate::log::{record_checksum, Head, BBSIZE, XLOG_HEADER_MAGIC, XLOG_REC_HEADER_SIZE};
use crate::superblock::Superblock;

/// The log's own layout constants, defined once in
/// [`crate::format::log_items`] alongside the structures this encoder
/// does not write yet. Re-exported rather than restated: a magic number
/// that appears twice is a magic number that can disagree with itself.
pub use crate::format::log_items::{
    inode_log_format::{INODE_LOG_FORMAT_SIZE, XFS_ILOG_CORE},
    item_types::XFS_LI_INODE,
    log_dinode::{LOG_DINODE_SIZE, V2_LOG_DINODE_SIZE},
    op_header::{OP_HEADER_SIZE, XFS_TRANSACTION, XLOG_COMMIT_TRANS, XLOG_START_TRANS},
    trans_header::{TRANS_HEADER_SIZE, XFS_TRANS_HEADER_MAGIC},
};

use crate::format::log_items::log_dinode::flags2::{DI_FLAGS2_BIGTIME, DI_FLAGS2_NREXT64};
use crate::format::log_items::rec_header::XLOG_VERSION_2;

/// One operation: its flags and its payload.
pub struct Op {
    pub flags: u8,
    pub data: Vec<u8>,
}

/// The transaction header that opens every transaction's items.
///
/// `item_ops` counts **operations belonging to items**, not items, and
/// not the START and COMMIT that bracket them. An inode item spans two
/// operations and contributes two.
///
/// This is worth stating because the natural reading of the field's
/// name is wrong, and nothing catches it: a record with the wrong count
/// still encodes, still checksums, and is simply discarded on replay.
/// The arithmetic settles it — a create is 14 operations, START and
/// COMMIT and the header included, and its count reads 11 while the
/// transaction carries only 5 items.
///
/// For a lone inode-core change — format op plus core — the count is 2.
pub fn trans_header(tid: u32, kind: u32, item_ops: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(TRANS_HEADER_SIZE);
    v.extend_from_slice(&XFS_TRANS_HEADER_MAGIC.to_le_bytes());
    v.extend_from_slice(&kind.to_le_bytes());
    v.extend_from_slice(&tid.to_le_bytes());
    v.extend_from_slice(&item_ops.to_le_bytes());
    v
}

/// Where an inode's cluster buffer is, which is how a logged inode is
/// addressed.
///
/// A logged inode does not name its own disk address. It names the
/// cluster that holds it and its offset inside that cluster, and the
/// replayer reads the whole cluster to apply the change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InodeBuffer {
    /// Basic-block address of the cluster.
    pub blkno: u64,
    /// Length of the cluster, in basic blocks.
    pub len: u32,
    /// The inode's byte offset within the cluster.
    pub boffset: u32,
}

impl InodeBuffer {
    /// Locate the cluster holding the inode at device offset `at`.
    ///
    /// The alignment is absolute — the inode's offset on the device
    /// truncated to a whole cluster — rather than relative to the
    /// allocation group, which is the thing worth checking against
    /// intuition. Confirmed across all four allocation groups of a
    /// multi-group filesystem.
    pub fn containing(at: u64, cluster_bytes: u32) -> Self {
        let cluster = u64::from(cluster_bytes);
        let start = at - at % cluster;
        InodeBuffer {
            blkno: start / BBSIZE as u64,
            len: cluster_bytes / BBSIZE as u32,
            boffset: (at - start) as u32,
        }
    }
}

/// `xfs_inode_log_format` — which inode is being logged, which parts of
/// it, and where its cluster buffer lives.
///
/// The buffer is not decoration. With those three fields zero the record
/// checksums, is found, is trusted, and fails in recovery reading block
/// zero for zero bytes — an I/O error that names neither the inode nor
/// the record.
pub fn inode_log_format(ino: u64, fields: u32, buffer: &InodeBuffer) -> Vec<u8> {
    use crate::format::log_items::inode_log_format::offsets as at;

    let mut v = vec![0u8; INODE_LOG_FORMAT_SIZE];
    v[at::TYPE..at::TYPE + 2].copy_from_slice(&XFS_LI_INODE.to_le_bytes());
    // The item spans two operations: this format, then the core.
    v[at::SIZE..at::SIZE + 2].copy_from_slice(&2u16.to_le_bytes());
    v[at::FIELDS..at::FIELDS + 4].copy_from_slice(&fields.to_le_bytes());
    v[at::INO..at::INO + 8].copy_from_slice(&ino.to_le_bytes());
    v[at::BLKNO..at::BLKNO + 8].copy_from_slice(&buffer.blkno.to_le_bytes());
    v[at::LEN..at::LEN + 4].copy_from_slice(&buffer.len.to_le_bytes());
    v[at::BOFFSET..at::BOFFSET + 4].copy_from_slice(&buffer.boffset.to_le_bytes());
    v
}

/// `xfs_inode_log_format` for an item that logs a data fork too.
///
/// `dsize` is the fork's own length, not its operation's — the operation
/// is padded to four bytes and this field is not.
pub fn inode_log_format_with_fork(
    ino: u64,
    fields: u32,
    buffer: &InodeBuffer,
    dsize: u16,
) -> Vec<u8> {
    use crate::format::log_items::inode_log_format::offsets as at;

    let mut v = inode_log_format(ino, fields, buffer);
    // Three operations: this format, the core, then the fork.
    v[at::SIZE..at::SIZE + 2].copy_from_slice(&3u16.to_le_bytes());
    v[at::DSIZE..at::DSIZE + 2].copy_from_slice(&dsize.to_le_bytes());
    v
}

/// Assemble a record's payload from its operations.
///
/// Every operation carries the same transaction id; that is what ties
/// them together across a record boundary.
pub fn payload(tid: u32, ops: &[Op]) -> Vec<u8> {
    let mut out = Vec::new();
    for op in ops {
        out.extend_from_slice(&tid.to_be_bytes());
        out.extend_from_slice(&(op.data.len() as u32).to_be_bytes());
        out.push(XFS_TRANSACTION);
        out.push(op.flags);
        out.extend_from_slice(&[0, 0]); // oh_res2
        out.extend_from_slice(&op.data);
    }
    out
}

/// Where a record is going, and what the log looked like before it.
pub struct Placement {
    /// Basic block within the log at which the record starts.
    pub block: u32,
    /// Cycle this record belongs to.
    pub cycle: u32,
    /// Start block of the previous record, or `u32::MAX` for none.
    pub prev_block: u32,
    /// Oldest record still needed, as a log sequence number.
    pub tail_lsn: u64,
    /// The filesystem's UUID, which every record carries.
    pub uuid: [u8; 16],
    /// The log's iclog size, from an existing record's `h_size`.
    pub iclog_size: u32,
}

/// Encode one complete record: header block, then the stamped payload.
///
/// Returns the bytes to write at `placement.block`, padded to whole
/// basic blocks. The checksum is computed last, over the header struct
/// and the stamped payload, because that is what the kernel verifies.
pub fn encode_record(placement: &Placement, num_logops: u32, payload: &[u8]) -> Vec<u8> {
    let padded = payload.len().div_ceil(BBSIZE) * BBSIZE;
    let mut data = vec![0u8; padded];
    data[..payload.len()].copy_from_slice(payload);

    let mut header = vec![0u8; BBSIZE];
    header[0..4].copy_from_slice(&XLOG_HEADER_MAGIC.to_be_bytes());
    header[4..8].copy_from_slice(&placement.cycle.to_be_bytes());
    header[8..12].copy_from_slice(&XLOG_VERSION_2.to_be_bytes());
    header[12..16].copy_from_slice(&(payload.len() as u32).to_be_bytes());
    let lsn = (u64::from(placement.cycle) << 32) | u64::from(placement.block);
    header[16..24].copy_from_slice(&lsn.to_be_bytes());
    header[24..32].copy_from_slice(&placement.tail_lsn.to_be_bytes());
    header[36..40].copy_from_slice(&placement.prev_block.to_be_bytes());
    header[40..44].copy_from_slice(&num_logops.to_be_bytes());
    // `h_fmt`, and it must be this machine's order rather than a
    // constant: the items below are memory images, and a reader on the
    // other byte order has to be able to tell that it may not replay
    // them. `crate::log::inspect` refuses on exactly this field.
    header[300..304].copy_from_slice(&crate::log::format::native().to_be_bytes());
    header[304..320].copy_from_slice(&placement.uuid);
    header[320..324].copy_from_slice(&placement.iclog_size.to_be_bytes());

    // Stamp each payload block's first word with the cycle, keeping the
    // displaced word in the header. A reader undoes this to recover the
    // payload; the checksum covers the stamped form.
    let blocks = padded / BBSIZE;
    for k in 0..blocks.min(64) {
        let at = k * BBSIZE;
        let original: [u8; 4] = data[at..at + 4].try_into().expect("4 bytes");
        header[44 + k * 4..48 + k * 4].copy_from_slice(&original);
        data[at..at + 4].copy_from_slice(&placement.cycle.to_be_bytes());
    }

    let crc = record_checksum(&header, &data[..payload.len()]);
    header[32..36].copy_from_slice(&crc.to_le_bytes());

    let mut out = header;
    out.extend_from_slice(&data);
    debug_assert!(out.len() >= XLOG_REC_HEADER_SIZE);
    out
}

/// Convert an inode's on-disk bytes into the form the log stores.
///
/// The two are the same structure at the same offsets. They differ in
/// exactly two ways, and both are easy to get wrong in a way that parses
/// cleanly and is entirely incorrect:
///
/// - **Byte order.** The on-disk inode is big-endian throughout. The log
///   copy is **native**-endian — the published format document says so
///   outright: *"All on-disk values are in big-endian format except the
///   journaling log which is in native endian format."* So this is a
///   swap on a little-endian host and a copy on a big-endian one, which
///   is why it goes through `to_native` rather than reversing
///   unconditionally. Only the UUID and the padding are carried across
///   untouched in either case.
/// - **The checksum is blank.** `di_crc` is a real value on disk and
///   always zero in the log. Whatever replays the record recomputes it,
///   so copying the disk value across would be wrong even though nothing
///   would complain.
///
/// A version-2 inode logs a **96-byte** core rather than 176, so the
/// length is taken from `di_version` rather than assumed.
///
/// **Only the core is converted.** A fork logged alongside it — a
/// shortform directory, an extent list — stays big-endian inside a
/// native-endian record, which is not what anyone would guess and is
/// why this function stops at the core rather than taking the whole
/// inode. Observed directly: a shortform directory's parent inode read
/// `00 00 00 80` for 128 in a record whose surrounding core was
/// little-endian.
///
/// Two features move fields, and both are read from `di_flags2` in the
/// inode itself rather than from the superblock, because it is the
/// inode's own encoding that matters:
///
/// - `bigtime` stores each timestamp as one 64-bit count; without it
///   they are a pair of 32-bit halves, which swap differently.
/// - `nrext64` moves the data-extent count to offset 24 as a 64-bit
///   field and the attribute count to 76, leaving 80 as padding.
pub fn log_dinode_from_disk(raw: &[u8]) -> std::result::Result<Vec<u8>, &'static str> {
    if raw.len() < V2_LOG_DINODE_SIZE {
        return Err("inode is shorter than a v2 core");
    }
    let version = raw[offsets::VERSION];
    let size = match version {
        1 | 2 => V2_LOG_DINODE_SIZE,
        3 => LOG_DINODE_SIZE,
        _ => return Err("unrecognised inode version"),
    };
    if raw.len() < size {
        return Err("inode is shorter than its version's core");
    }

    let flags2 = if version >= 3 {
        u64::from_be_bytes(
            raw[offsets::FLAGS2..offsets::FLAGS2 + 8]
                .try_into()
                .unwrap(),
        )
    } else {
        0
    };
    let bigtime = flags2 & DI_FLAGS2_BIGTIME != 0;
    let nrext64 = flags2 & DI_FLAGS2_NREXT64 != 0;

    let mut out = raw[..size].to_vec();
    if cfg!(target_endian = "little") {
        // The disk is big-endian and the log is native, so on a
        // little-endian host every multi-byte field is reversed. On a
        // big-endian host the two agree and the copy stands as it is.
        for &(at, width) in field_layout(version, bigtime, nrext64) {
            if at + width > size {
                continue;
            }
            out[at..at + width].reverse();
        }
    }
    if version >= 3 {
        out[offsets::CRC..offsets::CRC + 4].copy_from_slice(&[0, 0, 0, 0]);
    }
    Ok(out)
}

/// Turn a logged inode core back into the on-disk bytes it came from.
///
/// [`log_dinode_from_disk`] applied twice, not a second table to keep in
/// step: reversing a field's bytes a second time is the identity, so the
/// inverse inherits the forward one's layout exactly — which matters because
/// the two layouts move fields under `bigtime` and `nrext64`.
///
/// Two fields come back wrong on purpose and the caller must fix both:
///
/// - `di_crc` is zeroed by the writer and cannot be recovered from the core
///   anyway, since it covers the whole inode rather than these bytes;
/// - `di_lsn` must not be taken from the logged core at all. Upstream says so
///   outright (`xfs_log_dinode`'s field comment: "should never be used for
///   recovery sequencing, nor should it be recovered into the on-disk inode"),
///   and the replayer stamps the transaction's own sequence number instead.
///
/// That is [`crate::apply`]'s job, which is why this returns bytes rather than
/// writing them.
pub fn log_dinode_to_disk(log: &[u8]) -> std::result::Result<Vec<u8>, &'static str> {
    if log.len() < V2_LOG_DINODE_SIZE {
        return Err("logged inode core is shorter than a v2 core");
    }
    let version = log[offsets::VERSION];
    let size = match version {
        1 | 2 => V2_LOG_DINODE_SIZE,
        3 => LOG_DINODE_SIZE,
        _ => return Err("unrecognised inode version"),
    };
    if log.len() < size {
        return Err("logged inode core is shorter than its version's core");
    }
    let flags2 = if version >= 3 {
        // Read the way the writer left it: `log_dinode_from_disk` reverses
        // this field on a little-endian host, so on one it comes back as
        // little-endian. On a big-endian host the two orders agree, as they
        // did on the way out.
        let raw: [u8; 8] = log[offsets::FLAGS2..offsets::FLAGS2 + 8]
            .try_into()
            .unwrap();
        if cfg!(target_endian = "little") {
            u64::from_le_bytes(raw)
        } else {
            u64::from_be_bytes(raw)
        }
    } else {
        0
    };
    let bigtime = flags2 & DI_FLAGS2_BIGTIME != 0;
    let nrext64 = flags2 & DI_FLAGS2_NREXT64 != 0;

    let mut out = log[..size].to_vec();
    if cfg!(target_endian = "little") {
        for &(at, width) in field_layout(version, bigtime, nrext64) {
            if at + width > size {
                continue;
            }
            out[at..at + width].reverse();
        }
    }
    Ok(out)
}

/// Offsets within the inode core, shared by the on-disk and logged forms.
mod offsets {
    pub const VERSION: usize = 4;
    pub const CRC: usize = 100;
    pub const FLAGS2: usize = 120;
}

/// Every multi-byte field, as `(offset, width)`.
///
/// Padding and the UUID are absent deliberately: they are the only spans
/// carried across without swapping, so leaving them out of the table is
/// what makes them stay put.
fn field_layout(version: u8, bigtime: bool, nrext64: bool) -> &'static [(usize, usize)] {
    // The core up to offset 96, which is all a v2 inode has.
    const COMMON: &[(usize, usize)] = &[
        (0, 2),  // di_magic
        (2, 2),  // di_mode
        (6, 2),  // unused
        (8, 4),  // di_uid
        (12, 4), // di_gid
        (16, 4), // di_nlink
        (20, 2), // di_projid_lo
        (22, 2), // di_projid_hi
        (56, 8), // di_size
        (64, 8), // di_nblocks
        (72, 4), // di_extsize
        (84, 4), // di_dmevmask
        (88, 2), // di_dmstate
        (90, 2), // di_flags
        (92, 4), // di_gen
    ];
    // v3 adds everything past di_next_unlinked.
    const V3_TAIL: &[(usize, usize)] = &[
        (96, 4),  // di_next_unlinked
        (104, 8), // di_changecount
        (112, 8), // di_lsn
        (120, 8), // di_flags2
        (128, 4), // di_cowextsize
        (152, 8), // di_ino
    ];

    // Built once per shape rather than per call. The combinations are
    // few and fixed, so the alternative is allocating on every inode.
    macro_rules! shape {
        ($name:ident, $extra:expr) => {{
            static TABLE: std::sync::OnceLock<Vec<(usize, usize)>> = std::sync::OnceLock::new();
            TABLE
                .get_or_init(|| {
                    let mut v = COMMON.to_vec();
                    v.extend_from_slice($extra);
                    v
                })
                .as_slice()
        }};
    }

    // Timestamps: one 64-bit count under bigtime, two 32-bit halves
    // otherwise. They swap differently, so the shape has to know.
    let times: &[(usize, usize)] = if bigtime {
        &[(32, 8), (40, 8), (48, 8), (144, 8)]
    } else {
        &[
            (32, 4),
            (36, 4),
            (40, 4),
            (44, 4),
            (48, 4),
            (52, 4),
            (144, 4),
            (148, 4),
        ]
    };
    // Extent counts move under nrext64.
    let counts: &[(usize, usize)] = if nrext64 {
        &[(24, 8), (76, 4)]
    } else {
        &[(76, 4), (80, 2)]
    };

    match (version >= 3, bigtime, nrext64) {
        (false, _, _) => {
            // A v2 core stops at 96 and has no bigtime or nrext64.
            shape!(
                v2,
                &[
                    (32, 4),
                    (36, 4),
                    (40, 4),
                    (44, 4),
                    (48, 4),
                    (52, 4),
                    (76, 4),
                    (80, 2)
                ]
            )
        }
        (true, true, false) => shape!(v3_bt, &[V3_TAIL, times, counts].concat()),
        (true, true, true) => shape!(v3_bt_nr, &[V3_TAIL, times, counts].concat()),
        (true, false, false) => shape!(v3_lt, &[V3_TAIL, times, counts].concat()),
        (true, false, true) => shape!(v3_lt_nr, &[V3_TAIL, times, counts].concat()),
    }
}

/// The transaction header's type field.
///
/// It read `0x28` in all 53 transactions measured, across twelve
/// different filesystem operations, so it does not identify the
/// operation. It identifies the checkpoint — see
/// [`crate::log_write`]'s note on what a record actually contains.
pub const XFS_TRANS_CHECKPOINT: u32 = 0x28;

/// Write one checkpoint into the log at `head`.
///
/// Returns the sequence number the record was given, which is what
/// orders it against everything already there.
///
/// # What this does not do
///
/// It does not wrap. A record that will not fit between `head.block`
/// and the end of the ring is refused rather than split, because the
/// two halves would need separate cycle numbers and a reader that gets
/// that wrong replays a record that was never committed. A log with
/// room only past the wrap therefore reports full.
///
/// It also does not move the tail. `h_tail_lsn` is set to the record's
/// own sequence number, which is correct precisely while this is the
/// only outstanding checkpoint — the case a driver that commits one
/// operation at a time is always in, and a claim that stops being true
/// the moment it commits two without an intervening replay.
///
/// # Errors
///
/// [`Error::UnsupportedFeature`] when the record does not fit before the
/// wrap or exceeds the in-core buffer size, and [`Error::Io`] from the
/// device.
pub fn append_at(
    device: &dyn fs_core::BlockDevice,
    sb: &Superblock,
    head: &Head,
    tid: u32,
    ops: &[Op],
) -> Result<u64> {
    let payload = payload(tid, ops);

    // The kernel sizes its in-core buffers from `h_size` and reads a
    // record into one of them, so a payload larger than a buffer less
    // its header block cannot be read back however well it is written.
    let max_payload = head.iclog_size as usize - BBSIZE;
    if payload.len() > max_payload {
        return Err(Error::UnsupportedFeature(format!(
            "the checkpoint is {} bytes and the log's records hold at most {max_payload}; \
             splitting one across records is not implemented",
            payload.len()
        )));
    }

    let record_blocks = 1 + payload.len().div_ceil(BBSIZE);
    if record_blocks > head.free_blocks as usize {
        return Err(Error::UnsupportedFeature(format!(
            "the checkpoint needs {record_blocks} basic blocks and only {} remain before \
             the log wraps; writing across the wrap is not implemented",
            head.free_blocks
        )));
    }

    // `h_len` is the operations rounded up to a whole basic block, not
    // their exact length. The kernel writes it that way — a record of
    // 608 bytes of operations records 1024 — and the trailing zeros read
    // back as nothing, because `h_num_logops` says when to stop.
    let padded = payload.len().div_ceil(BBSIZE) * BBSIZE;
    let mut payload = payload;
    payload.resize(padded, 0);

    let lsn = (u64::from(head.cycle) << 32) | u64::from(head.block);
    let placement = Placement {
        block: head.block,
        cycle: head.cycle,
        prev_block: head.prev_block,
        tail_lsn: lsn,
        uuid: sb.uuid,
        iclog_size: head.iclog_size,
    };
    let bytes = encode_record(&placement, ops.len() as u32, &payload);

    let at = sb.fsblock_offset(sb.logstart) + u64::from(head.block) * BBSIZE as u64;
    device.write_at(at, &bytes)?;
    device.flush()?;
    Ok(lsn)
}

/// Find the head and write one checkpoint there.
///
/// `build` receives the transaction id, because the id has to be woven
/// into every operation and is derived from where the record lands —
/// which is not known until the head has been found.
///
/// # Errors
///
/// As [`crate::log::head`] and [`append_at`].
pub fn append<F>(device: &dyn fs_core::BlockDevice, sb: &Superblock, build: F) -> Result<u64>
where
    F: FnOnce(u32) -> Vec<Op>,
{
    let head = crate::log::head(device, sb)?;
    let tid = transaction_id(&head);
    append_at(device, sb, &head, tid, &build(tid))
}

/// A transaction id for a checkpoint written at `head`.
///
/// The id only has to tie a checkpoint's operations to each other and
/// differ from its neighbours', so deriving it from the position gives
/// both properties without a counter to keep. The high bit is set so it
/// can never come out zero, which the kernel treats as no transaction at
/// all.
fn transaction_id(head: &Head) -> u32 {
    0x8000_0000 | (head.cycle.rotate_left(16) ^ head.block) & 0x7fff_ffff
}

impl Filesystem {
    /// Log a new core for `ino` without writing it to the inode itself.
    ///
    /// `disk_core` is the core in its **on-disk** big-endian form, as
    /// [`Filesystem::read_inode_raw`] returns it; the conversion to
    /// what the log stores happens here.
    ///
    /// The inode on disk is deliberately left alone. That is what the
    /// log is for: the record is the durable statement of the change,
    /// and whatever mounts the filesystem next applies it. It also makes
    /// the operation checkable — if the core on disk changes, something
    /// replayed the record, and if it does not, nothing did.
    ///
    /// # Errors
    ///
    /// [`Error::ReadOnly`] unless opened with [`Filesystem::mount_rw`],
    /// and as [`append`] otherwise.
    pub fn log_inode_core(&self, ino: u64, disk_core: &[u8]) -> Result<u64> {
        let Some(device) = self.writable.as_ref() else {
            return Err(Error::ReadOnly);
        };
        let core = log_dinode_from_disk(disk_core).map_err(|why| {
            Error::UnsupportedFeature(format!("inode {ino} cannot be logged: {why}"))
        })?;
        let buffer =
            InodeBuffer::containing(self.inode_offset(ino)?, self.sb.inode_cluster_bytes());

        // Every refusal this operation has is behind us and the next
        // statement writes, so the mount's one checkpoint is claimed
        // here rather than on the way in: a refusal must not spend it.
        // See `Filesystem::begin_checkpoint`.
        self.begin_checkpoint()?;
        append(device.as_ref(), &self.sb, |tid| {
            vec![
                Op {
                    flags: XLOG_START_TRANS,
                    data: Vec::new(),
                },
                Op {
                    flags: 0,
                    data: trans_header(tid, XFS_TRANS_CHECKPOINT, 2),
                },
                Op {
                    flags: 0,
                    data: inode_log_format(ino, XFS_ILOG_CORE, &buffer),
                },
                Op {
                    flags: 0,
                    data: core,
                },
                Op {
                    flags: XLOG_COMMIT_TRANS,
                    data: Vec::new(),
                },
            ]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The buffer is the inode's address truncated to a whole cluster,
    /// with the remainder becoming the offset inside it — and the
    /// truncation is against the device, not the allocation group.
    #[test]
    fn the_cluster_contains_the_inode_at_its_own_offset() {
        // 16 KiB clusters: what a v5 filesystem with 512-byte inodes has.
        const CLUSTER: u32 = 16384;

        for (at, blkno, boffset) in [
            (0x10000u64, 128u64, 0u32), // exactly on a cluster boundary
            (0x10a00, 128, 2560),       // the 6th inode of that cluster
            (0x13e00, 128, 15872),      // the last inode that still fits
            (0x14000, 160, 0),          // the next cluster along
        ] {
            let got = InodeBuffer::containing(at, CLUSTER);
            assert_eq!(
                got,
                InodeBuffer {
                    blkno,
                    len: CLUSTER / BBSIZE as u32,
                    boffset,
                },
                "inode at {at:#x}"
            );
            assert_eq!(
                got.blkno * BBSIZE as u64 + u64::from(got.boffset),
                at,
                "the buffer and the offset within it must add back up to the inode"
            );
        }
    }

    /// An identifier that ties a checkpoint's operations together, and
    /// which the kernel treats as absent if it is zero.
    #[test]
    fn a_transaction_id_is_never_zero() {
        for cycle in [0u32, 1, 0xffff_ffff] {
            for block in [0u32, 1, 0xffff_ffff] {
                let head = Head {
                    block,
                    cycle,
                    prev_block: 0,
                    free_blocks: 1024,
                    iclog_size: 32768,
                };
                assert_ne!(transaction_id(&head), 0, "cycle {cycle}, block {block}");
            }
        }
    }
}
