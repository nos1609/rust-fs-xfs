//! rust-fs-xfs — pure-Rust XFS filesystem driver.
//!
//! Exposes a stable C ABI (`fs_xfs_*`) so FFI consumers (Swift/C/Go/…)
//! can link `libfs_xfs.a` and `#include "fs_xfs.h"`.
//!
//! # Byte order
//!
//! XFS is **big-endian on disk** on every host. This is the one thing to
//! keep in mind when reading this crate alongside its little-endian
//! sisters (ext4, Btrfs, NTFS): every on-disk integer is decoded with
//! `from_be_bytes`.
//!
//! # Status
//!
//! Read path, v5 first. See the README for the supported-feature matrix.
//!
//! Architecture:
//! - [`error`] — driver error type, mapped to errno by the C ABI
//! - [`endian`] — on-disk integer decoding; the byte-order rule lives here
//! - [`superblock`] — superblock parse + validation, geometry helpers
//! - [`ag`] — allocation group headers (AGF/AGI), with v5 identity checks
//! - [`inode`] — on-disk inode core, forks, timestamps
//! - [`extent`] — data fork extent records and the bmbt
//! - [`dir`] — directory formats (short form, block, leaf, node)
//! - [`fs`] — mounted filesystem handle: lookup, read, iterate
//! - [`capi`] — C ABI exports matching `include/fs_xfs.h`

#![deny(unsafe_op_in_unsafe_fn)]

pub mod ag;
pub mod alloc_btree;
pub mod bmbt;
pub mod buf_write;
pub mod capi;
pub mod create;
pub mod dir;
pub mod dir_block;
pub mod dir_write;
pub mod endian;
pub mod error;
pub mod extent;
pub mod file_write;
pub mod format;
pub mod fs;
pub mod group_write;
pub mod inode;
pub mod inode_btree;
pub mod log;
pub mod log_write;
pub mod overlay;
pub mod super_write;
pub mod superblock;
pub mod truncate;
pub mod unlink;
pub mod write;

pub use error::{Error, Result};
pub use fs::Filesystem;
pub use superblock::Superblock;
