//! Module de compression pour PMG
//!
//! Ce module fournit la compression/décompression en temps réel
//! pour optimiser l'usage mémoire et les transferts.

pub mod algorithms;
pub mod buffer_pool;
pub mod compressor;
pub mod error;
pub mod streaming;

pub use algorithms::{CompressionAlgorithm, CompressionStats};
pub use buffer_pool::BufferPool;
pub use compressor::{CompressionConfig, Compressor};
pub use error::{CompressionError, CompressionResult};
pub use streaming::{StreamingCompressor, StreamingDecompressor};

/// Version du module de compression
pub const COMPRESSION_VERSION: &str = "0.1.0";
