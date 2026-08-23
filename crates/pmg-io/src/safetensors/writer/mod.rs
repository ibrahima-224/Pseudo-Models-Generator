// Copyright (C) 2024 PMG Contributors
// This file is part of PMG (Pseudo-Model Generator).
//
// PMG is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// PMG is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with PMG.  If not, see <https://www.gnu.org/licenses/>.

//! Sous-module contenant les writers Safetensors (shard unique et sharding automatique).
//!
//! Ce module fournit :
//! - [`ShardWriter`] : writer pour un seul shard Safetensors
//! - [`SafetensorsWriter`] : writer avec sharding automatique
//! - [`OptimizedSafetensorsWriter`] : writer optimisé avec BufWriter 64MB
//! - [`DEFAULT_MAX_SHARD_SIZE`] et [`DEFAULT_CHUNK_SIZE`] : constantes de configuration

// Sous-modules
mod buf_writer;
mod buf_writer_utils;
mod chunk_writer;
mod safetensors_writer;
mod shard;
pub mod zero_copy;

// Ré-exports publics pour maintenir la compatibilité avec l'ancienne API.
// BufferPool est remplacé par UnifiedBufferPool du module pool
pub use chunk_writer::{
    ChunkWriteResult, ChunkWriter, ChunkWriterMetrics, DEFAULT_MAX_POOL_MEMORY, MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
};
pub use safetensors_writer::SafetensorsWriter;
pub use shard::{ShardWriter, DEFAULT_CHUNK_SIZE, DEFAULT_MAX_SHARD_SIZE};
pub use zero_copy::{
    TensorWriteError, TensorWriterConfig, WriterMetrics, WriterState, ZeroCopyTensorWriter,
};

// Tests unitaires
#[cfg(test)]
mod tests;
