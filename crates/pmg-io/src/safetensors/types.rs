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

//! Types partagés pour le module Safetensors.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Type de donnée d'un tenseur (correspond aux dtypes Safetensors).
///
/// Ce type supporte à la fois les formats standard (minuscules) et
/// les formats alternatifs (majuscules) pour une compatibilité maximale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    #[serde(alias = "f32", alias = "F32")]
    F32,
    #[serde(alias = "f16", alias = "F16")]
    F16,
    #[serde(alias = "bf16", alias = "BF16")]
    BF16,
    #[serde(alias = "f8_e4m3", alias = "F8_E4M3")]
    F8E4M3,
    #[serde(alias = "f8_e5m2", alias = "F8_E5M2")]
    F8E5M2,
    #[serde(alias = "i8", alias = "I8")]
    I8,
    #[serde(alias = "i16", alias = "I16")]
    I16,
    #[serde(alias = "i32", alias = "I32")]
    I32,
    #[serde(alias = "i64", alias = "I64")]
    I64,
    #[serde(alias = "u8", alias = "U8")]
    U8,
    #[serde(alias = "u16", alias = "U16")]
    U16,
    #[serde(alias = "u32", alias = "U32")]
    U32,
    #[serde(alias = "u64", alias = "U64")]
    U64,
    #[serde(alias = "bool", alias = "BOOL")]
    Bool,
}

impl DType {
    /// Retourne la taille en octets de ce dtype.
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F16 | DType::I16 | DType::U16 | DType::BF16 => 2,
            DType::F8E4M3 | DType::F8E5M2 | DType::I8 | DType::U8 | DType::Bool => 1,
            DType::I64 | DType::U64 => 8,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "F32"),
            DType::F16 => write!(f, "F16"),
            DType::BF16 => write!(f, "BF16"),
            DType::F8E4M3 => write!(f, "F8_E4M3"),
            DType::F8E5M2 => write!(f, "F8_E5M2"),
            DType::I8 => write!(f, "I8"),
            DType::I16 => write!(f, "I16"),
            DType::I32 => write!(f, "I32"),
            DType::I64 => write!(f, "I64"),
            DType::U8 => write!(f, "U8"),
            DType::U16 => write!(f, "U16"),
            DType::U32 => write!(f, "U32"),
            DType::U64 => write!(f, "U64"),
            DType::Bool => write!(f, "BOOL"),
        }
    }
}

/// Forme d'un tenseur (dimensions).
pub type Shape = Vec<u64>;

/// Entrée d'en-tête pour un tenseur dans le JSON header Safetensors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorHeaderEntry {
    pub dtype: DType,
    pub shape: Shape,
    pub data_offsets: [u64; 2], // [begin, end] dans le buffer de données
}

/// Résultat de la finalisation d'un shard.
#[derive(Debug, Clone)]
pub struct ShardResult {
    /// Nombre de tenseurs dans ce shard.
    pub tensor_count: usize,
    /// Taille totale du buffer de données (octets).
    pub buffer_size: u64,
    /// Taille totale du shard (header + buffer).
    pub shard_size: u64,
}

/// Informations sur un tenseur pour l'index global.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shard_index: usize,
    pub dtype: DType,
    pub shape: Shape,
    pub data_offsets: [u64; 2],
    pub generated_bytes: u64,
}

/// Index global model.safetensors.index.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorsIndex {
    pub metadata: IndexMetadata,
    pub weight_map: std::collections::BTreeMap<String, String>,
}

/// Métadonnées de l'index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    pub total_size: u64,
}

/// Erreurs spécifiques au module Safetensors.
#[derive(Debug, thiserror::Error)]
pub enum SafetensorsError {
    #[error("erreur d'écriture I/O : {0}")]
    Io(#[from] std::io::Error),
    #[error(
        "taille du header dépasse la réserve ({reserved} octets réservés, {needed} nécessaires)"
    )]
    HeaderReserveExceeded { reserved: u64, needed: u64 },
    #[error("taille du header trop grande ({size} octets, maximum {max})")]
    HeaderTooLarge { size: u64, max: u64 },
    #[error("dépassement d'arithmétique : {0}")]
    Overflow(String),
    #[error("nom de tenseur duplicué : {0}")]
    DuplicateTensorName(String),
    #[error("écriture partielle du tenseur {name} : écrit {written}, attendu {expected}")]
    PartialTensorWrite {
        name: String,
        written: u64,
        expected: u64,
    },
    #[error("dépassement de la taille maximale du shard ({shard_size} > {max_size})")]
    ShardSizeExceeded { shard_size: u64, max_size: u64 },
    #[error("tenseur trop grand pour un shard unique ({size} octets)")]
    TensorTooLargeForShard { size: u64 },
    #[error("shard non finalisé")]
    ShardNotFinalized,
    #[error("erreur de sérialisation JSON : {0}")]
    Json(#[from] serde_json::Error),
}

/// Résultat spécifique au module Safetensors.
pub type SafetensorsResult<T> = Result<T, SafetensorsError>;
