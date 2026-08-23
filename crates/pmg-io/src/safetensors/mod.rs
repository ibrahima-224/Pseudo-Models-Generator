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

//! Module Safetensors — writer streaming avec support du sharding et reader header-only.
//!
//! Ce module implémente l'écriture de fichiers Safetensors selon le format
//! spécifié, avec les caractéristiques suivantes :
//!
//! - **Streaming** : les tenseurs sont écrits par chunks, sans jamais charger
//!   le tenseur complet en mémoire.
//! - **Mémoire bornée** : `O(chunk_size)` où `chunk_size` est la taille maximale
//!   d'un shard.
//! - **Sharding** : support de la division en plusieurs fichiers .safetensors.
//! - **Index** : génération de `model.safetensors.index.json` avec les
//!   correspondances tenseur → shard.
//! - **Zero-payload** : le writer ne lit jamais les poids sources, il les
//!   écrit directement en streaming.
//! - **Reader header-only** : lecture des métadonnées sans télécharger le payload
//!   complet (données tensorielles).
//!
//! # Structure des fichiers de sortie
//!
//! ```text
//! <output>/
//! ├── model-00001-of-00003.safetensors
//! ├── model-00002-of-00003.safetensors
//! ├── model-00003-of-00003.safetensors
//! └── model.safetensors.index.json
//! ```
//!
//! # Exemple
//!
//! ```rust,no_run
//! use pmg_io::safetensors::{SafetensorsWriter, DType};
//! use std::path::PathBuf;
//!
//! let output_dir = PathBuf::from("model_output");
//! let mut writer = SafetensorsWriter::new(output_dir, 5 * 1024 * 1024 * 1024);
//!
//! // Écriture d'un tenseur en streaming
//! let data = vec![0u8; 1024]; // Données simulées
//! writer.write_tensor("model.layer.weight", &data, DType::F32, &[32, 32]).unwrap();
//!
//! // Finalisation et écriture de l'index
//! let index = writer.finish().unwrap();
//! ```
//!
//! # Exemple de lecture header-only (Zero-Payload)
//!
//! ```rust
//! use pmg_io::safetensors::{read_header_from};
//! use std::io::Cursor;
//!
//! // Exemple de fichier Safetensors (header + payload)
//! let header_json = r#"{"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
//! let padded_json = format!("{:<width$}", header_json, width = 8);
//! let header_size = padded_json.len() as u64;
//!
//! let mut file = Vec::new();
//! file.extend_from_slice(&header_size.to_le_bytes());
//! file.extend_from_slice(padded_json.as_bytes());
//! file.extend_from_slice(&vec![0u8; 24]); // Payload fictif
//!
//! let mut cursor = Cursor::new(file);
//! let reader = read_header_from(&mut cursor).unwrap();
//!
//! // Liste des tenseurs sans lire les données
//! for (name, entry) in reader.metadata_only() {
//!     println!("{}: {:?} {:?}", name, entry.dtype, entry.shape);
//! }
//! ```

mod header;
mod index;
mod reader;
mod types;
pub mod writer;

// Réexportations publiques
pub use header::{
    build_header, estimate_header_reserve, header_size_with_padding, pad_header, MAX_HEADER_SIZE,
};
pub use index::{shard_name, write_index};
pub use reader::{read_header_from, SafetensorsReader};
pub use types::{
    DType, IndexMetadata, SafetensorsError, SafetensorsIndex, SafetensorsResult, Shape,
    ShardResult, TensorHeaderEntry, TensorInfo,
};
pub use writer::zero_copy::{
    TensorWriteError, TensorWriterConfig, WriterMetrics, WriterState, ZeroCopyTensorWriter,
};
pub use writer::{
    ChunkWriteResult, ChunkWriter, ChunkWriterMetrics, SafetensorsWriter, ShardWriter,
    DEFAULT_CHUNK_SIZE, DEFAULT_MAX_POOL_MEMORY, DEFAULT_MAX_SHARD_SIZE, MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_safetensors_module_compiles() {
        // Test de compilation du module
        let _ = DType::F32;
        let _ = DEFAULT_MAX_SHARD_SIZE;
    }

    #[test]
    fn test_integration_write_and_read() {
        let dir = tempdir().unwrap();
        let output_dir = dir.path().to_path_buf();

        // Crée un writer avec shard de 100 octets max
        let mut writer = SafetensorsWriter::new(output_dir.clone(), 100);

        // Écrit plusieurs tenseurs (12 f32 = 48 octets chacun)
        let data1 = vec![0u8; 48];
        let data2 = vec![0u8; 48];
        let data3 = vec![0u8; 48];

        writer
            .write_tensor("model.layer1.weight", &data1, DType::F32, &[12])
            .unwrap();
        writer
            .write_tensor("model.layer1.bias", &data2, DType::F32, &[12])
            .unwrap();
        writer
            .write_tensor("model.layer2.weight", &data3, DType::F32, &[12])
            .unwrap();

        let index = writer.finish().unwrap();

        // Écrit l'index
        let index_path = output_dir.join("model.safetensors.index.json");
        write_index(&index, &index_path).unwrap();

        // Vérifie les fichiers générés
        let files: Vec<_> = fs::read_dir(&output_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();

        assert!(files.len() >= 3); // Au moins 2 shards + index

        // Vérifie que l'index est valide
        let index_content = fs::read_to_string(&index_path).unwrap();
        let parsed_index: SafetensorsIndex = serde_json::from_str(&index_content).unwrap();
        assert_eq!(parsed_index.weight_map.len(), 3);
    }

    #[test]
    fn test_read_header_only_zero_payload() {
        use std::io::Cursor;

        // Crée un fichier Safetensors minimal
        let header_json = r#"{"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let padded_json = format!("{:<width$}", header_json, width = 8); // Padding à 8 octets
        let header_size = padded_json.len() as u64;

        let mut file = Vec::new();
        file.extend_from_slice(&header_size.to_le_bytes());
        file.extend_from_slice(padded_json.as_bytes());
        // Ajoute 24 octets de données (payload)
        file.extend_from_slice(&[0u8; 24]);

        let mut cursor = Cursor::new(file);
        let reader = read_header_from(&mut cursor).unwrap();

        // Vérifie que le reader a bien lu les métadonnées
        assert_eq!(reader.tensor_count(), 1);
        assert_eq!(reader.buffer_size, 24);

        let metadata = reader.metadata_only();
        assert_eq!(metadata.len(), 1);
        let (name, entry) = metadata[0];
        assert_eq!(name, "weight");
        assert_eq!(entry.dtype, DType::F32);
        assert_eq!(entry.shape, vec![2, 3]);
        assert_eq!(entry.data_offsets, [0, 24]);
    }
}
