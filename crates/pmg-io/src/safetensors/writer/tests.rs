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

//! Tests unitaires pour les writers Safetensors.

use super::*;
use crate::safetensors::types::DType;
use std::fs;
use tempfile::tempdir;

#[test]
fn test_shard_writer_basic() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test.safetensors");

    let mut writer = ShardWriter::new(path.clone(), 1024).unwrap();

    // Écrit un petit tenseur
    let data = vec![0u8; 24]; // 6 f32 = 24 octets
    let shape = vec![2, 3];
    writer.begin_tensor("weight", DType::F32, &shape).unwrap();
    writer.write_chunk(&data).unwrap();
    writer.end_tensor().unwrap();

    let result = writer.finalize().unwrap();
    assert_eq!(result.tensor_count, 1);
    assert_eq!(result.buffer_size, 24);

    // Vérifie que le fichier existe
    assert!(path.exists());

    // Lit le header pour vérifier
    let file_content = fs::read(&path).unwrap();
    assert!(file_content.len() > 8); // Au moins le header_size
}

#[test]
fn test_safetensors_writer_sharding() {
    let dir = tempdir().unwrap();

    let mut writer = SafetensorsWriter::new(dir.path().to_path_buf(), 100); // Shard de 100 octets max

    // Écrit des tenseurs qui forceront le sharding (12 f32 = 48 octets chacun)
    let data1 = vec![0u8; 48];
    let data2 = vec![0u8; 48];
    let data3 = vec![0u8; 48];

    writer
        .write_tensor("tensor1", &data1, DType::F32, &[12])
        .unwrap();
    writer
        .write_tensor("tensor2", &data2, DType::F32, &[12])
        .unwrap();
    writer
        .write_tensor("tensor3", &data3, DType::F32, &[12])
        .unwrap();

    let index = writer.finish().unwrap();

    // Vérifie que l'index contient les 3 tenseurs
    assert_eq!(index.weight_map.len(), 3);
    assert!(index.weight_map.contains_key("tensor1"));
    assert!(index.weight_map.contains_key("tensor2"));
    assert!(index.weight_map.contains_key("tensor3"));

    // Vérifie que les fichiers existent
    let files: Vec<_> = fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().unwrap_or_default() == "safetensors")
        .collect();

    assert!(files.len() >= 2); // Au moins 2 shards
}
