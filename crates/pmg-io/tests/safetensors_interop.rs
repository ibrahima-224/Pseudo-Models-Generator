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

//! Tests d'interopérabilité avec la crate safetensors officielle.
//!
//! Vérifie que les fichiers générés par notre writer sont lisibles
//! par la crate safetensors et que les métadonnées correspondent.

use pmg_io::safetensors::{DType, SafetensorsIndex, SafetensorsWriter};
use std::fs;
use tempfile::tempdir;

#[test]
fn test_single_shard_interop() {
    let dir = tempdir().unwrap();
    let output_dir = dir.path().to_path_buf();

    // Crée un writer avec un shard unique (grand max_shard_size)
    let mut writer = SafetensorsWriter::new(output_dir.clone(), 1024 * 1024);

    // Écrit quelques tenseurs
    let data1 = [
        1.0f32.to_le_bytes(),
        2.0f32.to_le_bytes(),
        3.0f32.to_le_bytes(),
    ]
    .concat();
    let data2 = [
        4.0f32.to_le_bytes(),
        5.0f32.to_le_bytes(),
        6.0f32.to_le_bytes(),
    ]
    .concat();

    writer
        .write_tensor("layer1.weight", &data1, DType::F32, &[3])
        .unwrap();
    writer
        .write_tensor("layer1.bias", &data2, DType::F32, &[3])
        .unwrap();

    let index = writer.finish().unwrap();

    // Écrit l'index
    let index_path = output_dir.join("model.safetensors.index.json");
    pmg_io::safetensors::write_index(&index, &index_path).unwrap();

    // Lis le shard généré
    let shard_path = output_dir.join("model-00001-of-00001.safetensors");
    assert!(shard_path.exists(), "Le shard devrait exister");

    // Utilise la crate safetensors officielle pour lire
    let file_data = fs::read(&shard_path).unwrap();
    let safetensors = safetensors::SafeTensors::deserialize(&file_data).unwrap();

    // Vérifie que les tenseurs sont présents
    let tensors = safetensors.tensors();
    assert_eq!(tensors.len(), 2);

    // Vérifie les noms
    let names: Vec<_> = tensors.iter().map(|(name, _)| name.to_string()).collect();
    assert!(names.contains(&"layer1.weight".to_string()));
    assert!(names.contains(&"layer1.bias".to_string()));

    // Vérifie les métadonnées de l'index
    assert_eq!(index.weight_map.len(), 2);
    assert_eq!(index.metadata.total_size, 24); // 2 tenseurs * 3 floats * 4 octets
}

#[test]
fn test_multi_shard_interop() {
    let dir = tempdir().unwrap();
    let output_dir = dir.path().to_path_buf();

    // Crée un writer avec des shards de 20 octets max
    let mut writer = SafetensorsWriter::new(output_dir.clone(), 20);

    // Écrit des tenseurs qui forceront le sharding
    let data1 = vec![0u8; 12]; // 3 f32 = 12 octets
    let data2 = vec![0u8; 12]; // 3 f32 = 12 octets
    let data3 = vec![0u8; 12]; // 3 f32 = 12 octets

    writer
        .write_tensor("tensor1", &data1, DType::F32, &[3])
        .unwrap();
    writer
        .write_tensor("tensor2", &data2, DType::F32, &[3])
        .unwrap();
    writer
        .write_tensor("tensor3", &data3, DType::F32, &[3])
        .unwrap();

    let index = writer.finish().unwrap();

    // Écrit l'index
    let index_path = output_dir.join("model.safetensors.index.json");
    pmg_io::safetensors::write_index(&index, &index_path).unwrap();

    // Vérifie les shards générés
    let files: Vec<_> = fs::read_dir(&output_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().unwrap_or_default() == "safetensors")
        .collect();

    assert!(files.len() >= 2, "Devrait avoir au moins 2 shards");

    // Lis chaque shard avec la crate officielle
    for file in &files {
        let shard_data = fs::read(file.path()).unwrap();
        let safetensors = safetensors::SafeTensors::deserialize(&shard_data).unwrap();

        // Chaque shard devrait avoir au moins 1 tenseur
        assert!(!safetensors.tensors().is_empty());
    }

    // Vérifie la cohérence de l'index
    let index_content = fs::read_to_string(&index_path).unwrap();
    let parsed_index: SafetensorsIndex = serde_json::from_str(&index_content).unwrap();

    // Tous les tenseurs de l'index devraient exister dans les shards
    for (tensor_name, shard_name) in &parsed_index.weight_map {
        let shard_path = output_dir.join(shard_name);
        assert!(
            shard_path.exists(),
            "Le shard {} devrait exister",
            shard_name
        );

        let shard_data = fs::read(&shard_path).unwrap();
        let safetensors = safetensors::SafeTensors::deserialize(&shard_data).unwrap();

        // Le tenseur devrait exister dans le shard
        let tensor = safetensors.tensor(tensor_name);
        assert!(
            tensor.is_ok(),
            "Le tenseur {} devrait exister dans {}",
            tensor_name,
            shard_name
        );
    }
}

#[test]
fn test_dtype_interop() {
    let dir = tempdir().unwrap();
    let output_dir = dir.path().to_path_buf();

    let mut writer = SafetensorsWriter::new(output_dir.clone(), 1024 * 1024);

    // Teste différents dtypes
    let data_f32 = [1.0f32.to_le_bytes(), 2.0f32.to_le_bytes()].concat();
    let data_i32 = [1i32.to_le_bytes(), 2i32.to_le_bytes()].concat();
    let data_f16 = vec![0u8; 4]; // 2 f16 = 4 octets

    writer
        .write_tensor("f32_tensor", &data_f32, DType::F32, &[2])
        .unwrap();
    writer
        .write_tensor("i32_tensor", &data_i32, DType::I32, &[2])
        .unwrap();
    writer
        .write_tensor("f16_tensor", &data_f16, DType::F16, &[2])
        .unwrap();

    let index = writer.finish().unwrap();

    // Écrit l'index
    let index_path = output_dir.join("model.safetensors.index.json");
    pmg_io::safetensors::write_index(&index, &index_path).unwrap();

    // Lis avec la crate officielle
    let shard_path = output_dir.join("model-00001-of-00001.safetensors");
    let file_data = fs::read(&shard_path).unwrap();
    let safetensors = safetensors::SafeTensors::deserialize(&file_data).unwrap();

    // Vérifie que les tenseurs sont présents avec les bons dtypes
    let tensors = safetensors.tensors();
    assert_eq!(tensors.len(), 3);

    for (name, tensor_info) in &tensors {
        match name.as_str() {
            "f32_tensor" => assert_eq!(tensor_info.dtype(), safetensors::Dtype::F32),
            "i32_tensor" => assert_eq!(tensor_info.dtype(), safetensors::Dtype::I32),
            "f16_tensor" => assert_eq!(tensor_info.dtype(), safetensors::Dtype::F16),
            _ => panic!("Nom de tenseur inattendu: {}", name),
        }
    }
}
