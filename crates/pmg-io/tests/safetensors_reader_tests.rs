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

//! Tests pour le reader header-only Zero-Payload de Safetensors.

use pmg_io::safetensors::{
    read_header_from, write_index, DType, SafetensorsError, SafetensorsWriter,
};
use std::io::{Cursor, Read, Seek, SeekFrom};
use tempfile::tempdir;

/// Helper pour calculer le padding Safetensors (doit être identique à reader.rs)
fn pad_header(json: &str) -> String {
    let len = json.len();
    let padding = (8 - (len % 8)) % 8;
    if padding == 0 {
        json.to_string()
    } else {
        format!("{}{}", json, " ".repeat(padding))
    }
}

/// Mock Read qui panique si le payload (après le header) est lu.
/// Cela vérifie que le reader Zero-Payload ne lit jamais les données tensorielles.
struct PayloadPanicReader {
    data: Vec<u8>,
    position: usize,
}

impl PayloadPanicReader {
    fn new(data: Vec<u8>) -> Self {
        Self { data, position: 0 }
    }
}

impl Read for PayloadPanicReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        // Vérifie si on essaie de lire au-delà du header
        // Le header se termine à HEADER_SIZE_BYTES + header_size
        // Nous ne devrions jamais lire au-delà de cette position
        let header_end = 8 + get_header_size(&self.data) as usize;
        if self.position >= header_end {
            panic!(
                "PayloadPanicReader: tentative de lecture du payload à la position {} (header se termine à {})",
                self.position, header_end
            );
        }

        let remaining = self.data.len() - self.position;
        let to_read = std::cmp::min(buf.len(), remaining);
        buf[..to_read].copy_from_slice(&self.data[self.position..self.position + to_read]);
        self.position += to_read;
        Ok(to_read)
    }
}

impl Seek for PayloadPanicReader {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        match pos {
            SeekFrom::Start(offset) => {
                self.position = offset as usize;
                Ok(offset)
            },
            SeekFrom::End(offset) => {
                self.position = (self.data.len() as i64 + offset) as usize;
                Ok(self.position as u64)
            },
            SeekFrom::Current(offset) => {
                self.position = (self.position as i64 + offset) as usize;
                Ok(self.position as u64)
            },
        }
    }
}

/// Helper pour obtenir la taille du header depuis les premiers octets.
fn get_header_size(data: &[u8]) -> u64 {
    if data.len() < 8 {
        return 0;
    }
    u64::from_le_bytes([
        data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
    ])
}

/// Crée un fichier Safetensors minimal pour les tests.
fn make_valid_safetensors_file() -> Vec<u8> {
    let header_json = r#"{"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
    let padded_json = pad_header(header_json);
    let header_size = padded_json.len() as u64;

    let mut file = Vec::new();
    file.extend_from_slice(&header_size.to_le_bytes());
    file.extend_from_slice(padded_json.as_bytes());
    file.extend_from_slice(&[0u8; 24]); // Payload de 24 octets
    file
}

#[test]
fn test_zero_payload_no_payload_read() {
    let data = make_valid_safetensors_file();
    let mut reader = PayloadPanicReader::new(data);

    // Cette lecture ne devrait pas paniquer car elle ne lit que le header
    let safetensors_reader = read_header_from(&mut reader).unwrap();

    assert_eq!(safetensors_reader.tensor_count(), 1);
    assert_eq!(safetensors_reader.buffer_size, 24);
}

#[test]
fn test_corruption_header_too_small() {
    // Fichier trop petit pour contenir le header_size
    let data = vec![0u8; 4];
    let mut cursor = Cursor::new(data);
    let result = read_header_from(&mut cursor);

    assert!(result.is_err());
    match result {
        Err(SafetensorsError::Io(io_err)) => {
            assert_eq!(io_err.kind(), std::io::ErrorKind::UnexpectedEof);
        },
        _ => panic!("Erreur inattendue: {:?}", result),
    }
}

#[test]
fn test_corruption_header_size_zero() {
    let mut data = Vec::new();
    data.extend_from_slice(&0u64.to_le_bytes()); // header_size = 0
    data.extend_from_slice(&[0u8; 100]); // Données fictives

    let mut cursor = Cursor::new(data);
    let result = read_header_from(&mut cursor);

    assert!(result.is_err());
    match result {
        Err(SafetensorsError::Io(io_err)) => {
            assert!(io_err.to_string().contains("header_size est zéro"));
        },
        _ => panic!("Erreur inattendue: {:?}", result),
    }
}

#[test]
fn test_corruption_header_too_large() {
    let mut data = Vec::new();
    // header_size > MAX_HEADER_SIZE (8 MiB)
    data.extend_from_slice(&(9u64 * 1024 * 1024).to_le_bytes());
    data.extend_from_slice(&[0u8; 100]);

    let mut cursor = Cursor::new(data);
    let result = read_header_from(&mut cursor);

    assert!(result.is_err());
    match result {
        Err(SafetensorsError::HeaderTooLarge { size, max }) => {
            assert_eq!(size, 9 * 1024 * 1024);
            assert_eq!(max, 8 * 1024 * 1024);
        },
        _ => panic!("Erreur inattendue: {:?}", result),
    }
}

#[test]
fn test_corruption_invalid_json() {
    let header_json = "invalid json";
    let padded_json = pad_header(header_json);
    let header_size = padded_json.len() as u64;

    let mut file = Vec::new();
    file.extend_from_slice(&header_size.to_le_bytes());
    file.extend_from_slice(padded_json.as_bytes());
    file.extend_from_slice(&[0u8; 24]);

    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);

    assert!(result.is_err());
    match result {
        Err(SafetensorsError::Json(_)) => {}, // OK
        _ => panic!("Erreur inattendue: {:?}", result),
    }
}

#[test]
fn test_corruption_offsets_out_of_bounds() {
    // Offset end > buffer_size
    let header_json = r#"{"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,100]}}"#;
    let padded_json = pad_header(header_json);
    let header_size = padded_json.len() as u64;

    let mut file = Vec::new();
    file.extend_from_slice(&header_size.to_le_bytes());
    file.extend_from_slice(padded_json.as_bytes());
    file.extend_from_slice(&[0u8; 24]); // Buffer de 24 octets seulement

    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);

    assert!(result.is_err());
    match result {
        Err(SafetensorsError::Io(io_err)) => {
            assert!(io_err.to_string().contains("offsets hors limites"));
        },
        _ => panic!("Erreur inattendue: {:?}", result),
    }
}

#[test]
fn test_corruption_offsets_begin_greater_than_end() {
    let header_json = r#"{"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[10,5]}}"#;
    let padded_json = pad_header(header_json);
    let header_size = padded_json.len() as u64;

    let mut file = Vec::new();
    file.extend_from_slice(&header_size.to_le_bytes());
    file.extend_from_slice(padded_json.as_bytes());
    file.extend_from_slice(&[0u8; 24]);

    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);

    assert!(result.is_err());
    match result {
        Err(SafetensorsError::Io(io_err)) => {
            assert!(io_err.to_string().contains("offsets invalides"));
        },
        _ => panic!("Erreur inattendue: {:?}", result),
    }
}

#[test]
fn test_corruption_tensor_size_mismatch() {
    // Shape [2,3] avec F32 = 24 octets, mais offsets indiquent 30 octets
    let header_json = r#"{"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,30]}}"#;
    let padded_json = pad_header(header_json);
    let header_size = padded_json.len() as u64;

    let mut file = Vec::new();
    file.extend_from_slice(&header_size.to_le_bytes());
    file.extend_from_slice(padded_json.as_bytes());
    file.extend_from_slice(&[0u8; 30]); // Buffer de 30 octets

    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);

    assert!(result.is_err());
    match result {
        Err(SafetensorsError::Io(io_err)) => {
            assert!(io_err.to_string().contains("taille du tenseur"));
        },
        _ => panic!("Erreur inattendue: {:?}", result),
    }
}

#[test]
fn test_integration_with_writer() {
    let dir = tempdir().unwrap();
    let output_dir = dir.path().to_path_buf();

    // Crée un shard avec deux tenseurs
    let shard_path = output_dir.join("model-00001-of-00001.safetensors");
    let mut writer = SafetensorsWriter::new(output_dir.clone(), 1024);

    // Premier tenseur: F32, 2x3 = 24 octets
    let data1 = [1.0f32; 6]; // 6 floats = 24 octets
                             // Conversion sûre : les floats F32 sont de 4 octets et sans padding
    let data1_bytes = unsafe {
        std::slice::from_raw_parts(
            data1.as_ptr() as *const u8,
            data1.len() * std::mem::size_of::<f32>(),
        )
    };
    writer
        .write_tensor("weight", data1_bytes, DType::F32, &[2, 3])
        .unwrap();

    // Deuxième tenseur: F32, 4 = 4 octets
    let data2 = [2.0f32]; // 1 float = 4 octets
    let data2_bytes = unsafe {
        std::slice::from_raw_parts(
            data2.as_ptr() as *const u8,
            data2.len() * std::mem::size_of::<f32>(),
        )
    };
    writer
        .write_tensor("bias", data2_bytes, DType::F32, &[1])
        .unwrap();

    let index = writer.finish().unwrap();

    // Écrit l'index (non nécessaire pour ce test mais pour la complétude)
    let index_path = output_dir.join("model.safetensors.index.json");
    write_index(&index, &index_path).unwrap();

    // Lit le header du shard avec le reader Zero-Payload
    let mut file = std::fs::File::open(&shard_path).unwrap();
    let reader = read_header_from(&mut file).unwrap();

    // Vérifie les métadonnées
    assert_eq!(reader.tensor_count(), 2);
    assert!(reader.header_size > 0);

    let metadata = reader.metadata_only();
    assert_eq!(metadata.len(), 2);

    // Le reader retourne les tenseurs dans l'ordre alphabétique (BTreeMap)
    // Le writer a écrit "weight" en premier (offsets [0, 24]) puis "bias" (offsets [24, 28])
    // Donc dans le header, "bias" a les offsets [24, 28] et "weight" [0, 24]
    let (name1, entry1) = &metadata[0];
    assert_eq!(*name1, "bias");
    assert_eq!(entry1.dtype, DType::F32);
    assert_eq!(entry1.shape, vec![1]);
    assert_eq!(entry1.data_offsets, [24, 28]); // Après le weight

    let (name2, entry2) = &metadata[1];
    assert_eq!(*name2, "weight");
    assert_eq!(entry2.dtype, DType::F32);
    assert_eq!(entry2.shape, vec![2, 3]);
    assert_eq!(entry2.data_offsets, [0, 24]); // Premier tenseur écrit

    // Vérifie que le buffer_size est correct (24 + 4 = 28 octets)
    assert_eq!(reader.buffer_size, 28);
}

#[test]
fn test_metadata_only_multiple_tensors() {
    let header_json = r#"{
        "layer1.weight": {"dtype":"F32","shape":[10,10],"data_offsets":[0,400]},
        "layer1.bias": {"dtype":"F32","shape":[10],"data_offsets":[400,440]},
        "layer2.weight": {"dtype":"F16","shape":[5,5],"data_offsets":[440,490]}
    }"#;
    let padded_json = pad_header(header_json);
    let header_size = padded_json.len() as u64;

    let mut file = Vec::new();
    file.extend_from_slice(&header_size.to_le_bytes());
    file.extend_from_slice(padded_json.as_bytes());
    file.extend_from_slice(&vec![0u8; 490]); // Buffer de 490 octets

    let mut cursor = Cursor::new(file);
    let reader = read_header_from(&mut cursor).unwrap();

    assert_eq!(reader.tensor_count(), 3);

    let metadata = reader.metadata_only();
    assert_eq!(metadata.len(), 3);

    // Vérifie l'ordre alphabétique
    assert_eq!(metadata[0].0, "layer1.bias");
    assert_eq!(metadata[1].0, "layer1.weight");
    assert_eq!(metadata[2].0, "layer2.weight");

    // Vérifie les types
    assert_eq!(metadata[0].1.dtype, DType::F32);
    assert_eq!(metadata[1].1.dtype, DType::F32);
    assert_eq!(metadata[2].1.dtype, DType::F16);
}

#[test]
fn test_header_with_metadata_key() {
    // Le JSON peut contenir une clé "__metadata__" qui doit être ignorée
    let header_json = r#"{
        "__metadata__": {"version": "1.0"},
        "weight": {"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}
    }"#;
    let padded_json = pad_header(header_json);
    let header_size = padded_json.len() as u64;

    let mut file = Vec::new();
    file.extend_from_slice(&header_size.to_le_bytes());
    file.extend_from_slice(padded_json.as_bytes());
    file.extend_from_slice(&[0u8; 24]);

    let mut cursor = Cursor::new(file);
    let reader = read_header_from(&mut cursor).unwrap();

    // Le "__metadata__" n'est pas un tenseur, donc tensor_count = 1
    assert_eq!(reader.tensor_count(), 1);
    assert!(reader.header.contains_key("weight"));
    // Note: "__metadata__" est dans le header mais n'est pas un TensorHeaderEntry
    // car il n'a pas le même format. Selon la spec, il doit être toléré.
}

#[test]
fn test_empty_file() {
    let data = Vec::new();
    let mut cursor = Cursor::new(data);
    let result = read_header_from(&mut cursor);

    assert!(result.is_err());
}

#[test]
fn test_only_header_size_no_json() {
    let mut data = Vec::new();
    data.extend_from_slice(&8u64.to_le_bytes()); // header_size = 8
                                                 // Pas de JSON après

    let mut cursor = Cursor::new(data);
    let result = read_header_from(&mut cursor);

    assert!(result.is_err());
    match result {
        Err(SafetensorsError::Io(io_err)) => {
            assert_eq!(io_err.kind(), std::io::ErrorKind::UnexpectedEof);
        },
        _ => panic!("Erreur inattendue: {:?}", result),
    }
}

#[test]
fn test_truncated_json() {
    let header_json = r#"{"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}};"#;
    let padded_json = pad_header(header_json);
    let header_size = padded_json.len() as u64;

    let mut file = Vec::new();
    file.extend_from_slice(&header_size.to_le_bytes());
    file.extend_from_slice(padded_json.as_bytes());
    file.extend_from_slice(&[0u8; 24]);

    let mut cursor = Cursor::new(file);
    let result = read_header_from(&mut cursor);

    // Le JSON est invalide à cause du point-virgule
    assert!(result.is_err());
    match result {
        Err(SafetensorsError::Json(_)) => {}, // OK
        _ => panic!("Erreur inattendue: {:?}", result),
    }
}
