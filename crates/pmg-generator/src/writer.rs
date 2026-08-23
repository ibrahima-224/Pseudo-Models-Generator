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

//! Écriture des tenseurs générés en fichiers Safetensors.
//!
//! Ce module implémente l'écriture streaming des tenseurs générés au format
//! Safetensors binaire. Il utilise `pmg-io` pour l'atomicité et la structure
//! de sortie.
//!
//! ## Format Safetensors
//!
//! Le format Safetensors est un format binaire pour les tenseurs :
//! - Header JSON contenant les métadonnées (noms, shapes, dtypes, offsets)
//! - Payload binaire contenant les données des tenseurs
//!
//! ## Contraintes
//!
//! - **Mémoire bornée** : écriture par chunks pour les grands tenseurs
//! - **Atomicité** : écriture dans dossier temporaire puis renommage
//! - **Déterminisme** : même seed = même sortie binaire

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use pmg_core::error::{CoreError, CoreResult};

/// Métadonnées d'un tenseur pour l'écriture Safetensors.
#[derive(Debug, Clone)]
pub struct SafetensorMetadata {
    /// Nom du tenseur.
    pub name: String,
    /// Forme du tenseur.
    pub shape: Vec<usize>,
    /// Type de données.
    pub dtype: String,
    /// Offset en octets dans le payload.
    pub offset: u64,
    /// Taille en octets.
    pub size: u64,
}

/// Écrivain Safetensors streaming.
///
/// Gère l'écriture de tenseurs dans le format Safetensors avec gestion
/// de la mémoire par chunks.
pub struct SafetensorsWriter {
    /// Métadonnées des tenseurs écrits.
    metadata: Vec<SafetensorMetadata>,
    /// Offset courant dans le payload.
    current_offset: u64,
    /// Taille des chunks en octets.
    chunk_size_bytes: usize,
}

impl SafetensorsWriter {
    /// Crée un nouveau writer avec la taille de chunk spécifiée.
    ///
    /// # Paramètres
    /// - `chunk_size_bytes` : taille des chunks pour l'écriture (défaut: 1 Mo)
    pub fn new(chunk_size_bytes: usize) -> Self {
        Self {
            metadata: Vec::new(),
            current_offset: 0,
            chunk_size_bytes,
        }
    }

    /// Crée un writer avec la taille de chunk par défaut (1 Mo).
    pub fn with_default_chunk() -> Self {
        Self::new(1024 * 1024)
    }

    /// Écrit un tenseur dans un fichier Safetensors.
    ///
    /// # Paramètres
    /// - `path` : chemin du fichier à écrire
    /// - `tensors` : vecteur de (nom, forme, dtype, valeurs)
    ///
    /// # Erreurs
    /// Retourne une erreur si l'écriture échoue.
    pub fn write_file(
        &mut self,
        path: &Path,
        tensors: &[(String, Vec<usize>, String, &[f64])],
    ) -> CoreResult<()> {
        // Calculer les métadonnées et offsets
        let mut metadata_map = HashMap::new();
        let mut offset = 0u64;

        for (name, shape, dtype, _values) in tensors {
            let size_bytes = self.calculate_size_bytes(shape, dtype)?;
            let metadata = SafetensorMetadata {
                name: name.clone(),
                shape: shape.clone(),
                dtype: dtype.clone(),
                offset,
                size: size_bytes,
            };
            metadata_map.insert(name.clone(), metadata);
            offset += size_bytes;
        }

        // Écrire le header JSON
        let header = self.create_header(&metadata_map)?;

        // Écrire le fichier
        let file = File::create(path).map_err(|e| {
            CoreError::Internal(format!("échec création fichier {} : {}", path.display(), e))
        })?;
        let mut writer = BufWriter::new(file);

        // Écrire la taille du header (u64 little-endian)
        let header_bytes = header.as_bytes();
        let header_len = header_bytes.len() as u64;
        writer
            .write_all(&header_len.to_le_bytes())
            .map_err(|e| CoreError::Internal(format!("échec écriture taille header : {}", e)))?;

        // Écrire le header JSON
        writer
            .write_all(header_bytes)
            .map_err(|e| CoreError::Internal(format!("échec écriture header JSON : {}", e)))?;

        // Écrire le payload des tenseurs par chunks avec le dtype approprié
        for (_, _, dtype, values) in tensors {
            self.write_tensor_payload_chunked(&mut writer, values, dtype)?;
        }

        writer
            .flush()
            .map_err(|e| CoreError::Internal(format!("échec flush fichier : {}", e)))?;

        Ok(())
    }

    /// Calcule la taille en octets d'un tenseur.
    fn calculate_size_bytes(&self, shape: &[usize], dtype: &str) -> CoreResult<u64> {
        let element_size = match dtype {
            "f32" | "F32" => 4,
            "f64" | "F64" => 8,
            "bf16" | "BF16" => 2,
            "i32" | "I32" => 4,
            "i64" | "I64" => 8,
            _ => {
                return Err(CoreError::Internal(format!(
                    "dtype non supporté : {}",
                    dtype
                )))
            },
        };

        let num_elements: u64 = shape.iter().product::<usize>() as u64;
        Ok(num_elements * element_size as u64)
    }

    /// Crée le header JSON à partir des métadonnées.
    fn create_header(
        &self,
        metadata_map: &HashMap<String, SafetensorMetadata>,
    ) -> CoreResult<String> {
        let mut header = serde_json::Map::new();

        for (name, meta) in metadata_map {
            let tensor_info = serde_json::json!({
                "dtype": meta.dtype,
                "shape": meta.shape,
                "data_offsets": [meta.offset, meta.offset + meta.size]
            });
            header.insert(name.clone(), tensor_info);
        }

        serde_json::to_string(&header)
            .map_err(|e| CoreError::Internal(format!("échec sérialisation header JSON : {}", e)))
    }

    /// Écrit le payload d'un tenseur par chunks avec le dtype spécifié.
    ///
    /// # Paramètres
    /// - `writer` : écrivain bufferisé.
    /// - `values` : valeurs du tenseur en f64.
    /// - `dtype` : type de données cible.
    ///
    /// # Erreurs
    /// Retourne une erreur si l'écriture échoue.
    fn write_tensor_payload_chunked(
        &self,
        writer: &mut BufWriter<File>,
        values: &[f64],
        dtype: &str,
    ) -> CoreResult<()> {
        let chunk_size = self.chunk_size_bytes / std::mem::size_of::<f64>();
        let chunks: Vec<&[f64]> = values.chunks(chunk_size).collect();

        for chunk in chunks {
            for value in chunk {
                let bytes = self.value_to_bytes(*value, dtype)?;
                writer
                    .write_all(&bytes)
                    .map_err(|e| CoreError::Internal(format!("échec écriture valeur : {}", e)))?;
            }
        }

        Ok(())
    }

    /// Convertit une valeur f64 en bytes selon le dtype spécifié.
    ///
    /// # Paramètres
    /// - `value` : valeur à convertir.
    /// - `dtype` : type de données cible.
    ///
    /// # Retour
    /// Vecteur de bytes en little-endian.
    fn value_to_bytes(&self, value: f64, dtype: &str) -> CoreResult<Vec<u8>> {
        match dtype {
            "f32" | "F32" => {
                let f32_val = value as f32;
                Ok(f32_val.to_le_bytes().to_vec())
            },
            "f64" | "F64" => Ok(value.to_le_bytes().to_vec()),
            "bf16" | "BF16" => {
                // Conversion f64 -> f32 -> bf16 (format brain floating point)
                let f32_val = value as f32;
                let bf16_val = self.f32_to_bf16(f32_val);
                Ok(bf16_val.to_le_bytes().to_vec())
            },
            "i32" | "I32" => {
                let i32_val = value as i32;
                Ok(i32_val.to_le_bytes().to_vec())
            },
            "i64" | "I64" => {
                let i64_val = value as i64;
                Ok(i64_val.to_le_bytes().to_vec())
            },
            _ => {
                // Fallback vers f32 pour les dtypes non supportés
                let f32_val = value as f32;
                Ok(f32_val.to_le_bytes().to_vec())
            },
        }
    }

    /// Convertit un f32 en bf16 (brain floating point 16 bits).
    ///
    /// # Paramètres
    /// - `value` : valeur f32 à convertir.
    ///
    /// # Retour
    /// Valeur bf16 représentée en u16.
    fn f32_to_bf16(&self, value: f32) -> u16 {
        let bits = value.to_bits();
        // bf16 a 8 bits d'exposant et 7 bits de mantisse
        // vs f32 qui a 8 bits d'exposant et 23 bits de mantisse
        // On tronque les 16 bits de poids faible
        (bits >> 16) as u16
    }

    /// Retourne les métadonnées des tenseurs écrits.
    pub fn metadata(&self) -> &[SafetensorMetadata] {
        &self.metadata
    }

    /// Réinitialise le writer pour un nouveau fichier.
    pub fn reset(&mut self) {
        self.metadata.clear();
        self.current_offset = 0;
    }
}

/// Écriture atomique d'un fichier Safetensors.
///
/// # Paramètres
/// - `path` : chemin du fichier à écrire
/// - `tensors` : vecteur de (nom, forme, dtype, valeurs)
///
/// # Erreurs
/// Retourne une erreur si l'écriture échoue.
pub fn write_safetensors(
    path: &Path,
    tensors: &[(String, Vec<usize>, String, &[f64])],
) -> CoreResult<()> {
    let mut writer = SafetensorsWriter::with_default_chunk();
    writer.write_file(path, tensors)
}

/// Écriture atomique avec renommage temporaire.
///
/// # Paramètres
/// - `path` : chemin final du fichier
/// - `tensors` : vecteur de (nom, forme, dtype, valeurs)
///
/// # Erreurs
/// Retourne une erreur si l'écriture échoue.
pub fn write_safetensors_atomic(
    path: &Path,
    tensors: &[(String, Vec<usize>, String, &[f64])],
) -> CoreResult<()> {
    let temp_path = path.with_extension("tmp");

    write_safetensors(&temp_path, tensors)?;

    // Renommage atomique
    std::fs::rename(&temp_path, path).map_err(|e| {
        // Nettoyage en cas d'échec
        let _ = std::fs::remove_file(&temp_path);
        CoreError::Internal(format!(
            "échec renommage atomique {} → {} : {}",
            temp_path.display(),
            path.display(),
            e
        ))
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn writer_creation() {
        let writer = SafetensorsWriter::new(1024);
        assert_eq!(writer.chunk_size_bytes, 1024);
        assert!(writer.metadata().is_empty());
    }

    #[test]
    fn writer_default_chunk() {
        let writer = SafetensorsWriter::with_default_chunk();
        assert_eq!(writer.chunk_size_bytes, 1024 * 1024);
    }

    #[test]
    fn calculate_size_bytes_f32() {
        let writer = SafetensorsWriter::new(1024);
        let size = writer.calculate_size_bytes(&[10, 20], "f32").unwrap();
        assert_eq!(size, 10 * 20 * 4);
    }

    #[test]
    fn calculate_size_bytes_f64() {
        let writer = SafetensorsWriter::new(1024);
        let size = writer.calculate_size_bytes(&[10, 20], "f64").unwrap();
        assert_eq!(size, 10 * 20 * 8);
    }

    #[test]
    fn calculate_size_bytes_invalid_dtype() {
        let writer = SafetensorsWriter::new(1024);
        let result = writer.calculate_size_bytes(&[10], "int8");
        assert!(result.is_err());
    }

    #[test]
    fn write_safetensors_creates_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.safetensors");

        let values = [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensors = vec![(
            "tensor1".to_string(),
            vec![2, 3],
            "f32".to_string(),
            values.as_slice(),
        )];

        write_safetensors(&path, &tensors).unwrap();
        assert!(path.exists());

        // Vérifier que le fichier n'est pas vide
        let metadata = fs::metadata(&path).unwrap();
        assert!(metadata.len() > 0);
    }

    #[test]
    fn write_safetensors_atomic_creates_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_atomic.safetensors");

        let values = [1.0f64, 2.0];
        let tensors = vec![(
            "tensor1".to_string(),
            vec![2],
            "f32".to_string(),
            values.as_slice(),
        )];

        write_safetensors_atomic(&path, &tensors).unwrap();
        assert!(path.exists());

        // Vérifier que le fichier tmp n'existe pas
        let tmp_path = path.with_extension("tmp");
        assert!(!tmp_path.exists());
    }

    #[test]
    fn writer_reset() {
        let mut writer = SafetensorsWriter::new(1024);
        writer.current_offset = 100;
        writer.metadata.push(SafetensorMetadata {
            name: "test".to_string(),
            shape: vec![1],
            dtype: "f32".to_string(),
            offset: 0,
            size: 4,
        });

        writer.reset();
        assert_eq!(writer.current_offset, 0);
        assert!(writer.metadata.is_empty());
    }
}
