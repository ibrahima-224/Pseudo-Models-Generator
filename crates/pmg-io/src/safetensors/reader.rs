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

//! Reader header-only (Zero-Payload) pour les fichiers Safetensors.
//!
//! Ce module permet de lire les métadonnées (header JSON) d'un fichier Safetensors
//! sans télécharger le payload complet (données tensorielles). C'est la seule API
//! de lecture utilisée par `espec`/`validate`/`compare`.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};

use super::header::MAX_HEADER_SIZE;
use super::types::{SafetensorsError, SafetensorsResult, TensorHeaderEntry};

/// Taille de l'en-tête u64 en début de fichier (little-endian).
const HEADER_SIZE_BYTES: usize = 8;

/// Structure représentant un fichier Safetensors lu en mode header-only.
///
/// Contient uniquement les métadonnées du fichier, sans les données tensorielles.
/// Permet de lister les tenseurs, leurs types, formes et offsets sans charger le payload.
#[derive(Debug, Clone)]
pub struct SafetensorsReader {
    /// Taille du header JSON déclarée dans le fichier (en octets, incluant le padding).
    pub header_size: u64,
    /// Mappe ordonnée des noms de tenseurs vers leurs métadonnées.
    pub header: BTreeMap<String, TensorHeaderEntry>,
    /// Taille du buffer de données (payload) en octets.
    pub buffer_size: u64,
    /// Taille totale du fichier en octets.
    pub file_size: u64,
}

impl SafetensorsReader {
    /// Retourne la liste des tenseurs avec leurs métadonnées.
    ///
    /// Utile pour lister les tenseurs sans accéder aux données brutes.
    pub fn metadata_only(&self) -> Vec<(&str, &TensorHeaderEntry)> {
        self.header
            .iter()
            .map(|(name, entry)| (name.as_str(), entry))
            .collect()
    }

    /// Retourne le nombre de tenseurs dans le fichier.
    pub fn tensor_count(&self) -> usize {
        self.header.len()
    }

    /// Vérifie que les offsets de tous les tenseurs sont valides.
    fn validate_offsets(&self) -> SafetensorsResult<()> {
        for (name, entry) in &self.header {
            let (begin, end) = (entry.data_offsets[0], entry.data_offsets[1]);

            // Vérifie que begin <= end
            if begin > end {
                return Err(SafetensorsError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "offsets invalides pour le tenseur '{}': begin={} > end={}",
                        name, begin, end
                    ),
                )));
            }

            // Vérifie que end ne dépasse pas la taille du buffer
            if end > self.buffer_size {
                return Err(SafetensorsError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "offsets hors limites pour le tenseur '{}': end={} > buffer_size={}",
                        name, end, self.buffer_size
                    ),
                )));
            }

            // Vérifie que generated_bytes = shape.iter().product::<u64>() * dtype.size_bytes()
            let element_count: u64 = entry.shape.iter().product();
            let expected_bytes = element_count
                .checked_mul(entry.dtype.size_bytes() as u64)
                .ok_or_else(|| {
                    SafetensorsError::Overflow(format!("calcul de la taille du tenseur '{}'", name))
                })?;

            let actual_bytes = end - begin;
            if actual_bytes != expected_bytes {
                return Err(SafetensorsError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "taille du tenseur '{}' incohérente: generated_bytes={}, attendu={}",
                        name, actual_bytes, expected_bytes
                    ),
                )));
            }
        }

        Ok(())
    }
}

/// Lit le header d'un fichier Safetensors sans lire le payload.
///
/// Cette fonction implémente le principe Zero-Payload : elle ne lit que les
/// métadonnées (8 octets pour la taille du header + header JSON), sans jamais
/// accéder aux données tensorielles.
///
/// # Paramètres
/// - `r` : flux de lecture (doit implémenter `Read + Seek`).
///
/// # Retour
/// Un `SafetensorsReader` contenant les métadonnées du fichier.
///
/// # Erreurs
/// Retourne une erreur si :
/// - Le header_size est invalide (0 ou > MAX_HEADER_SIZE)
/// - Le JSON est malformé ou contient des champs inconnus
/// - Les offsets des tenseurs sont invalides ou hors limites
/// - Le fichier est tronqué
pub fn read_header_from<R: Read + Seek>(r: &mut R) -> SafetensorsResult<SafetensorsReader> {
    // Récupère la taille totale du fichier
    let file_size = r.seek(SeekFrom::End(0)).map_err(SafetensorsError::Io)?;
    r.seek(SeekFrom::Start(0)).map_err(SafetensorsError::Io)?;

    // Vérifie que le fichier contient au moins les 8 octets du header_size
    if file_size < HEADER_SIZE_BYTES as u64 {
        return Err(SafetensorsError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!(
                "fichier trop petit: {} octets, minimum {} octets",
                file_size, HEADER_SIZE_BYTES
            ),
        )));
    }

    // Lit le header_size (u64 little-endian)
    let mut header_size_buf = [0u8; HEADER_SIZE_BYTES];
    r.read_exact(&mut header_size_buf)
        .map_err(SafetensorsError::Io)?;
    let header_size = u64::from_le_bytes(header_size_buf);

    // Vérifie que header_size est dans les limites
    if header_size == 0 {
        return Err(SafetensorsError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "header_size est zéro",
        )));
    }
    if header_size > MAX_HEADER_SIZE {
        return Err(SafetensorsError::HeaderTooLarge {
            size: header_size,
            max: MAX_HEADER_SIZE,
        });
    }

    // Vérifie que le fichier contient le header complet
    let header_end = HEADER_SIZE_BYTES as u64 + header_size;
    if file_size < header_end {
        return Err(SafetensorsError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!(
                "header tronqué: fichier {} octets, header_size {} octets",
                file_size, header_size
            ),
        )));
    }

    // Lit le header JSON (y compris le padding)
    let mut header_buf = vec![0u8; header_size as usize];
    r.read_exact(&mut header_buf)
        .map_err(SafetensorsError::Io)?;

    // Convertit en string UTF-8 (le padding contient des espaces, donc valide)
    let header_str = String::from_utf8(header_buf).map_err(|e| {
        SafetensorsError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("header non UTF-8: {}", e),
        ))
    })?;

    // Parse le JSON en tant que mappe de valeurs brutes pour ignorer les champs non-tenseurs
    let raw_header: BTreeMap<String, serde_json::Value> = serde_json::from_str(&header_str)?;

    // Convertit uniquement les entrées qui sont des tenseurs (contiennent "dtype")
    let mut header = BTreeMap::new();
    for (key, value) in raw_header {
        // Ignore les métadonnées et autres champs non-tenseurs
        if let Ok(entry) = serde_json::from_value::<TensorHeaderEntry>(value) {
            header.insert(key, entry);
        }
    }

    // Calcule la taille du buffer (payload)
    let buffer_size = file_size.checked_sub(header_end).ok_or_else(|| {
        SafetensorsError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "taille du buffer négative",
        ))
    })?;

    // Crée le reader
    let reader = SafetensorsReader {
        header_size,
        header,
        buffer_size,
        file_size,
    };

    // Valide les offsets
    reader.validate_offsets()?;

    Ok(reader)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn make_valid_safetensors_file() -> Vec<u8> {
        // Header JSON minimal avec un tenseur F32 de 2x3 (24 octets)
        let header_json = r#"{"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        // Calcule le padding selon la spec : (8 - (len % 8)) % 8
        let json_len = header_json.len();
        let padding = (8 - (json_len % 8)) % 8;
        let padded_json = format!("{}{}", header_json, " ".repeat(padding));
        let header_size = padded_json.len() as u64;

        let mut file = Vec::new();
        file.extend_from_slice(&header_size.to_le_bytes());
        file.extend_from_slice(padded_json.as_bytes());
        // Ajoute 24 octets de données (payload)
        file.extend_from_slice(&[0u8; 24]);

        file
    }

    #[test]
    fn test_read_header_valid() {
        let data = make_valid_safetensors_file();
        let mut cursor = Cursor::new(data);
        let reader = read_header_from(&mut cursor).unwrap();

        // Calcule la taille attendue du header
        let header_json = r#"{"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let json_len = header_json.len();
        let padding = (8 - (json_len % 8)) % 8;
        let expected_header_size = (json_len + padding) as u64;

        assert_eq!(reader.header_size, expected_header_size);
        assert_eq!(reader.tensor_count(), 1);
        assert_eq!(reader.buffer_size, 24);
        assert_eq!(reader.file_size, 8 + expected_header_size + 24);

        let metadata = reader.metadata_only();
        assert_eq!(metadata.len(), 1);
        let (name, entry) = metadata[0];
        assert_eq!(name, "weight");
        assert_eq!(entry.dtype, super::super::types::DType::F32);
        assert_eq!(entry.shape, vec![2, 3]);
        assert_eq!(entry.data_offsets, [0, 24]);
    }

    #[test]
    fn test_read_header_too_small_file() {
        let data = vec![0u8; 4]; // Trop petit
        let mut cursor = Cursor::new(data);
        let result = read_header_from(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_header_zero_size() {
        let mut data = Vec::new();
        data.extend_from_slice(&0u64.to_le_bytes());
        let mut cursor = Cursor::new(data);
        let result = read_header_from(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_header_invalid_json() {
        let header_json = "invalid json";
        let json_len = header_json.len();
        let padding = (8 - (json_len % 8)) % 8;
        let padded_json = format!("{}{}", header_json, " ".repeat(padding));
        let header_size = padded_json.len() as u64;

        let mut file = Vec::new();
        file.extend_from_slice(&header_size.to_le_bytes());
        file.extend_from_slice(padded_json.as_bytes());
        file.extend_from_slice(&[0u8; 24]); // Payload fictif

        let mut cursor = Cursor::new(file);
        let result = read_header_from(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_header_invalid_offsets() {
        // Header avec offset end > buffer_size
        let header_json = r#"{"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,100]}}"#;
        let json_len = header_json.len();
        let padding = (8 - (json_len % 8)) % 8;
        let padded_json = format!("{}{}", header_json, " ".repeat(padding));
        let header_size = padded_json.len() as u64;

        let mut file = Vec::new();
        file.extend_from_slice(&header_size.to_le_bytes());
        file.extend_from_slice(padded_json.as_bytes());
        file.extend_from_slice(&[0u8; 24]); // Buffer de 24 octets seulement

        let mut cursor = Cursor::new(file);
        let result = read_header_from(&mut cursor);
        assert!(result.is_err());
    }
}
