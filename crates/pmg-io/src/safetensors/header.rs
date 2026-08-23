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

//! Construction du header JSON Safetensors.

use std::collections::BTreeMap;

use super::types::{SafetensorsError, SafetensorsResult, TensorHeaderEntry};

/// Taille maximale du header (8 MiB) comme spécifié dans l'architecture.
pub const MAX_HEADER_SIZE: u64 = 8 * 1024 * 1024;

/// Construit le header JSON Safetensors à partir des métadonnées des tenseurs.
///
/// # Paramètres
/// - `tensors` : mappe ordonnée (BTreeMap) des noms vers les entrées d'en-tête.
///
/// # Retour
/// Le contenu JSON du header, ou une erreur si le header dépasse la taille maximale.
pub fn build_header(tensors: &BTreeMap<String, TensorHeaderEntry>) -> SafetensorsResult<String> {
    // Sérialise en JSON compact
    let json = serde_json::to_string(tensors).map_err(SafetensorsError::Json)?;

    // Vérifie la taille maximale
    let size = json.len() as u64;
    if size > MAX_HEADER_SIZE {
        return Err(SafetensorsError::HeaderTooLarge {
            size,
            max: MAX_HEADER_SIZE,
        });
    }

    Ok(json)
}

/// Calcule la taille du header avec padding d'alignement à 8 octets.
///
/// Le format Safetensors exige que la taille déclarée (`header_size`) inclue
/// le padding d'alignement à 8 octets. Le contenu du padding doit être des
/// caractères d'espacement JSON valides (espaces).
pub fn header_size_with_padding(json_len: usize) -> u64 {
    let padding = (8 - (json_len % 8)) % 8;
    (json_len + padding) as u64
}

/// Calcule la réserve nécessaire pour le header.
///
/// Estimation : somme des tailles de nom + 128 octets par tenseur + marge.
pub fn estimate_header_reserve(tensor_names: &[&str]) -> u64 {
    let mut total = 0u64;
    for name in tensor_names {
        total += name.len() as u64 + 128;
    }
    total + 64 * 1024 // marge de 64 KiB
}

/// Applique le padding au JSON pour l'alignement à 8 octets.
///
/// Retourne le JSON avec des espaces ajoutés à la fin pour atteindre
/// une taille multiple de 8.
pub fn pad_header(json: &str) -> String {
    let len = json.len();
    let padding = (8 - (len % 8)) % 8;
    if padding == 0 {
        json.to_string()
    } else {
        format!("{}{}", json, " ".repeat(padding))
    }
}

#[cfg(test)]
mod tests {
    use super::super::types::DType;
    use super::*;
    use std::collections::BTreeMap;

    fn make_test_tensor_entry() -> TensorHeaderEntry {
        TensorHeaderEntry {
            dtype: DType::F32,
            shape: vec![2, 3],
            data_offsets: [0, 24],
        }
    }

    #[test]
    fn test_build_header_basic() {
        let mut tensors = BTreeMap::new();
        tensors.insert("weight".to_string(), make_test_tensor_entry());

        let json = build_header(&tensors).unwrap();
        assert!(json.contains("\"dtype\":\"F32\""));
        assert!(json.contains("\"shape\":[2,3]"));
        assert!(json.contains("\"data_offsets\":[0,24]"));
    }

    #[test]
    fn test_header_size_with_padding() {
        assert_eq!(header_size_with_padding(0), 0);
        assert_eq!(header_size_with_padding(1), 8);
        assert_eq!(header_size_with_padding(7), 8);
        assert_eq!(header_size_with_padding(8), 8);
        assert_eq!(header_size_with_padding(9), 16);
    }

    #[test]
    fn test_pad_header() {
        assert_eq!(pad_header(""), "");
        assert_eq!(pad_header("abc"), "abc     "); // 3 + 5 = 8
        assert_eq!(pad_header("12345678"), "12345678");
    }

    #[test]
    fn test_estimate_header_reserve() {
        let names = vec!["weight", "bias"];
        let reserve = estimate_header_reserve(&names);
        // 6 + 128 + 4 + 128 + 64*1024
        assert!(reserve > 64 * 1024);
    }
}
