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

//! Inspection des headers Safetensors.
//!
//! Ce module lit les en-têtes des fichiers Safetensors **sans charger les données
//! de poids**, conformément au principe Zero-Payload. Il extrait les métadonnées
//! de chaque tenseur (nom, dtype, shape, data_offsets) et calcule la taille N
//! et l'estimation Size.
//!
//! # Format Safetensors
//!
//! Un fichier Safetensors commence par :
//! - 8 octets : longueur du header JSON (little-endian)
//! - Header JSON : dictionnaire des tenseurs avec leurs métadonnées
//!
//! Le header JSON contient pour chaque tenseur :
//! - `dtype` : type de données (ex: "B16", "F32")
//! - `shape` : tableau des dimensions
//! - `data_offsets` : [offset_début, offset_fin] en octets
//!
//! # Exemple
//!
//! ```rust
//! use pmg_inspect::safetensors_inspector::inspect_safetensors_headers;
//!
//! // Inspection d'un fichier Safetensors (chemin fictif)
//! // let headers = inspect_safetensors_headers("path/to/model").unwrap();
//! // for header in headers {
//! //     println!("Shard : {}, Tenseurs : {}", header.file_path.display(), header.tensor_count());
//! // }
//! ```

use std::path::{Path, PathBuf};

use pmg_core::dtype::DType;
use pmg_core::shape::Shape;

use crate::InspectError;

/// Métadonnées d'un tenseur extraites du header Safetensors.
#[derive(Debug, Clone)]
pub struct TensorHeader {
    /// Nom complet du tenseur (ex: "model.layers.0.self_attn.q_proj.weight").
    pub name: String,
    /// Type de données.
    pub dtype: DType,
    /// Shape (dimensions) du tenseur.
    pub shape: Shape,
    /// Offsets de données [début, fin] en octets dans le fichier.
    pub data_offsets: [u64; 2],
}

impl TensorHeader {
    /// Calcule le nombre d'éléments N du tenseur (produit des dimensions).
    pub fn num_elements(&self) -> u64 {
        self.shape.num_elements().unwrap_or(0)
    }

    /// Calcule la taille en octets du tenseur (N × taille du dtype).
    pub fn size_bytes(&self) -> u64 {
        let element_size = self.dtype.size_bytes().unwrap_or(0);
        self.num_elements() * element_size
    }

    /// Vérifie la cohérence entre data_offsets et size_bytes.
    pub fn is_consistent(&self) -> bool {
        let declared_size = self.data_offsets[1] - self.data_offsets[0];
        declared_size == self.size_bytes()
    }
}

impl std::fmt::Display for TensorHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} : {:?} {:?} ({} éléments, {} octets)",
            self.name,
            self.dtype,
            self.shape,
            self.num_elements(),
            self.size_bytes()
        )
    }
}

/// Header complet d'un fichier Safetensors.
#[derive(Debug, Clone)]
pub struct SafetensorsHeader {
    /// Chemin vers le fichier Safetensors.
    pub file_path: PathBuf,
    /// Tenseurs contenus dans ce fichier.
    pub tensors: Vec<TensorHeader>,
    /// Taille totale du fichier en octets.
    pub file_size: u64,
    /// Taille du header JSON en octets.
    pub header_size: u64,
}

impl SafetensorsHeader {
    /// Nombre de tenseurs dans ce fichier.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Taille totale des données de poids en octets.
    pub fn total_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.size_bytes()).sum()
    }

    /// Densité du fichier (données utiles / taille totale).
    pub fn density(&self) -> f64 {
        if self.file_size == 0 {
            0.0
        } else {
            self.total_bytes() as f64 / self.file_size as f64
        }
    }

    /// Vérifie que tous les tenseurs sont cohérents.
    pub fn validate(&self) -> bool {
        self.tensors.iter().all(|t| t.is_consistent())
    }
}

impl std::fmt::Display for SafetensorsHeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Fichier : {}", self.file_path.display())?;
        writeln!(f, "Tenseurs : {}", self.tensor_count())?;
        writeln!(f, "Taille fichier : {} octets", self.file_size)?;
        writeln!(f, "Taille header : {} octets", self.header_size)?;
        writeln!(f, "Taille données : {} octets", self.total_bytes())?;
        writeln!(f, "Densité : {:.2}%", self.density() * 100.0)?;
        if self.validate() {
            writeln!(f, "Validation : ✓ cohérent")?;
        } else {
            writeln!(f, "Validation : ✗ incohérent")?;
        }
        Ok(())
    }
}

/// Inspecte les headers Safetensors de tous les fichiers .safetensors
/// dans le répertoire du modèle.
///
/// # Paramètres
/// - `model_path` : chemin vers le répertoire contenant le modèle.
///
/// # Erreurs
/// Retourne une erreur si un fichier est invalide ou illisible.
pub fn inspect_safetensors_headers(
    model_path: &Path,
) -> Result<Vec<SafetensorsHeader>, InspectError> {
    let mut headers = Vec::new();

    // Liste tous les fichiers .safetensors dans le répertoire
    let entries =
        std::fs::read_dir(model_path).map_err(|e| InspectError::Io(e, model_path.to_path_buf()))?;

    for entry in entries {
        let entry = entry.map_err(|e| InspectError::Io(e, model_path.to_path_buf()))?;
        let path = entry.path();

        // Ne traiter que les fichiers .safetensors
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            let header = inspect_single_safetensors(&path)?;
            headers.push(header);
        }
    }

    // Tri par nom de fichier pour une présentation cohérente
    headers.sort_by(|a, b| a.file_path.cmp(&b.file_path));

    Ok(headers)
}

/// Inspecte le header d'un fichier Safetensors unique.
fn inspect_single_safetensors(file_path: &Path) -> Result<SafetensorsHeader, InspectError> {
    // Lecture de la taille du fichier
    let metadata =
        std::fs::metadata(file_path).map_err(|e| InspectError::Io(e, file_path.to_path_buf()))?;
    let file_size = metadata.len();

    if file_size < 8 {
        return Err(InspectError::InvalidSafetensors(
            file_path.to_path_buf(),
            "Fichier trop petit pour contenir un header".to_string(),
        ));
    }

    // Ouverture du fichier en lecture seule (jamais en écriture)
    let mut file =
        std::fs::File::open(file_path).map_err(|e| InspectError::Io(e, file_path.to_path_buf()))?;

    use std::io::{Read, Seek, SeekFrom};

    // Lecture de la longueur du header (8 octets, little-endian)
    let mut header_len_bytes = [0u8; 8];
    file.read_exact(&mut header_len_bytes)
        .map_err(|e| InspectError::Io(e, file_path.to_path_buf()))?;
    let header_len = u64::from_le_bytes(header_len_bytes);

    // Vérification que le header ne dépasse pas le fichier
    if header_len + 8 > file_size {
        return Err(InspectError::InvalidSafetensors(
            file_path.to_path_buf(),
            format!("Header trop grand : {} + 8 > {}", header_len, file_size),
        ));
    }

    // Lecture du header JSON
    let mut header_json = vec![0u8; header_len as usize];
    file.seek(SeekFrom::Start(8))
        .map_err(|e| InspectError::Io(e, file_path.to_path_buf()))?;
    file.read_exact(&mut header_json)
        .map_err(|e| InspectError::Io(e, file_path.to_path_buf()))?;

    // Parsing du JSON
    let json: serde_json::Value = serde_json::from_slice(&header_json)
        .map_err(|e| InspectError::Json(e, file_path.to_path_buf()))?;

    // Extraction des métadonnées de chaque tenseur
    let mut tensors = Vec::new();

    if let Some(obj) = json.as_object() {
        for (name, tensor_info) in obj {
            // Ignorer la clé "__metadata__" si présente
            if name == "__metadata__" {
                continue;
            }

            // Extraction du dtype
            let dtype_str = tensor_info
                .get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    InspectError::InvalidSafetensors(
                        file_path.to_path_buf(),
                        format!("Tenseur '{}' : dtype manquant", name),
                    )
                })?;

            let dtype = parse_dtype(dtype_str).ok_or_else(|| {
                InspectError::InvalidSafetensors(
                    file_path.to_path_buf(),
                    format!("Tenseur '{}' : dtype inconnu '{}'", name, dtype_str),
                )
            })?;

            // Extraction de la shape
            let shape_array = tensor_info
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    InspectError::InvalidSafetensors(
                        file_path.to_path_buf(),
                        format!("Tenseur '{}' : shape manquante", name),
                    )
                })?;

            let dims: Vec<u64> = shape_array
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    v.as_u64().ok_or_else(|| {
                        InspectError::InvalidSafetensors(
                            file_path.to_path_buf(),
                            format!("Tenseur '{}' : dimension {} invalide", name, i),
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;

            let shape = Shape::new(dims).map_err(|e| {
                InspectError::InvalidSafetensors(
                    file_path.to_path_buf(),
                    format!("Tenseur '{}' : shape invalide : {}", name, e),
                )
            })?;

            // Extraction des data_offsets
            let offsets_array = tensor_info
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    InspectError::InvalidSafetensors(
                        file_path.to_path_buf(),
                        format!("Tenseur '{}' : data_offsets manquants", name),
                    )
                })?;

            if offsets_array.len() != 2 {
                return Err(InspectError::InvalidSafetensors(
                    file_path.to_path_buf(),
                    format!(
                        "Tenseur '{}' : data_offsets doit avoir 2 éléments, obtenu {}",
                        name,
                        offsets_array.len()
                    ),
                ));
            }

            let offsets: [u64; 2] = [
                offsets_array[0].as_u64().ok_or_else(|| {
                    InspectError::InvalidSafetensors(
                        file_path.to_path_buf(),
                        format!("Tenseur '{}' : offset début invalide", name),
                    )
                })?,
                offsets_array[1].as_u64().ok_or_else(|| {
                    InspectError::InvalidSafetensors(
                        file_path.to_path_buf(),
                        format!("Tenseur '{}' : offset fin invalide", name),
                    )
                })?,
            ];

            // Vérification de la cohérence des offsets
            if offsets[0] >= offsets[1] {
                return Err(InspectError::InvalidSafetensors(
                    file_path.to_path_buf(),
                    format!(
                        "Tenseur '{}' : offset début ({}) >= offset fin ({})",
                        name, offsets[0], offsets[1]
                    ),
                ));
            }

            tensors.push(TensorHeader {
                name: name.clone(),
                dtype,
                shape,
                data_offsets: offsets,
            });
        }
    }

    // Tri des tenseurs par nom pour une présentation cohérente
    tensors.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(SafetensorsHeader {
        file_path: file_path.to_path_buf(),
        tensors,
        file_size,
        header_size: header_len + 8,
    })
}

/// Inspecte le header d'un fichier Safetensors unique (fonction publique).
///
/// # Paramètres
/// - `file_path` : chemin vers le fichier .safetensors.
///
/// # Retourne
/// Le header du fichier ou une erreur si le fichier est invalide ou illisible.
pub fn inspect_single_safetensors_file(
    file_path: &Path,
) -> Result<SafetensorsHeader, InspectError> {
    inspect_single_safetensors(file_path)
}

/// Parse un dtype Safetensors en DType PMG.
///
/// Cette fonction est insensible à la casse pour une compatibilité maximale
/// avec les différents générateurs de fichiers Safetensors.
fn parse_dtype(s: &str) -> Option<DType> {
    // Conversion en majuscules pour une comparaison insensible à la casse
    let upper = s.to_uppercase();
    match upper.as_str() {
        "BOOL" => Some(DType::Bool),
        "I8" => Some(DType::I8),
        "I16" => Some(DType::I16),
        "I32" => Some(DType::I32),
        "I64" => Some(DType::I64),
        "U8" => Some(DType::U8),
        "U16" => Some(DType::U16),
        "U32" => Some(DType::U32),
        "U64" => Some(DType::U64),
        "F16" => Some(DType::F16),
        "BF16" => Some(DType::Bf16),
        "F32" => Some(DType::F32),
        "F64" => Some(DType::F64),
        "F8_E4M3" | "F8_E4M3FN" => Some(DType::F8E4M3),
        "F8_E5M2" => Some(DType::F8E5M2),
        "F8_E8M0" | "F8_E8M0FNU" => Some(DType::F8E8M0),
        "F6E2M3" => Some(DType::F6E2M3),
        "F6E3M2" => Some(DType::F6E3M2),
        "F4" => Some(DType::F4),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_header_calculations() {
        let shape = Shape::new(vec![1024, 1024]).unwrap();
        let header = TensorHeader {
            name: "test.weight".to_string(),
            dtype: DType::F32,
            shape,
            data_offsets: [0, 1024 * 1024 * 4],
        };

        assert_eq!(header.num_elements(), 1024 * 1024);
        assert_eq!(header.size_bytes(), 1024 * 1024 * 4);
        assert!(header.is_consistent());
    }

    #[test]
    fn test_tensor_header_inconsistent() {
        let shape = Shape::new(vec![100, 100]).unwrap();
        let header = TensorHeader {
            name: "test.weight".to_string(),
            dtype: DType::F32,
            shape,
            data_offsets: [0, 100 * 100 * 2], // Mauvais dtype : F32 = 4 octets, pas 2
        };

        assert_eq!(header.num_elements(), 10000);
        assert_eq!(header.size_bytes(), 40000);
        assert!(!header.is_consistent());
    }

    #[test]
    fn test_safetensors_header_density() {
        let header = SafetensorsHeader {
            file_path: PathBuf::from("test.safetensors"),
            tensors: Vec::new(),
            file_size: 1000,
            header_size: 100,
        };

        assert_eq!(header.tensor_count(), 0);
        assert_eq!(header.total_bytes(), 0);
        assert_eq!(header.density(), 0.0);
    }

    #[test]
    fn test_parse_dtype() {
        assert_eq!(parse_dtype("F32"), Some(DType::F32));
        assert_eq!(parse_dtype("BF16"), Some(DType::Bf16));
        assert_eq!(parse_dtype("I8"), Some(DType::I8));
        assert_eq!(parse_dtype("UNKNOWN"), None);
    }

    #[test]
    fn test_inspect_missing_directory() {
        let temp_dir = std::env::temp_dir().join("pmg_test_safetensors_missing");
        let result = inspect_safetensors_headers(&temp_dir);
        assert!(result.is_err());
    }

    #[test]
    fn test_inspect_invalid_file() {
        let temp_dir = std::env::temp_dir().join("pmg_test_safetensors_invalid");
        std::fs::create_dir_all(&temp_dir).unwrap();
        let file_path = temp_dir.join("test.safetensors");
        std::fs::write(&file_path, "invalid content").unwrap();

        let result = inspect_safetensors_headers(&temp_dir);
        assert!(result.is_err());

        // Nettoyage
        std::fs::remove_file(file_path).unwrap();
        std::fs::remove_dir(temp_dir).unwrap();
    }
}
