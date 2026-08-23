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

//! Construction du mapping tensor → fichier shard.
//!
//! Ce module construit l'index qui associe chaque tenseur au fichier shard
//! qui le contient, en se basant sur les headers Safetensors extraits.
//!
//! # Exemple
//!
//! ```rust
//! use pmg_inspect::index_inspector::build_shard_index;
//!
//! // Construction de l'index (chemin fictif)
//! // let index = build_shard_index("path/to/model", &headers).unwrap();
//! // let shard = index.shard_for_tensor("model.layers.0.weight");
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::safetensors_inspector::SafetensorsHeader;
use crate::InspectError;

/// Index des shards : mapping tensor → fichier shard.
#[derive(Debug, Clone)]
pub struct ShardIndex {
    /// Mapping nom de tenseur → chemin du shard.
    tensor_to_shard: BTreeMap<String, PathBuf>,
    /// Mapping chemin du shard → liste des noms de tenseurs.
    shard_to_tensors: BTreeMap<PathBuf, Vec<String>>,
    /// Nombre total de tenseurs indexés.
    total_tensors: usize,
}

impl ShardIndex {
    /// Crée un nouvel index vide.
    pub fn new() -> Self {
        Self {
            tensor_to_shard: BTreeMap::new(),
            shard_to_tensors: BTreeMap::new(),
            total_tensors: 0,
        }
    }

    /// Ajoute un tenseur à l'index.
    pub fn add_tensor(&mut self, tensor_name: &str, shard_path: &Path) {
        self.tensor_to_shard
            .insert(tensor_name.to_string(), shard_path.to_path_buf());
        self.shard_to_tensors
            .entry(shard_path.to_path_buf())
            .or_default()
            .push(tensor_name.to_string());
        self.total_tensors += 1;
    }

    /// Retourne le chemin du shard pour un tenseur donné.
    pub fn shard_for_tensor(&self, tensor_name: &str) -> Option<&Path> {
        self.tensor_to_shard.get(tensor_name).map(|p| p.as_path())
    }

    /// Retourne la liste des tenseurs dans un shard donné.
    pub fn tensors_in_shard(&self, shard_path: &Path) -> Option<&[String]> {
        self.shard_to_tensors.get(shard_path).map(|v| v.as_slice())
    }

    /// Nombre total de tenseurs dans l'index.
    pub fn total_tensors(&self) -> usize {
        self.total_tensors
    }

    /// Nombre de shards dans l'index.
    pub fn shard_count(&self) -> usize {
        self.shard_to_tensors.len()
    }

    /// Liste de tous les noms de tenseurs (triés).
    pub fn all_tensor_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tensor_to_shard.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    /// Liste de tous les chemins de shards (triés).
    pub fn all_shard_paths(&self) -> Vec<&Path> {
        let mut paths: Vec<&Path> = self.shard_to_tensors.keys().map(|p| p.as_path()).collect();
        paths.sort();
        paths
    }

    /// Vérifie que l'est cohérent (chaque tenseur a un shard).
    pub fn validate(&self) -> bool {
        // Vérification bidirectionnelle
        for (tensor, shard) in &self.tensor_to_shard {
            if let Some(tensors) = self.shard_to_tensors.get(shard) {
                if !tensors.contains(tensor) {
                    return false;
                }
            } else {
                return false;
            }
        }
        true
    }
}

impl Default for ShardIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ShardIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Index des shards :")?;
        writeln!(f, "  Shards : {}", self.shard_count())?;
        writeln!(f, "  Tenseurs : {}", self.total_tensors())?;
        writeln!(f)?;
        for shard_path in self.all_shard_paths() {
            writeln!(f, "Shard : {}", shard_path.display())?;
            if let Some(tensors) = self.tensors_in_shard(shard_path) {
                for tensor in tensors {
                    writeln!(f, "  - {}", tensor)?;
                }
            }
        }
        Ok(())
    }
}

/// Construit l'index des shards à partir des headers Safetensors.
///
/// # Paramètres
/// - `model_path` : chemin vers le répertoire contenant le modèle.
/// - `headers` : headers Safetensors extraits.
///
/// # Retour
/// Un `ShardIndex` contenant le mapping tensor → shard.
pub fn build_shard_index(
    _model_path: &Path,
    headers: &[SafetensorsHeader],
) -> Result<ShardIndex, InspectError> {
    let mut index = ShardIndex::new();

    for header in headers {
        for tensor in &header.tensors {
            index.add_tensor(&tensor.name, &header.file_path);
        }
    }

    // Vérification de cohérence
    if !index.validate() {
        return Err(InspectError::InvalidIndex(
            "Index incohérent : mapping bidirectionnel invalide".to_string(),
        ));
    }

    Ok(index)
}

/// Statistiques sur un shard individuel.
#[derive(Debug, Clone)]
pub struct ShardStats {
    /// Chemin du shard.
    pub path: PathBuf,
    /// Nombre de tenseurs.
    pub tensor_count: usize,
    /// Taille totale des données en octets.
    pub total_bytes: u64,
    /// Types de données présents.
    pub dtypes: Vec<pmg_core::DType>,
}

impl std::fmt::Display for ShardStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Shard : {}", self.path.display())?;
        writeln!(f, "  Tenseurs : {}", self.tensor_count)?;
        writeln!(f, "  Taille : {} octets", self.total_bytes)?;
        writeln!(f, "  dtypes : {:?}", self.dtypes)?;
        Ok(())
    }
}

/// Calcule les statistiques pour chaque shard.
pub fn compute_shard_stats(headers: &[SafetensorsHeader]) -> Vec<ShardStats> {
    headers
        .iter()
        .map(|header| {
            let mut dtypes = Vec::new();
            for tensor in &header.tensors {
                if !dtypes.contains(&tensor.dtype) {
                    dtypes.push(tensor.dtype);
                }
            }
            ShardStats {
                path: header.file_path.clone(),
                tensor_count: header.tensor_count(),
                total_bytes: header.total_bytes(),
                dtypes,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors_inspector::TensorHeader;
    use pmg_core::dtype::DType;
    use pmg_core::shape::Shape;
    use std::path::PathBuf;

    fn create_test_header(file_path: &str, tensor_names: &[&str]) -> SafetensorsHeader {
        let tensors: Vec<TensorHeader> = tensor_names
            .iter()
            .enumerate()
            .map(|(i, name)| TensorHeader {
                name: name.to_string(),
                dtype: DType::F32,
                shape: Shape::new(vec![100, 100]).unwrap(),
                data_offsets: [i as u64 * 40000, (i + 1) as u64 * 40000],
            })
            .collect();

        SafetensorsHeader {
            file_path: PathBuf::from(file_path),
            tensors,
            file_size: tensor_names.len() as u64 * 40000 + 100,
            header_size: 100,
        }
    }

    #[test]
    fn test_shard_index_creation() {
        let mut index = ShardIndex::new();
        index.add_tensor("tensor1", Path::new("shard1.safetensors"));
        index.add_tensor("tensor2", Path::new("shard1.safetensors"));
        index.add_tensor("tensor3", Path::new("shard2.safetensors"));

        assert_eq!(index.total_tensors(), 3);
        assert_eq!(index.shard_count(), 2);
        assert!(index.validate());
    }

    #[test]
    fn test_shard_for_tensor() {
        let mut index = ShardIndex::new();
        index.add_tensor("tensor1", Path::new("shard1.safetensors"));
        index.add_tensor("tensor2", Path::new("shard2.safetensors"));

        assert_eq!(
            index.shard_for_tensor("tensor1"),
            Some(Path::new("shard1.safetensors"))
        );
        assert_eq!(
            index.shard_for_tensor("tensor2"),
            Some(Path::new("shard2.safetensors"))
        );
        assert_eq!(index.shard_for_tensor("tensor3"), None);
    }

    #[test]
    fn test_tensors_in_shard() {
        let mut index = ShardIndex::new();
        index.add_tensor("tensor1", Path::new("shard1.safetensors"));
        index.add_tensor("tensor2", Path::new("shard1.safetensors"));
        index.add_tensor("tensor3", Path::new("shard2.safetensors"));

        let tensors_shard1 = index
            .tensors_in_shard(Path::new("shard1.safetensors"))
            .unwrap();
        assert_eq!(tensors_shard1.len(), 2);
        assert!(tensors_shard1.contains(&"tensor1".to_string()));
        assert!(tensors_shard1.contains(&"tensor2".to_string()));

        let tensors_shard2 = index
            .tensors_in_shard(Path::new("shard2.safetensors"))
            .unwrap();
        assert_eq!(tensors_shard2.len(), 1);
        assert!(tensors_shard2.contains(&"tensor3".to_string()));
    }

    #[test]
    fn test_build_shard_index() {
        let headers = vec![
            create_test_header("model-00001.safetensors", &["tensor1", "tensor2"]),
            create_test_header("model-00002.safetensors", &["tensor3", "tensor4"]),
        ];

        let index = build_shard_index(Path::new("/fake/model"), &headers).unwrap();
        assert_eq!(index.total_tensors(), 4);
        assert_eq!(index.shard_count(), 2);
        assert!(index.validate());
    }

    #[test]
    fn test_compute_shard_stats() {
        let headers = vec![
            create_test_header("shard1.safetensors", &["tensor1", "tensor2"]),
            create_test_header("shard2.safetensors", &["tensor3"]),
        ];

        let stats = compute_shard_stats(&headers);
        assert_eq!(stats.len(), 2);
        assert_eq!(stats[0].tensor_count, 2);
        assert_eq!(stats[1].tensor_count, 1);
    }
}
