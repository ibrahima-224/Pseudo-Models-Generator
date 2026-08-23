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

//! Statistiques physiques du modèle.
//!
//! Ce module calcule les statistiques physiques à partir des headers Safetensors :
//! mémoire totale (Memory ≈ Σ Size_i), taille brute/théorique, dtype, densité,
//! répartition par couche/shard.
//!
//! # Exemple
//!
//! ```rust
//! use pmg_inspect::physical_stats::compute_physical_stats;
//!
//! // Calcul des statistiques physiques (données fictives)
//! // let stats = compute_physical_stats(&headers, &structural);
//! // println!("Mémoire totale : {} octets", stats.total_memory_bytes);
//! ```

use std::collections::BTreeMap;
use std::collections::HashMap;

use crate::safetensors_inspector::SafetensorsHeader;
use crate::structural_stats::StructuralStats;
use pmg_core::dtype::DType;

/// Statistiques physiques d'un modèle.
#[derive(Debug, Clone, Default)]
pub struct PhysicalStats {
    /// Mémoire totale estimée en octets (somme des sizes des tenseurs).
    pub total_memory_bytes: u64,
    /// Taille brute totale des fichiers Safetensors en octets.
    pub total_file_size: u64,
    /// Taille théorique basée sur les shapes et dtypes.
    pub theoretical_size_bytes: u64,
    /// Type de données principal (le plus fréquent).
    pub primary_dtype: Option<DType>,
    /// Densité moyenne (données utiles / taille brute).
    pub average_density: f64,
    /// Répartition de la mémoire par couche.
    pub memory_per_layer: BTreeMap<u64, u64>,
    /// Répartition de la mémoire par shard.
    pub memory_per_shard: BTreeMap<String, u64>,
    /// Répartition de la mémoire par type de tenseur.
    pub memory_per_tensor_type: BTreeMap<String, u64>,
    /// Nombre total de paramètres (éléments).
    pub total_parameters: u64,
    /// Taille moyenne par paramètre en octets.
    pub average_bytes_per_parameter: f64,
}

impl std::fmt::Display for PhysicalStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Statistiques physiques :")?;
        writeln!(
            f,
            "  Mémoire totale : {} octets ({:.2} GiB)",
            self.total_memory_bytes,
            self.total_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        )?;
        writeln!(f, "  Taille brute : {} octets", self.total_file_size)?;
        writeln!(
            f,
            "  Taille théorique : {} octets",
            self.theoretical_size_bytes
        )?;
        if let Some(dtype) = self.primary_dtype {
            writeln!(f, "  Type principal : {:?}", dtype)?;
        }
        writeln!(
            f,
            "  Densité moyenne : {:.2}%",
            self.average_density * 100.0
        )?;
        writeln!(f, "  Paramètres : {}", self.total_parameters)?;
        writeln!(
            f,
            "  Octets/paramètre : {:.2}",
            self.average_bytes_per_parameter
        )?;
        writeln!(f)?;
        if !self.memory_per_layer.is_empty() {
            writeln!(f, "Mémoire par couche :")?;
            for (layer, bytes) in &self.memory_per_layer {
                writeln!(f, "  Couche {} : {} octets", layer, bytes)?;
            }
        }
        if !self.memory_per_shard.is_empty() {
            writeln!(f)?;
            writeln!(f, "Mémoire par shard :")?;
            for (shard, bytes) in &self.memory_per_shard {
                writeln!(f, "  {} : {} octets", shard, bytes)?;
            }
        }
        if !self.memory_per_tensor_type.is_empty() {
            writeln!(f)?;
            writeln!(f, "Mémoire par type de tenseur :")?;
            for (tensor_type, bytes) in &self.memory_per_tensor_type {
                writeln!(f, "  {} : {} octets", tensor_type, bytes)?;
            }
        }
        Ok(())
    }
}

/// Calcule les statistiques physiques à partir des headers et des stats structurelles.
///
/// # Paramètres
/// - `headers` : headers Safetensors extraits.
/// - `structural` : statistiques structurelles (peut être partiellement rempli).
pub fn compute_physical_stats(
    headers: &[SafetensorsHeader],
    structural: &StructuralStats,
) -> PhysicalStats {
    let mut stats = PhysicalStats::default();

    // Calcul des totaux de base
    for header in headers {
        stats.total_file_size += header.file_size;
        stats.memory_per_shard.insert(
            header
                .file_path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string(),
            header.total_bytes(),
        );

        for tensor in &header.tensors {
            let tensor_size = tensor.size_bytes();
            stats.total_memory_bytes += tensor_size;
            stats.theoretical_size_bytes += tensor_size;
            stats.total_parameters += tensor.num_elements();

            // Répartition par couche
            if let Some(layer_num) = extract_layer_number(&tensor.name) {
                *stats.memory_per_layer.entry(layer_num).or_insert(0) += tensor_size;
            }

            // Répartition par type de tenseur
            let tensor_type = classify_tensor_type(&tensor.name);
            *stats.memory_per_tensor_type.entry(tensor_type).or_insert(0) += tensor_size;
        }
    }

    // Calcul de la densité moyenne
    if stats.total_file_size > 0 {
        stats.average_density = stats.total_memory_bytes as f64 / stats.total_file_size as f64;
    }

    // Calcul de la taille moyenne par paramètre
    if stats.total_parameters > 0 {
        stats.average_bytes_per_parameter =
            stats.total_memory_bytes as f64 / stats.total_parameters as f64;
    }

    // Détermination du dtype principal
    let mut dtype_counts = HashMap::new();
    for header in headers {
        for tensor in &header.tensors {
            *dtype_counts.entry(tensor.dtype).or_insert(0) += 1;
        }
    }
    if let Some((dtype, _)) = dtype_counts.iter().max_by_key(|(_, &count)| count) {
        stats.primary_dtype = Some(*dtype);
    }

    // Utilisation des stats structurelles si disponibles
    if structural.total_tensors > 0 {
        // Les stats structurelles peuvent fournir des informations complémentaires
        // mais nous utilisons principalement les headers pour les calculs physiques
    }

    stats
}

/// Extrait le numéro de couche à partir du nom d'un tenseur.
fn extract_layer_number(name: &str) -> Option<u64> {
    let patterns = ["layers.", "layer."];
    for pattern in &patterns {
        if let Some(pos) = name.find(pattern) {
            let start = pos + pattern.len();
            let remaining = &name[start..];
            let num_str: String = remaining
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            if let Ok(num) = num_str.parse::<u64>() {
                return Some(num);
            }
        }
    }
    None
}

/// Classifie le type de tenseur à partir de son nom.
fn classify_tensor_type(name: &str) -> String {
    if name.contains("q_proj") || name.contains("k_proj") || name.contains("v_proj") {
        "attention_qkv".to_string()
    } else if name.contains("o_proj") {
        "attention_output".to_string()
    } else if name.contains("gate_proj") {
        "mlp_gate".to_string()
    } else if name.contains("up_proj") {
        "mlp_up".to_string()
    } else if name.contains("down_proj") {
        "mlp_down".to_string()
    } else if name.contains("embed_tokens") {
        "embedding".to_string()
    } else if name.contains("lm_head") {
        "lm_head".to_string()
    } else if name.contains("norm") || name.contains("layernorm") || name.contains("rmsnorm") {
        "normalization".to_string()
    } else if name.contains("bias") {
        "bias".to_string()
    } else {
        "other".to_string()
    }
}

/// Calcule la distribution des dtypes dans le modèle.
pub fn compute_dtype_distribution(headers: &[SafetensorsHeader]) -> HashMap<DType, u64> {
    let mut distribution = HashMap::new();
    for header in headers {
        for tensor in &header.tensors {
            *distribution.entry(tensor.dtype).or_insert(0) += tensor.num_elements();
        }
    }
    distribution
}

/// Calcule la taille totale par dtype.
pub fn compute_memory_by_dtype(headers: &[SafetensorsHeader]) -> HashMap<DType, u64> {
    let mut memory_by_dtype = HashMap::new();
    for header in headers {
        for tensor in &header.tensors {
            let size = tensor.size_bytes();
            *memory_by_dtype.entry(tensor.dtype).or_insert(0) += size;
        }
    }
    memory_by_dtype
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
    fn test_compute_physical_stats() {
        let headers = vec![
            create_test_header(
                "shard1.safetensors",
                &[
                    "model.layers.0.self_attn.q_proj.weight",
                    "model.layers.0.mlp.gate_proj.weight",
                ],
            ),
            create_test_header(
                "shard2.safetensors",
                &["model.embed_tokens.weight", "model.norm.weight"],
            ),
        ];

        let structural = StructuralStats::default();
        let stats = compute_physical_stats(&headers, &structural);

        assert_eq!(stats.total_memory_bytes, 4 * 40000); // 4 tenseurs × 40000 octets
                                                         // total_file_size = 2 shards × (2 tenseurs × 40000 + 100 header) = 2 × 80100 = 160200
        assert_eq!(stats.total_file_size, 160200);
        assert_eq!(stats.total_parameters, 4 * 10000); // 4 tenseurs × 10000 éléments
        assert_eq!(stats.primary_dtype, Some(DType::F32));
        assert!(stats.average_density > 0.0 && stats.average_density <= 1.0);
        assert!(stats.average_bytes_per_parameter > 0.0);
        assert_eq!(stats.memory_per_shard.len(), 2);
    }

    #[test]
    fn test_compute_dtype_distribution() {
        let headers = vec![create_test_header(
            "shard1.safetensors",
            &["tensor1", "tensor2"],
        )];

        let distribution = compute_dtype_distribution(&headers);
        assert_eq!(distribution.get(&DType::F32), Some(&20000)); // 2 × 10000 éléments
    }

    #[test]
    fn test_compute_memory_by_dtype() {
        let headers = vec![create_test_header(
            "shard1.safetensors",
            &["tensor1", "tensor2"],
        )];

        let memory_by_dtype = compute_memory_by_dtype(&headers);
        assert_eq!(memory_by_dtype.get(&DType::F32), Some(&80000)); // 2 × 40000 octets
    }

    #[test]
    fn test_classify_tensor_type() {
        assert_eq!(
            classify_tensor_type("model.layers.0.self_attn.q_proj.weight"),
            "attention_qkv"
        );
        assert_eq!(
            classify_tensor_type("model.layers.0.mlp.gate_proj.weight"),
            "mlp_gate"
        );
        assert_eq!(
            classify_tensor_type("model.embed_tokens.weight"),
            "embedding"
        );
        assert_eq!(classify_tensor_type("model.norm.weight"), "normalization");
        // Note: q_proj.bias est classé comme attention_qkv car le nom contient q_proj
        // avant d'être vérifié pour bias. C'est le comportement attendu car ce tenseur
        // est lié à l'attention.
        assert_eq!(
            classify_tensor_type("model.layers.0.self_attn.q_proj.bias"),
            "attention_qkv"
        );
        // gate_proj.bias est classé comme mlp_gate car gate_proj est vérifié avant bias
        assert_eq!(
            classify_tensor_type("model.layers.0.mlp.gate_proj.bias"),
            "mlp_gate"
        );
        // Un bias pur (sans q_proj/k_proj/v_proj/gate_proj) est bien classé comme bias
        assert_eq!(classify_tensor_type("model.layers.0.bias"), "bias");
    }
}
