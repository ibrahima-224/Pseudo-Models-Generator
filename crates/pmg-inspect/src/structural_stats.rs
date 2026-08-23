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

//! Statistiques structurelles du modèle.
//!
//! Ce module calcule les statistiques structurelles à partir des headers
//! Safetensors et de la configuration : nombre de tenseurs, couches, shards,
//! experts, dimensions, paramètres théoriques.
//!
//! # Exemple
//!
//! ```rust
//! use pmg_inspect::structural_stats::compute_structural_stats;
//!
//! // Calcul des statistiques (données fictives)
//! // let stats = compute_structural_stats(&config, &headers, &index);
//! // println!("Nombre de couches : {}", stats.num_layers);
//! ```

use std::collections::{BTreeMap, BTreeSet};

use crate::config_inspector::ConfigInspection;
use crate::index_inspector::ShardIndex;
use crate::safetensors_inspector::SafetensorsHeader;
use pmg_core::dtype::DType;

/// Statistiques structurelles d'un modèle.
#[derive(Debug, Clone, Default)]
pub struct StructuralStats {
    /// Nombre total de tenseurs.
    pub total_tensors: u64,
    /// Nombre de couches détectées (à partir des noms de tenseurs).
    pub num_layers: u64,
    /// Nombre de shards (fichiers Safetensors).
    pub num_shards: u64,
    /// Nombre d'experts MoE détectés.
    pub num_experts: u64,
    /// Dimensions uniques détectées (hidden_size, intermediate_size, etc.).
    pub dimensions: BTreeMap<String, u64>,
    /// Types de données uniques présents.
    pub dtypes: Vec<DType>,
    /// Nombre de paramètres théoriques (somme des N × taille_dtype).
    pub total_parameters: u64,
    /// Nombre total d'éléments (somme des N).
    pub total_elements: u64,
    /// Répartition des tenseurs par couche.
    pub tensors_per_layer: BTreeMap<u64, u64>,
    /// Répartition des tenseurs par type de rôle (attention, MLP, etc.).
    pub tensors_by_role: BTreeMap<String, u64>,
}

impl std::fmt::Display for StructuralStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Statistiques structurelles :")?;
        writeln!(f, "  Tenseurs totaux : {}", self.total_tensors)?;
        writeln!(f, "  Couches : {}", self.num_layers)?;
        writeln!(f, "  Shards : {}", self.num_shards)?;
        writeln!(f, "  Experts : {}", self.num_experts)?;
        writeln!(f, "  Paramètres : {}", self.total_parameters)?;
        writeln!(f, "  Éléments totaux : {}", self.total_elements)?;
        writeln!(f)?;
        writeln!(f, "Dimensions :")?;
        for (name, value) in &self.dimensions {
            writeln!(f, "  {} : {}", name, value)?;
        }
        writeln!(f)?;
        writeln!(f, "Types de données : {:?}", self.dtypes)?;
        if !self.tensors_per_layer.is_empty() {
            writeln!(f)?;
            writeln!(f, "Répartition par couche :")?;
            for (layer, count) in &self.tensors_per_layer {
                writeln!(f, "  Couche {} : {} tenseurs", layer, count)?;
            }
        }
        if !self.tensors_by_role.is_empty() {
            writeln!(f)?;
            writeln!(f, "Répartition par rôle :")?;
            for (role, count) in &self.tensors_by_role {
                writeln!(f, "  {} : {} tenseurs", role, count)?;
            }
        }
        Ok(())
    }
}

/// Calcule les statistiques structurelles à partir des données d'inspection.
///
/// # Paramètres
/// - `config` : inspection de la configuration (peut être None).
/// - `headers` : headers Safetensors extraits.
/// - `index` : index des shards (peut être None).
pub fn compute_structural_stats(
    config: &Option<ConfigInspection>,
    headers: &[SafetensorsHeader],
    _index: &Option<ShardIndex>,
) -> StructuralStats {
    let mut stats = StructuralStats {
        num_shards: headers.len() as u64,
        ..Default::default()
    };

    // Extraction des informations à partir des headers
    let mut layer_numbers = BTreeSet::new();
    let mut expert_numbers = BTreeSet::new();
    let mut dimensions = BTreeMap::new();
    let mut dtypes = Vec::new();
    let mut total_parameters = 0u64;
    let mut total_elements = 0u64;
    let mut tensors_per_layer = BTreeMap::new();
    let mut tensors_by_role = BTreeMap::new();

    for header in headers {
        for tensor in &header.tensors {
            stats.total_tensors += 1;

            // Extraction du numéro de couche à partir du nom
            if let Some(layer_num) = extract_layer_number(&tensor.name) {
                layer_numbers.insert(layer_num);
                *tensors_per_layer.entry(layer_num).or_insert(0) += 1;
            }

            // Extraction du numéro d'expert
            if let Some(expert_num) = extract_expert_number(&tensor.name) {
                expert_numbers.insert(expert_num);
            }

            // Extraction des dimensions
            extract_dimensions(&tensor.name, &tensor.shape, &mut dimensions);

            // Collecte des dtypes (uniques)
            if !dtypes.contains(&tensor.dtype) {
                dtypes.push(tensor.dtype);
            }

            // Calcul des paramètres
            let num_elements = tensor.num_elements();
            total_elements += num_elements;
            total_parameters += num_elements;

            // Classification par rôle
            let role = classify_tensor_role(&tensor.name);
            *tensors_by_role.entry(role).or_insert(0) += 1;
        }
    }

    // Mise à jour des statistiques
    stats.num_layers = layer_numbers.len() as u64;
    stats.num_experts = expert_numbers.len() as u64;
    stats.dimensions = dimensions;
    stats.dtypes = dtypes;
    stats.total_parameters = total_parameters;
    stats.total_elements = total_elements;
    stats.tensors_per_layer = tensors_per_layer;
    stats.tensors_by_role = tensors_by_role;

    // Si la configuration est disponible, on peut compléter certaines informations
    if let Some(config) = config {
        // Utiliser les dimensions de la configuration si disponibles
        if config.hidden_size > 0 {
            stats
                .dimensions
                .insert("hidden_size".to_string(), config.hidden_size);
        }
        if let Some(intermediate_size) = config.intermediate_size {
            stats
                .dimensions
                .insert("intermediate_size".to_string(), intermediate_size);
        }
        stats
            .dimensions
            .insert("vocab_size".to_string(), config.vocab_size);
        stats.dimensions.insert(
            "num_attention_heads".to_string(),
            config.num_attention_heads,
        );
        stats.dimensions.insert(
            "num_key_value_heads".to_string(),
            config.num_key_value_heads,
        );

        // Si la configuration indique un certain nombre de couches, on le vérifie
        if config.num_layers > 0 && stats.num_layers == 0 {
            stats.num_layers = config.num_layers;
        }
    }

    stats
}

/// Extrait le numéro de couche à partir du nom d'un tenseur.
fn extract_layer_number(name: &str) -> Option<u64> {
    // Patterns courants : "model.layers.0.weight", "layer.0.self_attn.q_proj.weight"
    let patterns = ["layers.", "layer."];
    for pattern in &patterns {
        if let Some(pos) = name.find(pattern) {
            let start = pos + pattern.len();
            let remaining = &name[start..];
            // Extraire le numéro jusqu'au prochain caractère non numérique
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

/// Extrait le numéro d'expert à partir du nom d'un tenseur.
fn extract_expert_number(name: &str) -> Option<u64> {
    // Patterns courants : "model.layers.0.mlp.experts.0.weight"
    let patterns = ["experts.", "expert."];
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

/// Extrait les dimensions interesting à partir du nom et de la shape.
fn extract_dimensions(name: &str, shape: &pmg_core::Shape, dims: &mut BTreeMap<String, u64>) {
    // Extraction des dimensions standard
    if shape.dims().len() == 2 {
        let dim0 = shape.dims()[0];
        let dim1 = shape.dims()[1];

        // Classification basée sur le nom du tenseur
        if name.contains("q_proj") || name.contains("k_proj") || name.contains("v_proj") {
            dims.insert("attention_dim".to_string(), dim0);
        } else if name.contains("o_proj") {
            dims.insert("output_dim".to_string(), dim0);
        } else if name.contains("gate_proj") || name.contains("up_proj") {
            dims.insert("mlp_intermediate".to_string(), dim0);
        } else if name.contains("down_proj") {
            dims.insert("mlp_output".to_string(), dim0);
        } else if name.contains("embed_tokens") {
            dims.insert("embed_dim".to_string(), dim0);
            dims.insert("vocab_size".to_string(), dim1);
        } else if name.contains("lm_head") {
            dims.insert("lm_head_dim".to_string(), dim0);
        }
    }
}

/// Classifie le rôle d'un tenseur à partir de son nom.
fn classify_tensor_role(name: &str) -> String {
    if name.contains("self_attn") || name.contains("attention") {
        "attention".to_string()
    } else if name.contains("mlp") || name.contains("ffn") {
        "mlp".to_string()
    } else if name.contains("embed") {
        "embedding".to_string()
    } else if name.contains("norm") || name.contains("layernorm") || name.contains("rmsnorm") {
        "normalization".to_string()
    } else if name.contains("lm_head") {
        "lm_head".to_string()
    } else if name.contains("gate") || name.contains("up") || name.contains("down") {
        "mlp".to_string()
    } else if name.contains("q_proj")
        || name.contains("k_proj")
        || name.contains("v_proj")
        || name.contains("o_proj")
    {
        "attention".to_string()
    } else {
        "other".to_string()
    }
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
    fn test_extract_layer_number() {
        assert_eq!(extract_layer_number("model.layers.0.weight"), Some(0));
        assert_eq!(
            extract_layer_number("model.layers.15.self_attn.q_proj.weight"),
            Some(15)
        );
        assert_eq!(
            extract_layer_number("layer.5.mlp.gate_proj.weight"),
            Some(5)
        );
        assert_eq!(extract_layer_number("embed_tokens.weight"), None);
    }

    #[test]
    fn test_extract_expert_number() {
        assert_eq!(
            extract_expert_number("model.layers.0.mlp.experts.0.weight"),
            Some(0)
        );
        assert_eq!(
            extract_expert_number("model.layers.0.mlp.experts.7.weight"),
            Some(7)
        );
        assert_eq!(
            extract_expert_number("model.layers.0.mlp.gate_proj.weight"),
            None
        );
    }

    #[test]
    fn test_classify_tensor_role() {
        assert_eq!(
            classify_tensor_role("model.layers.0.self_attn.q_proj.weight"),
            "attention"
        );
        assert_eq!(
            classify_tensor_role("model.layers.0.mlp.gate_proj.weight"),
            "mlp"
        );
        assert_eq!(
            classify_tensor_role("model.embed_tokens.weight"),
            "embedding"
        );
        assert_eq!(classify_tensor_role("model.norm.weight"), "normalization");
        assert_eq!(classify_tensor_role("model.lm_head.weight"), "lm_head");
    }

    #[test]
    fn test_compute_structural_stats() {
        let headers = vec![
            create_test_header(
                "shard1.safetensors",
                &[
                    "model.layers.0.self_attn.q_proj.weight",
                    "model.layers.0.mlp.gate_proj.weight",
                    "model.layers.1.self_attn.q_proj.weight",
                    "model.layers.1.mlp.gate_proj.weight",
                ],
            ),
            create_test_header(
                "shard2.safetensors",
                &["model.embed_tokens.weight", "model.norm.weight"],
            ),
        ];

        let stats = compute_structural_stats(&None, &headers, &None);

        assert_eq!(stats.total_tensors, 6);
        assert_eq!(stats.num_layers, 2);
        assert_eq!(stats.num_shards, 2);
        assert!(stats.dtypes.contains(&DType::F32));
        assert_eq!(stats.total_parameters, 6 * 10000); // 6 tenseurs × 100×100
        assert_eq!(stats.tensors_per_layer.get(&0), Some(&2));
        assert_eq!(stats.tensors_per_layer.get(&1), Some(&2));
        // Il y a 2 tenseurs q_proj (un par couche) → attention = 2
        assert_eq!(stats.tensors_by_role.get("attention"), Some(&2));
        assert_eq!(stats.tensors_by_role.get("mlp"), Some(&2));
    }
}
