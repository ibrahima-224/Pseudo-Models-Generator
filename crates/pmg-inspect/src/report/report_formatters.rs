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

//! Formateurs et conversions pour les types de rapport.
//!
//! Ce module contient les implémentations `From` et `Display` pour les types
//! JSON de rapport, ainsi que les fonctions utilitaires de formatage.

use crate::architecture::ArchitectureSummary;
use crate::config_inspector::ConfigInspection;
use crate::index_inspector::ShardIndex;
use crate::physical_stats::PhysicalStats;
use crate::safetensors_inspector::SafetensorsHeader;
use crate::structural_stats::StructuralStats;

use super::types::{
    ArchitectureSummaryJson, ConfigInspectionJson, MoEConfigJson, PhysicalStatsJson,
    SafetensorsHeaderJson, ShardIndexJson, StructuralStatsJson,
};

// ============================================================================
// Conversions depuis les types internes vers les types JSON
// ============================================================================

impl From<&ConfigInspection> for ConfigInspectionJson {
    fn from(config: &ConfigInspection) -> Self {
        // Conversion de la provenance Origin -> String
        let provenance = config
            .provenance
            .iter()
            .map(|(k, v)| (k.clone(), format!("{:?}", v)))
            .collect();

        Self {
            config_path: config.config_path.clone(),
            model_type: config.model_type.clone(),
            architectures: config.architectures.clone(),
            hidden_size: config.hidden_size,
            num_layers: config.num_layers,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            intermediate_size: config.intermediate_size,
            vocab_size: config.vocab_size,
            dtype: format!("{:?}", config.dtype),
            attention_type: format!("{:?}", config.attention_type),
            max_position_embeddings: config.max_position_embeddings,
            moe: config.moe.as_ref().map(MoEConfigJson::from),
            provenance,
        }
    }
}

impl From<&pmg_core::MoeConfig> for MoEConfigJson {
    fn from(moe: &pmg_core::MoeConfig) -> Self {
        Self {
            n_routed_experts: moe.n_routed_experts,
            n_shared_experts: moe.n_shared_experts,
            experts_per_tok: moe.experts_per_tok,
            router_dtype: format!("{:?}", moe.router_dtype),
            routed_scaling_factor: moe.routed_scaling_factor,
            norm_topk_prob: moe.norm_topk_prob,
            topk_method: moe.topk_method.clone(),
            first_k_dense_replace: moe.first_k_dense_replace,
            layer_types: moe.layer_types.clone(),
            expert_dtype: moe.expert_dtype.map(|d| format!("{:?}", d)),
        }
    }
}

impl From<&SafetensorsHeader> for SafetensorsHeaderJson {
    fn from(header: &SafetensorsHeader) -> Self {
        Self {
            file_path: header.file_path.clone(),
            tensor_count: header.tensor_count(),
            total_bytes: header.total_bytes(),
            file_size: header.file_size,
            header_size: header.header_size,
            tensors: None, // Par défaut, pas de détails des tenseurs
        }
    }
}

impl From<&ShardIndex> for ShardIndexJson {
    fn from(index: &ShardIndex) -> Self {
        Self {
            total_tensors: index.total_tensors(),
            shard_count: index.shard_count(),
            shards: index
                .all_shard_paths()
                .iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect(),
            tensor_to_shard: None, // Par défaut, pas de mapping détaillé
        }
    }
}

impl From<&StructuralStats> for StructuralStatsJson {
    fn from(stats: &StructuralStats) -> Self {
        Self {
            total_tensors: stats.total_tensors,
            num_layers: stats.num_layers,
            num_shards: stats.num_shards,
            num_experts: stats.num_experts,
            total_parameters: stats.total_parameters,
            total_elements: stats.total_elements,
            dimensions: stats.dimensions.clone(),
            dtypes: stats.dtypes.iter().map(|d| format!("{:?}", d)).collect(),
        }
    }
}

impl From<&PhysicalStats> for PhysicalStatsJson {
    fn from(stats: &PhysicalStats) -> Self {
        Self {
            total_memory_bytes: stats.total_memory_bytes,
            total_file_size: stats.total_file_size,
            theoretical_size_bytes: stats.theoretical_size_bytes,
            primary_dtype: stats.primary_dtype.map(|d| format!("{:?}", d)),
            average_density: stats.average_density,
            total_parameters: stats.total_parameters,
            average_bytes_per_parameter: stats.average_bytes_per_parameter,
        }
    }
}

impl From<&ArchitectureSummary> for ArchitectureSummaryJson {
    fn from(summary: &ArchitectureSummary) -> Self {
        Self {
            architecture_type: summary.architecture_type.clone(),
            attention_type: summary.attention_type.clone(),
            num_layers: summary.num_layers,
            hidden_size: summary.hidden_size,
            num_attention_heads: summary.num_attention_heads,
            num_key_value_heads: summary.num_key_value_heads,
            intermediate_size: summary.intermediate_size,
            vocab_size: summary.vocab_size,
            primary_dtype: summary.primary_dtype.clone(),
            total_parameters: summary.total_parameters,
            has_moe: summary.has_moe,
            num_experts: summary.num_experts,
            head_dim: summary.head_dim,
        }
    }
}

// ============================================================================
// Implémentations Display pour les types JSON
// ============================================================================

impl std::fmt::Display for ConfigInspectionJson {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Modèle : {}", self.model_type)?;
        writeln!(f, "Architectures : {:?}", self.architectures)?;
        writeln!(f, "Hidden size : {}", self.hidden_size)?;
        writeln!(f, "Couches : {}", self.num_layers)?;
        writeln!(f, "Têtes d'attention : {}", self.num_attention_heads)?;
        writeln!(f, "Têtes K/V : {}", self.num_key_value_heads)?;
        if let Some(inter) = self.intermediate_size {
            writeln!(f, "Taille intermédiaire : {}", inter)?;
        }
        writeln!(f, "Vocabulaire : {}", self.vocab_size)?;
        writeln!(f, "dtype : {}", self.dtype)?;
        writeln!(f, "Attention : {}", self.attention_type)?;
        writeln!(
            f,
            "Max position embeddings : {}",
            self.max_position_embeddings
        )?;
        if let Some(ref moe) = self.moe {
            writeln!(f, "MoE :")?;
            writeln!(f, "  Experts routés : {}", moe.n_routed_experts)?;
            writeln!(f, "  Experts partagés : {}", moe.n_shared_experts)?;
            writeln!(f, "  Experts par token : {}", moe.experts_per_tok)?;
        }
        Ok(())
    }
}

impl std::fmt::Display for SafetensorsHeaderJson {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Fichier : {}", self.file_path.display())?;
        writeln!(f, "Tenseurs : {}", self.tensor_count)?;
        writeln!(f, "Taille totale : {}", format_bytes(self.total_bytes))?;
        writeln!(f, "Taille fichier : {}", format_bytes(self.file_size))?;
        writeln!(f, "Taille header : {}", format_bytes(self.header_size))?;
        Ok(())
    }
}

impl std::fmt::Display for ShardIndexJson {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Tenseurs totaux : {}", self.total_tensors)?;
        writeln!(f, "Nombre de shards : {}", self.shard_count)?;
        writeln!(f, "Shards : {:?}", self.shards)?;
        Ok(())
    }
}

impl std::fmt::Display for StructuralStatsJson {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Tenseurs : {}", self.total_tensors)?;
        writeln!(f, "Couches : {}", self.num_layers)?;
        writeln!(f, "Shards : {}", self.num_shards)?;
        writeln!(f, "Experts : {}", self.num_experts)?;
        writeln!(f, "Paramètres : {}", format_number(self.total_parameters))?;
        writeln!(f, "Éléments : {}", format_number(self.total_elements))?;
        writeln!(f, "Dimensions : {:?}", self.dimensions)?;
        writeln!(f, "Types de données : {:?}", self.dtypes)?;
        Ok(())
    }
}

impl std::fmt::Display for PhysicalStatsJson {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Mémoire totale : {}",
            format_bytes(self.total_memory_bytes)
        )?;
        writeln!(
            f,
            "Taille fichiers : {}",
            format_bytes(self.total_file_size)
        )?;
        writeln!(
            f,
            "Taille théorique : {}",
            format_bytes(self.theoretical_size_bytes)
        )?;
        if let Some(ref dtype) = self.primary_dtype {
            writeln!(f, "Type principal : {}", dtype)?;
        }
        writeln!(f, "Densité moyenne : {:.4}", self.average_density)?;
        writeln!(f, "Paramètres : {}", format_number(self.total_parameters))?;
        writeln!(
            f,
            "Octets par paramètre : {:.2}",
            self.average_bytes_per_parameter
        )?;
        Ok(())
    }
}

impl std::fmt::Display for ArchitectureSummaryJson {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Architecture du modèle ===")?;
        writeln!(f, "Type : {}", self.architecture_type)?;
        writeln!(f, "Attention : {}", self.attention_type)?;
        writeln!(f, "Couches : {}", self.num_layers)?;
        writeln!(f, "Hidden size : {}", self.hidden_size)?;
        writeln!(f, "Têtes d'attention : {}", self.num_attention_heads)?;
        writeln!(f, "Têtes K/V : {}", self.num_key_value_heads)?;
        if let Some(inter) = self.intermediate_size {
            writeln!(f, "Taille intermédiaire : {}", inter)?;
        }
        writeln!(f, "Vocabulaire : {}", self.vocab_size)?;
        writeln!(f, "dtype : {}", self.primary_dtype)?;
        writeln!(f, "Paramètres : {}", format_number(self.total_parameters))?;
        if self.has_moe {
            writeln!(f, "MoE : oui")?;
            if let Some(experts) = self.num_experts {
                writeln!(f, "  Experts : {}", experts)?;
            }
        } else {
            writeln!(f, "MoE : non")?;
        }
        if let Some(head_dim) = self.head_dim {
            writeln!(f, "Taille par tête : {}", head_dim)?;
        }
        Ok(())
    }
}

// ============================================================================
// Fonctions utilitaires
// ============================================================================

/// Formate un nombre avec des séparateurs de milliers.
///
/// # Arguments
///
/// * `n` - Le nombre à formater.
///
/// # Retour
///
/// La chaîne formatée avec des underscores comme séparateurs.
pub fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push('_');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Formate une taille en octets en unité lisible.
///
/// # Arguments
///
/// * `bytes` - Le nombre d'octets à formater.
///
/// # Retour
///
/// La chaîne formatée avec l'unité appropriée (o, KiB, MiB, GiB, TiB).
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    const TB: u64 = 1024 * GB;

    if bytes >= TB {
        format!("{:.2} TiB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GiB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MiB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KiB", bytes as f64 / KB as f64)
    } else {
        format!("{} o", bytes)
    }
}
