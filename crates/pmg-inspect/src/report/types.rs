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

//! Sous-module contenant les types JSON pour les rapports d'inspection.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Rapport structuré d'inspection pour la sérialisation JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredReport {
    /// Chemin du modèle inspecté.
    pub model_path: PathBuf,
    /// Niveau de détail du rapport.
    pub level: String,
    /// Timestamp de l'inspection (optionnel).
    pub timestamp: Option<String>,
    /// Inspection de la configuration.
    pub config: Option<ConfigInspectionJson>,
    /// Headers Safetensors extraits.
    pub safetensors_headers: Vec<SafetensorsHeaderJson>,
    /// Index des shards (si disponible).
    pub shard_index: Option<ShardIndexJson>,
    /// Statistiques structurelles.
    pub structural: StructuralStatsJson,
    /// Statistiques physiques.
    pub physical: PhysicalStatsJson,
    /// Résumé architectural.
    pub architecture: ArchitectureSummaryJson,
}

/// Version JSON de l'inspection de configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigInspectionJson {
    /// Chemin du fichier config.
    pub config_path: PathBuf,
    /// Type de modèle.
    pub model_type: String,
    /// Architectures déclarées.
    pub architectures: Vec<String>,
    /// Taille cachée.
    pub hidden_size: u64,
    /// Nombre de couches.
    pub num_layers: u64,
    /// Nombre de têtes d'attention.
    pub num_attention_heads: u64,
    /// Nombre de têtes K/V.
    pub num_key_value_heads: u64,
    /// Taille intermédiaire.
    pub intermediate_size: Option<u64>,
    /// Taille du vocabulaire.
    pub vocab_size: u64,
    /// Type de données.
    pub dtype: String,
    /// Type d'attention.
    pub attention_type: String,
    /// Taille maximale des embeddings.
    pub max_position_embeddings: u64,
    /// Configuration MoE (si présente).
    pub moe: Option<MoEConfigJson>,
    /// Provenance des champs (sérialisé en String pour JSON).
    pub provenance: BTreeMap<String, String>,
}

/// Configuration MoE en format JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoEConfigJson {
    /// Nombre d'experts routés.
    pub n_routed_experts: u64,
    /// Nombre d'experts partagés.
    pub n_shared_experts: u64,
    /// Nombre d'experts par token.
    pub experts_per_tok: u64,
    /// Type de données du routeur.
    pub router_dtype: String,
    /// Facteur de mise à l'échelle routé.
    pub routed_scaling_factor: f64,
    /// Normalisation de la probabilité top-k.
    pub norm_topk_prob: bool,
    /// Méthode top-k.
    pub topk_method: String,
    /// Remplacement dense des premières couches.
    pub first_k_dense_replace: Option<u64>,
    /// Types de couches.
    pub layer_types: Vec<String>,
    /// Type de données des experts (optionnel).
    pub expert_dtype: Option<String>,
}

/// Version JSON d'un header Safetensors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetensorsHeaderJson {
    /// Chemin du fichier.
    pub file_path: PathBuf,
    /// Nombre de tenseurs.
    pub tensor_count: usize,
    /// Taille totale en octets.
    pub total_bytes: u64,
    /// Taille du fichier.
    pub file_size: u64,
    /// Taille du header.
    pub header_size: u64,
    /// Liste des tenseurs (si en mode verbose).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensors: Option<Vec<TensorInfoJson>>,
}

/// Informations sur un tenseur en format JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfoJson {
    /// Nom du tenseur.
    pub name: String,
    /// Type de données.
    pub dtype: String,
    /// Forme du tenseur.
    pub shape: Vec<u64>,
    /// Taille en octets.
    pub size_bytes: u64,
}

/// Index des shards en format JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardIndexJson {
    /// Nombre total de tenseurs.
    pub total_tensors: usize,
    /// Nombre de shards.
    pub shard_count: usize,
    /// Chemins des shards.
    pub shards: Vec<String>,
    /// Mapping tenseur → shard (si en mode verbose).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_to_shard: Option<BTreeMap<String, String>>,
}

/// Statistiques structurelles en format JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralStatsJson {
    /// Nombre total de tenseurs.
    pub total_tensors: u64,
    /// Nombre de couches.
    pub num_layers: u64,
    /// Nombre de shards.
    pub num_shards: u64,
    /// Nombre d'experts.
    pub num_experts: u64,
    /// Nombre total de paramètres.
    pub total_parameters: u64,
    /// Nombre total d'éléments.
    pub total_elements: u64,
    /// Dimensions des tenseurs.
    pub dimensions: BTreeMap<String, u64>,
    /// Types de données présents.
    pub dtypes: Vec<String>,
}

/// Statistiques physiques en format JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalStatsJson {
    /// Taille totale en mémoire (octets).
    pub total_memory_bytes: u64,
    /// Taille totale des fichiers.
    pub total_file_size: u64,
    /// Taille théorique (octets).
    pub theoretical_size_bytes: u64,
    /// Type de données principal.
    pub primary_dtype: Option<String>,
    /// Densité moyenne.
    pub average_density: f64,
    /// Nombre total de paramètres.
    pub total_parameters: u64,
    /// Taille moyenne par paramètre.
    pub average_bytes_per_parameter: f64,
}

/// Résumé architectural en format JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureSummaryJson {
    /// Type d'architecture.
    pub architecture_type: String,
    /// Type d'attention.
    pub attention_type: String,
    /// Nombre de couches.
    pub num_layers: u64,
    /// Taille cachée.
    pub hidden_size: u64,
    /// Nombre de têtes d'attention.
    pub num_attention_heads: u64,
    /// Nombre de têtes K/V.
    pub num_key_value_heads: u64,
    /// Taille intermédiaire.
    pub intermediate_size: Option<u64>,
    /// Taille du vocabulaire.
    pub vocab_size: u64,
    /// Type de données principal.
    pub primary_dtype: String,
    /// Nombre total de paramètres.
    pub total_parameters: u64,
    /// Présence de MoE.
    pub has_moe: bool,
    /// Nombre d'experts.
    pub num_experts: Option<u64>,
    /// Taille par tête.
    pub head_dim: Option<u64>,
}
