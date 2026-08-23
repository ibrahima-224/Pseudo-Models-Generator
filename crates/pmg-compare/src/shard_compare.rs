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

//! Comparaison de sharding — comparaison de la répartition des tenseurs.
//!
//! Ce module fournit des fonctions pour comparer le sharding
//! (répartition) des tenseurs entre deux modèles, incluant :
//! - le nombre de shards ;
//! - le mapping tensor → shard ;
//! - la taille des shards.
//!
//! # Responsabilités
//!
//! - Comparaison du nombre de shards ;
//! - Comparaison du mapping tensor → shard ;
//! - Comparaison de la taille des shards ;
//! - Calcul d'un score de similarité pour le sharding.
//!
//! # Conventions
//!
//! - La documentation est en français ;
//! - Les comparaisons sont déterministes.

use crate::comparison::ComparisonStatus;
use crate::diff::Diff;

/// Représente l'information de sharding d'un tenseur.
#[derive(Debug, Clone)]
pub struct ShardInfo {
    /// Nom du tenseur.
    pub tensor_name: String,
    /// Identifiant du shard (ou indices si multi-shard).
    pub shard_id: usize,
    /// Taille du shard en octets.
    pub byte_size: usize,
    /// Mapping optionnel : tenseur → shard(s).
    pub mapping: Option<Vec<String>>,
}

impl ShardInfo {
    /// Crée une nouvelle information de shard.
    pub fn new(tensor_name: String, shard_id: usize, byte_size: usize) -> Self {
        Self {
            tensor_name,
            shard_id,
            byte_size,
            mapping: None,
        }
    }

    /// Crée une information de shard avec mapping.
    pub fn with_mapping(
        tensor_name: String,
        shard_id: usize,
        byte_size: usize,
        mapping: Vec<String>,
    ) -> Self {
        Self {
            tensor_name,
            shard_id,
            byte_size,
            mapping: Some(mapping),
        }
    }
}

/// Configuration de sharding d'un modèle.
#[derive(Debug, Clone)]
pub struct ShardConfig {
    /// Nombre total de shards.
    pub total_shards: usize,
    /// Informations de sharding par tenseur.
    pub shards: Vec<ShardInfo>,
}

impl ShardConfig {
    /// Crée une nouvelle configuration de sharding.
    pub fn new(total_shards: usize, shards: Vec<ShardInfo>) -> Self {
        Self {
            total_shards,
            shards,
        }
    }
}

/// Résultat de la comparaison de deux configurations de sharding.
#[derive(Debug, Clone)]
pub struct ShardComparisonResult {
    /// Score de similarité (0.0 à 1.0).
    pub similarity_score: f64,
    /// Statut de la comparaison.
    pub status: ComparisonStatus,
    /// Différences détectées.
    pub differences: Vec<Diff>,
    /// Nombre total de tenseurs sharded comparés.
    pub total_tensors: usize,
    /// Nombre de tenseurs avec sharding identique.
    pub matching_tensors: usize,
    /// Différence de nombre de shards.
    pub shard_count_difference: Option<i64>,
}

impl Default for ShardComparisonResult {
    fn default() -> Self {
        Self {
            similarity_score: 1.0,
            status: ComparisonStatus::Match,
            differences: Vec::new(),
            total_tensors: 0,
            matching_tensors: 0,
            shard_count_difference: None,
        }
    }
}

impl ShardComparisonResult {
    /// Crée un nouveau résultat de comparaison.
    pub fn new(
        similarity_score: f64,
        status: ComparisonStatus,
        differences: Vec<Diff>,
        total_tensors: usize,
        matching_tensors: usize,
        shard_count_difference: Option<i64>,
    ) -> Self {
        Self {
            similarity_score,
            status,
            differences,
            total_tensors,
            matching_tensors,
            shard_count_difference,
        }
    }
}

/// Compare deux configurations de sharding.
///
/// # Entrées
/// - `original` : configuration de sharding originale ;
/// - `compared` : configuration de sharding à comparer.
///
/// # Sorties
/// Un [`ShardComparisonResult`] contenant les résultats de la comparaison.
///
/// # Exemple
///
/// ```
/// use pmg_compare::shard_compare::{compare_sharding, ShardConfig, ShardInfo};
///
/// let config1 = ShardConfig {
///     total_shards: 8,
///     shards: vec![
///         ShardInfo::new("layer1.weight".to_string(), 0, 1024),
///     ],
/// };
///
/// let config2 = ShardConfig {
///     total_shards: 8,
///     shards: vec![
///         ShardInfo::new("layer1.weight".to_string(), 0, 1024),
///     ],
/// };
///
/// let result = compare_sharding(&config1, &config2);
/// assert_eq!(result.similarity_score, 1.0);
/// assert!(result.differences.is_empty());
/// ```
pub fn compare_sharding(original: &ShardConfig, compared: &ShardConfig) -> ShardComparisonResult {
    let mut differences = Vec::new();
    let mut matching_count = 0;
    let mut total_count = 0;

    // Comparer le nombre de shards
    let shard_count_diff = if original.total_shards != compared.total_shards {
        let diff = compared.total_shards as i64 - original.total_shards as i64;
        differences.push(Diff::modified(
            "total_shards".to_string(),
            original.total_shards.to_string(),
            compared.total_shards.to_string(),
            "Nombre de shards différent".to_string(),
        ));
        Some(diff)
    } else {
        None
    };

    // Indexer les shards par nom de tenseur
    let original_map: std::collections::HashMap<&str, &ShardInfo> = original
        .shards
        .iter()
        .map(|s| (s.tensor_name.as_str(), s))
        .collect();
    let compared_map: std::collections::HashMap<&str, &ShardInfo> = compared
        .shards
        .iter()
        .map(|s| (s.tensor_name.as_str(), s))
        .collect();

    // Comparer les shards des tenseurs communs
    for (name, original_shard) in &original_map {
        if let Some(compared_shard) = compared_map.get(name) {
            total_count += 1;

            // Comparer les IDs de shards
            if original_shard.shard_id == compared_shard.shard_id {
                // Comparer les tailles
                if original_shard.byte_size == compared_shard.byte_size {
                    matching_count += 1;
                } else {
                    differences.push(Diff::modified(
                        name.to_string(),
                        format!("{} octets", original_shard.byte_size),
                        format!("{} octets", compared_shard.byte_size),
                        format!("Taille de shard différente pour {}", name),
                    ));
                }
            } else {
                differences.push(Diff::modified(
                    name.to_string(),
                    format!("shard {}", original_shard.shard_id),
                    format!("shard {}", compared_shard.shard_id),
                    format!("Mapping tensor → shard différent pour {}", name),
                ));
            }
        }
    }

    // Calcul du score de similarité
    let similarity_score = if total_count == 0 {
        1.0
    } else {
        matching_count as f64 / total_count as f64
    };

    // Déterminer le statut
    let status = if differences.is_empty() {
        ComparisonStatus::Match
    } else if similarity_score >= 0.8 {
        ComparisonStatus::Partial
    } else {
        ComparisonStatus::Different
    };

    ShardComparisonResult::new(
        similarity_score,
        status,
        differences,
        total_count,
        matching_count,
        shard_count_diff,
    )
}

/// Compare les sharding avec détection d'anomalies bloquantes.
///
/// # Entrées
/// - `original` : configuration de sharding originale ;
/// - `compared` : configuration de sharding à comparer.
///
/// # Sorties
/// Un tuple (résultat, anomalies bloquantes).
pub fn compare_sharding_with_anomalies(
    original: &ShardConfig,
    compared: &ShardConfig,
) -> (ShardComparisonResult, Vec<String>) {
    let result = compare_sharding(original, compared);
    let mut blocking_anomalies = Vec::new();

    // Vérifier les anomalies bloquantes
    // Un nombre de shards différent n'est pas forcément une erreur structurelle
    // mais peut affecter la compatibilité
    if let Some(diff) = result.shard_count_difference {
        if diff.abs() > 0 {
            blocking_anomalies.push(format!(
                "Nombre de shards différent: {} → {} (différence: {})",
                original.total_shards, compared.total_shards, diff
            ));
        }
    }

    // Vérifier les différences de mapping
    for diff in &result.differences {
        if diff
            .description
            .contains("Mapping tensor → shard différent")
        {
            blocking_anomalies.push(format!("Mapping incompatible: {}", diff.description));
        }
    }

    (result, blocking_anomalies)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_sharding_identical() {
        let config = ShardConfig::new(
            2,
            vec![
                ShardInfo::new("layer1.weight".to_string(), 0, 1000),
                ShardInfo::new("layer1.bias".to_string(), 1, 500),
            ],
        );
        let result = compare_sharding(&config, &config);
        assert_eq!(result.similarity_score, 1.0);
        assert!(result.differences.is_empty());
        assert_eq!(result.status, ComparisonStatus::Match);
    }

    #[test]
    fn compare_sharding_different_shard_count() {
        let original = ShardConfig::new(
            2,
            vec![
                ShardInfo::new("layer1.weight".to_string(), 0, 1000),
                ShardInfo::new("layer1.bias".to_string(), 1, 500),
            ],
        );
        let compared = ShardConfig::new(
            4,
            vec![
                ShardInfo::new("layer1.weight".to_string(), 0, 1000),
                ShardInfo::new("layer1.bias".to_string(), 1, 500),
            ],
        );
        let result = compare_sharding(&original, &compared);
        assert_eq!(result.similarity_score, 1.0); // Les tenseurs individuels sont identiques
        assert_eq!(result.differences.len(), 1); // Seulement la différence de nombre de shards
        assert!(result.shard_count_difference.is_some());
    }

    #[test]
    fn compare_sharding_different_mapping() {
        let original = ShardConfig::new(
            2,
            vec![
                ShardInfo::new("layer1.weight".to_string(), 0, 1000),
                ShardInfo::new("layer1.bias".to_string(), 1, 500),
            ],
        );
        let compared = ShardConfig::new(
            2,
            vec![
                ShardInfo::new("layer1.weight".to_string(), 1, 1000),
                ShardInfo::new("layer1.bias".to_string(), 0, 500),
            ],
        );
        let result = compare_sharding(&original, &compared);
        assert_eq!(result.similarity_score, 0.0);
        assert_eq!(result.differences.len(), 2);
    }
}
