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

//! Comparaison de configurations — comparaison des paramètres architecturaux.
//!
//! Ce module fournit des fonctions pour comparer les configurations
//! de modèles, en se concentrant sur les paramètres architecturaux clés :
//! vocab_size, hidden_size, num_layers, num_heads, num_experts, intermediate_size.
//!
//! # Responsabilités
//!
//! - Comparaison des paramètres architecturaux ;
//! - Détection des différences significatives ;
//! - Calcul d'un score de similarité pour les configurations.
//!
//! # Conventions
//!
//! - La documentation est en français ;
//! - Les comparaisons sont déterministes.

use crate::comparison::ComparisonStatus;
use crate::diff::{Diff, DiffType};

/// Représente une configuration de modèle à comparer.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Nom du modèle.
    pub name: String,
    /// Paramètres de configuration (clé, valeur).
    pub parameters: Vec<(String, ConfigValue)>,
}

/// Valeur de configuration (peut être de différents types).
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigValue {
    /// Valeur entière.
    Integer(i64),
    /// Valeur flottante.
    Float(f64),
    /// Valeur booléenne.
    Boolean(bool),
    /// Chaîne de caractères.
    String(String),
    /// Liste de valeurs.
    List(Vec<ConfigValue>),
}

impl std::fmt::Display for ConfigValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigValue::Integer(i) => write!(f, "{}", i),
            ConfigValue::Float(fl) => write!(f, "{:.6}", fl),
            ConfigValue::Boolean(b) => write!(f, "{}", b),
            ConfigValue::String(s) => write!(f, "\"{}\"", s),
            ConfigValue::List(_) => write!(f, "[...]"),
        }
    }
}

/// Résultat de la comparaison de deux configurations.
#[derive(Debug, Clone)]
pub struct ConfigComparisonResult {
    /// Score de similarité (0.0 à 1.0).
    pub similarity_score: f64,
    /// Statut de la comparaison.
    pub status: ComparisonStatus,
    /// Différences détectées.
    pub differences: Vec<Diff>,
    /// Nombre de paramètres comparés.
    pub parameter_count: usize,
    /// Nombre de paramètres identiques.
    pub matching_count: usize,
}

impl Default for ConfigComparisonResult {
    fn default() -> Self {
        Self {
            similarity_score: 1.0,
            status: ComparisonStatus::Match,
            differences: Vec::new(),
            parameter_count: 0,
            matching_count: 0,
        }
    }
}

impl ConfigComparisonResult {
    /// Crée un nouveau résultat de comparaison.
    pub fn new(
        similarity_score: f64,
        status: ComparisonStatus,
        differences: Vec<Diff>,
        parameter_count: usize,
        matching_count: usize,
    ) -> Self {
        Self {
            similarity_score,
            status,
            differences,
            parameter_count,
            matching_count,
        }
    }

    /// Vérifie si la comparaison est réussie.
    pub fn is_match(&self) -> bool {
        self.status == ComparisonStatus::Match
    }
}

/// Compare deux configurations de modèle.
///
/// # Entrées
/// - `original` : configuration originale ;
/// - `compared` : configuration à comparer.
///
/// # Sorties
/// Un [`ConfigComparisonResult`] contenant les résultats de la comparaison.
///
/// # Exemple
///
/// ```
/// use pmg_compare::config_compare::{compare_configs, ModelConfig, ConfigValue};
///
/// let config1 = ModelConfig {
///     name: "model_a".to_string(),
///     parameters: vec![
///         ("vocab_size".to_string(), ConfigValue::Integer(32000)),
///         ("hidden_size".to_string(), ConfigValue::Integer(4096)),
///         ("num_layers".to_string(), ConfigValue::Integer(32)),
///         ("num_heads".to_string(), ConfigValue::Integer(32)),
///         ("num_experts".to_string(), ConfigValue::Integer(8)),
///         ("intermediate_size".to_string(), ConfigValue::Integer(11008)),
///     ],
/// };
///
/// let config2 = ModelConfig {
///     name: "model_b".to_string(),
///     parameters: vec![
///         ("vocab_size".to_string(), ConfigValue::Integer(32000)),
///         ("hidden_size".to_string(), ConfigValue::Integer(4096)),
///         ("num_layers".to_string(), ConfigValue::Integer(32)),
///         ("num_heads".to_string(), ConfigValue::Integer(32)),
///         ("num_experts".to_string(), ConfigValue::Integer(8)),
///         ("intermediate_size".to_string(), ConfigValue::Integer(11008)),
///     ],
/// };
///
/// let result = compare_configs(&config1, &config2);
/// assert_eq!(result.similarity_score, 1.0);
/// assert!(result.differences.is_empty());
/// ```
pub fn compare_configs(original: &ModelConfig, compared: &ModelConfig) -> ConfigComparisonResult {
    let mut differences = Vec::new();
    let mut parameter_count = 0;
    let mut matching_count = 0;

    // Paramètres architecturaux à comparer
    let architecture_params = [
        "vocab_size",
        "hidden_size",
        "num_layers",
        "num_heads",
        "num_experts",
        "intermediate_size",
    ];

    // Indexer les paramètres par clé
    let original_params: std::collections::HashMap<&str, &ConfigValue> = original
        .parameters
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect();
    let compared_params: std::collections::HashMap<&str, &ConfigValue> = compared
        .parameters
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect();

    // Comparer les paramètres architecturaux
    for param_name in &architecture_params {
        parameter_count += 1;

        match (
            original_params.get(param_name),
            compared_params.get(param_name),
        ) {
            (Some(original_value), Some(compared_value)) => {
                if original_value == compared_value {
                    matching_count += 1;
                } else {
                    differences.push(Diff::new(
                        DiffType::Modified,
                        param_name.to_string(),
                        Some(original_value.to_string()),
                        Some(compared_value.to_string()),
                        format!("Paramètre '{}' différent", param_name),
                    ));
                }
            },
            (Some(original_value), None) => {
                differences.push(Diff::new(
                    DiffType::Removed,
                    param_name.to_string(),
                    Some(original_value.to_string()),
                    None,
                    format!(
                        "Paramètre '{}' présent uniquement dans l'original",
                        param_name
                    ),
                ));
            },
            (None, Some(compared_value)) => {
                differences.push(Diff::new(
                    DiffType::Added,
                    param_name.to_string(),
                    None,
                    Some(compared_value.to_string()),
                    format!(
                        "Paramètre '{}' présent uniquement dans la comparaison",
                        param_name
                    ),
                ));
            },
            (None, None) => {
                // Paramètre absent des deux modèles - pas de différence
                matching_count += 1;
            },
        }
    }

    // Calcul du score de similarité
    let similarity_score = if parameter_count == 0 {
        1.0
    } else {
        matching_count as f64 / parameter_count as f64
    };

    // Déterminer le statut
    let status = if differences.is_empty() {
        ComparisonStatus::Match
    } else if similarity_score >= 0.8 {
        ComparisonStatus::Partial
    } else {
        ComparisonStatus::Different
    };

    ConfigComparisonResult::new(
        similarity_score,
        status,
        differences,
        parameter_count,
        matching_count,
    )
}

/// Compare les configurations avec détection d'anomalies bloquantes.
///
/// # Entrées
/// - `original` : configuration originale ;
/// - `compared` : configuration à comparer.
///
/// # Sorties
/// Un tuple (résultat, anomalies bloquantes).
pub fn compare_configs_with_anomalies(
    original: &ModelConfig,
    compared: &ModelConfig,
) -> (ConfigComparisonResult, Vec<String>) {
    let result = compare_configs(original, compared);
    let mut blocking_anomalies = Vec::new();

    // Vérifier les anomalies bloquantes
    for diff in &result.differences {
        match diff.diff_type {
            DiffType::Removed => {
                blocking_anomalies.push(format!(
                    "Paramètre critique '{}' manquant dans le modèle comparé",
                    diff.path
                ));
            },
            // Certains paramètres modifiés sont bloquants
            DiffType::Modified if diff.path == "vocab_size" || diff.path == "hidden_size" => {
                blocking_anomalies.push(format!(
                    "Paramètre critique '{}' modifié: {} → {}",
                    diff.path,
                    diff.original_value.as_deref().unwrap_or("N/A"),
                    diff.compared_value.as_deref().unwrap_or("N/A")
                ));
            },
            DiffType::Modified => {},
            _ => {},
        }
    }

    (result, blocking_anomalies)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_configs_identical() {
        let config = ModelConfig {
            name: "test".to_string(),
            parameters: vec![
                ("vocab_size".to_string(), ConfigValue::Integer(32000)),
                ("hidden_size".to_string(), ConfigValue::Integer(4096)),
                ("num_layers".to_string(), ConfigValue::Integer(32)),
                ("num_heads".to_string(), ConfigValue::Integer(32)),
                ("num_experts".to_string(), ConfigValue::Integer(8)),
                ("intermediate_size".to_string(), ConfigValue::Integer(11008)),
            ],
        };
        let result = compare_configs(&config, &config);
        assert_eq!(result.similarity_score, 1.0);
        assert!(result.differences.is_empty());
        assert_eq!(result.status, ComparisonStatus::Match);
    }

    #[test]
    fn compare_configs_different_values() {
        let original = ModelConfig {
            name: "test".to_string(),
            parameters: vec![
                ("vocab_size".to_string(), ConfigValue::Integer(32000)),
                ("hidden_size".to_string(), ConfigValue::Integer(4096)),
            ],
        };
        let compared = ModelConfig {
            name: "test".to_string(),
            parameters: vec![
                ("vocab_size".to_string(), ConfigValue::Integer(32000)),
                ("hidden_size".to_string(), ConfigValue::Integer(8192)),
            ],
        };
        let result = compare_configs(&original, &compared);
        // 6 paramètres architecturaux, 1 différent → 5/6 = 0.833...
        assert!((result.similarity_score - 0.8333333333333334).abs() < 1e-10);
        assert_eq!(result.differences.len(), 1);
        assert_eq!(result.status, ComparisonStatus::Partial);
    }

    #[test]
    fn compare_configs_missing_parameter() {
        let original = ModelConfig {
            name: "test".to_string(),
            parameters: vec![
                ("vocab_size".to_string(), ConfigValue::Integer(32000)),
                ("hidden_size".to_string(), ConfigValue::Integer(4096)),
            ],
        };
        let compared = ModelConfig {
            name: "test".to_string(),
            parameters: vec![("vocab_size".to_string(), ConfigValue::Integer(32000))],
        };
        let result = compare_configs(&original, &compared);
        // 6 paramètres architecturaux, 1 manquant → 5/6 = 0.833...
        assert!((result.similarity_score - 0.8333333333333334).abs() < 1e-10);
        assert_eq!(result.differences.len(), 1);
        assert_eq!(result.differences[0].diff_type, DiffType::Removed);
    }

    #[test]
    fn test_compare_configs_with_anomalies() {
        let original = ModelConfig {
            name: "test".to_string(),
            parameters: vec![
                ("vocab_size".to_string(), ConfigValue::Integer(32000)),
                ("hidden_size".to_string(), ConfigValue::Integer(4096)),
            ],
        };
        let compared = ModelConfig {
            name: "test".to_string(),
            parameters: vec![("vocab_size".to_string(), ConfigValue::Integer(32000))],
        };
        let (result, anomalies) = compare_configs_with_anomalies(&original, &compared);
        // 6 paramètres architecturaux, 1 manquant → 5/6 = 0.833...
        assert!((result.similarity_score - 0.8333333333333334).abs() < 1e-10);
        assert!(!anomalies.is_empty());
        assert!(anomalies[0].contains("hidden_size"));
    }
}
