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

//! Comparaison d'architectures — détection de compatibilité.
//!
//! Ce module fournit des fonctions pour comparer les architectures
//! de modèles, en détectant si elles sont identiques, compatibles,
//! différentes ou inconnues.
//!
//! # Responsabilités
//!
//! - Détection du type d'architecture (IDENTIQUE, COMPATIBLE, DIFFÉRENTE, INCONNUE) ;
//! - Comparaison des propriétés architecturales ;
//! - Calcul d'un score de compatibilité architecturale.
//!
//! # Conventions
//!
//! - La documentation est en français ;
//! - Les comparaisons sont déterministes.

use crate::architecture_helpers::{
    compare_optional_i64, compare_optional_i64_non_blocking, extract_i64,
};
use crate::comparison::ComparisonStatus;
use crate::config_compare::{ConfigValue, ModelConfig};
use crate::diff::Diff;

/// Type d'architecture détecté.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArchitectureType {
    /// Les architectures sont identiques.
    Identical,
    /// Les architectures sont compatibles (peuvent être interchangées).
    Compatible,
    /// Les architectures sont différentes.
    Different,
    /// Le type d'architecture est inconnu.
    Unknown,
}

impl std::fmt::Display for ArchitectureType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArchitectureType::Identical => write!(f, "IDENTIQUE"),
            ArchitectureType::Compatible => write!(f, "COMPATIBLE"),
            ArchitectureType::Different => write!(f, "DIFFÉRENTE"),
            ArchitectureType::Unknown => write!(f, "INCONNUE"),
        }
    }
}

/// Résultat de la comparaison architecturale.
#[derive(Debug, Clone)]
pub struct ArchitectureComparisonResult {
    /// Type d'architecture détecté.
    pub architecture_type: ArchitectureType,
    /// Score de compatibilité (0.0 à 1.0).
    pub compatibility_score: f64,
    /// Statut de la comparaison.
    pub status: ComparisonStatus,
    /// Différences détectées.
    pub differences: Vec<Diff>,
    /// Propriétés architecturales comparées.
    pub properties_compared: usize,
    /// Propriétés compatibles.
    pub properties_compatible: usize,
}

impl Default for ArchitectureComparisonResult {
    fn default() -> Self {
        Self {
            architecture_type: ArchitectureType::Unknown,
            compatibility_score: 0.0,
            status: ComparisonStatus::Unknown,
            differences: Vec::new(),
            properties_compared: 0,
            properties_compatible: 0,
        }
    }
}

impl ArchitectureComparisonResult {
    /// Crée un nouveau résultat de comparaison.
    pub fn new(
        architecture_type: ArchitectureType,
        compatibility_score: f64,
        status: ComparisonStatus,
        differences: Vec<Diff>,
        properties_compared: usize,
        properties_compatible: usize,
    ) -> Self {
        Self {
            architecture_type,
            compatibility_score,
            status,
            differences,
            properties_compared,
            properties_compatible,
        }
    }

    /// Vérifie si les architectures sont identiques.
    pub fn is_identical(&self) -> bool {
        self.architecture_type == ArchitectureType::Identical
    }

    /// Vérifie si les architectures sont compatibles.
    pub fn is_compatible(&self) -> bool {
        matches!(
            self.architecture_type,
            ArchitectureType::Identical | ArchitectureType::Compatible
        )
    }
}

/// Propriétés architecturales à comparer.
pub struct ArchitectureProperties {
    /// Nombre de couches (num_layers).
    pub num_layers: Option<i64>,
    /// Dimension cachée (hidden_size).
    pub hidden_size: Option<i64>,
    /// Nombre de têtes d'attention (num_heads).
    pub num_heads: Option<i64>,
    /// Taille de vocabulaire (vocab_size).
    pub vocab_size: Option<i64>,
    /// Nombre d'experts (num_experts).
    pub num_experts: Option<i64>,
    /// Taille intermédiaire (intermediate_size).
    pub intermediate_size: Option<i64>,
}

impl ArchitectureProperties {
    /// Extrait les propriétés architecturales d'une configuration.
    pub fn from_config(config: &ModelConfig) -> Self {
        let params: std::collections::HashMap<&str, &ConfigValue> = config
            .parameters
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect();

        Self {
            num_layers: params.get("num_layers").copied().and_then(extract_i64),
            hidden_size: params.get("hidden_size").copied().and_then(extract_i64),
            num_heads: params.get("num_heads").copied().and_then(extract_i64),
            vocab_size: params.get("vocab_size").copied().and_then(extract_i64),
            num_experts: params.get("num_experts").copied().and_then(extract_i64),
            intermediate_size: params
                .get("intermediate_size")
                .copied()
                .and_then(extract_i64),
        }
    }

    /// Vérifie si deux propriétés sont compatibles.
    pub fn are_compatible(&self, other: &Self) -> (usize, usize, Vec<Diff>) {
        let mut compatible = 0;
        let mut total = 0;
        let mut diffs = Vec::new();

        // Comparer num_layers
        total += 1;
        let (comp, diff) = compare_optional_i64(
            "num_layers",
            self.num_layers,
            other.num_layers,
            "Nombre de couches manquant",
            "Nombre de couches différent",
        );
        if comp {
            compatible += 1;
        }
        if let Some(d) = diff {
            diffs.push(d);
        }

        // Comparer hidden_size
        total += 1;
        let (comp, diff) = compare_optional_i64(
            "hidden_size",
            self.hidden_size,
            other.hidden_size,
            "Dimension cachée manquante",
            "Dimension cachée différente",
        );
        if comp {
            compatible += 1;
        }
        if let Some(d) = diff {
            diffs.push(d);
        }

        // Comparer num_heads
        total += 1;
        let (comp, diff) = compare_optional_i64(
            "num_heads",
            self.num_heads,
            other.num_heads,
            "Nombre de têtes manquant",
            "Nombre de têtes différent",
        );
        if comp {
            compatible += 1;
        }
        if let Some(d) = diff {
            diffs.push(d);
        }

        // Comparer vocab_size
        total += 1;
        let (comp, diff) = compare_optional_i64(
            "vocab_size",
            self.vocab_size,
            other.vocab_size,
            "Taille de vocabulaire manquante",
            "Taille de vocabulaire différente",
        );
        if comp {
            compatible += 1;
        }
        if let Some(d) = diff {
            diffs.push(d);
        }

        // Comparer num_experts (non bloquant)
        total += 1;
        let (comp, diff) = compare_optional_i64_non_blocking(
            "num_experts",
            self.num_experts,
            other.num_experts,
            "Nombre d'experts différent",
        );
        if comp {
            compatible += 1;
        }
        if let Some(d) = diff {
            diffs.push(d);
        }

        // Comparer intermediate_size
        total += 1;
        let (comp, diff) = compare_optional_i64(
            "intermediate_size",
            self.intermediate_size,
            other.intermediate_size,
            "Taille intermédiaire manquante",
            "Taille intermédiaire différente",
        );
        if comp {
            compatible += 1;
        }
        if let Some(d) = diff {
            diffs.push(d);
        }

        (compatible, total, diffs)
    }
}

/// Compare les architectures de deux modèles.
///
/// # Entrées
/// - `original` : configuration originale ;
/// - `compared` : configuration à comparer.
///
/// # Sorties
/// Un [`ArchitectureComparisonResult`] contenant les résultats de la comparaison.
///
/// # Exemple
///
/// ```
/// use pmg_compare::config_compare::{ModelConfig, ConfigValue};
/// use pmg_compare::architecture_compare::compare_architectures;
///
/// let config1 = ModelConfig {
///     name: "model_a".to_string(),
///     parameters: vec![
///         ("num_layers".to_string(), ConfigValue::Integer(32)),
///         ("hidden_size".to_string(), ConfigValue::Integer(4096)),
///         ("num_heads".to_string(), ConfigValue::Integer(32)),
///         ("vocab_size".to_string(), ConfigValue::Integer(32000)),
///         ("num_experts".to_string(), ConfigValue::Integer(8)),
///         ("intermediate_size".to_string(), ConfigValue::Integer(11008)),
///     ],
/// };
///
/// let config2 = ModelConfig {
///     name: "model_b".to_string(),
///     parameters: vec![
///         ("num_layers".to_string(), ConfigValue::Integer(32)),
///         ("hidden_size".to_string(), ConfigValue::Integer(4096)),
///         ("num_heads".to_string(), ConfigValue::Integer(32)),
///         ("vocab_size".to_string(), ConfigValue::Integer(32000)),
///         ("num_experts".to_string(), ConfigValue::Integer(8)),
///         ("intermediate_size".to_string(), ConfigValue::Integer(11008)),
///     ],
/// };
///
/// let result = compare_architectures(&config1, &config2);
/// assert_eq!(result.architecture_type, pmg_compare::architecture_compare::ArchitectureType::Identical);
/// assert_eq!(result.compatibility_score, 1.0);
/// ```
pub fn compare_architectures(
    original: &ModelConfig,
    compared: &ModelConfig,
) -> ArchitectureComparisonResult {
    let props_original = ArchitectureProperties::from_config(original);
    let props_compared = ArchitectureProperties::from_config(compared);

    let (compatible, total, diffs) = props_original.are_compatible(&props_compared);

    // Calcul du score de compatibilité
    let compatibility_score = if total == 0 {
        1.0
    } else {
        compatible as f64 / total as f64
    };

    // Déterminer le type d'architecture
    let architecture_type = if diffs.is_empty() {
        ArchitectureType::Identical
    } else if compatibility_score >= 0.8 {
        ArchitectureType::Compatible
    } else if compatibility_score > 0.0 {
        ArchitectureType::Different
    } else {
        ArchitectureType::Unknown
    };

    // Déterminer le statut
    let status = match architecture_type {
        ArchitectureType::Identical => ComparisonStatus::Match,
        ArchitectureType::Compatible => ComparisonStatus::Partial,
        ArchitectureType::Different => ComparisonStatus::Different,
        ArchitectureType::Unknown => ComparisonStatus::Unknown,
    };

    ArchitectureComparisonResult::new(
        architecture_type,
        compatibility_score,
        status,
        diffs,
        total,
        compatible,
    )
}

/// Compare les architectures avec détection d'anomalies bloquantes.
///
/// # Entrées
/// - `original` : configuration originale ;
/// - `compared` : configuration à comparer.
///
/// # Sorties
/// Un tuple (résultat, anomalies bloquantes).
pub fn compare_architectures_with_anomalies(
    original: &ModelConfig,
    compared: &ModelConfig,
) -> (ArchitectureComparisonResult, Vec<String>) {
    let result = compare_architectures(original, compared);
    let mut blocking_anomalies = Vec::new();

    // Vérifier les anomalies bloquantes
    for diff in &result.differences {
        // Les différences sur num_layers, hidden_size, vocab_size sont bloquantes
        if diff.path == "num_layers" || diff.path == "hidden_size" || diff.path == "vocab_size" {
            blocking_anomalies.push(format!(
                "Architecture incompatible: {} - {}",
                diff.path, diff.description
            ));
        }
    }

    (result, blocking_anomalies)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn architecture_type_display() {
        assert_eq!(ArchitectureType::Identical.to_string(), "IDENTIQUE");
        assert_eq!(ArchitectureType::Compatible.to_string(), "COMPATIBLE");
        assert_eq!(ArchitectureType::Different.to_string(), "DIFFÉRENTE");
        assert_eq!(ArchitectureType::Unknown.to_string(), "INCONNUE");
    }

    #[test]
    fn compare_architectures_identical() {
        let config = ModelConfig {
            name: "test".to_string(),
            parameters: vec![
                ("num_layers".to_string(), ConfigValue::Integer(32)),
                ("hidden_size".to_string(), ConfigValue::Integer(4096)),
                ("num_heads".to_string(), ConfigValue::Integer(32)),
                ("vocab_size".to_string(), ConfigValue::Integer(32000)),
                ("num_experts".to_string(), ConfigValue::Integer(8)),
                ("intermediate_size".to_string(), ConfigValue::Integer(11008)),
            ],
        };
        let result = compare_architectures(&config, &config);
        assert_eq!(result.architecture_type, ArchitectureType::Identical);
        assert_eq!(result.compatibility_score, 1.0);
        assert!(result.differences.is_empty());
    }

    #[test]
    fn compare_architectures_compatible() {
        let original = ModelConfig {
            name: "test".to_string(),
            parameters: vec![
                ("num_layers".to_string(), ConfigValue::Integer(32)),
                ("hidden_size".to_string(), ConfigValue::Integer(4096)),
                ("num_heads".to_string(), ConfigValue::Integer(32)),
                ("vocab_size".to_string(), ConfigValue::Integer(32000)),
                ("num_experts".to_string(), ConfigValue::Integer(8)),
                ("intermediate_size".to_string(), ConfigValue::Integer(11008)),
            ],
        };
        let compared = ModelConfig {
            name: "test".to_string(),
            parameters: vec![
                ("num_layers".to_string(), ConfigValue::Integer(32)),
                ("hidden_size".to_string(), ConfigValue::Integer(4096)),
                ("num_heads".to_string(), ConfigValue::Integer(32)),
                ("vocab_size".to_string(), ConfigValue::Integer(32000)),
                ("num_experts".to_string(), ConfigValue::Integer(16)),
                ("intermediate_size".to_string(), ConfigValue::Integer(11008)),
            ],
        };
        let result = compare_architectures(&original, &compared);
        // 5/6 paramètres identiques → 0.833...
        assert_eq!(result.architecture_type, ArchitectureType::Compatible);
        assert!((result.compatibility_score - 0.8333333333333334).abs() < 1e-10);
    }

    #[test]
    fn compare_architectures_different() {
        let original = ModelConfig {
            name: "test".to_string(),
            parameters: vec![
                ("num_layers".to_string(), ConfigValue::Integer(32)),
                ("hidden_size".to_string(), ConfigValue::Integer(4096)),
                ("num_heads".to_string(), ConfigValue::Integer(32)),
                ("vocab_size".to_string(), ConfigValue::Integer(32000)),
                ("num_experts".to_string(), ConfigValue::Integer(8)),
                ("intermediate_size".to_string(), ConfigValue::Integer(11008)),
            ],
        };
        let compared = ModelConfig {
            name: "test".to_string(),
            parameters: vec![
                ("num_layers".to_string(), ConfigValue::Integer(24)),
                ("hidden_size".to_string(), ConfigValue::Integer(2048)),
                ("num_heads".to_string(), ConfigValue::Integer(16)),
                ("vocab_size".to_string(), ConfigValue::Integer(32000)),
                ("num_experts".to_string(), ConfigValue::Integer(8)),
                ("intermediate_size".to_string(), ConfigValue::Integer(11008)),
            ],
        };
        let result = compare_architectures(&original, &compared);
        assert_eq!(result.architecture_type, ArchitectureType::Different);
        assert!(result.compatibility_score < 0.8);
    }
}
