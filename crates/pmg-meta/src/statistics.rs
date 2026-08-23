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

//! Module `statistics` — Métriques agrégées de génération PMG.
//!
//! Fournit des structures pour les statistiques par tenseur, par couche
//! et globales, permettant de suivre la qualité et la performance de
//! la génération.
//!
//! ## Structure
//!
//! Les statistiques sont organisées hiérarchiquement :
//! - [`PmgStatistics`] : statistiques globales
//! - [`LayerStatistics`] : statistiques par couche
//! - [`TensorStatistics`] : statistiques par tenseur
//!
//! ## Utilisation
//!
//! ```rust
//! use pmg_meta::statistics::PmgStatistics;
//!
//! let mut stats = PmgStatistics::new("glm-5.2", "size-constrained", 42);
//! assert_eq!(stats.model, "glm-5.2");
//! println!("{}", stats.summary());
//! ```

use serde::{Deserialize, Serialize};

/// Statistiques pour un tenseur individuel.
///
/// Contient des métriques détaillées sur la distribution des valeurs
/// d'un tenseur spécifique.
///
/// # Exemple
///
/// ```rust
/// use pmg_meta::statistics::TensorStatistics;
///
/// let tensor = TensorStatistics {
///     name: "layer1.weight".to_string(),
///     parameter_count: 1000,
///     size_bytes: 4000,
///     dtype: "f32".to_string(),
///     min_value: Some(-0.5),
///     max_value: Some(0.5),
///     mean_value: Some(0.0),
///     std_value: Some(0.1),
///     zero_percentage: 10.0,
///     outlier_percentage: 2.0,
/// };
///
/// assert_eq!(tensor.name, "layer1.weight");
/// assert_eq!(tensor.parameter_count, 1000);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TensorStatistics {
    /// Nom du tenseur.
    pub name: String,
    /// Nombre total de paramètres dans le tenseur.
    pub parameter_count: u64,
    /// Taille en octets.
    pub size_bytes: u64,
    /// Type de données.
    pub dtype: String,
    /// Valeur minimale observée (si applicable).
    pub min_value: Option<f64>,
    /// Valeur maximale observée (si applicable).
    pub max_value: Option<f64>,
    /// Moyenne des valeurs (si applicable).
    pub mean_value: Option<f64>,
    /// Écart-type des valeurs (si applicable).
    pub std_value: Option<f64>,
    /// Pourcentage de valeurs nulles.
    pub zero_percentage: f64,
    /// Pourcentage de valeurs extrêmes (outliers).
    pub outlier_percentage: f64,
}

/// Statistiques pour une couche (layer) du modèle.
///
/// Agrège les statistiques de tous les tenseurs d'une couche.
///
/// # Exemple
///
/// ```rust
/// use pmg_meta::statistics::LayerStatistics;
///
/// let layer = LayerStatistics {
///     name: "layer1".to_string(),
///     tensor_count: 2,
///     parameter_count: 1000,
///     size_bytes: 4000,
///     tensors: vec![],
/// };
///
/// assert_eq!(layer.tensor_count, 2);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LayerStatistics {
    /// Nom de la couche.
    pub name: String,
    /// Nombre de tenseurs dans la couche.
    pub tensor_count: u32,
    /// Nombre total de paramètres.
    pub parameter_count: u64,
    /// Taille totale en octets.
    pub size_bytes: u64,
    /// Statistiques détaillées par tenseur.
    pub tensors: Vec<TensorStatistics>,
}

/// Statistiques globales de génération.
///
/// Contient toutes les métriques agrégées pour une génération complète.
///
/// # Exemple
///
/// ```rust
/// use pmg_meta::statistics::PmgStatistics;
///
/// let stats = PmgStatistics::new("glm-5.2", "size-constrained", 42);
/// assert_eq!(stats.model, "glm-5.2");
/// assert_eq!(stats.generation_mode, "size-constrained");
/// assert_eq!(stats.seed, 42);
///
/// // Affichage du résumé
/// let summary = stats.summary();
/// assert!(summary.contains("Statistiques PMG"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PmgStatistics {
    /// Version du schéma de statistiques.
    pub schema_version: u32,
    /// Identifiant du modèle.
    pub model: String,
    /// Mode de génération utilisé.
    pub generation_mode: String,
    /// Graine aléatoire.
    pub seed: u64,
    /// Horodatage UTC.
    pub timestamp_utc: String,
    /// Nombre total de couches.
    pub layer_count: u32,
    /// Nombre total de tenseurs.
    pub tensor_count: u64,
    /// Nombre total de paramètres.
    pub parameter_count: u64,
    /// Taille totale estimée en octets.
    pub estimated_size_bytes: u64,
    /// Taille réelle en octets.
    pub actual_size_bytes: u64,
    /// Statistiques par couche.
    pub layers: Vec<LayerStatistics>,
    /// Métriques de performance de génération.
    pub generation_metrics: GenerationMetrics,
}

/// Métriques de performance de la génération.
///
/// Mesure les performances du processus de génération.
///
/// # Exemple
///
/// ```rust
/// use pmg_meta::statistics::GenerationMetrics;
///
/// let metrics = GenerationMetrics {
///     total_duration_ms: 5000,
///     operations_count: 1000,
///     throughput_params_per_sec: 1_000_000.0,
///     peak_memory_bytes: 1024 * 1024 * 1024,
///     error_count: 0,
///     warning_count: 2,
/// };
///
/// assert_eq!(metrics.total_duration_ms, 5000);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GenerationMetrics {
    /// Durée totale de génération en millisecondes.
    pub total_duration_ms: u64,
    /// Nombre d'opérations effectuées.
    pub operations_count: u64,
    /// Débit moyen en paramètres par seconde.
    pub throughput_params_per_sec: f64,
    /// Utilisation maximale de la mémoire en octets.
    pub peak_memory_bytes: u64,
    /// Nombre d'erreurs rencontrées.
    pub error_count: u32,
    /// Nombre d'avertissements.
    pub warning_count: u32,
}

impl PmgStatistics {
    /// Crée des statistiques vides pour un modèle donné.
    ///
    /// # Exemple
    ///
    /// ```rust
    /// use pmg_meta::statistics::PmgStatistics;
    ///
    /// let stats = PmgStatistics::new("glm-5.2", "size-constrained", 42);
    /// assert_eq!(stats.schema_version, 1);
    /// assert_eq!(stats.layer_count, 0);
    /// ```
    pub fn new(model: &str, generation_mode: &str, seed: u64) -> Self {
        Self {
            schema_version: 1,
            model: model.to_string(),
            generation_mode: generation_mode.to_string(),
            seed,
            timestamp_utc: chrono::Utc::now().to_rfc3339(),
            layer_count: 0,
            tensor_count: 0,
            parameter_count: 0,
            estimated_size_bytes: 0,
            actual_size_bytes: 0,
            layers: Vec::new(),
            generation_metrics: GenerationMetrics {
                total_duration_ms: 0,
                operations_count: 0,
                throughput_params_per_sec: 0.0,
                peak_memory_bytes: 0,
                error_count: 0,
                warning_count: 0,
            },
        }
    }

    /// Ajoute une couche aux statistiques.
    ///
    /// Met à jour automatiquement les compteurs globaux.
    ///
    /// # Exemple
    ///
    /// ```rust
    /// use pmg_meta::statistics::{PmgStatistics, LayerStatistics};
    ///
    /// let mut stats = PmgStatistics::new("test", "full", 123);
    /// let layer = LayerStatistics {
    ///     name: "layer1".to_string(),
    ///     tensor_count: 2,
    ///     parameter_count: 1000,
    ///     size_bytes: 4000,
    ///     tensors: vec![],
    /// };
    ///
    /// stats.add_layer(layer);
    /// assert_eq!(stats.layer_count, 1);
    /// assert_eq!(stats.tensor_count, 2);
    /// ```
    pub fn add_layer(&mut self, layer: LayerStatistics) {
        self.tensor_count += layer.tensor_count as u64;
        self.parameter_count += layer.parameter_count;
        self.actual_size_bytes += layer.size_bytes;
        self.layers.push(layer);
        self.layer_count = self.layers.len() as u32;
    }

    /// Calcule le pourcentage moyen de zéros sur tous les tenseurs.
    ///
    /// Utile pour évaluer la sparsité des poids.
    ///
    /// # Exemple
    ///
    /// ```rust
    /// use pmg_meta::statistics::PmgStatistics;
    ///
    /// let stats = PmgStatistics::new("test", "full", 123);
    /// assert_eq!(stats.average_zero_percentage(), 0.0);
    /// ```
    pub fn average_zero_percentage(&self) -> f64 {
        let total_tensors: usize = self.layers.iter().map(|l| l.tensors.len()).sum();
        if total_tensors == 0 {
            return 0.0;
        }
        let total_zero: f64 = self
            .layers
            .iter()
            .flat_map(|l| &l.tensors)
            .map(|t| t.zero_percentage)
            .sum();
        total_zero / total_tensors as f64
    }

    /// Calcule le pourcentage moyen d'outliers sur tous les tenseurs.
    ///
    /// Utile pour détecter les distributions anormales.
    ///
    /// # Exemple
    ///
    /// ```rust
    /// use pmg_meta::statistics::PmgStatistics;
    ///
    /// let stats = PmgStatistics::new("test", "full", 123);
    /// assert_eq!(stats.average_outlier_percentage(), 0.0);
    /// ```
    pub fn average_outlier_percentage(&self) -> f64 {
        let total_tensors: usize = self.layers.iter().map(|l| l.tensors.len()).sum();
        if total_tensors == 0 {
            return 0.0;
        }
        let total_outlier: f64 = self
            .layers
            .iter()
            .flat_map(|l| &l.tensors)
            .map(|t| t.outlier_percentage)
            .sum();
        total_outlier / total_tensors as f64
    }

    /// Retourne un résumé textuel des statistiques.
    ///
    /// # Exemple
    ///
    /// ```rust
    /// use pmg_meta::statistics::PmgStatistics;
    ///
    /// let stats = PmgStatistics::new("glm-5.2", "full", 42);
    /// let summary = stats.summary();
    /// assert!(summary.contains("Statistiques PMG"));
    /// assert!(summary.contains("glm-5.2"));
    /// ```
    pub fn summary(&self) -> String {
        format!(
            "Statistiques PMG pour {} (mode {})\n\
             Couches: {}, tenseurs: {}, paramètres: {}\n\
             Taille estimée: {} octets, réelle: {} octets\n\
             Zéros moyens: {:.2}%, outliers moyens: {:.2}%\n\
             Durée génération: {} ms, débit: {:.0} params/s",
            self.model,
            self.generation_mode,
            self.layer_count,
            self.tensor_count,
            self.parameter_count,
            self.estimated_size_bytes,
            self.actual_size_bytes,
            self.average_zero_percentage(),
            self.average_outlier_percentage(),
            self.generation_metrics.total_duration_ms,
            self.generation_metrics.throughput_params_per_sec
        )
    }
}

impl std::fmt::Display for PmgStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistics_new() {
        let stats = PmgStatistics::new("glm-5.2", "size-constrained", 42);
        assert_eq!(stats.model, "glm-5.2");
        assert_eq!(stats.generation_mode, "size-constrained");
        assert_eq!(stats.seed, 42);
        assert_eq!(stats.schema_version, 1);
        assert_eq!(stats.layer_count, 0);
        assert_eq!(stats.tensor_count, 0);
        assert_eq!(stats.parameter_count, 0);
    }

    #[test]
    fn test_add_layer() {
        let mut stats = PmgStatistics::new("test-model", "full", 123);
        let layer = LayerStatistics {
            name: "layer1".to_string(),
            tensor_count: 2,
            parameter_count: 1000,
            size_bytes: 4000,
            tensors: vec![],
        };
        stats.add_layer(layer);
        assert_eq!(stats.layer_count, 1);
        assert_eq!(stats.tensor_count, 2);
        assert_eq!(stats.parameter_count, 1000);
        assert_eq!(stats.actual_size_bytes, 4000);
    }

    #[test]
    fn test_average_zero_percentage() {
        let mut stats = PmgStatistics::new("test-model", "full", 123);
        let tensor1 = TensorStatistics {
            name: "t1".to_string(),
            parameter_count: 100,
            size_bytes: 400,
            dtype: "f32".to_string(),
            min_value: None,
            max_value: None,
            mean_value: None,
            std_value: None,
            zero_percentage: 10.0,
            outlier_percentage: 5.0,
        };
        let tensor2 = TensorStatistics {
            name: "t2".to_string(),
            parameter_count: 200,
            size_bytes: 800,
            dtype: "f32".to_string(),
            min_value: None,
            max_value: None,
            mean_value: None,
            std_value: None,
            zero_percentage: 20.0,
            outlier_percentage: 10.0,
        };
        let layer = LayerStatistics {
            name: "layer1".to_string(),
            tensor_count: 2,
            parameter_count: 300,
            size_bytes: 1200,
            tensors: vec![tensor1, tensor2],
        };
        stats.add_layer(layer);
        assert_eq!(stats.average_zero_percentage(), 15.0);
        assert_eq!(stats.average_outlier_percentage(), 7.5);
    }

    #[test]
    fn test_statistics_serialization() {
        let stats = PmgStatistics::new("test-model", "full", 123);
        let json = serde_json::to_string_pretty(&stats).unwrap();
        let deserialized: PmgStatistics = serde_json::from_str(&json).unwrap();
        assert_eq!(stats, deserialized);
    }

    #[test]
    fn test_statistics_display() {
        let stats = PmgStatistics::new("glm-5.2", "size-constrained", 42);
        let display = stats.to_string();
        assert!(display.contains("Statistiques PMG"));
        assert!(display.contains("glm-5.2"));
        assert!(display.contains("size-constrained"));
    }
}
