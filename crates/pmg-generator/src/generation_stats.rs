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

//! Statistiques de génération.
//!
//! Ce module définit la structure `GenerationStats` qui collecte et agrège
//! les statistiques pendant la génération des tenseurs. Les statistiques
//! permettent de valider la qualité de la génération et de détecter les
//! anomalies.

use pmg_math::statistics::quantiles;

/// Statistiques de génération pour un ensemble de tenseurs.
///
/// Collecte les métriques pendant la génération et fournit des agrégats
/// pour la validation et le reporting.
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    /// Moyenne des valeurs.
    pub mean: f64,
    /// Variance des valeurs.
    pub variance: f64,
    /// Écart-type des valeurs.
    pub std_dev: f64,
    /// Valeur minimale.
    pub min: f64,
    /// Valeur maximale.
    pub max: f64,
    /// Quantiles calculés (25%, 50%, 75%).
    pub quantiles: Vec<f64>,
    /// Nombre total d'outliers détectés (> 3 écarts-types).
    pub outlier_count: usize,
    /// Nombre total de super-poids détectés (> 5 écarts-types).
    pub super_weight_count: usize,
    /// Nombre total de paramètres générés.
    pub parameter_count: u64,
    /// Nombre de tenseurs analysés.
    pub tensor_count: usize,
    /// Nombre de chunks traités.
    pub chunk_count: usize,
}

impl GenerationStats {
    /// Crée de nouvelles statistiques vides.
    pub fn new() -> Self {
        Self::default()
    }

    /// Met à jour les statistiques à partir d'un vecteur de valeurs.
    ///
    /// Cette méthode est conçue pour être appelée plusieurs fois avec des
    /// sous-ensembles de valeurs, et d'agrégés les résultats.
    pub fn update_from_values(&mut self, values: &[f64]) {
        if values.is_empty() {
            return;
        }

        let n = values.len() as f64;

        // Calculer la moyenne
        let sum: f64 = values.iter().sum();
        let new_mean = sum / n;

        // Mettre à jour la variance (formule de mise à jour en ligne)
        if self.tensor_count == 0 {
            // Premier appel
            self.mean = new_mean;
            let variance_sum: f64 = values.iter().map(|x| (x - new_mean).powi(2)).sum();
            self.variance = variance_sum / n;
        } else {
            // Mise à jour de la moyenne et variance avec formule de Welford
            let old_mean = self.mean;
            let old_count = self.parameter_count as f64;
            let new_count = old_count + n;

            // Nouvelle moyenne
            self.mean = (old_count * old_mean + sum) / new_count;

            // Nouvelle variance (formule de Welford)
            let delta = new_mean - old_mean;
            let delta2 = new_mean - self.mean;
            let variance_sum: f64 = values.iter().map(|x| (x - old_mean).powi(2)).sum();
            self.variance =
                (self.variance * old_count + variance_sum + delta * delta2 * n) / new_count;
        }

        // Mettre à jour l'écart-type
        self.std_dev = self.variance.sqrt();

        // Mettre à jour min/max
        let local_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let local_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if self.tensor_count == 0 {
            self.min = local_min;
            self.max = local_max;
        } else {
            self.min = self.min.min(local_min);
            self.max = self.max.max(local_max);
        }

        // Compter les outliers et super-poids
        if self.std_dev > 0.0 {
            for &x in values {
                let z = (x - self.mean).abs() / self.std_dev;
                if z > 3.0 {
                    self.outlier_count += 1;
                }
                if z > 5.0 {
                    self.super_weight_count += 1;
                }
            }
        }

        // Mettre à jour les compteurs
        self.parameter_count += values.len() as u64;
        self.tensor_count += 1;
    }

    /// Calcule les quantiles à partir des valeurs agrégées.
    ///
    /// Cette méthode nécessite de stocker toutes les valeurs, ce qui n'est
    /// pas fait ici pour des raisons de mémoire. Elle est donc un placeholder.
    /// Calcule les quantiles à partir des valeurs agrégées.
    ///
    /// Cette méthode nécessite de stocker toutes les valeurs, ce qui n'est
    /// pas fait ici pour des raisons de mémoire. Elle est donc un placeholder.
    pub fn compute_quantiles(&mut self, all_values: &[f64]) {
        // Si pas de valeurs, on garde un vecteur vide
        if all_values.is_empty() {
            self.quantiles = Vec::new();
            return;
        }
        // Probabilités standard : 5%, 25%, 50%, 75%, 95%
        let probs = [0.05, 0.25, 0.5, 0.75, 0.95];
        match quantiles(all_values, &probs) {
            Ok(q) => self.quantiles = q,
            Err(_) => {
                // En cas d'erreur (ne devrait pas arriver car on vérifie non vide), on garde vide
                self.quantiles = Vec::new();
            },
        }
    }

    /// Retourne le taux d'outliers en pourcentage.
    pub fn outlier_percentage(&self) -> f64 {
        if self.parameter_count == 0 {
            0.0
        } else {
            (self.outlier_count as f64 / self.parameter_count as f64) * 100.0
        }
    }

    /// Retourne le taux de super-poids en pourcentage.
    pub fn super_weight_percentage(&self) -> f64 {
        if self.parameter_count == 0 {
            0.0
        } else {
            (self.super_weight_count as f64 / self.parameter_count as f64) * 100.0
        }
    }

    /// Réinitialise les statistiques.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_creation() {
        let stats = GenerationStats::new();
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.variance, 0.0);
        assert_eq!(stats.std_dev, 0.0);
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
        assert_eq!(stats.outlier_count, 0);
        assert_eq!(stats.super_weight_count, 0);
        assert_eq!(stats.parameter_count, 0);
        assert_eq!(stats.tensor_count, 0);
    }

    #[test]
    fn stats_update_single_values() {
        let mut stats = GenerationStats::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        stats.update_from_values(&values);

        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.variance, 2.0); // (4+1+0+1+4)/5 = 2
        assert!((stats.std_dev - 2.0_f64.sqrt()).abs() < 1e-10);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.parameter_count, 5);
        assert_eq!(stats.tensor_count, 1);
    }

    #[test]
    fn stats_update_multiple_calls() {
        let mut stats = GenerationStats::new();

        // Premier appel
        let values1 = vec![1.0, 2.0, 3.0];
        stats.update_from_values(&values1);

        // Deuxième appel
        let values2 = vec![4.0, 5.0];
        stats.update_from_values(&values2);

        // Vérifier la moyenne globale (3.0)
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert_eq!(stats.parameter_count, 5);
        assert_eq!(stats.tensor_count, 2);
    }

    #[test]
    fn stats_outlier_detection() {
        let mut stats = GenerationStats::new();
        // Échantillon de 1000 valeurs sans outlier (distribution uniforme)
        let values: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        stats.update_from_values(&values);

        // Avec une distribution uniforme, aucune valeur ne devrait dépasser 3 sigma
        // (pour un échantillon suffisamment grand)
        assert_eq!(stats.outlier_count, 0);

        // Test avec un outlier évident
        let mut stats2 = GenerationStats::new();
        let mut values2: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        values2.push(100000.0); // Outlier évident
        stats2.update_from_values(&values2);

        // Devrait détecter au moins 1 outlier
        assert!(stats2.outlier_count >= 1);
    }

    #[test]
    fn stats_percentages() {
        let mut stats = GenerationStats::new();
        let values = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        stats.update_from_values(&values);

        // Aucun outlier ni super-poids
        assert_eq!(stats.outlier_percentage(), 0.0);
        assert_eq!(stats.super_weight_percentage(), 0.0);
    }

    #[test]
    fn stats_reset() {
        let mut stats = GenerationStats::new();
        let values = vec![1.0, 2.0, 3.0];
        stats.update_from_values(&values);

        stats.reset();

        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.parameter_count, 0);
        assert_eq!(stats.tensor_count, 0);
    }

    #[test]
    fn stats_quantiles_computation() {
        let mut stats = GenerationStats::new();
        // Données connues : de 1 à 100
        let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        stats.update_from_values(&values);

        // Appeler compute_quantiles avec les probabilités standard
        stats.compute_quantiles(&values);

        // Vérifier que les quantiles sont calculés (5%, 25%, 50%, 75%, 95%)
        assert_eq!(stats.quantiles.len(), 5);
        // Le 5ème centile : index = 0.05 * 99 = 4.95 → entre 5 et 6 → 5.95
        assert!((stats.quantiles[0] - 5.95).abs() < 0.01);
        // Le 25ème centile : index = 24.75 → entre 25 et 26 → 25.75
        assert!((stats.quantiles[1] - 25.75).abs() < 0.01);
        // Le 50ème centile (médiane) : index = 49.5 → entre 50 et 51 → 50.5
        assert!((stats.quantiles[2] - 50.5).abs() < 0.01);
        // Le 75ème centile : index = 74.25 → entre 75 et 76 → 75.25
        assert!((stats.quantiles[3] - 75.25).abs() < 0.01);
        // Le 95ème centile : index = 94.05 → entre 95 et 96 → 95.05
        assert!((stats.quantiles[4] - 95.05).abs() < 0.01);
    }

    #[test]
    fn stats_quantiles_empty_data() {
        let mut stats = GenerationStats::new();
        let values: Vec<f64> = Vec::new();

        // compute_quantiles avec données vides ne devrait pas planter
        stats.compute_quantiles(&values);

        // Les quantiles devraient rester vides
        assert!(stats.quantiles.is_empty());
    }

    #[test]
    fn stats_quantiles_single_value() {
        let mut stats = GenerationStats::new();
        let values = vec![42.0];
        stats.update_from_values(&values);

        stats.compute_quantiles(&values);

        // Avec une seule valeur, tous les quantiles devraient être 42.0
        assert_eq!(stats.quantiles.len(), 5);
        for &q in &stats.quantiles {
            assert!((q - 42.0).abs() < 1e-10);
        }
    }
}
