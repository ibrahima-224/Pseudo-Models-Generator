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

//! Pipeline de génération déterministe.
//!
//! Ce module définit l'ordonnancement des étapes de transformation des valeurs
//! d'un tenseur. Le pipeline ne réimplémente pas les mathématiques, mais coordonne
//! l'application séquentielle des opérations dans l'ordre correct :
//!
//! ```text
//! Distribution → Corrélation → Bas-rang → Outliers → Super-poids
//! ```
//!
//! Chaque étape peut être activée ou désactivée, permettant des configurations
//! de génération flexibles tout en préservant le déterminisme.

use crate::error::GeneratorResult;
use crate::pipeline_config::{
    CorrelationConfig, DistributionConfig, LowRankConfig, OutliersConfig, PipelineGlobalConfig,
    SuperWeightsConfig,
};
use crate::pipeline_steps::{
    apply_correlation, apply_distribution, apply_low_rank, apply_outliers, apply_super_weights,
};
use pmg_math::rng::DeterministicRng;

// Import conditionnel pour le support GPU
#[cfg(feature = "gpu-acceleration")]
use pmg_gpu::{GpuAccelerated, GpuDevice, NormalGenerationAccelerated};

/// Étape du pipeline de génération.
///
/// Chaque variante correspond à une opération mathématique spécifique
/// appliquée aux valeurs du tenseur. Les étapes sont exécutées dans
/// un ordre fixe pour garantir la reproductibilité.
///
/// # Exemple
///
/// ```
/// use pmg_generator::PipelineStep;
///
/// assert_eq!(PipelineStep::Distribution.name(), "Distribution");
/// assert_eq!(PipelineStep::Distribution.order(), 0);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineStep {
    /// Étape de distribution : échantillonnage selon une distribution statistique.
    Distribution,
    /// Étape de corrélation : application de corrélations structurelles.
    Correlation,
    /// Étape de bas-rang : réduction de rang pour les matrices de poids.
    LowRank,
    /// Étape d'outliers : injection de valeurs aberrantes contrôlées.
    Outliers,
    /// Étape de super-poids : injection de super-poids pour les couches critiques.
    SuperWeights,
}

impl PipelineStep {
    /// Retourne le nom de l'étape.
    pub fn name(&self) -> &'static str {
        match self {
            PipelineStep::Distribution => "Distribution",
            PipelineStep::Correlation => "Corrélation",
            PipelineStep::LowRank => "Bas-rang",
            PipelineStep::Outliers => "Outliers",
            PipelineStep::SuperWeights => "Super-poids",
        }
    }

    /// Retourne le nom lisible en français de l'étape.
    pub fn display_name(&self) -> &'static str {
        self.name()
    }

    /// Retourne une description détaillée de l'étape.
    pub fn description(&self) -> &'static str {
        match self {
            PipelineStep::Distribution => {
                "Échantillonnage initial selon la distribution configurée"
            },
            PipelineStep::Correlation => {
                "Application de corrélations structurelles entre les éléments"
            },
            PipelineStep::LowRank => "Réduction de rang pour les matrices de poids (bas-rang)",
            PipelineStep::Outliers => "Injection de valeurs aberrantes contrôlées (outliers)",
            PipelineStep::SuperWeights => "Injection de super-poids pour les couches critiques",
        }
    }

    /// Retourne l'ordre d'exécution de l'étape (0-based).
    ///
    /// L'ordre est fixe et correspond à la séquence mathématique correcte.
    pub fn order(&self) -> u8 {
        match self {
            PipelineStep::Distribution => 0,
            PipelineStep::Correlation => 1,
            PipelineStep::LowRank => 2,
            PipelineStep::Outliers => 3,
            PipelineStep::SuperWeights => 4,
        }
    }
}

/// Résultat de l'application d'une étape du pipeline.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Nom de l'étape appliquée.
    pub step_name: String,
    /// Nombre d'éléments modifiés.
    pub elements_modified: usize,
    /// Métriques spécifiques à l'étape (clés-valeurs).
    pub metrics: std::collections::HashMap<String, f64>,
}

/// Pipeline de génération pour un tenseur.
///
/// Contient la séquence d'étapes à appliquer et fournit des méthodes
/// pour exécuter le pipeline sur des vecteurs de valeurs. Le pipeline
/// garantit l'ordre d'exécution correct et la reproductibilité.
///
/// # Exemple
///
/// ```
/// use pmg_generator::{GenerationPipeline, PipelineStep};
///
/// let mut pipeline = GenerationPipeline::empty();
/// pipeline.add_step(PipelineStep::Distribution);
/// pipeline.add_step(PipelineStep::Outliers);
///
/// assert_eq!(pipeline.step_count(), 2);
/// assert!(pipeline.has_step(&PipelineStep::Distribution));
/// assert!(pipeline.has_step(&PipelineStep::Outliers));
/// assert!(!pipeline.has_step(&PipelineStep::Correlation));
/// ```
#[derive(Debug, Clone)]
pub struct GenerationPipeline {
    /// Étapes actives du pipeline, dans l'ordre d'exécution.
    steps: Vec<PipelineStep>,
    /// Configuration globale du pipeline.
    config: PipelineGlobalConfig,
    /// Device GPU optionnel pour l'accélération
    #[cfg(feature = "gpu-acceleration")]
    device: Option<GpuDevice>,
}

impl GenerationPipeline {
    /// Crée un pipeline avec toutes les étapes activées dans l'ordre correct.
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_generator::GenerationPipeline;
    ///
    /// let pipeline = GenerationPipeline::full();
    /// assert_eq!(pipeline.step_count(), 5);
    /// ```
    pub fn full() -> Self {
        Self {
            steps: vec![
                PipelineStep::Distribution,
                PipelineStep::Correlation,
                PipelineStep::LowRank,
                PipelineStep::Outliers,
                PipelineStep::SuperWeights,
            ],
            config: PipelineGlobalConfig::default(),
            #[cfg(feature = "gpu-acceleration")]
            device: None,
        }
    }

    /// Crée un pipeline vide (aucune étape).
    pub fn empty() -> Self {
        Self {
            steps: Vec::new(),
            config: PipelineGlobalConfig::default(),
            #[cfg(feature = "gpu-acceleration")]
            device: None,
        }
    }

    /// Crée un pipeline avec uniquement la distribution (étape de base).
    pub fn distribution_only() -> Self {
        Self {
            steps: vec![PipelineStep::Distribution],
            config: PipelineGlobalConfig::default(),
            #[cfg(feature = "gpu-acceleration")]
            device: None,
        }
    }

    /// Ajoute une étape au pipeline.
    ///
    /// L'ajout est fait à la fin, mais l'ordre d'exécution est toujours
    /// déterminé par l'ordre interne des étapes.
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_generator::{GenerationPipeline, PipelineStep};
    ///
    /// let mut pipeline = GenerationPipeline::empty();
    /// pipeline.add_step(PipelineStep::Outliers);
    /// pipeline.add_step(PipelineStep::Distribution);
    ///
    /// // L'ordre d'exécution est corrigé
    /// let steps = pipeline.steps();
    /// assert_eq!(steps[0], PipelineStep::Distribution);
    /// assert_eq!(steps[1], PipelineStep::Outliers);
    /// ```
    pub fn add_step(&mut self, step: PipelineStep) {
        if !self.steps.contains(&step) {
            self.steps.push(step);
            // Trier par ordre d'exécution
            self.steps.sort_by_key(|s| s.order());
        }
    }

    /// Supprime une étape du pipeline.
    pub fn remove_step(&mut self, step: &PipelineStep) {
        self.steps.retain(|s| s != step);
    }

    /// Vérifie si une étape est présente dans le pipeline.
    pub fn has_step(&self, step: &PipelineStep) -> bool {
        self.steps.contains(step)
    }

    /// Retourne le nombre d'étapes actives.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Retourne une référence aux étapes dans l'ordre d'exécution.
    pub fn steps(&self) -> &[PipelineStep] {
        &self.steps
    }

    /// Exécute le pipeline sur un vecteur de valeurs.
    ///
    /// # Paramètres
    /// - `values` : vecteur de valeurs à transformer (modifié en place)
    /// - `seed` : seed pour les opérations stochastiques
    ///
    /// # Retourne
    /// Un vecteur de `StepResult` contenant les métriques de chaque étape appliquée.
    ///
    /// # Erreurs
    /// Retourne une erreur si une étape échoue.
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_generator::GenerationPipeline;
    ///
    /// let pipeline = GenerationPipeline::full();
    /// let mut values = vec![0.0; 100];
    /// let results = pipeline.execute(&mut values, 42).unwrap();
    /// assert_eq!(results.len(), 5);
    /// ```
    pub fn execute(&self, values: &mut [f64], seed: u64) -> GeneratorResult<Vec<StepResult>> {
        let mut results = Vec::new();

        // Crée un RNG déterministe à partir du seed
        let mut rng = DeterministicRng::from_seed(derive_seed_from_u64(seed));

        for step in &self.steps {
            let result = self.apply_step(
                step,
                values,
                &mut rng,
                #[cfg(feature = "gpu-acceleration")]
                self.device.as_ref(),
            )?;
            results.push(result);
        }

        Ok(results)
    }

    /// Applique une étape spécifique aux valeurs.
    ///
    /// Cette méthode délègue l'application de chaque étape aux fonctions
    /// d'implémentation appropriées dans `pipeline_steps`.
    fn apply_step(
        &self,
        step: &PipelineStep,
        values: &mut [f64],
        rng: &mut DeterministicRng,
        #[cfg(feature = "gpu-acceleration")] device: Option<&GpuDevice>,
    ) -> GeneratorResult<StepResult> {
        match step {
            PipelineStep::Distribution => {
                // Utiliser le GPU pour la génération normale si disponible
                #[cfg(feature = "gpu-acceleration")]
                {
                    if let Some(dev) = device {
                        // Utiliser NormalGenerationAccelerated avec le device GPU
                        let generator = NormalGenerationAccelerated::new(rng.seed());
                        let input = (
                            values.len(),
                            self.config.distribution.mean,
                            self.config.distribution.std,
                        );

                        // Exécuter sur GPU avec fallback CPU automatique
                        match generator.execute(&input, Some(dev)) {
                            Ok(new_values) => {
                                // Remplacer les valeurs existantes
                                values.copy_from_slice(&new_values);

                                let mut metrics = std::collections::HashMap::new();
                                metrics.insert("distribution_type".to_string(), 1.0);
                                metrics.insert("mean".to_string(), self.config.distribution.mean);
                                metrics.insert("std".to_string(), self.config.distribution.std);
                                metrics.insert("gpu_accelerated".to_string(), 1.0);

                                return Ok(StepResult {
                                    step_name: PipelineStep::Distribution.name().to_string(),
                                    elements_modified: values.len(),
                                    metrics,
                                });
                            },
                            Err(_) => {
                                // Fallback sur CPU en cas d'erreur GPU
                                log::warn!("Échec de l'accélération GPU pour la distribution, fallback CPU");
                            },
                        }
                    }
                }

                // Fallback CPU (ou si pas de GPU disponible)
                apply_distribution(values, &self.config.distribution, rng)
            },
            PipelineStep::Correlation => apply_correlation(values, &self.config.correlation, rng),
            PipelineStep::LowRank => apply_low_rank(values, &self.config.low_rank),
            PipelineStep::Outliers => apply_outliers(values, &self.config.outliers),
            PipelineStep::SuperWeights => apply_super_weights(values, &self.config.super_weights),
        }
    }

    /// Valide la cohérence du pipeline.
    ///
    /// Vérifie que les étapes sont dans l'ordre correct et qu'il n'y a pas de doublons.
    pub fn validate(&self) -> GeneratorResult<()> {
        // Vérifie que les étapes sont dans l'ordre correct
        for (i, step) in self.steps.iter().enumerate() {
            if step.order() as usize != i {
                return Err(crate::error::GeneratorError::PipelineError(format!(
                    "étape '{}' à la position {} au lieu de {}",
                    step.name(),
                    i,
                    step.order()
                )));
            }
        }
        Ok(())
    }
    /// Retourne une référence à la configuration du pipeline.
    pub fn config(&self) -> &PipelineGlobalConfig {
        &self.config
    }

    /// Définit la configuration du pipeline.
    pub fn set_config(&mut self, config: PipelineGlobalConfig) {
        self.config = config;
    }

    /// Définit la configuration de l'étape de distribution.
    pub fn set_distribution_config(&mut self, config: DistributionConfig) {
        self.config.distribution = config;
    }

    /// Définit la configuration de l'étape de corrélation.
    pub fn set_correlation_config(&mut self, config: CorrelationConfig) {
        self.config.correlation = config;
    }

    /// Définit la configuration de l'étape de bas-rang.
    pub fn set_low_rank_config(&mut self, config: LowRankConfig) {
        self.config.low_rank = config;
    }

    /// Définit la configuration de l'étape d'outliers.
    pub fn set_outliers_config(&mut self, config: OutliersConfig) {
        self.config.outliers = config;
    }

    /// Définit la configuration de l'étape de super-poids.
    pub fn set_super_weights_config(&mut self, config: SuperWeightsConfig) {
        self.config.super_weights = config;
    }
}

/// Fonction utilitaire pour dériver un seed à partir d'un u64.
fn derive_seed_from_u64(seed: u64) -> [u8; 32] {
    let mut result = [0u8; 32];
    let bytes = seed.to_le_bytes();
    result[..8].copy_from_slice(&bytes);
    // Ajoute un mélange simple pour améliorer la distribution
    for i in 8..32 {
        result[i] = result[i % 8].wrapping_add(i as u8);
    }
    result
}

impl Default for GenerationPipeline {
    fn default() -> Self {
        Self::full()
    }
}

// Méthodes pour la gestion du device GPU (uniquement avec la feature gpu-acceleration)
#[cfg(feature = "gpu-acceleration")]
impl GenerationPipeline {
    /// Définit le device GPU à utiliser pour l'accélération
    pub fn set_device(&mut self, device: GpuDevice) {
        self.device = Some(device);
    }

    /// Retourne une référence au device GPU si disponible
    pub fn device(&self) -> Option<&GpuDevice> {
        self.device.as_ref()
    }

    /// Vérifie si un device GPU est disponible
    pub fn has_device(&self) -> bool {
        self.device.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_step_order() {
        assert_eq!(PipelineStep::Distribution.order(), 0);
        assert_eq!(PipelineStep::Correlation.order(), 1);
        assert_eq!(PipelineStep::LowRank.order(), 2);
        assert_eq!(PipelineStep::Outliers.order(), 3);
        assert_eq!(PipelineStep::SuperWeights.order(), 4);
    }

    #[test]
    fn pipeline_full_has_all_steps() {
        let pipeline = GenerationPipeline::full();
        assert_eq!(pipeline.step_count(), 5);
        assert!(pipeline.has_step(&PipelineStep::Distribution));
        assert!(pipeline.has_step(&PipelineStep::Correlation));
        assert!(pipeline.has_step(&PipelineStep::LowRank));
        assert!(pipeline.has_step(&PipelineStep::Outliers));
        assert!(pipeline.has_step(&PipelineStep::SuperWeights));
    }

    #[test]
    fn pipeline_empty_has_no_steps() {
        let pipeline = GenerationPipeline::empty();
        assert_eq!(pipeline.step_count(), 0);
    }

    #[test]
    fn pipeline_distribution_only() {
        let pipeline = GenerationPipeline::distribution_only();
        assert_eq!(pipeline.step_count(), 1);
        assert!(pipeline.has_step(&PipelineStep::Distribution));
    }

    #[test]
    fn pipeline_add_step_maintains_order() {
        let mut pipeline = GenerationPipeline::empty();
        pipeline.add_step(PipelineStep::SuperWeights);
        pipeline.add_step(PipelineStep::Distribution);
        pipeline.add_step(PipelineStep::Outliers);

        let steps = pipeline.steps();
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0], PipelineStep::Distribution);
        assert_eq!(steps[1], PipelineStep::Outliers);
        assert_eq!(steps[2], PipelineStep::SuperWeights);
    }

    #[test]
    fn pipeline_remove_step() {
        let mut pipeline = GenerationPipeline::full();
        pipeline.remove_step(&PipelineStep::LowRank);

        assert_eq!(pipeline.step_count(), 4);
        assert!(!pipeline.has_step(&PipelineStep::LowRank));
    }

    #[test]
    fn pipeline_execute_empty() {
        let pipeline = GenerationPipeline::empty();
        let mut values = vec![1.0, 2.0, 3.0];
        let results = pipeline.execute(&mut values, 42).unwrap();
        assert_eq!(results.len(), 0);
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn pipeline_execute_distribution_only() {
        let pipeline = GenerationPipeline::distribution_only();
        let mut values = vec![1.0, 2.0, 3.0];
        let original_values = values.clone();
        let results = pipeline.execute(&mut values, 42).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].step_name, "Distribution");
        assert_eq!(results[0].elements_modified, 3);
        // Vérifie que les valeurs ont été modifiées
        assert_ne!(values, original_values);
    }

    #[test]
    fn pipeline_execute_full() {
        let pipeline = GenerationPipeline::full();
        // Utiliser une taille qui est un produit de deux entiers (2×2=4) pour la matrice bas-rang
        let mut values = vec![1.0, 2.0, 3.0, 100.0]; // 4 éléments
        let results = pipeline.execute(&mut values, 42).unwrap();

        assert_eq!(results.len(), 5);
        // Vérifier que les étapes sont dans le bon ordre
        assert_eq!(results[0].step_name, "Distribution");
        assert_eq!(results[1].step_name, "Corrélation");
        assert_eq!(results[2].step_name, "Bas-rang");
        assert_eq!(results[3].step_name, "Outliers");
        assert_eq!(results[4].step_name, "Super-poids");
    }

    #[test]
    fn pipeline_config_default() {
        let pipeline = GenerationPipeline::full();
        let config = pipeline.config();

        // Vérifie les valeurs par défaut
        assert_eq!(config.distribution.mean, 0.0);
        assert_eq!(config.distribution.std, 1.0);
        assert_eq!(config.correlation.dim, 1);
        assert_eq!(config.low_rank.energy_threshold, 0.9);
        assert_eq!(config.low_rank.rank_reduction, 0.5);
        assert_eq!(config.outliers.threshold_k, 3.0);
        assert_eq!(config.super_weights.threshold_k, 5.0);
    }

    #[test]
    fn pipeline_config_modification() {
        let mut pipeline = GenerationPipeline::full();

        // Modifie la configuration
        let new_config = PipelineGlobalConfig {
            distribution: DistributionConfig {
                mean: 1.0,
                std: 2.0,
            },
            ..PipelineGlobalConfig::default()
        };

        pipeline.set_config(new_config);

        // Vérifie que la configuration a été mise à jour
        let config = pipeline.config();
        assert_eq!(config.distribution.mean, 1.0);
        assert_eq!(config.distribution.std, 2.0);
    }
}
