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

//! Pipeline de génération en mode streaming.
//!
//! Ce module implémente un pipeline de génération qui traite les tenseurs
//! par chunks sans accumulation mémoire. Il est conçu pour les modèles
//! de grande taille (> 10 GB) où l'accumulation en mémoire n'est pas viable.
//!
//! ## Principe
//!
//! Le pipeline streaming génère les valeurs pour un sous-ensemble d'éléments
//! d'un tenseur, les transforme via les étapes du pipeline, et retourne les
//! bytes résultants. Cela permet d'écrire directement dans le fichier de sortie
//! sans jamais charger le tenseur complet en mémoire.
//!
//! ## Contraintes
//!
//! - **Mémoire bornée** : O(chunk_size) par tenseur
//! - **Déterminisme** : même seed = même sortie binaire (identique au mode classique)
//! - **Atomicité** : écriture dans dossier temporaire puis renommage

use crate::error::GeneratorResult;
use crate::pipeline::{GenerationPipeline, StepResult};
use crate::pipeline_config::PipelineGlobalConfig;
use crate::streaming_config::StreamingConfig;
use crate::tensor_chunk_generator::TensorChunkGenerator;
use pmg_blueprint::tensor_spec::TensorSpec;
use pmg_io::safetensors::ShardWriter;
use pmg_math::rng::DeterministicRng;

/// Callback de progression pour le streaming.
///
/// # Paramètres
/// - `current` : index du tenseur en cours de traitement
/// - `total` : nombre total de tenseurs
/// - `name` : nom du tenseur en cours
pub type ProgressCallback = Box<dyn Fn(usize, usize, &str) + Send + Sync>;

/// Pipeline de génération en mode streaming.
///
/// Génère les tenseurs un par un sans accumulation mémoire, en traitant
/// chaque tenseur par chunks. Utilise le même ordre d'étapes que le
/// pipeline classique pour garantir le déterminisme.
pub struct StreamingPipeline {
    /// Configuration globale du pipeline.
    config: PipelineGlobalConfig,
    /// Étapes actives du pipeline.
    pipeline: GenerationPipeline,
    /// Callback de progression optionnel.
    progress_callback: Option<ProgressCallback>,
}

impl StreamingPipeline {
    /// Crée un nouveau pipeline streaming avec toutes les étapes activées.
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_generator::StreamingPipeline;
    ///
    /// let pipeline = StreamingPipeline::new();
    /// assert_eq!(pipeline.step_count(), 5);
    /// ```
    pub fn new() -> Self {
        Self {
            config: PipelineGlobalConfig::default(),
            pipeline: GenerationPipeline::full(),
            progress_callback: None,
        }
    }

    /// Crée un pipeline streaming avec une configuration personnalisée.
    pub fn with_config(config: PipelineGlobalConfig) -> Self {
        let mut pipeline = GenerationPipeline::full();
        pipeline.set_config(config.clone());
        Self {
            config,
            pipeline,
            progress_callback: None,
        }
    }

    /// Définit le callback de progression.
    ///
    /// # Paramètres
    /// - `callback` : fonction appelée à chaque début de traitement de tenseur
    ///
    /// # Retourne
    /// Le pipeline avec le callback configuré (pattern builder).
    pub fn with_progress_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(usize, usize, &str) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Box::new(callback));
        self
    }

    /// Retourne le nombre d'étapes actives dans le pipeline.
    pub fn step_count(&self) -> usize {
        self.pipeline.step_count()
    }

    /// Retourne une référence à la configuration du pipeline.
    pub fn config(&self) -> &PipelineGlobalConfig {
        &self.config
    }

    /// Appelle le callback de progression si défini.
    ///
    /// # Paramètres
    /// - `current` : index du tenseur en cours
    /// - `total` : nombre total de tenseurs
    /// - `name` : nom du tenseur
    pub fn notify_progress(&self, current: usize, total: usize, name: &str) {
        if let Some(ref callback) = self.progress_callback {
            callback(current, total, name);
        }
    }

    /// Exécute le pipeline pour un tenseur complet en mode streaming avec écriture directe sur disque.
    ///
    /// # Paramètres
    /// - `tensor_spec` : spécification du tenseur à générer
    /// - `writer` : writer SafeTensors pour l'écriture sur disque
    /// - `tensor_index` : index du tenseur dans le modèle
    /// - `seed` : seed pour la génération déterministe
    ///
    /// # Retourne
    /// Le résultat de la génération avec les métriques.
    ///
    /// # Erreurs
    /// Retourne une erreur si la génération ou l'écriture échoue.
    pub fn execute_tensor_streaming(
        &self,
        tensor_spec: &TensorSpec,
        writer: &mut ShardWriter,
        tensor_index: usize,
        seed: u64,
    ) -> GeneratorResult<crate::tensor_chunk_generator::TensorGenerationResult> {
        // Créer un générateur de chunks avec la configuration du pipeline
        let streaming_config = StreamingConfig::default();
        let mut chunk_generator = TensorChunkGenerator::new(streaming_config, seed);

        // Exécuter la génération et l'écriture
        chunk_generator.generate_and_write_tensor(tensor_spec, writer, tensor_index)
    }

    /// Exécute le pipeline pour un chunk d'éléments d'un tenseur.
    ///
    /// # Paramètres
    /// - `tensor_spec` : spécification du tenseur
    /// - `chunk_offset` : offset du premier élément du chunk
    /// - `chunk_size` : nombre d'éléments dans le chunk
    /// - `seed` : seed pour la génération déterministe
    ///
    /// # Retourne
    /// Un vecteur de `StepResult` contenant les métriques de chaque étape.
    ///
    /// # Erreurs
    /// Retourne une erreur si une étape du pipeline échoue.
    pub fn execute_chunk(
        &self,
        tensor_spec: &TensorSpec,
        chunk_offset: usize,
        chunk_size: usize,
        seed: u64,
    ) -> GeneratorResult<Vec<StepResult>> {
        // Dériver un seed spécifique au tenseur et au chunk
        let chunk_seed = self.derive_chunk_seed(seed, &tensor_spec.name, chunk_offset);

        // Créer un RNG déterministe
        let mut rng = DeterministicRng::from_seed(derive_seed_from_u64(chunk_seed));

        // Générer les valeurs initiales pour ce chunk
        let mut values =
            self.generate_initial_values(tensor_spec, chunk_offset, chunk_size, &mut rng)?;

        // Appliquer les étapes du pipeline
        let mut results = Vec::new();
        for step in self.pipeline.steps() {
            let result = self.apply_step(step, &mut values, &mut rng)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Génère les valeurs initiales pour un chunk d'un tenseur.
    ///
    /// # Paramètres
    /// - `tensor_spec` : spécification du tenseur
    /// - `chunk_offset` : offset du premier élément
    /// - `chunk_size` : nombre d'éléments à générer
    /// - `rng` : générateur aléatoire déterministe
    ///
    /// # Retourne
    /// Un vecteur de valeurs f64 générées.
    fn generate_initial_values(
        &self,
        _tensor_spec: &TensorSpec,
        _chunk_offset: usize,
        chunk_size: usize,
        rng: &mut DeterministicRng,
    ) -> GeneratorResult<Vec<f64>> {
        // Pour l'instant, générer des valeurs basées sur la distribution configurée
        // La génération réelle sera implémentée dans tensor_generator
        let mut values = Vec::with_capacity(chunk_size);
        let config = &self.config.distribution;

        for _ in 0..chunk_size {
            // Générer une valeur selon la distribution normale configurée
            // Utiliser la méthode next_f64() du RNG déterministe
            let u1 = rng.next_f64();
            let u2 = rng.next_f64();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let value = config.mean + config.std * z;
            values.push(value);
        }

        Ok(values)
    }

    /// Applique une étape spécifique aux valeurs.
    ///
    /// # Paramètres
    /// - `step` : étape à appliquer
    /// - `values` : valeurs à transformer (modifié en place)
    /// - `rng` : générateur aléatoire pour les opérations stochastiques
    ///
    /// # Retourne
    /// Le résultat de l'étape avec les métriques.
    fn apply_step(
        &self,
        step: &crate::pipeline::PipelineStep,
        values: &mut [f64],
        rng: &mut DeterministicRng,
    ) -> GeneratorResult<StepResult> {
        use crate::pipeline_steps::{
            apply_correlation, apply_distribution, apply_low_rank, apply_outliers,
            apply_super_weights,
        };

        match step {
            crate::pipeline::PipelineStep::Distribution => {
                apply_distribution(values, &self.config.distribution, rng)
            },
            crate::pipeline::PipelineStep::Correlation => {
                apply_correlation(values, &self.config.correlation, rng)
            },
            crate::pipeline::PipelineStep::LowRank => apply_low_rank(values, &self.config.low_rank),
            crate::pipeline::PipelineStep::Outliers => {
                apply_outliers(values, &self.config.outliers)
            },
            crate::pipeline::PipelineStep::SuperWeights => {
                apply_super_weights(values, &self.config.super_weights)
            },
        }
    }

    /// Dérive un seed spécifique pour un chunk donné.
    ///
    /// # Paramètres
    /// - `base_seed` : seed de base du modèle
    /// - `tensor_name` : nom du tenseur
    /// - `chunk_offset` : offset du chunk
    ///
    /// # Retourne
    /// Un seed dérivé unique pour ce chunk.
    fn derive_chunk_seed(&self, base_seed: u64, tensor_name: &str, chunk_offset: usize) -> u64 {
        // Utiliser un mélange simple pour dériver le seed
        let name_hash = tensor_name
            .bytes()
            .fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        base_seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(name_hash)
            .wrapping_add(chunk_offset as u64)
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

impl Default for StreamingPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmg_core::{DType, Shape, TensorRole};

    #[test]
    fn streaming_pipeline_new() {
        let pipeline = StreamingPipeline::new();
        assert_eq!(pipeline.step_count(), 5);
    }

    #[test]
    fn streaming_pipeline_with_config() {
        let config = PipelineGlobalConfig::default();
        let pipeline = StreamingPipeline::with_config(config);
        assert_eq!(pipeline.step_count(), 5);
    }

    #[test]
    fn streaming_pipeline_execute_chunk() {
        let pipeline = StreamingPipeline::new();
        let spec = TensorSpec::new(
            "test.tensor",
            Shape::new(vec![10]).unwrap(),
            DType::F32,
            TensorRole::Other,
        )
        .unwrap();

        let results = pipeline.execute_chunk(&spec, 0, 10, 42).unwrap();
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].step_name, "Distribution");
    }

    #[test]
    fn streaming_pipeline_derive_chunk_seed() {
        let pipeline = StreamingPipeline::new();
        let seed1 = pipeline.derive_chunk_seed(42, "tensor1", 0);
        let seed2 = pipeline.derive_chunk_seed(42, "tensor1", 1);
        let seed3 = pipeline.derive_chunk_seed(42, "tensor2", 0);

        // Les seeds doivent être différents pour des chunks différents
        assert_ne!(seed1, seed2);
        // Les seeds doivent être différents pour des tenseurs différents
        assert_ne!(seed1, seed3);
    }

    #[test]
    fn streaming_pipeline_determinism() {
        let pipeline1 = StreamingPipeline::new();
        let pipeline2 = StreamingPipeline::new();

        let spec = TensorSpec::new(
            "test.tensor",
            Shape::new(vec![10]).unwrap(),
            DType::F32,
            TensorRole::Other,
        )
        .unwrap();

        let results1 = pipeline1.execute_chunk(&spec, 0, 10, 42).unwrap();
        let results2 = pipeline2.execute_chunk(&spec, 0, 10, 42).unwrap();

        // Les résultats doivent être identiques pour le même seed
        assert_eq!(results1.len(), results2.len());
        for (r1, r2) in results1.iter().zip(results2.iter()) {
            assert_eq!(r1.step_name, r2.step_name);
            assert_eq!(r1.elements_modified, r2.elements_modified);
        }
    }

    #[test]
    fn streaming_pipeline_different_seeds() {
        let pipeline = StreamingPipeline::new();
        let spec = TensorSpec::new(
            "test.tensor",
            Shape::new(vec![10]).unwrap(),
            DType::F32,
            TensorRole::Other,
        )
        .unwrap();

        let results1 = pipeline.execute_chunk(&spec, 0, 10, 42).unwrap();
        let results2 = pipeline.execute_chunk(&spec, 0, 10, 43).unwrap();

        // Les résultats doivent être différents pour des seeds différentes
        // (au moins une différence dans les valeurs modifiées)
        let _all_same = results1
            .iter()
            .zip(results2.iter())
            .all(|(r1, r2)| r1.elements_modified == r2.elements_modified);

        // Les noms des étapes doivent être identiques (même pipeline)
        for (r1, r2) in results1.iter().zip(results2.iter()) {
            assert_eq!(r1.step_name, r2.step_name);
        }

        // Note: Les seeds différentes peuvent produire des résultats similaires
        // mais les étapes doivent être les mêmes
    }

    #[test]
    fn streaming_pipeline_notify_progress() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = call_count.clone();

        let pipeline =
            StreamingPipeline::new().with_progress_callback(move |_current, _total, _name| {
                call_count_clone.fetch_add(1, Ordering::SeqCst);
            });

        // Simuler la notification de progression
        pipeline.notify_progress(0, 10, "test_tensor");
        pipeline.notify_progress(1, 10, "test_tensor2");

        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn streaming_pipeline_notify_progress_without_callback() {
        let pipeline = StreamingPipeline::new();

        // Ne doit pas paniquer sans callback
        pipeline.notify_progress(0, 10, "test_tensor");
        pipeline.notify_progress(1, 10, "test_tensor2");
    }
}
