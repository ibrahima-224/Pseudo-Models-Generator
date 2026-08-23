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

//! Générateur complet de modèle.
//!
//! Ce module définit la structure `ModelGeneratorComplete` qui parcourt le blueprint
//! complet (Embedding → Layer 0 → Layer N → Final Norm → LM Head) et produit
//! tous les tenseurs du modèle en utilisant le pipeline de génération.
//!
//! ## Modes de génération
//!
//! - **Mode classique** (`generate_all`) : génère tous les tenseurs en mémoire
//! - **Mode streaming** (`generate_and_write`) : écrit chaque tenseur directement
//!   dans un fichier Safetensors, éliminant l'accumulation en mémoire pour les
//!   grands modèles (> 10 GB).

use crate::context::GenerationContext;

/// Type alias pour le callback de progression.
/// Prend (tensor_index, total_tensors, tensor_name) en paramètres.
type ProgressCallback<'a> = Option<&'a dyn Fn(usize, usize, &str)>;
use crate::error::GeneratorResult;
use crate::generation_stats::GenerationStats;
use crate::layer_generator::LayerGenerator;
use crate::pipeline::{GenerationPipeline, StepResult};
use crate::seed_plan::GeneratorSeedPlan;
use crate::tensor_generator::TensorGenerator;
use pmg_blueprint::ModelBlueprint;

/// Helpers pour le streaming tension par tension.
pub mod streaming;

/// Résultat de la génération d'un tenseur du modèle.
#[derive(Debug, Clone)]
pub struct ModelTensorResult {
    /// Nom du tenseur.
    pub name: String,
    /// Valeurs générées.
    pub values: Vec<f64>,
    /// Résultats du pipeline.
    pub pipeline_results: Vec<StepResult>,
    /// Catégorie du tenseur (embedding, layer, norm, lm_head, extra).
    pub category: String,
    /// Index de la couche (si applicable).
    pub layer_index: Option<usize>,
}

/// Générateur complet de modèle.
///
/// Parcourt le blueprint complet et produit tous les tenseurs du modèle
/// en utilisant le pipeline de génération. Préserve les relations structurelles
/// entre les tenseurs d'une même couche.
pub struct ModelGeneratorComplete {
    /// Blueprint du modèle.
    blueprint: ModelBlueprint,
    /// Seed globale de génération.
    seed: u64,
    /// Version du générateur.
    generation_version: String,
    /// Pipeline de génération.
    pipeline: GenerationPipeline,
    /// Taille des chunks.
    chunk_size: usize,
    /// Budget tensoriel disponible (optionnel, en octets).
    tensor_budget: Option<u64>,
}

impl ModelGeneratorComplete {
    /// Crée un nouveau générateur complet.
    ///
    /// # Paramètres
    /// - `blueprint` : blueprint du modèle
    /// - `seed` : seed globale de génération
    /// - `generation_version` : version du générateur
    /// - `pipeline` : pipeline de génération à appliquer
    /// - `chunk_size` : taille des chunks pour la génération
    pub fn new(
        blueprint: ModelBlueprint,
        seed: u64,
        generation_version: impl Into<String>,
        pipeline: GenerationPipeline,
        chunk_size: usize,
    ) -> Self {
        Self {
            blueprint,
            seed,
            generation_version: generation_version.into(),
            pipeline,
            chunk_size,
            tensor_budget: None,
        }
    }

    /// Crée un nouveau générateur complet avec un budget tensoriel.
    pub fn with_budget(
        blueprint: ModelBlueprint,
        seed: u64,
        generation_version: impl Into<String>,
        pipeline: GenerationPipeline,
        chunk_size: usize,
        tensor_budget: Option<u64>,
    ) -> Self {
        Self {
            blueprint,
            seed,
            generation_version: generation_version.into(),
            pipeline,
            chunk_size,
            tensor_budget,
        }
    }

    /// Définit le budget tensoriel disponible.
    pub fn set_tensor_budget(&mut self, budget: Option<u64>) {
        self.tensor_budget = budget;
    }

    /// Retourne le budget tensoriel disponible.
    pub fn tensor_budget(&self) -> Option<u64> {
        self.tensor_budget
    }

    /// Génère tous les tenseurs du modèle.
    ///
    /// # Retourne
    /// Un vecteur de `ModelTensorResult` contenant tous les tenseurs générés.
    ///
    /// # Erreurs
    /// Retourne une erreur si la génération d'un tenseur échoue.
    pub fn generate_all(&self) -> GeneratorResult<Vec<ModelTensorResult>> {
        let mut results = Vec::new();

        // 1. Générer les embeddings
        for (tensor_index, tensor_spec) in self.blueprint.embeddings.iter().enumerate() {
            let result = self.generate_tensor(tensor_spec, "embedding", None, tensor_index)?;
            results.push(result);
        }

        // 2. Générer les couches
        for (layer_index, layer_spec) in self.blueprint.layers.iter().enumerate() {
            let layer_gen = LayerGenerator::new(
                layer_spec.clone(),
                self.seed,
                &self.blueprint.id,
                &self.generation_version,
                self.pipeline.clone(),
                self.chunk_size,
            );

            let layer_results = layer_gen.generate_all()?;
            for (name, values, pipeline_results) in layer_results.into_iter() {
                results.push(ModelTensorResult {
                    name,
                    values,
                    pipeline_results,
                    category: "layer".to_string(),
                    layer_index: Some(layer_index),
                });
            }
        }

        // 3. Générer la norme finale
        for (tensor_index, tensor_spec) in self.blueprint.final_norm.iter().enumerate() {
            let result = self.generate_tensor(tensor_spec, "final_norm", None, tensor_index)?;
            results.push(result);
        }

        // 4. Générer la tête de langage
        for (tensor_index, tensor_spec) in self.blueprint.lm_head.iter().enumerate() {
            let result = self.generate_tensor(tensor_spec, "lm_head", None, tensor_index)?;
            results.push(result);
        }

        // 5. Générer les tenseurs supplémentaires
        for (tensor_index, tensor_spec) in self.blueprint.extra_tensors.iter().enumerate() {
            let result = self.generate_tensor(tensor_spec, "extra", None, tensor_index)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Génère un tenseur individuel du modèle.
    ///
    /// # Paramètres
    /// - `tensor_spec` : spécification du tenseur
    /// - `category` : catégorie du tenseur (embedding, layer, norm, lm_head, extra)
    /// - `layer_index` : index de la couche (si applicable)
    /// - `tensor_index` : index du tenseur dans la catégorie
    fn generate_tensor(
        &self,
        tensor_spec: &pmg_blueprint::tensor_spec::TensorSpec,
        category: &str,
        layer_index: Option<usize>,
        tensor_index: usize,
    ) -> GeneratorResult<ModelTensorResult> {
        let num_elements = tensor_spec.num_elements()? as usize;

        // Créer le contexte de génération
        let context = GenerationContext::new(
            self.seed,
            &self.blueprint.id,
            &self.generation_version,
            layer_index,
            tensor_index,
            0, // chunk_index initial
            &tensor_spec.name,
            num_elements,
            self.chunk_size,
        );

        // Créer le plan de seed
        let seed_plan =
            GeneratorSeedPlan::new(self.seed, &self.blueprint.id, &self.generation_version);

        // Générer les valeurs initiales avec le budget tensoriel si disponible
        let tensor_gen = TensorGenerator::new(tensor_spec.clone(), seed_plan, self.tensor_budget);
        let mut values = tensor_gen.generate()?;

        // Appliquer le pipeline
        let pipeline_results = self.pipeline.execute(&mut values, context.tensor_seed())?;

        Ok(ModelTensorResult {
            name: tensor_spec.name.clone(),
            values,
            pipeline_results,
            category: category.to_string(),
            layer_index,
        })
    }

    /// Calcule les statistiques globales du modèle.
    ///
    /// # Paramètres
    /// - `results` : résultats de la génération de tous les tenseurs
    ///
    /// # Retourne
    /// Statistiques agrégées du modèle.
    pub fn compute_stats(&self, results: &[ModelTensorResult]) -> GenerationStats {
        let mut stats = GenerationStats::new();
        let mut all_values = Vec::new();

        for result in results {
            // Collecter toutes les valeurs pour le calcul des quantiles
            all_values.extend(&result.values);
            stats.update_from_values(&result.values);
            // Mettre à jour les compteurs à partir des résultats du pipeline
            for step_result in &result.pipeline_results {
                if let Some(&count) = step_result.metrics.get("outlier_count") {
                    stats.outlier_count += count as usize;
                }
                if let Some(&count) = step_result.metrics.get("super_weight_count") {
                    stats.super_weight_count += count as usize;
                }
            }
        }

        // Calculer les quantiles à partir de toutes les valeurs collectées
        stats.compute_quantiles(&all_values);

        stats
    }

    /// Retourne le blueprint.
    pub fn blueprint(&self) -> &ModelBlueprint {
        &self.blueprint
    }

    /// Retourne la seed globale.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Retourne la version du générateur.
    pub fn generation_version(&self) -> &str {
        &self.generation_version
    }

    /// Génère et écrit chaque tenseur directement dans le writer (mode streaming).
    ///
    /// Élimine l'accumulation en mémoire pour les grands modèles en écrivant
    /// chaque tenseur directement dans le fichier Safetensors.
    ///
    /// # Paramètres
    /// - `writer` : writer Safetensors pour l'écriture
    /// - `progress_callback` : callback optionnel pour le suivi de progression
    ///
    /// # Retourne
    /// Statistiques de génération.
    ///
    /// # Erreurs
    /// Retourne une erreur si la génération ou l'écriture échoue.
    pub fn generate_and_write(
        &self,
        writer: &mut pmg_io::safetensors::ShardWriter,
        progress_callback: ProgressCallback<'_>,
    ) -> GeneratorResult<GenerationStats> {
        let mut stats = GenerationStats::new();
        let total_tensors = streaming::count_total_tensors(&self.blueprint);
        let mut current_tensor = 0;

        // 1. Générer et écrire les embeddings
        for tensor_spec in &self.blueprint.embeddings {
            current_tensor += 1;
            if let Some(callback) = progress_callback {
                callback(current_tensor, total_tensors, &tensor_spec.name);
            }
            self.generate_and_write_tensor(tensor_spec, "embedding", None, writer, &mut stats)?;
        }

        // 2. Générer et écrire les couches
        for (layer_index, layer_spec) in self.blueprint.layers.iter().enumerate() {
            // Tenseurs de la couche (attention + mlp + norms + hyper_connections + MoE)
            for tensor_spec in layer_spec.all_tensors() {
                current_tensor += 1;
                if let Some(callback) = progress_callback {
                    callback(current_tensor, total_tensors, &tensor_spec.name);
                }
                self.generate_and_write_tensor(
                    tensor_spec,
                    "layer",
                    Some(layer_index),
                    writer,
                    &mut stats,
                )?;
            }
        }

        // 3. Générer et écrire la norme finale
        for tensor_spec in &self.blueprint.final_norm {
            current_tensor += 1;
            if let Some(callback) = progress_callback {
                callback(current_tensor, total_tensors, &tensor_spec.name);
            }
            self.generate_and_write_tensor(tensor_spec, "final_norm", None, writer, &mut stats)?;
        }

        // 4. Générer et écrire la tête de langage
        for tensor_spec in &self.blueprint.lm_head {
            current_tensor += 1;
            if let Some(callback) = progress_callback {
                callback(current_tensor, total_tensors, &tensor_spec.name);
            }
            self.generate_and_write_tensor(tensor_spec, "lm_head", None, writer, &mut stats)?;
        }

        // 5. Générer et écrire les tenseurs supplémentaires
        for tensor_spec in &self.blueprint.extra_tensors {
            current_tensor += 1;
            if let Some(callback) = progress_callback {
                callback(current_tensor, total_tensors, &tensor_spec.name);
            }
            self.generate_and_write_tensor(tensor_spec, "extra", None, writer, &mut stats)?;
        }

        Ok(stats)
    }

    /// Génère et écrit un tenseur individuel en mode streaming.
    ///
    /// # Paramètres
    /// - `tensor_spec` : spécification du tenseur
    /// - `category` : catégorie du tenseur
    /// - `layer_index` : index de la couche (si applicable)
    /// - `writer` : writer Safetensors
    /// - `stats` : statistiques à mettre à jour
    fn generate_and_write_tensor(
        &self,
        tensor_spec: &pmg_blueprint::tensor_spec::TensorSpec,
        _category: &str,
        layer_index: Option<usize>,
        writer: &mut pmg_io::safetensors::ShardWriter,
        stats: &mut GenerationStats,
    ) -> GeneratorResult<()> {
        let num_elements = tensor_spec.num_elements()? as usize;

        // Créer le contexte de génération
        let context = GenerationContext::new(
            self.seed,
            &self.blueprint.id,
            &self.generation_version,
            layer_index, // Déjà Option<usize>
            0,           // tensor_index
            0,           // chunk_index initial
            &tensor_spec.name,
            num_elements,
            self.chunk_size,
        );

        // Créer le plan de seed
        let seed_plan =
            GeneratorSeedPlan::new(self.seed, &self.blueprint.id, &self.generation_version);

        // Générer les valeurs initiales
        let tensor_gen = TensorGenerator::new(tensor_spec.clone(), seed_plan, self.tensor_budget);
        let mut values = tensor_gen.generate()?;

        // Appliquer le pipeline
        let _pipeline_results = self.pipeline.execute(&mut values, context.tensor_seed())?;

        // Écrire le tenseur en streaming
        streaming::write_tensor_streaming(tensor_spec, &values, writer, stats)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmg_blueprint::architecture::ArchitectureKind;
    use pmg_blueprint::naming::NamingRules;
    use pmg_blueprint::tensor_spec::TensorSpec;
    use pmg_core::model_config::glm52_test_config;
    use pmg_core::{DType, Shape, TensorRole};

    fn test_blueprint() -> ModelBlueprint {
        let config = glm52_test_config();
        let mut bp = ModelBlueprint::new(
            "glm-5.2",
            ArchitectureKind::MoETransformer,
            config,
            NamingRules::glm52(),
        );

        // Ajouter un embedding
        bp.embeddings.push(
            TensorSpec::new(
                "model.embed_tokens.weight",
                Shape::new(vec![100, 64]).unwrap(),
                DType::F32,
                TensorRole::Embedding,
            )
            .unwrap(),
        );

        // Ajouter une couche
        let mut layer =
            pmg_blueprint::layer::LayerSpec::new(0, pmg_blueprint::layer::LayerKind::Dense);
        layer.attention.push(
            TensorSpec::new(
                "model.layers.0.self_attn.q_proj.weight",
                Shape::new(vec![64, 64]).unwrap(),
                DType::F32,
                TensorRole::AttentionQuery,
            )
            .unwrap(),
        );
        bp.layers.push(layer);

        // Ajouter une norme finale
        bp.final_norm.push(
            TensorSpec::new(
                "model.norm.weight",
                Shape::new(vec![64]).unwrap(),
                DType::F32,
                TensorRole::Norm,
            )
            .unwrap(),
        );

        // Ajouter une tête de langage
        bp.lm_head.push(
            TensorSpec::new(
                "lm_head.weight",
                Shape::new(vec![100, 64]).unwrap(),
                DType::F32,
                TensorRole::LmHead,
            )
            .unwrap(),
        );

        bp
    }

    #[test]
    fn model_generator_creation() {
        let blueprint = test_blueprint();
        let pipeline = GenerationPipeline::full();

        let gen = ModelGeneratorComplete::new(blueprint, 42, "1.0.0", pipeline, 256);

        assert_eq!(gen.seed(), 42);
        assert_eq!(gen.generation_version(), "1.0.0");
    }

    #[test]
    fn model_generator_generate_all() {
        let blueprint = test_blueprint();
        let pipeline = GenerationPipeline::full();

        let gen = ModelGeneratorComplete::new(blueprint, 42, "1.0.0", pipeline, 256);

        let results = gen.generate_all().unwrap();
        // 1 embedding + 1 couche (1 tenseur) + 1 norme + 1 lm_head = 4 tenseurs
        assert_eq!(results.len(), 4);

        // Vérifier les catégories
        assert_eq!(results[0].category, "embedding");
        assert_eq!(results[1].category, "layer");
        assert_eq!(results[2].category, "final_norm");
        assert_eq!(results[3].category, "lm_head");

        // Vérifier les index de couche
        assert_eq!(results[0].layer_index, None);
        assert_eq!(results[1].layer_index, Some(0));
        assert_eq!(results[2].layer_index, None);
        assert_eq!(results[3].layer_index, None);
    }

    #[test]
    fn model_generator_stats() {
        let blueprint = test_blueprint();
        let pipeline = GenerationPipeline::full();

        let gen = ModelGeneratorComplete::new(blueprint, 42, "1.0.0", pipeline, 256);

        let results = gen.generate_all().unwrap();
        let stats = gen.compute_stats(&results);

        // Vérifier le nombre total de paramètres
        assert_eq!(stats.parameter_count, 100 * 64 + 64 * 64 + 64 + 100 * 64);
        assert!(stats.mean != 0.0 || stats.variance != 0.0);
    }

    #[test]
    fn model_generator_deterministic() {
        let blueprint1 = test_blueprint();
        let blueprint2 = test_blueprint();
        let pipeline = GenerationPipeline::full();

        let gen1 = ModelGeneratorComplete::new(blueprint1, 42, "1.0.0", pipeline.clone(), 256);

        let gen2 = ModelGeneratorComplete::new(blueprint2, 42, "1.0.0", pipeline, 256);

        let results1 = gen1.generate_all().unwrap();
        let results2 = gen2.generate_all().unwrap();

        assert_eq!(results1.len(), results2.len());
        for (i, (r1, r2)) in results1.iter().zip(results2.iter()).enumerate() {
            assert_eq!(r1.name, r2.name, "Les noms du tenseur {} diffèrent", i);
            assert_eq!(
                r1.values, r2.values,
                "Les valeurs du tenseur {} diffèrent",
                i
            );
            assert_eq!(
                r1.category, r2.category,
                "Les catégories du tenseur {} diffèrent",
                i
            );
        }
    }
}
