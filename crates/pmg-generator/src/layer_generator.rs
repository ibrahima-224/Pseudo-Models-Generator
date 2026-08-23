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

//! Générateur de couche de transformeur.
//!
//! Ce module regroupe les tenseurs d'une couche (q_proj, k_proj, v_proj, o_proj,
//! gate_proj, up_proj, down_proj, etc.) en préservant les relations structurelles.
//! Il utilise le pipeline de génération et le contexte pour produire des tenseurs
//! déterministes et cohérents.

use pmg_blueprint::layer::LayerSpec;
use pmg_blueprint::tensor_spec::TensorSpec;

use crate::context::GenerationContext;
use crate::error::GeneratorResult;
use crate::generation_stats::GenerationStats;
use crate::pipeline::{GenerationPipeline, StepResult};
use crate::seed_plan::GeneratorSeedPlan;
use crate::tensor_generator::TensorGenerator;

/// Type pour le résultat de génération d'un tenseur de couche.
pub type LayerTensorResult = (String, Vec<f64>, Vec<StepResult>);

/// Générateur pour une couche de transformeur.
///
/// Regroupe les tenseurs d'une couche et préserve leurs relations structurelles.
/// Chaque tenseur est généré individuellement mais avec des seeds dérivées
/// de manière cohérente pour maintenir les corrélations.
pub struct LayerGenerator {
    /// Spécification de la couche (blueprint).
    layer_spec: LayerSpec,
    /// Seed globale de génération.
    seed: u64,
    /// Identifiant du modèle.
    model_name: String,
    /// Version du générateur.
    generation_version: String,
    /// Pipeline de génération à appliquer.
    pipeline: GenerationPipeline,
    /// Taille des chunks pour la génération.
    chunk_size: usize,
    /// Budget tensoriel disponible (optionnel, en octets).
    tensor_budget: Option<u64>,
}

impl LayerGenerator {
    /// Crée un nouveau générateur pour une couche.
    ///
    /// # Paramètres
    /// - `layer_spec` : spécification de la couche (blueprint)
    /// - `seed` : seed globale de génération
    /// - `model_name` : identifiant du modèle
    /// - `generation_version` : version du générateur
    /// - `pipeline` : pipeline de génération à appliquer
    /// - `chunk_size` : taille des chunks pour la génération
    pub fn new(
        layer_spec: LayerSpec,
        seed: u64,
        model_name: impl Into<String>,
        generation_version: impl Into<String>,
        pipeline: GenerationPipeline,
        chunk_size: usize,
    ) -> Self {
        Self {
            layer_spec,
            seed,
            model_name: model_name.into(),
            generation_version: generation_version.into(),
            pipeline,
            chunk_size,
            tensor_budget: None,
        }
    }

    /// Crée un nouveau générateur pour une couche avec un budget tensoriel.
    pub fn with_budget(
        layer_spec: LayerSpec,
        seed: u64,
        model_name: impl Into<String>,
        generation_version: impl Into<String>,
        pipeline: GenerationPipeline,
        chunk_size: usize,
        tensor_budget: Option<u64>,
    ) -> Self {
        Self {
            layer_spec,
            seed,
            model_name: model_name.into(),
            generation_version: generation_version.into(),
            pipeline,
            chunk_size,
            tensor_budget,
        }
    }

    /// Génère tous les tenseurs de la couche.
    ///
    /// # Retourne
    /// Un vecteur de tuples (nom du tenseur, valeurs générées, résultats du pipeline).
    ///
    /// # Erreurs
    /// Retourne une erreur si la génération d'un tenseur échoue.
    pub fn generate_all(&self) -> GeneratorResult<Vec<LayerTensorResult>> {
        let mut results = Vec::new();

        // Générer les tenseurs d'attention
        for (tensor_index, tensor_spec) in self.layer_spec.attention.iter().enumerate() {
            let result = self.generate_tensor(tensor_spec, tensor_index)?;
            results.push(result);
        }

        // Générer les tenseurs du MLP
        for (tensor_index, tensor_spec) in self.layer_spec.mlp.iter().enumerate() {
            let result = self.generate_tensor(tensor_spec, tensor_index)?;
            results.push(result);
        }

        // Générer les normes
        for (tensor_index, tensor_spec) in self.layer_spec.norms.iter().enumerate() {
            let result = self.generate_tensor(tensor_spec, tensor_index)?;
            results.push(result);
        }

        // Générer les hyper-connections
        for (tensor_index, tensor_spec) in self.layer_spec.hyper_connections.iter().enumerate() {
            let result = self.generate_tensor(tensor_spec, tensor_index)?;
            results.push(result);
        }

        // Générer les tenseurs MoE si présents
        if let Some(moe_block) = &self.layer_spec.moe_block {
            // Routeur
            let result = self.generate_tensor(&moe_block.router, 0)?;
            results.push(result);

            // Experts partagés
            for (tensor_index, tensor_spec) in moe_block.shared_experts.iter().enumerate() {
                let result = self.generate_tensor(tensor_spec, tensor_index)?;
                results.push(result);
            }

            // Experts routés
            for (expert_index, expert) in moe_block.routed_experts.iter().enumerate() {
                let result_up = self.generate_tensor(&expert.up, expert_index * 3)?;
                let result_gate = self.generate_tensor(&expert.gate, expert_index * 3 + 1)?;
                let result_down = self.generate_tensor(&expert.down, expert_index * 3 + 2)?;
                results.push(result_up);
                results.push(result_gate);
                results.push(result_down);
            }
        }

        Ok(results)
    }

    /// Génère un tenseur individuel de la couche.
    ///
    /// # Paramètres
    /// - `tensor_spec` : spécification du tenseur
    /// - `tensor_index` : index du tenseur dans la catégorie (attention, mlp, etc.)
    ///
    /// # Retourne
    /// Un tuple (nom du tenseur, valeurs générées, résultats du pipeline).
    fn generate_tensor(
        &self,
        tensor_spec: &TensorSpec,
        tensor_index: usize,
    ) -> GeneratorResult<(String, Vec<f64>, Vec<StepResult>)> {
        let num_elements = tensor_spec.num_elements()? as usize;

        // Créer le contexte de génération
        let context = GenerationContext::new(
            self.seed,
            &self.model_name,
            &self.generation_version,
            Some(self.layer_spec.index as usize),
            tensor_index,
            0, // chunk_index initial
            &tensor_spec.name,
            num_elements,
            self.chunk_size,
        );

        // Créer le plan de seed
        let seed_plan =
            GeneratorSeedPlan::new(self.seed, &self.model_name, &self.generation_version);

        // Générer les valeurs initiales avec le budget tensoriel si disponible
        let tensor_gen = TensorGenerator::new(tensor_spec.clone(), seed_plan, self.tensor_budget);
        let mut values = tensor_gen.generate()?;

        // Appliquer le pipeline
        let pipeline_results = self.pipeline.execute(&mut values, context.tensor_seed())?;

        Ok((tensor_spec.name.clone(), values, pipeline_results))
    }

    /// Calcule les statistiques de la couche.
    ///
    /// # Paramètres
    /// - `tuples` : vecteur de (nom, valeurs, résultats) pour chaque tenseur
    ///
    /// # Retourne
    /// Statistiques agrégées de la couche.
    pub fn compute_stats(&self, tuples: &[(String, Vec<f64>, Vec<StepResult>)]) -> GenerationStats {
        let mut stats = GenerationStats::new();
        let mut all_values = Vec::new();

        for (_name, values, _step_results) in tuples {
            // Collecter toutes les valeurs pour le calcul des quantiles
            all_values.extend(values);
            stats.update_from_values(values);
            // Mettre à jour les compteurs d'outliers et super-poids à partir des résultats du pipeline
            for step_result in _step_results {
                if let Some(&count) = step_result.metrics.get("outlier_count") {
                    stats.outlier_count += count as usize;
                }
                if let Some(&count) = step_result.metrics.get("super_weight_count") {
                    stats.super_weight_count += count as usize;
                }
            }
            // Note: update_from_values already updates parameter_count
        }

        // Calculer les quantiles à partir de toutes les valeurs collectées
        stats.compute_quantiles(&all_values);

        stats
    }

    /// Retourne la spécification de la couche.
    pub fn layer_spec(&self) -> &LayerSpec {
        &self.layer_spec
    }

    /// Retourne la seed globale.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Retourne l'identifiant du modèle.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Retourne le budget tensoriel disponible.
    pub fn tensor_budget(&self) -> Option<u64> {
        self.tensor_budget
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmg_blueprint::architecture::ArchitectureKind;
    use pmg_blueprint::naming::NamingRules;
    use pmg_blueprint::ModelBlueprint;
    use pmg_core::model_config::glm52_test_config;

    fn test_layer_spec() -> LayerSpec {
        let config = glm52_test_config();
        let mut bp = ModelBlueprint::new(
            "glm-5.2",
            ArchitectureKind::MoETransformer,
            config,
            NamingRules::glm52(),
        );

        // Ajouter un tenseur d'embedding pour avoir un blueprint valide
        bp.embeddings.push(
            TensorSpec::new(
                "model.embed_tokens.weight",
                pmg_core::Shape::new(vec![100, 64]).unwrap(),
                pmg_core::DType::F32,
                pmg_core::TensorRole::Embedding,
            )
            .unwrap(),
        );

        // Créer une couche de test
        let mut layer = LayerSpec::new(0, pmg_blueprint::layer::LayerKind::Dense);
        layer.attention.push(
            TensorSpec::new(
                "model.layers.0.self_attn.q_proj.weight",
                pmg_core::Shape::new(vec![64, 64]).unwrap(),
                pmg_core::DType::F32,
                pmg_core::TensorRole::AttentionQuery,
            )
            .unwrap(),
        );
        layer.mlp.push(
            TensorSpec::new(
                "model.layers.0.mlp.gate_proj.weight",
                pmg_core::Shape::new(vec![128, 64]).unwrap(),
                pmg_core::DType::F32,
                pmg_core::TensorRole::MlpGate,
            )
            .unwrap(),
        );
        layer
    }

    #[test]
    fn layer_generator_creation() {
        let layer_spec = test_layer_spec();
        let pipeline = GenerationPipeline::full();

        let gen = LayerGenerator::new(layer_spec, 42, "glm-5.2", "1.0.0", pipeline, 256);

        assert_eq!(gen.seed(), 42);
        assert_eq!(gen.model_name(), "glm-5.2");
    }

    #[test]
    fn layer_generator_generate_all() {
        let layer_spec = test_layer_spec();
        let pipeline = GenerationPipeline::full();

        let gen = LayerGenerator::new(layer_spec, 42, "glm-5.2", "1.0.0", pipeline, 256);

        let results = gen.generate_all().unwrap();
        assert_eq!(results.len(), 2); // q_proj + gate_proj

        // Vérifier que les noms sont corrects
        assert_eq!(results[0].0, "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(results[1].0, "model.layers.0.mlp.gate_proj.weight");

        // Vérifier les tailles
        assert_eq!(results[0].1.len(), 64 * 64);
        assert_eq!(results[1].1.len(), 128 * 64);
    }

    #[test]
    fn layer_generator_stats() {
        let layer_spec = test_layer_spec();
        let pipeline = GenerationPipeline::full();

        let gen = LayerGenerator::new(layer_spec, 42, "glm-5.2", "1.0.0", pipeline, 256);

        let results = gen.generate_all().unwrap();
        let stats = gen.compute_stats(&results);

        assert_eq!(stats.parameter_count, 64 * 64 + 128 * 64);
        assert!(stats.mean != 0.0 || stats.variance != 0.0); // Au moins une statistique calculée
    }

    #[test]
    fn layer_generator_deterministic() {
        let layer_spec1 = test_layer_spec();
        let layer_spec2 = test_layer_spec();
        let pipeline = GenerationPipeline::full();

        let gen1 = LayerGenerator::new(layer_spec1, 42, "glm-5.2", "1.0.0", pipeline.clone(), 256);

        let gen2 = LayerGenerator::new(layer_spec2, 42, "glm-5.2", "1.0.0", pipeline, 256);

        let results1 = gen1.generate_all().unwrap();
        let results2 = gen2.generate_all().unwrap();

        // Vérifier que les résultats sont identiques
        assert_eq!(results1.len(), results2.len());
        for (i, (r1, r2)) in results1.iter().zip(results2.iter()).enumerate() {
            assert_eq!(r1.0, r2.0, "Les noms du tenseur {} diffèrent", i);
            assert_eq!(r1.1, r2.1, "Les valeurs du tenseur {} diffèrent", i);
        }
    }
}
