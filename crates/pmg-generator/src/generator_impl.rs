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

//! Implémentations lourdes du générateur.
//!
//! Ce module contient les méthodes de génération, validation et conversion
//! qui nécessitent des imports supplémentaires et une logique complexe.
//! Les structures de base et les méthodes simples sont dans `generator_core`.

use pmg_blueprint::{plan_blueprint, Plan};
use pmg_injector::tensor_injector::TensorInjector;

use crate::chunk::collect_all_chunks;
use crate::error::{GeneratorError, GeneratorResult};
use crate::generation_report::{DistributionStats, GenerationReport, InjectionStats};
use crate::generation_validator::{GenerationValidator, ValidationResult};
use crate::seed_plan::GeneratorSeedPlan;
use crate::tensor_generator::TensorGenerator;

use super::generator_core::{GeneratedTensor, ModelGenerator};

impl ModelGenerator {
    /// Planifie les tenseurs à partir du blueprint.
    ///
    /// Cette méthode valide le blueprint et crée un plan de génération
    /// contenant la liste des tenseurs à générer.
    ///
    /// # Erreurs
    ///
    /// Retourne une erreur si le blueprint n'est pas défini ou invalide.
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_generator::ModelGenerator;
    /// use pmg_blueprint::{ArchitectureKind, ModelBlueprint, NamingRules};
    /// use pmg_core::model_config::glm52_test_config;
    ///
    /// let mut gen = ModelGenerator::with_seed(42, "glm-5.2");
    /// let mut bp = ModelBlueprint::new(
    ///     "glm-5.2",
    ///     ArchitectureKind::MoETransformer,
    ///     glm52_test_config(),
    ///     NamingRules::glm52(),
    /// );
    ///
    /// // Ajout d'un tenseur d'embedding
    /// bp.embeddings.push(
    ///     pmg_blueprint::TensorSpec::new(
    ///         "model.embed_tokens.weight",
    ///         pmg_core::Shape::new(vec![100, 64]).expect("dimensions valides"),
    ///         pmg_core::DType::F32,
    ///         pmg_core::TensorRole::Embedding,
    ///     )
    ///     .expect("spécification valide"),
    /// );
    ///
    /// gen.set_blueprint(bp);
    /// let plan = gen.plan().expect("plan généré avec succès");
    /// assert_eq!(plan.tensors.len(), 1);
    /// ```
    pub fn plan(&mut self) -> GeneratorResult<&Plan> {
        if let Some(ref blueprint) = self.blueprint {
            let plan = plan_blueprint(blueprint)?;
            self.plan = Some(plan);
            Ok(self
                .plan
                .as_ref()
                .ok_or_else(|| GeneratorError::InvalidModelConfig("plan non disponible".into()))?)
        } else {
            Err(GeneratorError::InvalidModelConfig(
                "aucun blueprint défini".into(),
            ))
        }
    }

    /// Calcule le budget tensoriel à partir du plan et de la configuration.
    ///
    /// Cette méthode est appelée avant la génération pour déterminer le budget
    /// disponible en fonction du budget total, des en-têtes et des métadonnées.
    ///
    /// # Arguments
    ///
    /// * `total_budget` - Budget total en octets (B).
    ///
    /// # Retour
    ///
    /// Le budget tensoriel disponible en octets (W).
    pub fn calculate_tensor_budget(&self, total_budget: u64) -> GeneratorResult<u64> {
        let plan = self
            .plan
            .as_ref()
            .ok_or_else(|| GeneratorError::InvalidModelConfig("plan non disponible".into()))?;

        let header_est = self.budget_planner.estimate_headers(plan);

        // Récupérer la config du blueprint si disponible, sinon utiliser une estimation par défaut
        let metadata_est = if let Some(ref blueprint) = self.blueprint {
            self.budget_planner.estimate_metadata(&blueprint.config)
        } else {
            // Estimation par défaut : 24 Ko
            24 * 1024
        };

        let tensor_budget =
            self.budget_planner
                .calculate_budget(total_budget, header_est, metadata_est);

        // Vérifier la suffisance du budget pour le mode full-structural
        let required_budget = plan.parameter_count * 4; // Estimation : 4 octets par paramètre (F32)
        self.budget_planner.check_budget_for_mode(
            &self.generation_mode,
            tensor_budget,
            required_budget,
        )?;

        Ok(tensor_budget)
    }

    /// Génère un seul tenseur à partir de sa spécification.
    ///
    /// Cette méthode applique le pipeline complet de génération :
    /// 1. Génération des valeurs initiales via le RNG déterministe
    /// 2. Application des injections structurelles (outliers, super-poids, etc.)
    /// 3. Application des étapes du pipeline (distribution, corrélation, etc.)
    ///
    /// # Erreurs
    ///
    /// Retourne une erreur si la génération ou les injections échouent.
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_generator::ModelGenerator;
    /// use pmg_blueprint::{ArchitectureKind, ModelBlueprint, NamingRules};
    /// use pmg_core::model_config::glm52_test_config;
    ///
    /// let mut gen = ModelGenerator::with_seed(42, "glm-5.2");
    /// let mut bp = ModelBlueprint::new(
    ///     "glm-5.2",
    ///     ArchitectureKind::MoETransformer,
    ///     glm52_test_config(),
    ///     NamingRules::glm52(),
    /// );
    ///
    /// bp.embeddings.push(
    ///     pmg_blueprint::TensorSpec::new(
    ///         "model.embed_tokens.weight",
    ///         pmg_core::Shape::new(vec![100, 64]).expect("dimensions valides"),
    ///         pmg_core::DType::F32,
    ///         pmg_core::TensorRole::Embedding,
    ///     )
    ///     .expect("spécification valide"),
    /// );
    ///
    /// gen.set_blueprint(bp);
    /// gen.plan().expect("plan généré avec succès");
    ///
    /// let spec = &gen.plan_ref().expect("plan disponible").tensors[0];
    /// let tensor = gen.generate_tensor(spec).expect("génération réussie");
    ///
    /// assert_eq!(tensor.num_elements, 100 * 64);
    /// assert_eq!(tensor.values.len(), 100 * 64);
    /// ```
    pub fn generate_tensor(
        &self,
        spec: &pmg_blueprint::TensorSpec,
    ) -> GeneratorResult<GeneratedTensor> {
        let seed_plan = GeneratorSeedPlan::new(
            self.config.seed,
            &self.config.model_id,
            &self.config.generation_version,
        );

        // Générer les valeurs initiales avec le budget tensoriel si disponible
        let gen = TensorGenerator::new(spec.clone(), seed_plan, self.tensor_budget);
        let mut values = gen.generate()?;

        // Appliquer l'injection
        let layer_id = spec.layer_id.map(|l| l as u32);
        let tensor_seed = pmg_math::rng::derive_seed(&pmg_math::rng::SeedPlan {
            seed_global: self.config.seed,
            model_id: &self.config.model_id,
            tensor_name: &spec.name,
            layer_id,
            generation_version: &self.config.generation_version,
        });

        let injector =
            TensorInjector::from_seed(spec, self.config.injection_policy.clone(), tensor_seed);
        injector.apply_to(&mut values)?;

        // Appliquer le pipeline de génération (Sprint 12)
        // Convertir la seed [u8; 32] en u64 pour le pipeline
        let pipeline_seed = u64::from_le_bytes(tensor_seed[..8].try_into().map_err(|e| {
            GeneratorError::Internal(format!("échec de conversion de la seed : {}", e))
        })?);
        let _pipeline_results = self.pipeline.execute(&mut values, pipeline_seed)?;

        Ok(GeneratedTensor {
            name: spec.name.clone(),
            values,
            num_elements: spec.num_elements()? as usize,
        })
    }

    /// Génère tous les tenseurs du plan en mémoire.
    ///
    /// # Avertissement Mémoire (IMPORTANT)
    ///
    /// Cette fonction accumule **TOUS** les tenseurs en mémoire simultanément.
    /// Pour un modèle de 10 Go, elle allocera ~10 Go de RAM.
    ///
    /// **Pour les modèles dépassant 1 Go, utilisez impérativement :**
    /// - `execute_pipeline_output_streaming()` — streaming tension par tension
    /// - `generate_streaming_to_disk()` — écriture directe sur disque
    ///
    /// Cette fonction est conservée pour la rétrocompatibilité et les petits modèles.
    ///
    /// # Erreurs
    ///
    /// Retourne une erreur si la génération d'un tenseur échoue.
    ///
    /// # Exemple
    ///
    /// ```ignore
    /// // ⚠️ Déconseillé pour les gros modèles
    /// let tensors = gen.generate_all().expect("génération réussie");
    /// ```
    #[deprecated(
        since = "2.0.0",
        note = "Utilisez generate_streaming_to_disk() ou execute_pipeline_output_streaming() pour éviter l'accumulation en mémoire"
    )]
    pub fn generate_all(&self) -> GeneratorResult<Vec<GeneratedTensor>> {
        // Avertissement runtime pour les gros modèles
        eprintln!(
            "⚠️  AVERTISSEMENT: generate_all() accumule tous les tenseurs en mémoire. \
             Pour les gros modèles, utilisez generate_streaming_to_disk()."
        );
        let plan = self
            .plan
            .as_ref()
            .ok_or_else(|| GeneratorError::InvalidModelConfig("plan non disponible".into()))?;

        let mut tensors = Vec::new();
        for spec in &plan.tensors {
            let tensor = self.generate_tensor(spec)?;
            tensors.push(tensor);
        }

        Ok(tensors)
    }

    /// Génère les tenseurs par chunks (streaming).
    pub fn generate_chunked(
        &self,
        spec: &pmg_blueprint::TensorSpec,
    ) -> GeneratorResult<Vec<crate::chunk::TensorChunk>> {
        let seed_plan = GeneratorSeedPlan::new(
            self.config.seed,
            &self.config.model_id,
            &self.config.generation_version,
        );

        let total_size = spec.num_elements()? as usize;
        let layer_id = spec.layer_id.map(|l| l as u32);
        let tensor_seed = seed_plan.derive_tensor_seed(&spec.name, layer_id);

        collect_all_chunks(
            total_size,
            self.config.chunk_size,
            move |chunk_id, start, end| {
                let chunk_seed = GeneratorSeedPlan::derive_chunk_seed(&tensor_seed, chunk_id);
                let mut rng = pmg_math::rng::DeterministicRng::from_seed(chunk_seed);
                let mut values = Vec::with_capacity(end - start);
                for _ in 0..(end - start) {
                    values.push(rng.next_f64());
                }
                Ok(values)
            },
        )
    }

    /// Génère le rapport de génération.
    ///
    /// Le rapport contient des statistiques sur la génération : nombre de tenseurs,
    /// paramètres, couches, et métriques de distribution/injection.
    ///
    /// # Erreurs
    ///
    /// Retourne une erreur si le plan n'est pas disponible.
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_generator::ModelGenerator;
    /// use pmg_blueprint::{ArchitectureKind, ModelBlueprint, NamingRules};
    /// use pmg_core::model_config::glm52_test_config;
    ///
    /// let mut gen = ModelGenerator::with_seed(42, "glm-5.2");
    /// let mut bp = ModelBlueprint::new(
    ///     "glm-5.2",
    ///     ArchitectureKind::MoETransformer,
    ///     glm52_test_config(),
    ///     NamingRules::glm52(),
    /// );
    ///
    /// bp.embeddings.push(
    ///     pmg_blueprint::TensorSpec::new(
    ///         "model.embed_tokens.weight",
    ///         pmg_core::Shape::new(vec![100, 64]).expect("dimensions valides"),
    ///         pmg_core::DType::F32,
    ///         pmg_core::TensorRole::Embedding,
    ///     )
    ///     .expect("spécification valide"),
    /// );
    ///
    /// gen.set_blueprint(bp);
    /// gen.plan().expect("plan généré avec succès");
    ///
    /// let report = gen.generate_report().expect("rapport généré");
    /// assert_eq!(report.seed, 42);
    /// assert_eq!(report.num_tensors, 1);
    /// assert_eq!(report.parameter_count, 100 * 64);
    /// ```
    pub fn generate_report(&self) -> GeneratorResult<GenerationReport> {
        let plan = self
            .plan
            .as_ref()
            .ok_or_else(|| GeneratorError::InvalidModelConfig("plan non disponible".into()))?;

        let mut report = GenerationReport::new(&self.config.model_id, self.config.seed);
        report.num_tensors = plan.tensors.len() as u64;
        report.parameter_count = plan.parameter_count;

        // Compter les couches uniques
        let mut layers = std::collections::BTreeSet::new();
        for spec in &plan.tensors {
            if let Some(layer_id) = spec.layer_id {
                layers.insert(layer_id);
            }
        }
        report.num_layers = layers.len() as u64;

        // Statistiques de distribution (à implémenter plus complètement)
        report.distribution_stats = DistributionStats {
            normal_pct: 90.0,
            student_t_pct: 5.0,
            pareto_pct: 3.0,
            other_pct: 2.0,
            total_analyzed: report.num_tensors,
        };

        // Statistiques d'injection (à implémenter plus complètement)
        report.injection_stats = InjectionStats {
            outlier_pct: 1.0,
            low_rank_layers: 0,
            correlation_enabled: false,
            total_analyzed: report.num_tensors,
        };

        Ok(report)
    }

    /// Valide la génération.
    ///
    /// Cette méthode vérifie la cohérence de la génération en comparant
    /// le rapport de génération avec les spécifications du plan.
    ///
    /// # Erreurs
    ///
    /// Retourne une erreur si la génération du rapport ou la validation échoue.
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_generator::ModelGenerator;
    /// use pmg_blueprint::{ArchitectureKind, ModelBlueprint, NamingRules};
    /// use pmg_core::model_config::glm52_test_config;
    ///
    /// let mut gen = ModelGenerator::with_seed(42, "glm-5.2");
    /// let mut bp = ModelBlueprint::new(
    ///     "glm-5.2",
    ///     ArchitectureKind::MoETransformer,
    ///     glm52_test_config(),
    ///     NamingRules::glm52(),
    /// );
    ///
    /// bp.embeddings.push(
    ///     pmg_blueprint::TensorSpec::new(
    ///         "model.embed_tokens.weight",
    ///         pmg_core::Shape::new(vec![100, 64]).expect("dimensions valides"),
    ///         pmg_core::DType::F32,
    ///         pmg_core::TensorRole::Embedding,
    ///     )
    ///     .expect("spécification valide"),
    /// );
    ///
    /// gen.set_blueprint(bp);
    /// gen.plan().expect("plan généré avec succès");
    ///
    /// let result = gen.validate().expect("validation réussie");
    /// assert!(result.success);
    /// ```
    pub fn validate(&self) -> GeneratorResult<ValidationResult> {
        let report = self.generate_report()?;
        let plan = self
            .plan
            .as_ref()
            .ok_or_else(|| GeneratorError::InvalidModelConfig("plan non disponible".into()))?;

        let validator = GenerationValidator::new(report, plan.tensors.clone());
        validator.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmg_blueprint::{ArchitectureKind, ModelBlueprint, NamingRules};
    use pmg_core::model_config::glm52_test_config;

    fn test_blueprint() -> ModelBlueprint {
        let config = glm52_test_config();
        let mut bp = ModelBlueprint::new(
            "glm-5.2",
            ArchitectureKind::MoETransformer,
            config,
            NamingRules::glm52(),
        );
        // Ajouter un tenseur de test
        bp.embeddings.push(
            pmg_blueprint::TensorSpec::new(
                "model.embed_tokens.weight",
                pmg_core::Shape::new(vec![100, 64]).unwrap(),
                pmg_core::DType::F32,
                pmg_core::TensorRole::Embedding,
            )
            .unwrap(),
        );
        bp
    }

    #[test]
    fn plan_generation() {
        let mut gen = ModelGenerator::with_seed(42, "glm-5.2");
        let blueprint = test_blueprint();
        gen.set_blueprint(blueprint);
        let plan = gen.plan().unwrap();
        assert_eq!(plan.tensors.len(), 1);
        assert_eq!(plan.tensors[0].name, "model.embed_tokens.weight");
    }

    #[test]
    fn generate_single_tensor() {
        let mut gen = ModelGenerator::with_seed(42, "glm-5.2");
        let blueprint = test_blueprint();
        gen.set_blueprint(blueprint);
        gen.plan().unwrap();

        let spec = &gen.plan_ref().unwrap().tensors[0];
        let tensor = gen.generate_tensor(spec).unwrap();
        assert_eq!(tensor.num_elements, 100 * 64);
        assert_eq!(tensor.values.len(), 100 * 64);
    }

    #[test]
    fn generate_all_tensors() {
        let mut gen = ModelGenerator::with_seed(42, "glm-5.2");
        let blueprint = test_blueprint();
        gen.set_blueprint(blueprint);
        gen.plan().unwrap();

        let tensors = gen.generate_all().unwrap();
        assert_eq!(tensors.len(), 1);
    }

    #[test]
    fn generate_report() {
        let mut gen = ModelGenerator::with_seed(42, "glm-5.2");
        let blueprint = test_blueprint();
        gen.set_blueprint(blueprint);
        gen.plan().unwrap();

        let report = gen.generate_report().unwrap();
        assert_eq!(report.seed, 42);
        assert_eq!(report.num_tensors, 1);
        assert_eq!(report.parameter_count, 100 * 64);
    }

    #[test]
    fn validate_generation() {
        let mut gen = ModelGenerator::with_seed(42, "glm-5.2");
        let blueprint = test_blueprint();
        gen.set_blueprint(blueprint);
        gen.plan().unwrap();

        let result = gen.validate().unwrap();
        assert!(result.success);
    }
}
