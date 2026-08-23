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

//! Helpers réutilisables pour les benchmarks du projet PMG.
//!
//! Ce module fournit des fonctions utilitaires pour la configuration
//! et l'exécution des benchmarks, incluant la création de spécifications
//! de tenseurs, de plans de seed et de configurations de validation.

use std::collections::BTreeMap;

use pmg_blueprint::architecture::ArchitectureKind;
use pmg_blueprint::layer::{LayerKind, LayerSpec};
use pmg_blueprint::naming::NamingRules;
use pmg_blueprint::tensor_spec::{DistributionFamily, TensorSpec};
use pmg_core::dtype::DType;
use pmg_core::model_config::{AttentionKind, ModelConfig};
use pmg_core::shape::Shape;
use pmg_core::TensorRole;
use pmg_generator::seed_plan::GeneratorSeedPlan;
use pmg_validate::types::ValidationConfig;

/// Crée un GeneratorSeedPlan pour les benchmarks de génération.
///
/// # Paramètres
/// - `seed_global` : seed globale
/// - `model_id` : identifiant du modèle
/// - `generation_version` : version du générateur
///
/// # Retourne
/// Un `GeneratorSeedPlan` configuré pour les benchmarks.
pub fn create_seed_plan(
    seed_global: u64,
    model_id: &str,
    generation_version: &str,
) -> GeneratorSeedPlan {
    GeneratorSeedPlan::new(seed_global, model_id, generation_version)
}

/// Crée un TensorSpec pour les benchmarks.
///
/// # Paramètres
/// - `name` : nom du tenseur
/// - `rows` : nombre de lignes
/// - `cols` : nombre de colonnes
///
/// # Retourne
/// Un `TensorSpec` configuré pour les benchmarks.
pub fn create_tensor_spec(name: &str, rows: usize, cols: usize) -> TensorSpec {
    TensorSpec::new(
        name,
        Shape::new(vec![rows as u64, cols as u64]).unwrap(),
        DType::F32,
        TensorRole::MlpGate,
    )
    .unwrap()
}

/// Crée un TensorSpec avec une distribution spécifique.
///
/// # Paramètres
/// - `name` : nom du tenseur
/// - `rows` : nombre de lignes
/// - `cols` : nombre de colonnes
/// - `family` : famille de distribution
///
/// # Retourne
/// Un `TensorSpec` configuré avec la distribution donnée.
pub fn create_tensor_spec_with_distribution(
    name: &str,
    rows: usize,
    cols: usize,
    family: DistributionFamily,
) -> TensorSpec {
    let mut spec = create_tensor_spec(name, rows, cols);
    spec.distribution.family = family;
    spec
}

/// Crée une ValidationConfig avec des valeurs typiques pour les benchmarks.
///
/// # Retourne
/// Une `ValidationConfig` configurée pour les benchmarks.
pub fn create_validation_config() -> ValidationConfig {
    ValidationConfig {
        outlier_threshold: 3.0,
        energy_threshold: 0.9,
        statistical_tolerance: 0.1,
        check_structural: true,
        check_statistical: true,
        check_outliers: true,
        check_distribution: false,
        check_metadata: false,
    }
}

/// Crée des données de test pour les benchmarks.
///
/// # Paramètres
/// - `size` : nombre d'éléments
///
/// # Retourne
/// Un vecteur de `f64` contenant des données de test.
pub fn create_test_data(size: usize) -> Vec<f64> {
    (0..size).map(|i| i as f64 * 0.001).collect()
}

/// Crée des données de test avec distribution normale simulée.
///
/// # Paramètres
/// - `size` : nombre d'éléments
/// - `mean` : moyenne
/// - `std` : écart-type
///
/// # Retourne
/// Un vecteur de `f64` contenant des données de test.
pub fn create_normal_test_data(size: usize, mean: f64, std: f64) -> Vec<f64> {
    // Simulation d'une distribution normale sans dépendre à rand
    (0..size)
        .map(|i| {
            let t = i as f64 / size as f64;
            mean + std * (2.0 * t - 1.0) * 3.0
        })
        .collect()
}

/// Crée un ModelBlueprint simplifié pour les benchmarks.
///
/// # Retourne
/// Un `ModelBlueprint` configuré pour les benchmarks.
pub fn create_model_blueprint_for_bench() -> pmg_blueprint::ModelBlueprint {
    use pmg_blueprint::ModelBlueprint;

    let config = ModelConfig {
        model_type: "bench_model".to_string(),
        architectures: vec!["BenchForCausalLM".to_string()],
        hidden_size: 512,
        num_layers: 2,
        num_attention_heads: 8,
        num_key_value_heads: 8,
        head_dim: None,
        qk_head_dim: None,
        v_head_dim: None,
        intermediate_size: Some(1024),
        moe_intermediate_size: None,
        vocab_size: 1000,
        max_position_embeddings: 2048,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        tie_word_embeddings: false,
        moe: None,
        attention_type: AttentionKind::Dense,
        hyper_connections: false,
        dtype_declared: DType::F32,
        extras: BTreeMap::new(),
        provenance: BTreeMap::new(),
    };

    let naming_rules = NamingRules::glm52();

    let mut blueprint = ModelBlueprint::new(
        "bench-model",
        ArchitectureKind::DenseTransformer,
        config,
        naming_rules,
    );

    // Embedding
    blueprint
        .embeddings
        .push(create_tensor_spec("model.embed_tokens.weight", 1000, 512));

    // Couches
    for i in 0..2 {
        let mut layer = LayerSpec::new(i, LayerKind::Dense);
        // Attention tensors
        layer.attention.push(create_tensor_spec(
            &format!("model.layers.{}.self_attn.q_proj.weight", i),
            512,
            512,
        ));
        layer.attention.push(create_tensor_spec(
            &format!("model.layers.{}.self_attn.k_proj.weight", i),
            512,
            512,
        ));
        layer.attention.push(create_tensor_spec(
            &format!("model.layers.{}.self_attn.v_proj.weight", i),
            512,
            512,
        ));
        layer.attention.push(create_tensor_spec(
            &format!("model.layers.{}.self_attn.o_proj.weight", i),
            512,
            512,
        ));
        // MLP tensors
        layer.mlp.push(create_tensor_spec(
            &format!("model.layers.{}.mlp.gate.weight", i),
            1024,
            512,
        ));
        layer.mlp.push(create_tensor_spec(
            &format!("model.layers.{}.mlp.up.weight", i),
            1024,
            512,
        ));
        layer.mlp.push(create_tensor_spec(
            &format!("model.layers.{}.mlp.down.weight", i),
            512,
            1024,
        ));
        // Norms
        layer.norms.push(create_tensor_spec(
            &format!("model.layers.{}.input_layernorm.weight", i),
            1,
            512,
        ));
        layer.norms.push(create_tensor_spec(
            &format!("model.layers.{}.post_attention_layernorm.weight", i),
            1,
            512,
        ));

        blueprint.layers.push(layer);
    }

    // Final norm
    blueprint
        .final_norm
        .push(create_tensor_spec("model.norm.weight", 1, 512));

    // LM head
    blueprint
        .lm_head
        .push(create_tensor_spec("lm_head.weight", 1000, 512));

    blueprint
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_seed_plan() {
        let plan = create_seed_plan(42, "bench-model", "0.1.0");
        assert_eq!(plan.seed_global, 42);
        assert_eq!(plan.model_id, "bench-model");
        assert_eq!(plan.generation_version, "0.1.0");
    }

    #[test]
    fn test_create_tensor_spec() {
        let spec = create_tensor_spec("test.tensor", 100, 64);
        assert_eq!(spec.name, "test.tensor");
        assert_eq!(spec.shape.dims(), &[100, 64]);
        assert_eq!(spec.dtype, DType::F32);
    }

    #[test]
    fn test_create_validation_config() {
        let config = create_validation_config();
        assert_eq!(config.outlier_threshold, 3.0);
        assert!(config.check_structural);
        assert!(config.check_statistical);
        assert!(config.check_outliers);
    }

    #[test]
    fn test_create_test_data() {
        let data = create_test_data(100);
        assert_eq!(data.len(), 100);
        assert_eq!(data[0], 0.0);
        assert_eq!(data[99], 0.099);
    }

    #[test]
    fn test_create_model_blueprint() {
        let blueprint = create_model_blueprint_for_bench();
        assert_eq!(blueprint.id, "bench-model");
        assert_eq!(blueprint.layers.len(), 2);
        assert_eq!(blueprint.embeddings.len(), 1);
        assert_eq!(blueprint.final_norm.len(), 1);
        assert_eq!(blueprint.lm_head.len(), 1);
    }
}
