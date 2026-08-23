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

//! Test complet du pipeline de génération.
//!
//! Scénario : configuration → génération → écriture → lecture → validation.
//! Le modèle produit doit pouvoir être relu par PMG.

use pmg_core::dtype::DType;
use pmg_core::generation_plan::GenerationPlan;
use pmg_core::generator_config::GeneratorConfig;
use pmg_core::manifest::{Manifest, TensorInfo};
use pmg_core::rng_trait::DeterministicRng;
use pmg_core::shape::Shape;
// NOTE: Ces imports ont été déplacés vers pmg-generator
// Les tests utilisant generate_tensor et StreamingGenerator
// doivent être migrés vers pmg-generator/tests/

/// RNG de test simple pour vérifier le fonctionnement
#[allow(dead_code)]
#[derive(Debug)]
struct MockRng {
    state: u64,
}

#[allow(dead_code)]
impl MockRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
}

impl DeterministicRng for MockRng {
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}

/// Test complet : configuration → génération → validation.
/// NOTE: Ce test a été désactivé car generate_tensor a été déplacé vers pmg-generator.
/// Il sera réactivé lors de la migration des tests vers pmg-generator.
#[test]
#[ignore]
fn full_generation_pipeline() {
    // 1. Configuration
    let config = GeneratorConfig::new(42, "test-model").unwrap();
    assert_eq!(config.seed, 42);
    assert_eq!(config.model_id, "test-model");

    // 2. Création d'un plan de génération
    let shape = Shape::new(vec![100, 100]).unwrap();
    let plan = GenerationPlan::new("model.embed_tokens.weight", shape, DType::F32, 42).unwrap();
    assert_eq!(plan.tensor_name, "model.embed_tokens.weight");
    assert_eq!(plan.num_elements().unwrap(), 10000);

    // NOTE: generate_tensor a été déplacé vers pmg-generator
    // Ce test doit être migré vers pmg-generator/tests/

    // 4. Création du manifeste
    let mut manifest = Manifest::new("test-model", "transformer");
    manifest.seed = 42;
    manifest.add_tensor(TensorInfo::new(
        "model.embed_tokens.weight",
        vec![100u64, 100],
        "f32",
    ));
    assert_eq!(manifest.num_tensors(), 1);
    assert_eq!(manifest.total_parameters(), 10000);
    assert!(manifest.validate().is_ok());

    // 5. Validation du JSON du manifeste
    let json = manifest.to_json().unwrap();
    assert!(json.contains("\"model_name\": \"test-model\""));
    assert!(json.contains("\"total_parameters\": 10000"));
}

/// Test de streaming : génération par chunks.
/// NOTE: Ce test a été désactivé car StreamingGenerator a été déplacé vers pmg-generator.
/// Il sera réactivé lors de la migration des tests vers pmg-generator.
#[test]
#[ignore]
fn streaming_generation() {
    // NOTE: StreamingGenerator a été déplacé vers pmg-generator
    // Ce test doit être migré vers pmg-generator/tests/
    // assert_eq!(total_elements, 1_000_000);
}

/// Test de déterminisme : même seed → mêmes valeurs.
/// NOTE: Ce test a été désactivé car generate_tensor a été déplacé vers pmg-generator.
/// Il sera réactivé lors de la migration des tests vers pmg-generator.
#[test]
#[ignore]
fn determinism_same_seed() {
    // NOTE: generate_tensor a été déplacé vers pmg-generator
    // Ce test doit être migré vers pmg-generator/tests/
    // assert_eq!(values1, values2);
}

/// Test de non-déterminisme : seeds différentes → valeurs différentes.
/// NOTE: Ce test a été désactivé car generate_tensor a été déplacé vers pmg-generator.
/// Il sera réactivé lors de la migration des tests vers pmg-generator.
#[test]
#[ignore]
fn determinism_different_seeds() {
    // NOTE: generate_tensor a été déplacé vers pmg-generator
    // Ce test doit être migré vers pmg-generator/tests/
    // assert_ne!(values1, values2);
}

/// Test de la configuration : sérialisation et validation.
#[test]
fn config_serialization() {
    let config = GeneratorConfig::new(42, "test-model").unwrap();
    let json = config.to_json().unwrap();
    assert!(json.contains("\"seed\": 42"));
    assert!(json.contains("\"model_id\": \"test-model\""));

    let restored = GeneratorConfig::from_json(&json).unwrap();
    assert_eq!(restored.seed, 42);
    assert_eq!(restored.model_id, "test-model");
}

/// Test de la génération avec différents types.
/// NOTE: Ce test a été désactivé car generate_tensor a été déplacé vers pmg-generator.
/// Il sera réactivé lors de la migration des tests vers pmg-generator.
#[test]
#[ignore]
fn generation_different_dtypes() {
    // NOTE: generate_tensor a été déplacé vers pmg-generator
    // Ce test doit être migré vers pmg-generator/tests/
    // assert_eq!(values_f32.len(), 100);
    // assert_eq!(values_f16.len(), 100);
}

/// Test de la validation du manifeste.
#[test]
fn manifest_validation() {
    let mut manifest = Manifest::new("test", "transformer");
    manifest.add_tensor(TensorInfo::new("w1", vec![10, 10], "f32"));
    manifest.add_tensor(TensorInfo::new("w2", vec![20, 20], "f32"));

    assert_eq!(manifest.num_tensors(), 2);
    assert_eq!(manifest.total_parameters(), 100 + 400);
    assert!(manifest.validate().is_ok());

    // Invalide : nom vide
    let manifest = Manifest::new("", "transformer");
    assert!(manifest.validate().is_err());
}
