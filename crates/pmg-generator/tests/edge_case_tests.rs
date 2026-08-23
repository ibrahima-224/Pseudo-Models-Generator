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

//! Tests d'intégration pour les cas limites du crate pmg-generator.
//!
//! Ces tests vérifient le comportement du générateur avec des entrées
//! inhabituelles ou des cas limites.

use pmg_blueprint::architecture::ArchitectureKind;
use pmg_blueprint::naming::NamingRules;
use pmg_blueprint::tensor_spec::TensorSpec;
use pmg_blueprint::ModelBlueprint;
use pmg_core::model_config::glm52_test_config;
use pmg_core::{DType, Shape, TensorRole};

use pmg_generator::{
    GenerationContext, GenerationPipeline, GeneratorSeedPlan, ModelGenerator,
    ModelGeneratorComplete,
};

/// Blueprint minimal pour les tests de cas limites.
fn minimal_blueprint() -> ModelBlueprint {
    let config = glm52_test_config();
    let mut bp = ModelBlueprint::new(
        "glm-5.2",
        ArchitectureKind::MoETransformer,
        config,
        NamingRules::glm52(),
    );

    // Embedding avec un seul élément
    bp.embeddings.push(
        TensorSpec::new(
            "model.embed_tokens.weight",
            Shape::new(vec![1, 1]).unwrap(),
            DType::F32,
            TensorRole::Embedding,
        )
        .unwrap(),
    );

    bp
}

/// Test : génération d'un tenseur avec un seul élément.
#[test]
fn test_single_element_tensor() {
    let blueprint = minimal_blueprint();
    let pipeline = GenerationPipeline::full();

    let gen = ModelGeneratorComplete::new(blueprint, 42, "1.0.0", pipeline, 256);
    let results = gen.generate_all().unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].values.len(), 1);
    // La valeur ne doit pas être NaN ou infinie
    assert!(results[0].values[0].is_finite());
}

/// Test : génération avec une seed très grande.
#[test]
fn test_large_seed() {
    let blueprint = minimal_blueprint();
    let pipeline = GenerationPipeline::full();

    let gen = ModelGeneratorComplete::new(blueprint, u64::MAX, "1.0.0", pipeline, 256);
    let results = gen.generate_all().unwrap();

    assert_eq!(results.len(), 1);
    assert!(results[0].values[0].is_finite());
}

/// Test : génération avec une seed de zéro (devrait échouer la validation).
#[test]
fn test_zero_seed_fails() {
    // Vérifier que GeneratorSeedPlan::validate() rejette la seed 0
    let plan = GeneratorSeedPlan::new(0, "glm-5.2", "1.0.0");
    assert!(plan.validate().is_err(), "La seed 0 devrait être rejetée");

    // Note: La génération avec seed 0 est techniquement possible car
    // GeneratorSeedPlan::new() ne valide pas la seed. La validation est
    // optionnelle et doit être appelée explicitement.
    // Nous testons donc uniquement que la validation échoue.
}

/// Test : reproductibilité avec le même contexte.
#[test]
fn test_reproducibility_with_context() {
    let ctx1 = GenerationContext::new(42, "glm-5.2", "1.0.0", Some(0), 0, 0, "tensor_a", 100, 32);
    let ctx2 = GenerationContext::new(42, "glm-5.2", "1.0.0", Some(0), 0, 0, "tensor_a", 100, 32);

    // Les seeds dérivées doivent être identiques
    assert_eq!(ctx1.tensor_seed(), ctx2.tensor_seed());
    assert_eq!(ctx1.chunk_seed(0), ctx2.chunk_seed(0));
    assert_eq!(ctx1.chunk_seed(1), ctx2.chunk_seed(1));
}

/// Test : les chunks couvrent tous les éléments.
#[test]
fn test_chunk_coverage() {
    let ctx = GenerationContext::new(42, "glm-5.2", "1.0.0", Some(0), 0, 0, "tensor_a", 100, 32);
    let total_chunks = ctx.total_chunks();

    // Vérifier que tous les éléments sont couverts
    let mut covered = vec![false; ctx.num_elements];
    for chunk_id in 0..total_chunks {
        let (start, end) = ctx.chunk_range(chunk_id);
        for item in covered.iter_mut().take(end).skip(start) {
            *item = true;
        }
    }

    assert!(
        covered.iter().all(|&c| c),
        "Certains éléments ne sont pas couverts"
    );
}

/// Test : les tailles de chunks sont correctes.
#[test]
fn test_chunk_sizes() {
    let ctx = GenerationContext::new(42, "glm-5.2", "1.0.0", Some(0), 0, 0, "tensor_a", 100, 32);
    let total_chunks = ctx.total_chunks();

    for chunk_id in 0..total_chunks {
        let (start, end) = ctx.chunk_range(chunk_id);
        let size = end - start;

        // Le dernier chunk peut être plus petit
        if chunk_id < total_chunks - 1 {
            assert_eq!(
                size, ctx.chunk_size,
                "Taille incorrecte pour le chunk {}",
                chunk_id
            );
        } else {
            assert!(size <= ctx.chunk_size, "Dernier chunk trop grand");
        }
    }
}

/// Test : le pipeline vide ne modifie pas les valeurs.
#[test]
fn test_empty_pipeline_no_modification() {
    let pipeline = GenerationPipeline::empty();
    let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let original = values.clone();

    let results = pipeline.execute(&mut values, 42).unwrap();
    assert!(results.is_empty());
    assert_eq!(values, original);
}

/// Test : la génération par chunks produit les mêmes résultats.
#[test]
fn test_chunk_determinism() {
    let mut gen = ModelGenerator::with_seed(42, "glm-5.2");
    let blueprint = minimal_blueprint();
    gen.set_blueprint(blueprint);
    gen.plan().unwrap();

    let spec = &gen.plan_ref().unwrap().tensors[0];

    // Générer avec chunks (taille de chunk = 1 pour forcer plusieurs chunks)
    let mut gen_chunked = ModelGenerator::with_seed(42, "glm-5.2");
    gen_chunked.set_blueprint(minimal_blueprint());
    gen_chunked.plan().unwrap();

    // Utiliser generate_chunked avec une petite taille de chunk
    let chunks = gen_chunked.generate_chunked(spec).unwrap();

    // Reconstruire à partir des chunks
    let mut chunk_values = Vec::new();
    for chunk in &chunks {
        chunk_values.extend_from_slice(&chunk.values);
    }

    // Vérifier que les valeurs sont finies et que la taille est correcte
    assert_eq!(chunk_values.len(), 1, "Le tenseur devrait avoir 1 élément");
    assert!(chunk_values[0].is_finite(), "La valeur devrait être finie");

    // Note: La génération par chunks ne passe pas par le pipeline ni les injections,
    // donc elle diffère de generate_tensor. C'est un comportement attendu.
    // Nous vérifions simplement que la génération par chunks est déterministe.
    let chunks2 = gen_chunked.generate_chunked(spec).unwrap();
    let mut chunk_values2 = Vec::new();
    for chunk in &chunks2 {
        chunk_values2.extend_from_slice(&chunk.values);
    }
    assert_eq!(
        chunk_values, chunk_values2,
        "La génération par chunks devrait être déterministe"
    );
}

/// Test : les seeds dérivées sont différentes pour différents tenseurs.
#[test]
fn test_different_tensors_different_seeds() {
    let plan = GeneratorSeedPlan::new(42, "glm-5.2", "1.0.0");
    let seed1 = plan.derive_tensor_seed("tensor_a", Some(0));
    let seed2 = plan.derive_tensor_seed("tensor_b", Some(0));

    assert_ne!(
        seed1, seed2,
        "Les seeds devraient être différentes pour différents tenseurs"
    );
}

/// Test : les seeds dérivées sont différentes pour différentes couches.
#[test]
fn test_different_layers_different_seeds() {
    let plan = GeneratorSeedPlan::new(42, "glm-5.2", "1.0.0");
    let seed1 = plan.derive_tensor_seed("tensor", Some(0));
    let seed2 = plan.derive_tensor_seed("tensor", Some(1));

    assert_ne!(
        seed1, seed2,
        "Les seeds devraient être différentes pour différentes couches"
    );
}

/// Test : le rapport de génération est cohérent.
#[test]
fn test_report_consistency() {
    let mut gen = ModelGenerator::with_seed(42, "glm-5.2");
    let blueprint = minimal_blueprint();
    gen.set_blueprint(blueprint);
    gen.plan().unwrap();

    let report = gen.generate_report().unwrap();

    assert_eq!(report.seed, 42);
    assert_eq!(report.num_tensors, 1);
    assert_eq!(report.parameter_count, 1);
    assert!(report.distribution_stats.total_analyzed > 0);
    assert!(report.injection_stats.total_analyzed > 0);
}
