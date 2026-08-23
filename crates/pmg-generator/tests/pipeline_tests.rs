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

//! Tests d'intégration du pipeline de génération (Sprint 12).
//!
//! Ces tests vérifient le bon fonctionnement de l'orchestration de la génération,
//! la déterminisme, l'ordre du pipeline, et la collecte des statistiques.

use pmg_blueprint::architecture::ArchitectureKind;
use pmg_blueprint::layer::{LayerKind, LayerSpec};
use pmg_blueprint::naming::NamingRules;
use pmg_blueprint::tensor_spec::TensorSpec;
use pmg_blueprint::ModelBlueprint;
use pmg_core::model_config::glm52_test_config;
use pmg_core::{DType, Shape, TensorRole};

use pmg_generator::{
    GenerationContext, GenerationPipeline, LayerGenerator, ModelGenerator, ModelGeneratorComplete,
    PipelineStep,
};

/// Blueprint de test simple pour les tests d'intégration.
fn test_blueprint() -> ModelBlueprint {
    let config = glm52_test_config();
    let mut bp = ModelBlueprint::new(
        "glm-5.2",
        ArchitectureKind::MoETransformer,
        config,
        NamingRules::glm52(),
    );

    // Embedding
    bp.embeddings.push(
        TensorSpec::new(
            "model.embed_tokens.weight",
            Shape::new(vec![100, 64]).unwrap(),
            DType::F32,
            TensorRole::Embedding,
        )
        .unwrap(),
    );

    // Couche 0
    let mut layer0 = LayerSpec::new(0, LayerKind::Dense);
    layer0.attention.push(
        TensorSpec::new(
            "model.layers.0.self_attn.q_proj.weight",
            Shape::new(vec![64, 64]).unwrap(),
            DType::F32,
            TensorRole::AttentionQuery,
        )
        .unwrap(),
    );
    layer0.mlp.push(
        TensorSpec::new(
            "model.layers.0.mlp.gate_proj.weight",
            Shape::new(vec![128, 64]).unwrap(),
            DType::F32,
            TensorRole::MlpGate,
        )
        .unwrap(),
    );
    bp.layers.push(layer0);

    // Norme finale
    bp.final_norm.push(
        TensorSpec::new(
            "model.norm.weight",
            Shape::new(vec![64]).unwrap(),
            DType::F32,
            TensorRole::Norm,
        )
        .unwrap(),
    );

    // LM Head
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

/// Test : même seed → même sortie (déterminisme).
#[test]
fn test_same_seed_same_output() {
    let blueprint1 = test_blueprint();
    let blueprint2 = test_blueprint();
    let pipeline = GenerationPipeline::full();

    let gen1 = ModelGeneratorComplete::new(blueprint1, 42, "1.0.0", pipeline.clone(), 256);
    let gen2 = ModelGeneratorComplete::new(blueprint2, 42, "1.0.0", pipeline, 256);

    let results1 = gen1.generate_all().unwrap();
    let results2 = gen2.generate_all().unwrap();

    assert_eq!(results1.len(), results2.len());
    for (i, (r1, r2)) in results1.iter().zip(results2.iter()).enumerate() {
        assert_eq!(r1.name, r2.name, "Les noms diffèrent au tenseur {}", i);
        assert_eq!(
            r1.values, r2.values,
            "Les valeurs diffèrent au tenseur {}",
            i
        );
    }
}

/// Test : seeds différentes → sorties différentes.
#[test]
fn test_different_seed_different_output() {
    let blueprint1 = test_blueprint();
    let blueprint2 = test_blueprint();
    let pipeline = GenerationPipeline::full();

    let gen1 = ModelGeneratorComplete::new(blueprint1, 42, "1.0.0", pipeline.clone(), 256);
    let gen2 = ModelGeneratorComplete::new(blueprint2, 123, "1.0.0", pipeline, 256);

    let results1 = gen1.generate_all().unwrap();
    let results2 = gen2.generate_all().unwrap();

    assert_eq!(results1.len(), results2.len());
    // Au moins un tenseur doit être différent
    let mut any_different = false;
    for (r1, r2) in results1.iter().zip(results2.iter()) {
        if r1.values != r2.values {
            any_different = true;
            break;
        }
    }
    assert!(
        any_different,
        "Aucune différence détectée entre les générations avec des seeds différentes"
    );
}

/// Test : l'ordre du pipeline est respecté.
#[test]
fn test_pipeline_order() {
    let pipeline = GenerationPipeline::full();
    let steps = pipeline.steps();

    assert_eq!(steps.len(), 5);
    assert_eq!(steps[0], PipelineStep::Distribution);
    assert_eq!(steps[1], PipelineStep::Correlation);
    assert_eq!(steps[2], PipelineStep::LowRank);
    assert_eq!(steps[3], PipelineStep::Outliers);
    assert_eq!(steps[4], PipelineStep::SuperWeights);

    // Vérifier que l'ordre est le même même après ajout/suppression
    let mut pipeline2 = GenerationPipeline::empty();
    pipeline2.add_step(PipelineStep::SuperWeights);
    pipeline2.add_step(PipelineStep::Distribution);
    pipeline2.add_step(PipelineStep::Outliers);

    let steps2 = pipeline2.steps();
    assert_eq!(steps2[0], PipelineStep::Distribution);
    assert_eq!(steps2[1], PipelineStep::Outliers);
    assert_eq!(steps2[2], PipelineStep::SuperWeights);
}

/// Test : génération d'un tenseur individuel.
#[test]
fn test_tensor_generation() {
    let blueprint = test_blueprint();
    let mut gen = ModelGenerator::with_seed(42, "glm-5.2");
    gen.set_blueprint(blueprint);
    gen.plan().unwrap();

    let spec = &gen.plan_ref().unwrap().tensors[0];
    let tensor = gen.generate_tensor(spec).unwrap();

    assert_eq!(tensor.num_elements, 100 * 64);
    assert_eq!(tensor.values.len(), 100 * 64);

    // Vérifier que les valeurs ne sont pas toutes identiques
    let first = tensor.values[0];
    let different = tensor.values.iter().any(|&v| (v - first).abs() > 1e-10);
    assert!(different, "Les valeurs générées sont toutes identiques");
}

/// Test : génération d'une couche complète.
#[test]
fn test_layer_generation() {
    let blueprint = test_blueprint();
    let layer_spec = blueprint.layers[0].clone();
    let pipeline = GenerationPipeline::full();

    let layer_gen = LayerGenerator::new(layer_spec, 42, "glm-5.2", "1.0.0", pipeline, 256);

    let results = layer_gen.generate_all().unwrap();
    assert_eq!(results.len(), 2); // q_proj + gate_proj

    // Vérifier les noms
    assert_eq!(results[0].0, "model.layers.0.self_attn.q_proj.weight");
    assert_eq!(results[1].0, "model.layers.0.mlp.gate_proj.weight");

    // Vérifier les tailles
    assert_eq!(results[0].1.len(), 64 * 64);
    assert_eq!(results[1].1.len(), 128 * 64);
}

/// Test : génération du modèle complet.
#[test]
fn test_model_generation() {
    let blueprint = test_blueprint();
    let pipeline = GenerationPipeline::full();

    let gen = ModelGeneratorComplete::new(blueprint, 42, "1.0.0", pipeline, 256);
    let results = gen.generate_all().unwrap();

    // 1 embedding + 2 couches (q_proj + gate_proj) + 1 norme + 1 lm_head = 5 tenseurs
    assert_eq!(results.len(), 5);

    // Vérifier les catégories
    assert_eq!(results[0].category, "embedding");
    assert_eq!(results[1].category, "layer");
    assert_eq!(results[2].category, "layer");
    assert_eq!(results[3].category, "final_norm");
    assert_eq!(results[4].category, "lm_head");

    // Vérifier les index de couche
    assert_eq!(results[0].layer_index, None);
    assert_eq!(results[1].layer_index, Some(0));
    assert_eq!(results[2].layer_index, Some(0));
    assert_eq!(results[3].layer_index, None);
    assert_eq!(results[4].layer_index, None);
}

/// Test : collecte des statistiques.
#[test]
fn test_statistics_collection() {
    let blueprint = test_blueprint();
    let pipeline = GenerationPipeline::full();

    let gen = ModelGeneratorComplete::new(blueprint, 42, "1.0.0", pipeline, 256);
    let results = gen.generate_all().unwrap();
    let stats = gen.compute_stats(&results);

    // Vérifier que les statistiques sont raisonnables
    assert!(
        stats.mean.abs() < 10.0,
        "Moyenne trop élevée: {}",
        stats.mean
    );
    assert!(
        stats.variance >= 0.0,
        "Variance négative: {}",
        stats.variance
    );
    assert!(
        stats.std_dev >= 0.0,
        "Écart-type négatif: {}",
        stats.std_dev
    );
    assert!(stats.min <= stats.max, "min > max");
    assert!(stats.parameter_count > 0, "Nombre de paramètres nul");

    // Vérifier que les compteurs sont cohérents
    assert_eq!(
        stats.parameter_count,
        100 * 64 + 64 * 64 + 128 * 64 + 64 + 100 * 64
    );
    assert_eq!(stats.tensor_count, 5);
}

/// Test : le contexte génère des seeds déterministes.
#[test]
fn test_context_deterministic() {
    let ctx1 = GenerationContext::new(42, "glm-5.2", "1.0.0", Some(0), 0, 0, "tensor_a", 100, 32);
    let ctx2 = GenerationContext::new(42, "glm-5.2", "1.0.0", Some(0), 0, 0, "tensor_a", 100, 32);

    assert_eq!(ctx1.tensor_seed(), ctx2.tensor_seed());
    assert_eq!(ctx1.chunk_seed(0), ctx2.chunk_seed(0));
    assert_eq!(ctx1.total_chunks(), ctx2.total_chunks());
}

/// Test : les étapes du pipeline peuvent être activées/désactivées.
#[test]
fn test_pipeline_enable_disable() {
    let mut pipeline = GenerationPipeline::full();
    assert_eq!(pipeline.step_count(), 5);

    // Désactiver Corrélation
    pipeline.remove_step(&PipelineStep::Correlation);
    assert_eq!(pipeline.step_count(), 4);
    assert!(!pipeline.has_step(&PipelineStep::Correlation));

    // Réactiver Corrélation
    pipeline.add_step(PipelineStep::Correlation);
    assert_eq!(pipeline.step_count(), 5);
    assert!(pipeline.has_step(&PipelineStep::Correlation));
}

/// Test : le pipeline produit des résultats pour chaque étape.
#[test]
fn test_pipeline_results() {
    let pipeline = GenerationPipeline::full();
    let mut values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let results = pipeline.execute(&mut values, 42).unwrap();

    assert_eq!(results.len(), 5);
    assert_eq!(results[0].step_name, "Distribution");
    assert_eq!(results[1].step_name, "Corrélation");
    assert_eq!(results[2].step_name, "Bas-rang");
    assert_eq!(results[3].step_name, "Outliers");
    assert_eq!(results[4].step_name, "Super-poids");
}

/// Test : le générateur ModelGenerator utilise le pipeline.
#[test]
fn test_model_generator_with_pipeline() {
    let mut gen = ModelGenerator::with_seed(42, "glm-5.2");
    let blueprint = test_blueprint();
    gen.set_blueprint(blueprint);

    // Vérifier que le pipeline est défini par défaut
    assert_eq!(gen.pipeline().step_count(), 5);

    // Modifier le pipeline
    let mut new_pipeline = GenerationPipeline::empty();
    new_pipeline.add_step(PipelineStep::Distribution);
    gen.set_pipeline(new_pipeline);

    assert_eq!(gen.pipeline().step_count(), 1);
    assert!(gen.pipeline().has_step(&PipelineStep::Distribution));
}
