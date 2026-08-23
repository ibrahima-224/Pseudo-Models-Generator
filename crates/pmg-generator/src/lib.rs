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

//! Crate `pmg-generator` — orchestration de la génération déterministe.
//!
//! Chef d'orchestre du pipeline complet : `ModelConfig → Blueprint → TensorPlan →
//! TensorGenerator → Injector → Tensor`. Ce crate assemble les moteurs mathématiques,
//! le blueprint et l'injection pour produire des pseudo-modèles déterministes.
//!
//! ## Responsabilité
//!
//! - **`ModelGenerator`** : point d'entrée unique du pipeline de génération ;
//! - **`TensorGenerator`** : génération des valeurs initiales d'un tenseur ;
//! - **`GeneratorSeedPlan`** : dérivation hiérarchique des seeds (tenseur, chunk) ;
//! - **`ChunkIterator`** : découpage en blocs pour la gestion mémoire ;
//! - **`GenerationReport`** : rapport de génération structuré et sérialisable ;
//! - **`GenerationValidator`** : validation de la cohérence de la génération ;
//! - **helpers de déterminisme** : garantie et tests du déterminisme.
//!
//! ## Dépendances
//!
//! `pmg-core`, `pmg-blueprint`, `pmg-math`, `pmg-injector`, `pmg-io`, `pmg-models`,
//! `pmg-meta`. Ce crate ne dépend pas de `pmg-io` ou `pmg-models` pour la logique
//! de génération (ceux-ci sont squelettes).
//!
//! ## Pipeline
//!
//! ```text
//! ModelConfig
//!    ↓
//! Blueprint (validation)
//!    ↓
//! TensorPlan (planification déterministe)
//!    ↓
//! TensorGenerator (génération des valeurs initiales)
//!    ↓
//! Injector (injections structurelles)
//!    ↓
//! Tensor final
//! ```
//!
//! ## Déterminisme
//!
//! Toute génération passe par un RNG dérivé de seed. Les mêmes entrées
//! produisent exactement les mêmes sorties, bit à bit. Le découpage en
//! chunks préserve le déterminisme : la génération par chunks produit
//! les mêmes résultats qu'une génération non découpée.
//!
//! # Exemple simple
//!
//! ```
//! use pmg_generator::{ModelGenerator, GenerationConfig};
//! use pmg_core::model_config::glm52_test_config;
//! use pmg_blueprint::{ArchitectureKind, ModelBlueprint, NamingRules};
//!
//! // Configuration de génération
//! let config = GenerationConfig {
//!     seed: 42,
//!     model_id: "glm-5.2".to_string(),
//!     ..GenerationConfig::default()
//! };
//!
//! // Création du générateur
//! let mut gen = ModelGenerator::new(config);
//!
//! // Blueprint simple (un seul tenseur d'embedding)
//! let mut bp = ModelBlueprint::new(
//!     "glm-5.2",
//!     ArchitectureKind::MoETransformer,
//!     glm52_test_config(),
//!     NamingRules::glm52(),
//! );
//! bp.embeddings.push(
//!     pmg_blueprint::TensorSpec::new(
//!         "model.embed_tokens.weight",
//!         pmg_core::Shape::new(vec![100, 64]).unwrap(),
//!         pmg_core::DType::F32,
//!         pmg_core::TensorRole::Embedding,
//!     )
//!     .unwrap(),
//! );
//!
//! // Planification et génération
//! gen.set_blueprint(bp);
//! gen.plan().unwrap();
//! let tensors = gen.generate_all().unwrap();
//! assert_eq!(tensors.len(), 1);
//! ```
//!
//! # Exemple avec pipeline personnalisé
//!
//! ```
//! use pmg_generator::{ModelGenerator, GenerationConfig, GenerationPipeline, PipelineStep};
//! use pmg_core::model_config::glm52_test_config;
//! use pmg_blueprint::{ArchitectureKind, ModelBlueprint, NamingRules};
//!
//! // Configuration avec pipeline personnalisé
//! let mut pipeline = GenerationPipeline::empty();
//! pipeline.add_step(PipelineStep::Distribution);
//! pipeline.add_step(PipelineStep::Outliers);
//!
//! let config = GenerationConfig {
//!     seed: 123,
//!     model_id: "glm-5.2".to_string(),
//!     ..GenerationConfig::default()
//! };
//!
//! let mut gen = ModelGenerator::new(config);
//! gen.set_pipeline(pipeline);
//!
//! // Blueprint avec plusieurs tenseurs
//! let mut bp = ModelBlueprint::new(
//!     "glm-5.2",
//!     ArchitectureKind::MoETransformer,
//!     glm52_test_config(),
//!     NamingRules::glm52(),
//! );
//!
//! // Ajout d'un embedding
//! bp.embeddings.push(
//!     pmg_blueprint::TensorSpec::new(
//!         "model.embed_tokens.weight",
//!         pmg_core::Shape::new(vec![100, 64]).unwrap(),
//!         pmg_core::DType::F32,
//!         pmg_core::TensorRole::Embedding,
//!     )
//!     .unwrap(),
//! );
//!
//! // Ajout d'une couche
//! let mut layer = pmg_blueprint::layer::LayerSpec::new(0, pmg_blueprint::layer::LayerKind::Dense);
//! layer.attention.push(
//!     pmg_blueprint::TensorSpec::new(
//!         "model.layers.0.self_attn.q_proj.weight",
//!         pmg_core::Shape::new(vec![64, 64]).unwrap(),
//!         pmg_core::DType::F32,
//!         pmg_core::TensorRole::AttentionQuery,
//!     )
//!     .unwrap(),
//! );
//! bp.layers.push(layer);
//!
//! gen.set_blueprint(bp);
//! gen.plan().unwrap();
//!
//! // Génération avec validation
//! let tensors = gen.generate_all().unwrap();
//! let report = gen.generate_report().unwrap();
//! let validation = gen.validate().unwrap();
//!
//! assert!(validation.success);
//! assert_eq!(report.num_tensors, 2);
//! ```

pub mod async_pipeline;
pub mod budget;
pub mod chunk;
pub mod context;
pub mod deterministic;
pub mod distributed;
pub mod error;
pub mod generation_report;
pub mod generation_stats;
pub mod generation_validator;
pub mod generator_config;
pub mod generator_core;
pub mod generator_impl;
pub mod gpu_support;
pub mod iterator;
pub mod layer_generator;
pub mod lazy_iterator;
pub mod memory_manager;
pub mod model_generator;
pub mod optimized_generator;
pub mod output;
pub mod output_streaming;
pub mod pipeline;
pub mod pipeline_config;
pub mod pipeline_steps;
pub mod progress;
pub mod seed_plan;
pub mod streaming;
pub mod streaming_pipeline;
pub mod tensor_chunk_generator;
pub mod tensor_generator;
pub mod writer;

// Ré-exports pratiques pour les consommateurs.
pub use async_pipeline::{
    generate_model_async, AsyncConfig, AsyncPipeline, AsyncProgressCallback, TensorJob,
    TensorResult,
};
pub use budget::{BudgetError, BudgetPlanner, GenerationMode};
pub use chunk::{ChunkIterator, TensorChunk, DEFAULT_CHUNK_SIZE};
pub use deterministic::{
    assert_chunk_determinism, assert_different_seeds_different_results,
    assert_injection_determinism, assert_tensor_determinism,
};
pub use error::{GeneratorError, GeneratorResult};
pub use generation_report::{DistributionStats, GenerationReport, InjectionStats};
pub use generation_validator::{GenerationValidator, ValidationResult};
pub use generator_config::GeneratorConfig;
pub use generator_core::{GeneratedTensor, GenerationConfig, ModelGenerator};
pub use lazy_iterator::LazyBaseDistribution;
pub use memory_manager::BoundedMemoryManager;
pub use seed_plan::GeneratorSeedPlan;
pub use streaming::{Chunk, StreamingGenerator, StreamingStats};
pub use streaming_pipeline::StreamingPipeline;
pub use tensor_generator::TensorGenerator;
pub mod compression;
pub mod memory_monitor;
pub mod streaming_config;
pub use compression::{GeneratorCompressionConfig, GeneratorCompressor};
pub use writer::{
    write_safetensors, write_safetensors_atomic, SafetensorMetadata, SafetensorsWriter,
};

// Ré-exports des nouveaux modules du Sprint 12.
pub use context::GenerationContext;
pub use generation_stats::GenerationStats;
pub use layer_generator::LayerGenerator;
pub use model_generator::{ModelGeneratorComplete, ModelTensorResult};
pub use output::{
    execute_pipeline_output, execute_pipeline_output_streaming, PipelineOutputConfig,
    PipelineOutputResult,
};
pub use output_streaming::execute_full_pipeline_streaming;
pub use pipeline::{GenerationPipeline, PipelineStep, StepResult};
pub use pipeline_config::{
    CorrelationConfig, DistributionConfig, LowRankConfig, OutlierReplacementMode, OutliersConfig,
    PipelineGlobalConfig, SuperWeightsConfig,
};
pub use progress::{
    create_progress_tracker, format_duration, format_size, DisplayMode, ProgressStats,
    ProgressTracker,
};

// Ré-exports du module distribué.
pub use distributed::{
    DistributedConfig, DistributedCoordinator, DistributedError, DistributedResult,
    DistributedTask, DistributedWorker, GlobalStats, TaskStats, TaskStatus, WorkerInfo,
};

// Test de smoke pour vérifier que le crate compile.
#[cfg(test)]
mod tests {

    #[test]
    fn crate_compiles() {
        // Test trivial de compilation du crate.
        let _ = 0u64;
    }
}
