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

//! Tests unitaires pour le module policies.

use crate::policies::config::{
    CorrelationPolicy, DtypePolicy, GenerationPolicy, LayerPolicyGlm, LowRankPolicy, ModelPolicies,
    OutlierPolicy, SerializationPolicy, TensorRule,
};
use crate::policies::strategies::{
    CompressionStrategy, CorrelationStrategy, LowRankStrategy, OutlierStrategy, SeedStrategy,
};
use pmg_core::{DType, TensorRole};
use std::collections::BTreeMap;

#[test]
fn test_generation_policy_default() {
    let policy = GenerationPolicy::default_policy();
    assert_eq!(policy.chunk_elements, 1_048_576);
    assert_eq!(policy.seed_strategy, SeedStrategy::Global);
    assert!(policy.deterministic);
    assert_eq!(policy.generator_version, "1.0.0");
    assert!(policy.validate().is_ok());
}

#[test]
fn test_dtype_policy_uniform() {
    let policy = DtypePolicy::uniform(DType::Bf16);
    assert_eq!(policy.default, DType::Bf16);
    assert!(policy.overrides.is_empty());
    assert_eq!(
        policy.effective_dtype(TensorRole::AttentionQuery),
        DType::Bf16
    );
    assert!(policy.validate().is_ok());
}

#[test]
fn test_dtype_policy_with_overrides() {
    let mut overrides = BTreeMap::new();
    overrides.insert(TensorRole::AttentionQuery, DType::F16);
    overrides.insert(TensorRole::MlpGate, DType::F8E4M3);
    let policy = DtypePolicy {
        default: DType::Bf16,
        overrides,
    };
    assert_eq!(
        policy.effective_dtype(TensorRole::AttentionQuery),
        DType::F16
    );
    assert_eq!(policy.effective_dtype(TensorRole::MlpGate), DType::F8E4M3);
    assert_eq!(policy.effective_dtype(TensorRole::Norm), DType::Bf16);
    assert!(policy.validate().is_ok());
}

#[test]
fn test_layer_policy_default() {
    let policy = LayerPolicyGlm::default_for(5);
    assert_eq!(policy.structure_strength, 0.1);
    assert_eq!(policy.outlier_intensity, 1.0);
    assert_eq!(policy.outlier_density, 0.001);
    assert_eq!(policy.layer_seed_shift, 6);
    assert!(policy.validate().is_ok());
}

#[test]
fn test_outlier_policy_none() {
    let policy = OutlierPolicy::none();
    assert_eq!(policy.frequency, 0.0);
    assert_eq!(policy.scale, 1.0);
    assert_eq!(policy.strategy, OutlierStrategy::Multiplicative);
    assert!(policy.validate().is_ok());
}

#[test]
fn test_correlation_policy_none() {
    let policy = CorrelationPolicy::none();
    assert_eq!(policy.strength, 0.0);
    assert_eq!(policy.strategy, CorrelationStrategy::Pearson);
    assert!(policy.validate().is_ok());
}

#[test]
fn test_low_rank_policy_none() {
    let policy = LowRankPolicy::none();
    assert_eq!(policy.rank_threshold, 1);
    assert_eq!(policy.strategy, LowRankStrategy::Svd);
    assert_eq!(policy.alpha, 0.0);
    assert_eq!(policy.probability, 0.0);
    assert!(policy.validate().is_ok());
}

#[test]
fn test_serialization_policy_default() {
    let policy = SerializationPolicy::default_policy();
    assert_eq!(policy.shard_size, 10 * 1024 * 1024 * 1024);
    assert_eq!(policy.compression, CompressionStrategy::None);
    assert_eq!(policy.compression_level, 0);
    assert!(policy.validate().is_ok());
}

#[test]
fn test_tensor_rule_simple() {
    let rule = TensorRule::simple(
        "model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
        TensorRole::MlpGate,
    );
    assert_eq!(
        rule.pattern,
        "model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight"
    );
    assert_eq!(rule.role, TensorRole::MlpGate);
    assert!(rule.dtype_override.is_none());
    assert!(rule.outlier_override.is_none());
    assert!(rule.correlation_override.is_none());
    assert!(rule.low_rank_override.is_none());
    assert!(rule.validate().is_ok());
}

#[test]
fn test_tensor_rule_matches() {
    let rule = TensorRule::simple(
        "model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
        TensorRole::MlpGate,
    );

    assert!(rule.matches("model.layers.0.mlp.experts.5.gate_proj.weight"));
    assert!(rule.matches("model.layers.12.mlp.experts.255.gate_proj.weight"));
    assert!(!rule.matches("model.layers.0.self_attn.q_proj.weight"));
    assert!(!rule.matches("model.norm.weight"));
}

#[test]
fn test_model_policies_validate() {
    let policies = ModelPolicies {
        generation: GenerationPolicy::default_policy(),
        dtype: DtypePolicy::uniform(DType::Bf16),
        correlation: CorrelationPolicy::none(),
        low_rank: LowRankPolicy::none(),
        serialization: SerializationPolicy::default_policy(),
        tensor_rules: vec![
            TensorRule::simple("model.embed_tokens.weight", TensorRole::Embedding),
            TensorRule::simple("model.norm.weight", TensorRole::Norm),
        ],
    };

    assert!(policies.validate().is_ok());
}

#[test]
fn test_validation_errors() {
    // GenerationPolicy avec chunk_elements = 0
    let mut gen_policy = GenerationPolicy::default_policy();
    gen_policy.chunk_elements = 0;
    assert!(gen_policy.validate().is_err());

    // OutlierPolicy avec frequency > 1
    let mut outlier_policy = OutlierPolicy::none();
    outlier_policy.frequency = 1.5;
    assert!(outlier_policy.validate().is_err());

    // CorrelationPolicy avec strength = 1.0
    let mut corr_policy = CorrelationPolicy::none();
    corr_policy.strength = 1.0;
    assert!(corr_policy.validate().is_err());

    // LowRankPolicy avec rank_threshold = 0
    let mut lr_policy = LowRankPolicy::none();
    lr_policy.rank_threshold = 0;
    assert!(lr_policy.validate().is_err());

    // SerializationPolicy avec shard_size = 0
    let mut ser_policy = SerializationPolicy::default_policy();
    ser_policy.shard_size = 0;
    assert!(ser_policy.validate().is_err());
}

#[test]
fn test_seed_strategy_labels() {
    assert_eq!(SeedStrategy::Global.label_fr(), "globale");
    assert_eq!(SeedStrategy::PerTensor.label_fr(), "par tenseur");
    assert_eq!(SeedStrategy::PerLayer.label_fr(), "par couche");
    assert_eq!(SeedStrategy::PerExpert.label_fr(), "par expert");
}

#[test]
fn test_outlier_strategy_labels() {
    assert_eq!(OutlierStrategy::Multiplicative.label_fr(), "multiplicatif");
    assert_eq!(OutlierStrategy::Additive.label_fr(), "additif");
    assert_eq!(OutlierStrategy::Replacement.label_fr(), "remplacement");
    assert_eq!(OutlierStrategy::HeavyTail.label_fr(), "queues lourdes");
}

#[test]
fn test_correlation_strategy_labels() {
    assert_eq!(CorrelationStrategy::Pearson.label_fr(), "Pearson");
    assert_eq!(CorrelationStrategy::Spearman.label_fr(), "Spearman");
    assert_eq!(CorrelationStrategy::Kendall.label_fr(), "Kendall");
    assert_eq!(CorrelationStrategy::Block.label_fr(), "par blocs");
}

#[test]
fn test_low_rank_strategy_labels() {
    assert_eq!(LowRankStrategy::Svd.label_fr(), "SVD");
    assert_eq!(LowRankStrategy::Nmf.label_fr(), "NMF");
    assert_eq!(LowRankStrategy::Randomized.label_fr(), "aléatoire");
    assert_eq!(LowRankStrategy::Pca.label_fr(), "ACP");
}

#[test]
fn test_compression_strategy_labels() {
    assert_eq!(CompressionStrategy::None.label_fr(), "aucune");
    assert_eq!(CompressionStrategy::Zstd.label_fr(), "zstd");
    assert_eq!(CompressionStrategy::Lz4.label_fr(), "lz4");
    assert_eq!(CompressionStrategy::Gzip.label_fr(), "gzip");
}
