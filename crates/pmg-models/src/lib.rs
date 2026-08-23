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

//! Crate `pmg-models` — profils des modèles cibles supportés.
//!
//! Décrit les propriétés observées/estimées des modèles cibles v1.0
//! (**GLM-5.2** : 78 couches, 256+1 experts, top-8, vocab 154880 ;
//! **DeepSeek-V4-Flash** : 43 couches, 64 têtes, KV=1, head_dim 512,
//! 256+1 experts, top-6, vocab 129280, hyper-connections FP8/FP4 déclarés).
//!
//! ## Responsabilité
//!
//! - Trait `ModelProfile` : model_family, architecture, tensor_rules
//!   (pattern → rôle + politiques), generation_policy, dtype_policy,
//!   layer_policy (θ_l = f(l)), outlier_policy, correlation_policy,
//!   low_rank_policy, serialization_policy, provenance par propriété
//!   (EXACT/DERIVED/ESTIMATED/SYNTHETIC/UNKNOWN) ;
//! - deux implémentations : GLM-5.2 et DeepSeek-V4-Flash (valeurs issues des
//!   artefacts `Models/` et des informations publiées, sources consignées) ;
//! - aucun transfert de propriété d'un modèle à l'autre (pilier 13) ;
//! - profils statistiques chargeables depuis `profiles/*.json`, profils
//!   embarqués par défaut compilés dans la crate.
//!
//! ## Dépendances
//!
//! `pmg-core`, `pmg-blueprint`, `serde`, `serde_json`, `thiserror`.
//!
//! ## Statut
//!
//! Sprint 0 : squelette documenté, aucune API publique. Implémentation prévue
//! aux lots L2/L10 (voir `docs/architecture/08-plan-implementation.md` §6).
//!
//! # Exemple
//!
//! ```
//! use pmg_models::{Glm52Profile, ModelProfile};
//!
//! let profile = Glm52Profile::default_profile();
//! assert_eq!(profile.model_family(), "GLM");
//! assert_eq!(profile.num_layers(), 78);
//! ```

mod deepseek_v4_flash;
mod error;
mod glm52;
mod policies;
mod profile;

pub use deepseek_v4_flash::DeepseekV4FlashProfile;
pub use error::{ModelProfileError, Result};
pub use glm52::Glm52Profile;
pub use policies::{
    CompressionStrategy, CorrelationPolicy, CorrelationStrategy, DtypePolicy, GenerationPolicy,
    LayerPolicyGlm, LowRankPolicy, LowRankStrategy, ModelPolicies, OutlierPolicy, OutlierStrategy,
    SeedStrategy, SerializationPolicy, TensorRule,
};
pub use profile::{
    load_profile_from_file, validate_profile, MetadataSource, ModelProfile, ProfileData,
};

/// Re-export pour la commodité.
pub use error::Result as ModelsResult;

#[cfg(test)]
mod tests {
    use super::*;
    use pmg_core::{DType, TensorRole};

    #[test]
    fn test_skeleton_compiles() {
        // Test de compilation du squelette.
        let _ = 0u64;
    }

    #[test]
    fn test_glm52_default_profile() {
        let profile = Glm52Profile::default_profile();
        assert_eq!(profile.model_family(), "GLM");
        assert_eq!(profile.num_layers(), 78);
    }

    #[test]
    fn test_deepseek_v4_flash_default_profile() {
        let profile = DeepseekV4FlashProfile::default_profile();
        assert_eq!(profile.model_family(), "DeepSeek");
        assert_eq!(profile.num_layers(), 43);
    }

    #[test]
    fn test_metadata_source_display() {
        assert_eq!(MetadataSource::Exact.to_string(), "EXACT");
        assert_eq!(MetadataSource::Derived.to_string(), "DERIVED");
        assert_eq!(MetadataSource::Estimated.to_string(), "ESTIMATED");
        assert_eq!(MetadataSource::Synthetic.to_string(), "SYNTHETIC");
        assert_eq!(MetadataSource::Unknown.to_string(), "UNKNOWN");
    }

    #[test]
    fn test_load_profile_from_file_glm52() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../profiles/glm52.json");
        let profile = load_profile_from_file(&path).expect("chargement du profil GLM-5.2");
        assert_eq!(profile.model_family(), "GLM");
        assert_eq!(profile.num_layers(), 78);
    }

    #[test]
    fn test_load_profile_from_file_deepseek() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../profiles/deepseek_v4_flash.json");
        let profile =
            load_profile_from_file(&path).expect("chargement du profil DeepSeek-V4-Flash");
        assert_eq!(profile.model_family(), "DeepSeek");
        assert_eq!(profile.num_layers(), 43);
    }

    #[test]
    fn test_load_profile_unknown_model() {
        let dir = std::env::temp_dir().join("pmg_models_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("unknown.json");
        std::fs::write(
            &path,
            r#"{
                "model_type": "unknown_model",
                "architecture": "UnknownArch",
                "hidden_size": 1024,
                "num_attention_heads": 16,
                "num_hidden_layers": 12,
                "vocab_size": 32000,
                "max_position_embeddings": 4096,
                "tensor_patterns": []
            }"#,
        )
        .unwrap();
        let result = load_profile_from_file(&path);
        assert!(result.is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_glm52_profile_policies() {
        let profile = Glm52Profile::default_profile();

        // Vérification des politiques
        let gen_policy = profile.generation_policy();
        assert_eq!(gen_policy.chunk_elements, 1_048_576);
        assert!(gen_policy.deterministic);

        let dtype_policy = profile.dtype_policy();
        assert_eq!(dtype_policy.default, DType::Bf16);
        assert_eq!(
            dtype_policy.effective_dtype(TensorRole::AttentionQuery),
            DType::F16
        );

        let layer_policy = profile.layer_policy();
        assert_eq!(layer_policy.structure_strength, 0.1);

        let outlier_policy = profile.outlier_policy();
        assert_eq!(outlier_policy.frequency, 0.001);
        assert_eq!(outlier_policy.scale, 10.0);

        let corr_policy = profile.correlation_policy();
        assert_eq!(corr_policy.strength, 0.1);

        let lr_policy = profile.low_rank_policy();
        assert_eq!(lr_policy.probability, 0.0);

        let ser_policy = profile.serialization_policy();
        assert_eq!(ser_policy.shard_size, 10 * 1024 * 1024 * 1024);

        let tensor_rules = profile.tensor_rules();
        assert_eq!(tensor_rules.len(), 13);
        assert_eq!(tensor_rules[0].role, TensorRole::Embedding);
    }

    #[test]
    fn test_deepseek_v4_flash_profile_policies() {
        let profile = DeepseekV4FlashProfile::default_profile();

        // Vérification des politiques
        let gen_policy = profile.generation_policy();
        assert_eq!(gen_policy.chunk_elements, 1_048_576);

        let dtype_policy = profile.dtype_policy();
        assert_eq!(dtype_policy.default, DType::Bf16);
        assert_eq!(
            dtype_policy.effective_dtype(TensorRole::MoeRoutedExpert),
            DType::F8E4M3
        );

        let outlier_policy = profile.outlier_policy();
        assert_eq!(outlier_policy.frequency, 0.002);
        assert_eq!(outlier_policy.scale, 15.0);

        let tensor_rules = profile.tensor_rules();
        assert_eq!(tensor_rules.len(), 13);
    }

    #[test]
    fn test_tensor_rule_matching() {
        let rule = TensorRule::simple(
            "model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
            TensorRole::MlpGate,
        );

        // Test de correspondance
        assert!(rule.matches("model.layers.0.mlp.experts.5.gate_proj.weight"));
        assert!(rule.matches("model.layers.12.mlp.experts.255.gate_proj.weight"));
        assert!(!rule.matches("model.layers.0.self_attn.q_proj.weight"));
        assert!(!rule.matches("model.norm.weight"));
    }

    #[test]
    fn test_model_policies_validation() {
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
    fn test_policy_enums_labels() {
        // Test des libellés français
        assert_eq!(SeedStrategy::Global.label_fr(), "globale");
        assert_eq!(OutlierStrategy::Multiplicative.label_fr(), "multiplicatif");
        assert_eq!(CorrelationStrategy::Pearson.label_fr(), "Pearson");
        assert_eq!(LowRankStrategy::Svd.label_fr(), "SVD");
        assert_eq!(CompressionStrategy::None.label_fr(), "aucune");
    }
}
