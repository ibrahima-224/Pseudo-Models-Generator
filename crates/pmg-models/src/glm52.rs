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

//! Implémentation du profil pour le modèle GLM-5.2.

use std::collections::BTreeMap;

use pmg_core::{DType, TensorRole};

use crate::policies::{
    CorrelationPolicy, DtypePolicy, GenerationPolicy, LayerPolicyGlm, LowRankPolicy, OutlierPolicy,
    SerializationPolicy, TensorRule,
};
use crate::profile::{MetadataSource, ModelProfile, ProfileData};

/// Profil du modèle GLM-5.2 (architecture GlmMoeDsaForCausalLM).
///
/// Caractéristiques principales :
/// - 78 couches cachées
/// - Dimension cachée 6144
/// - 64 têtes d'attention
/// - 256 experts routés + 1 expert partagé (MoE)
/// - Vocabulaire 154880
/// - Contexte maximal 1 048 576 tokens
///
/// # Exemple
///
/// ```rust
/// use pmg_models::{Glm52Profile, ModelProfile};
///
/// // Création du profil par défaut
/// let profile = Glm52Profile::default_profile();
/// assert_eq!(profile.model_family(), "GLM");
/// assert_eq!(profile.num_layers(), 78);
///
/// // Validation du profil
/// assert!(profile.validate().is_ok());
/// ```
#[derive(Debug, Clone)]
pub struct Glm52Profile {
    data: ProfileData,
    generation_policy: GenerationPolicy,
    dtype_policy: DtypePolicy,
    layer_policy: LayerPolicyGlm,
    outlier_policy: OutlierPolicy,
    correlation_policy: CorrelationPolicy,
    low_rank_policy: LowRankPolicy,
    serialization_policy: SerializationPolicy,
    tensor_rules: Vec<TensorRule>,
}

impl Glm52Profile {
    /// Crée un nouveau profil GLM-5.2 avec les données fournies.
    pub fn from_data(data: ProfileData) -> Self {
        // Politiques par défaut pour GLM-5.2
        let generation_policy = GenerationPolicy::default_policy();

        // GLM-5.2 utilise principalement BF16, avec des overrides pour certains rôles
        let mut dtype_overrides = BTreeMap::new();
        dtype_overrides.insert(TensorRole::AttentionQuery, DType::F16);
        dtype_overrides.insert(TensorRole::AttentionKey, DType::F16);
        dtype_overrides.insert(TensorRole::AttentionValue, DType::F16);
        dtype_overrides.insert(TensorRole::MlpGate, DType::F16);
        dtype_overrides.insert(TensorRole::MlpUp, DType::F16);
        dtype_overrides.insert(TensorRole::MlpDown, DType::F16);

        let dtype_policy = DtypePolicy {
            default: DType::Bf16,
            overrides: dtype_overrides,
        };

        let layer_policy = LayerPolicyGlm::default_for(0);

        let outlier_policy = OutlierPolicy {
            frequency: 0.001,
            scale: 10.0,
            strategy: crate::policies::OutlierStrategy::Multiplicative,
            heavy_tail_df: 5.0,
        };

        let correlation_policy = CorrelationPolicy {
            strength: 0.1,
            strategy: crate::policies::CorrelationStrategy::Pearson,
        };

        let low_rank_policy = LowRankPolicy::none();

        let serialization_policy = SerializationPolicy::default_policy();

        // Règles de mapping pour GLM-5.2
        let tensor_rules = vec![
            TensorRule::simple("model.embed_tokens.weight", TensorRole::Embedding),
            TensorRule::simple(
                "model.layers.{layer}.self_attn.q_proj.weight",
                TensorRole::AttentionQuery,
            ),
            TensorRule::simple(
                "model.layers.{layer}.self_attn.k_proj.weight",
                TensorRole::AttentionKey,
            ),
            TensorRule::simple(
                "model.layers.{layer}.self_attn.v_proj.weight",
                TensorRole::AttentionValue,
            ),
            TensorRule::simple(
                "model.layers.{layer}.self_attn.o_proj.weight",
                TensorRole::AttentionOutput,
            ),
            TensorRule::simple(
                "model.layers.{layer}.mlp.gate_proj.weight",
                TensorRole::MlpGate,
            ),
            TensorRule::simple("model.layers.{layer}.mlp.up_proj.weight", TensorRole::MlpUp),
            TensorRule::simple(
                "model.layers.{layer}.mlp.down_proj.weight",
                TensorRole::MlpDown,
            ),
            TensorRule::simple(
                "model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight",
                TensorRole::MoeRoutedExpert,
            ),
            TensorRule::simple(
                "model.layers.{layer}.mlp.experts.{expert}.up_proj.weight",
                TensorRole::MoeRoutedExpert,
            ),
            TensorRule::simple(
                "model.layers.{layer}.mlp.experts.{expert}.down_proj.weight",
                TensorRole::MoeRoutedExpert,
            ),
            TensorRule::simple("model.norm.weight", TensorRole::Norm),
            TensorRule::simple("lm_head.weight", TensorRole::LmHead),
        ];

        Self {
            data,
            generation_policy,
            dtype_policy,
            layer_policy,
            outlier_policy,
            correlation_policy,
            low_rank_policy,
            serialization_policy,
            tensor_rules,
        }
    }

    /// Crée le profil par défaut intégré dans la crate.
    pub fn default_profile() -> Self {
        let data = ProfileData {
            model_type: "glm_moe_dsa".to_string(),
            architecture: "GlmMoeDsaForCausalLM".to_string(),
            hidden_size: 6144,
            num_attention_heads: 64,
            num_hidden_layers: 78,
            vocab_size: 154880,
            max_position_embeddings: 1048576,
            n_routed_experts: Some(256),
            n_shared_experts: Some(1),
            expert_capacity: Some(8), // top-8
            tensor_patterns: vec![
                "model.embed_tokens.weight".to_string(),
                "model.layers.{layer}.self_attn.q_proj.weight".to_string(),
                "model.layers.{layer}.self_attn.k_proj.weight".to_string(),
                "model.layers.{layer}.self_attn.v_proj.weight".to_string(),
                "model.layers.{layer}.self_attn.o_proj.weight".to_string(),
                "model.layers.{layer}.mlp.gate_proj.weight".to_string(),
                "model.layers.{layer}.mlp.up_proj.weight".to_string(),
                "model.layers.{layer}.mlp.down_proj.weight".to_string(),
                "model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight".to_string(),
                "model.layers.{layer}.mlp.experts.{expert}.up_proj.weight".to_string(),
                "model.layers.{layer}.mlp.experts.{expert}.down_proj.weight".to_string(),
                "model.norm.weight".to_string(),
                "lm_head.weight".to_string(),
            ],
            head_dim: None,     // GLM-5.2 n'utilise pas MLA
            kv_lora_rank: None, // GLM-5.2 n'utilise pas MLA
        };
        Self::from_data(data)
    }
}

impl ModelProfile for Glm52Profile {
    fn model_family(&self) -> &str {
        "GLM"
    }

    fn architecture(&self) -> &str {
        &self.data.architecture
    }

    fn num_layers(&self) -> u32 {
        self.data.num_hidden_layers
    }

    fn hidden_size(&self) -> u32 {
        self.data.hidden_size
    }

    fn num_attention_heads(&self) -> u32 {
        self.data.num_attention_heads
    }

    fn num_experts(&self) -> Option<u32> {
        self.data.n_routed_experts
    }

    fn expert_capacity(&self) -> Option<u32> {
        self.data.expert_capacity
    }

    fn head_dim(&self) -> Option<u32> {
        self.data.head_dim
    }

    fn kv_lora_rank(&self) -> Option<u32> {
        self.data.kv_lora_rank
    }

    fn vocab_size(&self) -> u32 {
        self.data.vocab_size
    }

    fn max_position_embeddings(&self) -> u32 {
        self.data.max_position_embeddings
    }

    fn tensor_names(&self) -> Vec<String> {
        self.data.tensor_patterns.clone()
    }

    fn metadata_source(&self) -> MetadataSource {
        MetadataSource::Exact
    }

    fn generation_policy(&self) -> &GenerationPolicy {
        &self.generation_policy
    }

    fn dtype_policy(&self) -> &DtypePolicy {
        &self.dtype_policy
    }

    fn layer_policy(&self) -> &LayerPolicyGlm {
        &self.layer_policy
    }

    fn outlier_policy(&self) -> &OutlierPolicy {
        &self.outlier_policy
    }

    fn correlation_policy(&self) -> &CorrelationPolicy {
        &self.correlation_policy
    }

    fn low_rank_policy(&self) -> &LowRankPolicy {
        &self.low_rank_policy
    }

    fn serialization_policy(&self) -> &SerializationPolicy {
        &self.serialization_policy
    }

    fn tensor_rules(&self) -> &[TensorRule] {
        &self.tensor_rules
    }
}

impl Glm52Profile {
    /// Valide la cohérence et les valeurs du profil GLM-5.2.
    ///
    /// # Erreurs
    ///
    /// Retourne une erreur si le profil contient des valeurs incohérentes,
    /// des champs manquants ou des valeurs hors plage.
    pub fn validate(&self) -> crate::error::Result<()> {
        // Validation spécifique au modèle GLM-5.2
        // Vérification de l'architecture
        if self.data.architecture != "GlmMoeDsaForCausalLM" {
            return Err(crate::error::ModelProfileError::InconsistentArchitecture {
                message: format!(
                    "architecture attendue 'GlmMoeDsaForCausalLM', trouvée '{}'",
                    self.data.architecture
                ),
            });
        }
        // Vérification du type de modèle
        if self.data.model_type != "glm_moe_dsa" {
            return Err(crate::error::ModelProfileError::InvalidValue {
                field: "model_type".to_string(),
                message: format!(
                    "type de modèle attendu 'glm_moe_dsa', trouvé '{}'",
                    self.data.model_type
                ),
            });
        }
        // Appel de la validation par défaut
        crate::profile::validate_profile(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glm52_profile_default() {
        let profile = Glm52Profile::default_profile();
        assert_eq!(profile.model_family(), "GLM");
        assert_eq!(profile.architecture(), "GlmMoeDsaForCausalLM");
        assert_eq!(profile.num_layers(), 78);
        assert_eq!(profile.hidden_size(), 6144);
        assert_eq!(profile.num_attention_heads(), 64);
        assert_eq!(profile.num_experts(), Some(256));
        assert_eq!(profile.expert_capacity(), Some(8));
        assert_eq!(profile.vocab_size(), 154880);
        assert_eq!(profile.max_position_embeddings(), 1048576);
        assert!(!profile.tensor_names().is_empty());
        assert_eq!(profile.metadata_source(), MetadataSource::Exact);
    }

    #[test]
    fn test_glm52_profile_validation_success() {
        let profile = Glm52Profile::default_profile();
        assert!(profile.validate().is_ok());
    }

    #[test]
    fn test_glm52_profile_validation_wrong_architecture() {
        let mut data = Glm52Profile::default_profile().data;
        data.architecture = "WrongArchitecture".to_string();
        let profile = Glm52Profile::from_data(data);
        let result = profile.validate();
        assert!(result.is_err());
        match result {
            Err(crate::error::ModelProfileError::InconsistentArchitecture { message }) => {
                assert!(message.contains("architecture attendue 'GlmMoeDsaForCausalLM'"));
            },
            _ => panic!("Erreur attendue: InconsistentArchitecture"),
        }
    }

    #[test]
    fn test_glm52_profile_validation_wrong_model_type() {
        let mut data = Glm52Profile::default_profile().data;
        data.model_type = "wrong_type".to_string();
        let profile = Glm52Profile::from_data(data);
        let result = profile.validate();
        assert!(result.is_err());
        match result {
            Err(crate::error::ModelProfileError::InvalidValue { field, message }) => {
                assert_eq!(field, "model_type");
                assert!(message.contains("type de modèle attendu 'glm_moe_dsa'"));
            },
            _ => panic!("Erreur attendue: InvalidValue"),
        }
    }

    #[test]
    fn test_glm52_profile_validation_invalid_hidden_size() {
        let mut data = Glm52Profile::default_profile().data;
        data.hidden_size = 0;
        let profile = Glm52Profile::from_data(data);
        let result = profile.validate();
        assert!(result.is_err());
        match result {
            Err(crate::error::ModelProfileError::InvalidValue { field, message }) => {
                assert_eq!(field, "hidden_size");
                assert!(message.contains("la dimension cachée doit être supérieure à 0"));
            },
            _ => panic!("Erreur attendue: InvalidValue"),
        }
    }

    #[test]
    fn test_glm52_profile_validation_inconsistent_hidden_heads() {
        let mut data = Glm52Profile::default_profile().data;
        data.hidden_size = 6145; // Non divisible par 64
        let profile = Glm52Profile::from_data(data);
        let result = profile.validate();
        assert!(result.is_err());
        match result {
            Err(crate::error::ModelProfileError::InconsistentArchitecture { message }) => {
                assert!(
                    message.contains("hidden_size")
                        && message.contains("divisible par num_attention_heads")
                );
            },
            _ => panic!("Erreur attendue: InconsistentArchitecture"),
        }
    }
}
