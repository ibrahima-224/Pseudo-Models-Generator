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

//! Configuration de modèle normalisée (`ModelConfig`).
//!
//! `ModelConfig` est le format **normalisé** interne de PMG, indépendant des
//! noms de champs des `config.json` sources (GLM, DeepSeek, autres). Il est
//! construit par le parser de `pmg-io` (Sprint 10) ; ce module fournit la
//! structure, la validation des invariants et des constructeurs de test
//! alignés sur les valeurs observées des deux modèles cibles.
//!
//! Référence : `docs/architecture/03-modeles-de-donnees.md` §2.5.
//!
//! # Exemple
//!
//! ```
//! use pmg_core::ModelConfig;
//! use std::collections::BTreeMap;
//!
//! let config = ModelConfig {
//!     model_type: "glm_moe_dsa".to_string(),
//!     architectures: vec!["GlmMoeDsaForCausalLM".to_string()],
//!     hidden_size: 6144,
//!     num_layers: 78,
//!     num_attention_heads: 64,
//!     num_key_value_heads: 64,
//!     head_dim: Some(192),
//!     qk_head_dim: Some(256),
//!     v_head_dim: Some(256),
//!     intermediate_size: Some(12288),
//!     moe_intermediate_size: Some(2048),
//!     vocab_size: 65536,
//!     max_position_embeddings: 16384,
//!     rms_norm_eps: 1e-6,
//!     rope_theta: 10000.0,
//!     tie_word_embeddings: false,
//!     moe: None,
//!     attention_type: pmg_core::model_config::AttentionKind::Dsa,
//!     hyper_connections: true,
//!     dtype_declared: pmg_core::DType::Bf16,
//!     extras: BTreeMap::new(),
//!     provenance: BTreeMap::new(),
//! };
//!
//! assert!(config.validate().is_ok());
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::dtype::DType;
use crate::error::{CoreError, CoreResult};
use crate::moe::MoeConfig;
use crate::origin::Origin;

/// Type d'attention d'un modèle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttentionKind {
    /// Attention dense classique (Q/K/V/O par tête).
    Dense,
    /// Grouped-Query Attention.
    Gqa,
    /// Multi-head Latent Attention (DeepSeek MLA).
    Mla,
    /// DeepSeek Sparse Attention (indexeur).
    Dsa,
    /// Attention hybride (dense + sparse selon les couches).
    Hybrid,
    /// Indexeur sparse seul.
    SparseIndexer,
}

/// # Exemple
///
/// ```
/// use pmg_core::model_config::AttentionKind;
///
/// let kind = AttentionKind::Dsa;
/// assert_eq!(format!("{:?}", kind), "Dsa");
/// ```
/// Configuration normalisée d'un modèle de transformeur.
///
/// Les champs optionnels valent `None` lorsque la valeur est absente de la
/// source — **jamais une valeur inventée** (champ absent → `None` +
/// provenance `UNKNOWN`).
///
/// # Exemple
///
/// ```
/// use pmg_core::ModelConfig;
///
/// let config = pmg_core::model_config::glm52_test_config();
/// assert_eq!(config.hidden_size, 6144);
/// assert_eq!(config.num_layers, 78);
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Type de modèle source (`glm_moe_dsa`, `deepseek_v4`…).
    pub model_type: String,
    /// Architectures déclarées (ex. `["GlmMoeDsaForCausalLM"]`).
    pub architectures: Vec<String>,
    /// Taille cachée (`hidden_size`).
    pub hidden_size: u64,
    /// Nombre de couches (`num_hidden_layers`).
    pub num_layers: u64,
    /// Nombre de têtes d'attention (`num_attention_heads`).
    pub num_attention_heads: u64,
    /// Nombre de têtes K/V (`num_key_value_heads`).
    pub num_key_value_heads: u64,
    /// Dimension par tête, `None` si `d_head = hidden_size / heads`.
    pub head_dim: Option<u64>,
    /// Dimension Q/K (DSA/hybride GLM/DeepSeek), optionnel.
    pub qk_head_dim: Option<u64>,
    /// Dimension V (optionnel).
    pub v_head_dim: Option<u64>,
    /// Taille intermédiaire du MLP dense (`intermediate_size`), optionnel.
    pub intermediate_size: Option<u64>,
    /// Taille intermédiaire des experts MoE (`moe_intermediate_size`).
    pub moe_intermediate_size: Option<u64>,
    /// Taille du vocabulaire.
    pub vocab_size: u64,
    /// Longueur maximale de contexte (`max_position_embeddings`).
    pub max_position_embeddings: u64,
    /// Epsilon de RMSNorm.
    pub rms_norm_eps: f64,
    /// Base thêta de RoPE.
    pub rope_theta: f64,
    /// Partage des embeddings avec la tête de langage.
    pub tie_word_embeddings: bool,
    /// Configuration MoE, `None` pour un modèle dense.
    pub moe: Option<MoeConfig>,
    /// Type d'attention.
    pub attention_type: AttentionKind,
    /// Hyper-connections (DeepSeek hc_*).
    pub hyper_connections: bool,
    /// Dtype déclaré dans la config (`bfloat16` → `Bf16`).
    pub dtype_declared: DType,
    /// Champs non normalisés conservés pour réémission fidèle.
    #[serde(default)]
    pub extras: BTreeMap<String, Value>,
    /// Provenance par champ normalisé.
    #[serde(default)]
    pub provenance: BTreeMap<String, Origin>,
}

impl ModelConfig {
    /// Valide les invariants de cohérence (§2.5) :
    /// - `num_layers >= 1`, `vocab_size >= 1`, `hidden_size >= 1` ;
    /// - divisibilité des têtes (sauf cas DSA/MLA à dimensions propres) ;
    /// - `rope_theta > 0`, `rms_norm_eps > 0` ;
    /// - invariants MoE délégués à [`MoeConfig::validate`].
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_core::ModelConfig;
    ///
    /// let mut config = pmg_core::model_config::glm52_test_config();
    /// assert!(config.validate().is_ok());
    ///
    /// config.num_layers = 0;
    /// assert!(config.validate().is_err());
    /// ```
    pub fn validate(&self) -> CoreResult<()> {
        if self.num_layers == 0 {
            return Err(CoreError::InvalidModelConfig(
                "num_layers doit être ≥ 1".into(),
            ));
        }
        if self.vocab_size == 0 {
            return Err(CoreError::InvalidModelConfig(
                "vocab_size doit être ≥ 1".into(),
            ));
        }
        if self.hidden_size == 0 {
            return Err(CoreError::InvalidModelConfig(
                "hidden_size doit être ≥ 1".into(),
            ));
        }
        if self.num_attention_heads == 0 {
            return Err(CoreError::InvalidModelConfig(
                "num_attention_heads doit être ≥ 1".into(),
            ));
        }
        if self.num_key_value_heads == 0 {
            return Err(CoreError::InvalidModelConfig(
                "num_key_value_heads doit être ≥ 1".into(),
            ));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(CoreError::InvalidModelConfig(format!(
                "rope_theta doit être fini et > 0 (obtenu {})",
                self.rope_theta
            )));
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(CoreError::InvalidModelConfig(format!(
                "rms_norm_eps doit être fini et > 0 (obtenu {})",
                self.rms_norm_eps
            )));
        }
        // Règle de divisibilité des têtes, réservée aux architectures à têtes
        // égales (Dense/GQA) où hidden_size = heads × head_dim.
        // MLA/DSA/hybrides ont des dimensions de projection propres
        // (qk_head_dim, v_head_dim, ou dimension latente) : la règle ne
        // s'applique pas — GLM : 64 × 192 = 12288 ≠ 6144 (DSA) ;
        // DeepSeek : 64 × 512 = 32768 ≠ 4096 (MLA).
        match self.attention_type {
            AttentionKind::Dense | AttentionKind::Gqa => {
                if let Some(head_dim) = self.head_dim {
                    let total =
                        self.num_attention_heads
                            .checked_mul(head_dim)
                            .ok_or_else(|| {
                                CoreError::Overflow("num_attention_heads × head_dim".into())
                            })?;
                    if total != self.hidden_size {
                        return Err(CoreError::IncompatibleHeads(format!(
                            "num_attention_heads ({}) × head_dim ({}) = {total} ≠ hidden_size ({})",
                            self.num_attention_heads, head_dim, self.hidden_size
                        )));
                    }
                } else if self.hidden_size % self.num_attention_heads != 0 {
                    return Err(CoreError::IncompatibleHeads(format!(
                        "hidden_size ({}) non divisible par num_attention_heads ({})",
                        self.hidden_size, self.num_attention_heads
                    )));
                }
            },
            _ => {},
        }
        if let Some(moe) = &self.moe {
            moe.validate()?;
        }
        Ok(())
    }

    /// Marque la provenance d'un champ normalisé (`OBSERVÉ`).
    pub fn mark_observed(&mut self, field: &str) {
        self.provenance.insert(field.to_string(), Origin::Observed);
    }

    /// Retourne la provenance d'un champ, `UNKNOWN` si absente.
    pub fn provenance_of(&self, field: &str) -> Origin {
        self.provenance
            .get(field)
            .copied()
            .unwrap_or(Origin::Unknown)
    }
}

/// Constructeur de test GLM-5.2 (valeurs observées dans `config.json`).
///
/// # Exemple
///
/// ```
/// use pmg_core::model_config::glm52_test_config;
///
/// let config = glm52_test_config();
/// assert_eq!(config.model_type, "glm_moe_dsa");
/// assert!(config.validate().is_ok());
/// ```
pub fn glm52_test_config() -> ModelConfig {
    let mut cfg = ModelConfig {
        model_type: "glm_moe_dsa".to_string(),
        architectures: vec!["GlmMoeDsaForCausalLM".to_string()],
        hidden_size: 6144,
        num_layers: 78,
        num_attention_heads: 64,
        num_key_value_heads: 64,
        head_dim: Some(192),
        qk_head_dim: Some(256),
        v_head_dim: Some(256),
        intermediate_size: Some(12288),
        moe_intermediate_size: Some(2048),
        vocab_size: 154880,
        max_position_embeddings: 1048576,
        rms_norm_eps: 1e-5,
        rope_theta: 8_000_000.0,
        tie_word_embeddings: false,
        moe: Some(crate::moe::glm52_moe_config()),
        attention_type: AttentionKind::Dsa,
        hyper_connections: false,
        dtype_declared: DType::Bf16,
        extras: BTreeMap::new(),
        provenance: BTreeMap::new(),
    };
    cfg.mark_observed("hidden_size");
    cfg.mark_observed("num_layers");
    cfg
}

/// Constructeur de test DeepSeek-V4-Flash (valeurs observées dans `config.json`).
pub fn deepseek_v4_flash_test_config() -> ModelConfig {
    let moe = MoeConfig {
        n_routed_experts: 256,
        n_shared_experts: 1,
        experts_per_tok: 6,
        router_dtype: DType::F32,
        routed_scaling_factor: 1.5,
        norm_topk_prob: true,
        topk_method: "noaux_tc".to_string(),
        first_k_dense_replace: None,
        layer_types: Vec::new(),
        expert_dtype: Some(DType::F4),
    };
    let mut cfg = ModelConfig {
        model_type: "deepseek_v4".to_string(),
        architectures: vec!["DeepseekV4ForCausalLM".to_string()],
        hidden_size: 4096,
        num_layers: 43,
        num_attention_heads: 64,
        num_key_value_heads: 1,
        head_dim: Some(512),
        qk_head_dim: None,
        v_head_dim: None,
        intermediate_size: None,
        moe_intermediate_size: Some(2048),
        vocab_size: 129280,
        max_position_embeddings: 1048576,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000.0,
        tie_word_embeddings: false,
        moe: Some(moe),
        attention_type: AttentionKind::Mla,
        hyper_connections: true,
        dtype_declared: DType::Bf16,
        extras: BTreeMap::new(),
        provenance: BTreeMap::new(),
    };
    cfg.mark_observed("vocab_size");
    cfg
}

#[cfg(test)]
mod tests {
    use super::{deepseek_v4_flash_test_config, glm52_test_config, ModelConfig};
    use crate::error::CoreError;

    #[test]
    fn glm52_config_validates() {
        // GLM : DSA → la règle heads×head_dim = hidden ne s'applique pas.
        let cfg = glm52_test_config();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn deepseek_config_validates() {
        // DeepSeek : MLA (attention_type = Mla) — la règle de divisibilité
        // des têtes ne s'applique pas : head_dim 512 est une dimension
        // latente compressée, pas d_head = hidden/heads.
        let cfg = deepseek_v4_flash_test_config();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn zero_layers_is_rejected() {
        let mut cfg = glm52_test_config();
        cfg.num_layers = 0;
        assert!(matches!(
            cfg.validate(),
            Err(CoreError::InvalidModelConfig(_))
        ));
    }

    #[test]
    fn zero_vocab_is_rejected() {
        let mut cfg = glm52_test_config();
        cfg.vocab_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn heads_divisibility_rule() {
        // Modèle dense classique : hidden 6144, 64 têtes, pas de head_dim
        // explicite → 6144 % 64 == 0 → valide.
        let mut cfg = glm52_test_config();
        cfg.attention_type = crate::model_config::AttentionKind::Dense;
        cfg.qk_head_dim = None;
        cfg.v_head_dim = None;
        cfg.head_dim = None;
        assert!(cfg.validate().is_ok());

        // 61 têtes → indivisible → IncompatibleHeads.
        cfg.num_attention_heads = 61;
        assert!(matches!(
            cfg.validate(),
            Err(CoreError::IncompatibleHeads(_))
        ));
    }

    #[test]
    fn head_dim_mismatch_is_rejected() {
        // En Dense, heads × head_dim = hidden est exigé : 64 × 100 ≠ 6144.
        let mut cfg = glm52_test_config();
        cfg.attention_type = crate::model_config::AttentionKind::Dense;
        cfg.qk_head_dim = None;
        cfg.v_head_dim = None;
        cfg.head_dim = Some(100);
        assert!(matches!(
            cfg.validate(),
            Err(CoreError::IncompatibleHeads(_))
        ));
    }

    #[test]
    fn invalid_rope_theta_is_rejected() {
        let mut cfg = glm52_test_config();
        cfg.rope_theta = 0.0;
        assert!(cfg.validate().is_err());
        cfg.rope_theta = f64::NAN;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn provenance_defaults_to_unknown() {
        let cfg = glm52_test_config();
        assert_eq!(
            cfg.provenance_of("hidden_size"),
            crate::origin::Origin::Observed
        );
        assert_eq!(
            cfg.provenance_of("num_key_value_heads"),
            crate::origin::Origin::Unknown
        );
    }

    #[test]
    fn serde_roundtrip() {
        let cfg = glm52_test_config();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: ModelConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back, cfg);
        // La provenance est conservée dans le JSON.
        assert!(json.contains("\"provenance\""));
    }
}
