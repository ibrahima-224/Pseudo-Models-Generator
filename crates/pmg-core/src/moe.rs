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

//! Configuration MoE (`MoeConfig`) — spécification `docs/architecture/03-modeles-de-donnees.md` §2.6.
//!
//! # Exemple
//!
//! ```
//! use pmg_core::MoeConfig;
//! use pmg_core::dtype::DType;
//!
//! let moe = MoeConfig {
//!     n_routed_experts: 256,
//!     n_shared_experts: 1,
//!     experts_per_tok: 8,
//!     router_dtype: DType::F32,
//!     routed_scaling_factor: 2.5,
//!     norm_topk_prob: true,
//!     topk_method: "noaux_tc".to_string(),
//!     first_k_dense_replace: Some(3),
//!     layer_types: vec!["dense".to_string(); 3].into_iter().chain(vec!["sparse".to_string(); 75]).collect(),
//!     expert_dtype: None,
//! };
//!
//! assert!(moe.validate().is_ok());
//! // SAFETY: total_experts() ne peut échouer que par débordement u64,
//! // ce qui est impossible avec les valeurs connues (256+1=257).
//! assert_eq!(moe.total_experts().unwrap(), 257);
//! ```

use serde::{Deserialize, Serialize};

use crate::dtype::DType;
use crate::error::{CoreError, CoreResult};

/// Configuration des experts d'un modèle Mixture-of-Experts.
///
/// Les champs suivent la nomenclature normalisée de la spécification
/// (indépendante des noms de `config.json` GLM/DeepSeek).
///
/// # Exemple
///
/// ```
/// use pmg_core::MoeConfig;
/// use pmg_core::dtype::DType;
///
/// let moe = MoeConfig {
///     n_routed_experts: 256,
///     n_shared_experts: 1,
///     experts_per_tok: 8,
///     router_dtype: DType::F32,
///     routed_scaling_factor: 2.5,
///     norm_topk_prob: true,
///     topk_method: "noaux_tc".to_string(),
///     first_k_dense_replace: Some(3),
///     layer_types: vec!["dense".to_string(); 3].into_iter().chain(vec!["sparse".to_string(); 75]).collect(),
///     expert_dtype: None,
/// };
///
/// // SAFETY: total_experts() ne peut échouer que par débordement u64,
/// // ce qui est impossible avec les valeurs connues (256+1=257).
/// assert_eq!(moe.total_experts().unwrap(), 257);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoeConfig {
    /// Nombre d'experts routés (`n_routed_experts`).
    pub n_routed_experts: u64,
    /// Nombre d'experts partagés (`n_shared_experts`).
    pub n_shared_experts: u64,
    /// Nombre d'experts sélectionnés par token (top-k, `num_experts_per_tok`).
    pub experts_per_tok: u64,
    /// Dtype du routeur (GLM : float32).
    pub router_dtype: DType,
    /// Facteur d'échelle du routage (GLM 2.5, DeepSeek 1.5).
    pub routed_scaling_factor: f64,
    /// Normalise les probabilités top-k (norm_topk_prob).
    pub norm_topk_prob: bool,
    /// Méthode de sélection top-k (`noaux_tc` pour GLM/DeepSeek).
    pub topk_method: String,
    /// Nombre de premières couches **denses** (GLM : 3) — `None` si absent.
    pub first_k_dense_replace: Option<u64>,
    /// Type de couche par couche (`dense`/`sparse`), longueur = `num_layers`.
    pub layer_types: Vec<String>,
    /// Dtype des experts (DeepSeek : fp4 quantifié, non émissible) — optionnel.
    pub expert_dtype: Option<DType>,
}

impl MoeConfig {
    /// Nombre total d'experts (routés + partagés).
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_core::moe::glm52_moe_config;
    ///
    /// let moe = glm52_moe_config();
    /// // SAFETY: total_experts() ne peut échouer que par débordement u64,
    /// // ce qui est impossible avec les valeurs connues (256+1=257).
    /// assert_eq!(moe.total_experts().unwrap(), 257);
    /// ```
    pub fn total_experts(&self) -> CoreResult<u64> {
        self.n_routed_experts
            .checked_add(self.n_shared_experts)
            .ok_or_else(|| {
                CoreError::Overflow(format!(
                    "débordement lors du calcul du total d'experts : {} + {}",
                    self.n_routed_experts, self.n_shared_experts
                ))
            })
    }

    /// Valide les invariants MoE :
    /// - `n_routed_experts >= 1` ;
    /// - `experts_per_tok <= n_routed_experts + n_shared_experts` ;
    /// - `routed_scaling_factor > 0` ;
    /// - si `layer_types` est non vide, sa longueur est cohérente avec
    ///   `first_k_dense_replace` (GLM : exactement k couches `dense`).
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_core::moe::glm52_moe_config;
    ///
    /// let moe = glm52_moe_config();
    /// assert!(moe.validate().is_ok());
    /// ```
    pub fn validate(&self) -> CoreResult<()> {
        if self.n_routed_experts == 0 {
            return Err(CoreError::InvalidMoeConfig(
                "n_routed_experts doit être ≥ 1".into(),
            ));
        }
        if self.experts_per_tok == 0 {
            return Err(CoreError::InvalidMoeConfig(
                "experts_per_tok (top-k) doit être ≥ 1".into(),
            ));
        }
        let total = self.total_experts()?;
        if self.experts_per_tok > total {
            return Err(CoreError::InvalidMoeConfig(format!(
                "experts_per_tok (top-k) = {} dépasse le total d'experts ({})",
                self.experts_per_tok, total
            )));
        }
        if !self.routed_scaling_factor.is_finite() || self.routed_scaling_factor <= 0.0 {
            return Err(CoreError::InvalidMoeConfig(format!(
                "routed_scaling_factor doit être fini et > 0 (obtenu {})",
                self.routed_scaling_factor
            )));
        }
        if let Some(k) = self.first_k_dense_replace {
            if k > self.layer_types.len() as u64 {
                return Err(CoreError::InvalidMoeConfig(format!(
                    "first_k_dense_replace = {k} mais seulement {} types de couches fournis",
                    self.layer_types.len()
                )));
            }
            let dense_count = self.layer_types.iter().filter(|t| *t == "dense").count() as u64;
            if dense_count != k {
                return Err(CoreError::InvalidMoeConfig(format!(
                    "first_k_dense_replace = {k} mais {dense_count} couches marquées 'dense'"
                )));
            }
        }
        Ok(())
    }
}

/// Configuration MoE de référence pour un modèle GLM-5.2 (valeurs observées
/// dans `Models/GLM-5.2/config.json`) — utile pour les tests et les profils.
///
/// # Exemple
///
/// ```
/// use pmg_core::moe::glm52_moe_config;
///
/// let moe = glm52_moe_config();
/// assert_eq!(moe.n_routed_experts, 256);
/// ```
pub fn glm52_moe_config() -> MoeConfig {
    MoeConfig {
        n_routed_experts: 256,
        n_shared_experts: 1,
        experts_per_tok: 8,
        router_dtype: DType::F32,
        routed_scaling_factor: 2.5,
        norm_topk_prob: true,
        topk_method: "noaux_tc".to_string(),
        first_k_dense_replace: Some(3),
        layer_types: glm52_layer_types(),
        expert_dtype: None,
    }
}

/// Types de couches GLM-5.2 (3 `dense` + 75 `sparse`), observés dans le config.
pub fn glm52_layer_types() -> Vec<String> {
    let mut v = Vec::with_capacity(78);
    v.extend(std::iter::repeat("dense".to_string()).take(3));
    v.extend(std::iter::repeat("sparse".to_string()).take(75));
    v
}

#[cfg(test)]
mod tests {
    use super::{glm52_layer_types, glm52_moe_config, MoeConfig};
    use crate::dtype::DType;
    use crate::error::CoreError;

    #[test]
    fn moe_config_examples_in_doc() {
        // Vérifie les exemples de la doc.
        let moe = glm52_moe_config();
        // SAFETY: total_experts() ne peut échouer que par débordement u64,
        // ce qui est impossible avec les valeurs connues (256+1=257).
        assert_eq!(moe.total_experts().unwrap(), 257);
        assert!(moe.validate().is_ok());
    }

    #[test]
    fn glm52_config_is_valid() {
        let moe = glm52_moe_config();
        assert!(moe.validate().is_ok());
        // SAFETY: total_experts() ne peut échouer que par débordement u64,
        // ce qui est impossible avec les valeurs connues (256+1=257).
        assert_eq!(moe.total_experts().unwrap(), 257);
        assert_eq!(moe.layer_types.len(), 78);
        assert_eq!(moe.first_k_dense_replace, Some(3));
    }

    #[test]
    fn topk_exceeds_total_experts_is_rejected() {
        let mut moe = glm52_moe_config();
        moe.experts_per_tok = 1000;
        assert!(matches!(
            moe.validate(),
            Err(CoreError::InvalidMoeConfig(_))
        ));
    }

    #[test]
    fn zero_routed_experts_is_rejected() {
        let mut moe = glm52_moe_config();
        moe.n_routed_experts = 0;
        assert!(moe.validate().is_err());
    }

    #[test]
    fn zero_topk_is_rejected() {
        let mut moe = glm52_moe_config();
        moe.experts_per_tok = 0;
        assert!(moe.validate().is_err());
    }

    #[test]
    fn invalid_scaling_factor_is_rejected() {
        let mut moe = glm52_moe_config();
        moe.routed_scaling_factor = -1.0;
        assert!(moe.validate().is_err());
    }

    #[test]
    fn first_k_dense_mismatch_is_rejected() {
        // 4 couches marquées 'dense' mais first_k = 3 → erreur.
        let mut moe = glm52_moe_config();
        moe.layer_types[0] = "sparse".to_string();
        moe.layer_types[1] = "dense".to_string();
        assert!(moe.validate().is_err());
    }

    #[test]
    fn layer_types_length_matches_num_layers() {
        // Contrat du §2.6 : layer_types longueur = num_layers (78 pour GLM).
        assert_eq!(glm52_layer_types().len(), 78);
        assert_eq!(
            glm52_layer_types().iter().filter(|t| *t == "dense").count(),
            3
        );
        assert_eq!(
            glm52_layer_types()
                .iter()
                .filter(|t| *t == "sparse")
                .count(),
            75
        );
    }

    #[test]
    fn serde_roundtrip() {
        let moe = glm52_moe_config();
        let json = serde_json::to_string(&moe).unwrap();
        assert_eq!(serde_json::from_str::<MoeConfig>(&json).unwrap(), moe);
    }

    #[test]
    fn expert_dtype_declared_but_not_emittable() {
        // DeepSeek : expert_dtype = F4 (quantifié, non émissible en v1.0).
        let mut moe = glm52_moe_config();
        moe.expert_dtype = Some(DType::F4);
        assert!(!DType::F4.is_supported_for_write());
        assert!(moe.validate().is_ok());
    }
}
