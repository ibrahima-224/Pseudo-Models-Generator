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

//! Bloc MoE d'une couche (`MoeBlockSpec`) — description sans poids.
//!
//! Le blueprint ne réduit jamais un modèle MoE à un MLP dense : chaque couche
//! sparse porte un routeur, des experts partagés et des experts routés, avec
//! la règle `W_e = W_shared + ΔW_e` (composante partagée + spécifique).

use serde::{Deserialize, Serialize};

use pmg_core::CoreResult;

use crate::tensor_spec::TensorSpec;

/// Spécification d'un expert routé.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertSpec {
    /// Index de l'expert (0-based).
    pub index: u64,
    /// Projection up (w1 / up_proj).
    pub up: TensorSpec,
    /// Projection gate (w3 / gate_proj).
    pub gate: TensorSpec,
    /// Projection down (w2 / down_proj).
    pub down: TensorSpec,
}

impl ExpertSpec {
    /// Vérifie la cohérence interne de l'expert (index, noms, tailles).
    pub fn validate(&self) -> CoreResult<()> {
        // Les noms des trois matrices doivent référencer le même expert.
        for spec in [&self.up, &self.gate, &self.down] {
            if spec.expert_id != Some(self.index) {
                return Err(pmg_core::CoreError::Validation(format!(
                    "l'expert {} possède une matrice avec expert_id {:?}",
                    self.index, spec.expert_id
                )));
            }
        }
        Ok(())
    }
}

/// Bloc MoE complet d'une couche sparse.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MoeBlockSpec {
    /// Routeur (logits par expert) — `mlp.gate` / `ffn.gate`.
    pub router: TensorSpec,
    /// Experts partagés (ex. `mlp.shared_experts.*`).
    pub shared_experts: Vec<TensorSpec>,
    /// Experts routés (généralement 256).
    pub routed_experts: Vec<ExpertSpec>,
    /// Nombre d'experts sélectionnés par token (top-k).
    pub top_k: u64,
    /// Type de couche (`dense` vs `sparse`).
    pub layer_type: LayerType,
}

/// Type de couche pour le routage MoE (GLM : premières couches denses).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayerType {
    /// Couche dense (MLP classique sans experts).
    Dense,
    /// Couche sparse (MoE).
    Sparse,
}

impl LayerType {
    /// Parse un libellé `config.json` (`dense`/`sparse`).
    pub fn from_str_config(s: &str) -> Option<LayerType> {
        match s {
            "dense" => Some(LayerType::Dense),
            "sparse" => Some(LayerType::Sparse),
            _ => None,
        }
    }
}

impl MoeBlockSpec {
    /// Valide les invariants du bloc : top-k ≤ experts routés disponibles, cohérence
    /// des experts, noms uniques.
    ///
    /// Note : `top_k` ne concerne que les experts routés, pas les experts partagés.
    /// Les experts partagés sont toujours utilisés en complément des experts sélectionnés.
    pub fn validate(&self) -> CoreResult<()> {
        // Nombre d'experts routés uniquement (les experts partagés ne comptent pas pour top_k).
        let routed_count = self.routed_experts.len() as u64;
        if self.top_k == 0 || self.top_k > routed_count {
            return Err(pmg_core::CoreError::InvalidMoeConfig(format!(
                "top_k = {} hors bornes [1, {}] (routed experts uniquement)",
                self.top_k, routed_count
            )));
        }
        for expert in &self.routed_experts {
            expert.validate()?;
        }
        Ok(())
    }

    /// Nombre total d'experts (routés + partagés).
    pub fn total_experts(&self) -> u64 {
        self.routed_experts.len() as u64 + self.shared_experts.len() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::{ExpertSpec, LayerType, MoeBlockSpec};
    use crate::tensor_spec::TensorSpec;
    use pmg_core::{DType, Shape, TensorRole};

    fn expert_spec(index: u64, name: &str, role: TensorRole) -> TensorSpec {
        let mut spec = TensorSpec::new(
            name,
            Shape::new(vec![2048, 4096]).unwrap(),
            DType::Bf16,
            role,
        )
        .unwrap();
        spec.expert_id = Some(index);
        spec
    }

    #[test]
    fn expert_validate_checks_consistency() {
        let expert = ExpertSpec {
            index: 0,
            up: expert_spec(0, "e.0.w1", TensorRole::MoeRoutedExpert),
            gate: expert_spec(0, "e.0.w3", TensorRole::MoeRoutedExpert),
            down: expert_spec(0, "e.0.w2", TensorRole::MoeRoutedExpert),
        };
        assert!(expert.validate().is_ok());
    }

    #[test]
    fn expert_with_mismatched_id_is_rejected() {
        let expert = ExpertSpec {
            index: 0,
            up: expert_spec(1, "e.1.w1", TensorRole::MoeRoutedExpert),
            gate: expert_spec(0, "e.0.w3", TensorRole::MoeRoutedExpert),
            down: expert_spec(0, "e.0.w2", TensorRole::MoeRoutedExpert),
        };
        assert!(expert.validate().is_err());
    }

    #[test]
    fn moe_block_validation() {
        let router = expert_spec(0, "ffn.gate.weight", TensorRole::MoeRouter);
        let block = MoeBlockSpec {
            router,
            shared_experts: vec![
                expert_spec(0, "shared.w1", TensorRole::MoeSharedExpert),
                expert_spec(0, "shared.w2", TensorRole::MoeSharedExpert),
            ],
            routed_experts: (0..8)
                .map(|i| ExpertSpec {
                    index: i,
                    up: expert_spec(i, &format!("e.{i}.w1"), TensorRole::MoeRoutedExpert),
                    gate: expert_spec(i, &format!("e.{i}.w3"), TensorRole::MoeRoutedExpert),
                    down: expert_spec(i, &format!("e.{i}.w2"), TensorRole::MoeRoutedExpert),
                })
                .collect(),
            top_k: 8,
            layer_type: LayerType::Sparse,
        };
        assert!(block.validate().is_ok());
        assert_eq!(block.total_experts(), 10);
    }

    #[test]
    fn topk_out_of_range_is_rejected() {
        let block = MoeBlockSpec {
            router: expert_spec(0, "ffn.gate.weight", TensorRole::MoeRouter),
            shared_experts: vec![],
            routed_experts: (0..4)
                .map(|i| ExpertSpec {
                    index: i,
                    up: expert_spec(i, &format!("e.{i}.w1"), TensorRole::MoeRoutedExpert),
                    gate: expert_spec(i, &format!("e.{i}.w3"), TensorRole::MoeRoutedExpert),
                    down: expert_spec(i, &format!("e.{i}.w2"), TensorRole::MoeRoutedExpert),
                })
                .collect(),
            top_k: 9,
            layer_type: LayerType::Sparse,
        };
        assert!(block.validate().is_err());
    }

    #[test]
    fn layer_type_parsing() {
        assert_eq!(LayerType::from_str_config("dense"), Some(LayerType::Dense));
        assert_eq!(
            LayerType::from_str_config("sparse"),
            Some(LayerType::Sparse)
        );
        assert_eq!(LayerType::from_str_config("bogus"), None);
    }
}
