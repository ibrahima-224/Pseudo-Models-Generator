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

//! Spécification d'une couche de transformeur (`LayerSpec`).
//!
//! Une couche regroupe l'attention, le MLP (dense ou MoE), les normalisations
//! et les éventuelles hyper-connections, avec une politique de génération
//! propre à la couche `θ_l = f(l)`.

use serde::{Deserialize, Serialize};

use pmg_core::{CoreError, CoreResult, TensorRole};

use crate::moe::MoeBlockSpec;
use crate::tensor_spec::TensorSpec;

/// Type de couche au niveau blueprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum LayerKind {
    /// Couche dense (attention + MLP dense).
    Dense,
    /// Couche à experts (attention + MoE).
    MoE,
    /// Couche hybride (attention hybride + MoE, hyper-connections).
    Hybrid,
}

/// Politique de génération d'une couche.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerPolicy {
    /// Force de structure de la couche (0..1).
    pub structure_strength: f64,
    /// Densité d'outliers de la couche.
    pub outlier_density: f64,
    /// Semence de dérivation propre à la couche (non nulle).
    pub layer_seed_shift: u64,
}

impl LayerPolicy {
    /// Politique par défaut (neutre, déterministe).
    pub fn default_for(index: u64) -> LayerPolicy {
        LayerPolicy {
            structure_strength: 0.1,
            outlier_density: 0.001,
            // Le décalage évite les seeds nulles et différencie les couches.
            layer_seed_shift: index.saturating_add(1),
        }
    }
}

/// Spécification complète d'une couche de transformeur.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerSpec {
    /// Index de la couche (0-based).
    pub index: u64,
    /// Type de couche (dense, MoE, hybride).
    pub kind: LayerKind,
    /// Tenseurs d'attention (q/k/v/o, ou q_a/q_b/kv_a/kv_b + indexeur).
    pub attention: Vec<TensorSpec>,
    /// Tenseurs du MLP (up/gate/down pour dense ; experts pour MoE).
    pub mlp: Vec<TensorSpec>,
    /// Bloc MoE si la couche est sparse (routeur + experts).
    pub moe_block: Option<MoeBlockSpec>,
    /// Normalisations (input_layernorm, post_attention_layernorm…).
    pub norms: Vec<TensorSpec>,
    /// Hyper-connections (DeepSeek hc_*).
    pub hyper_connections: Vec<TensorSpec>,
    /// Politique de génération de la couche.
    pub layer_policy: LayerPolicy,
}

impl LayerSpec {
    /// Construit une couche vide (index + politique), à remplir ensuite.
    pub fn new(index: u64, kind: LayerKind) -> LayerSpec {
        LayerSpec {
            index,
            kind,
            attention: Vec::new(),
            mlp: Vec::new(),
            moe_block: None,
            norms: Vec::new(),
            hyper_connections: Vec::new(),
            layer_policy: LayerPolicy::default_for(index),
        }
    }

    /// Tous les tenseurs de la couche (attention + mlp + norms + hc).
    ///
    /// L'ordre est stable : attention, MLP dense, normes, hyper-connections,
    /// puis le bloc MoE (routeur, shared, routés) pour les couches sparse.
    pub fn all_tensors(&self) -> Vec<&TensorSpec> {
        let mut out = Vec::new();
        out.extend(self.attention.iter());
        out.extend(self.mlp.iter());
        out.extend(self.norms.iter());
        out.extend(self.hyper_connections.iter());
        if let Some(moe) = &self.moe_block {
            out.push(&moe.router);
            out.extend(moe.shared_experts.iter());
            for expert in &moe.routed_experts {
                out.push(&expert.up);
                out.push(&expert.gate);
                out.push(&expert.down);
            }
        }
        out
    }

    /// Nombre total de tenseurs de la couche.
    pub fn tensor_count(&self) -> usize {
        self.all_tensors().len()
    }

    /// Valide la cohérence de la couche : index des tenseurs alignés sur
    /// `self.index`, invariants du bloc MoE.
    pub fn validate(&self) -> CoreResult<()> {
        for spec in self.all_tensors() {
            if spec.layer_id != Some(self.index) {
                return Err(CoreError::Validation(format!(
                    "le tenseur '{}' appartient à la couche {:?}, attendu couche {}",
                    spec.name, spec.layer_id, self.index
                )));
            }
        }
        if let Some(moe) = &self.moe_block {
            moe.validate()?;
        }
        Ok(())
    }
}

/// Construit les spécifications d'attention GLM-5.2 (DSA) pour une couche.
///
/// Motifs réels (index GLM) : `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`,
/// `kv_b_proj`, `o_proj` + indexeur.
pub fn glm_attention_specs(index: u64, hidden: u64, prefix: &str) -> Vec<TensorSpec> {
    let layer = format!("{prefix}model.layers.{index}.self_attn");
    let mut v = Vec::with_capacity(6);
    for (suffix, role) in [
        ("q_a_proj", TensorRole::AttentionQuery),
        ("q_b_proj", TensorRole::AttentionQuery),
        ("kv_a_proj_with_mqa", TensorRole::AttentionKey),
        ("kv_b_proj", TensorRole::AttentionKey),
        ("o_proj", TensorRole::AttentionOutput),
    ] {
        let mut spec = TensorSpec::new(
            format!("{layer}.{suffix}.weight"),
            pmg_core::Shape::new(vec![hidden, hidden]).unwrap(),
            pmg_core::DType::Bf16,
            role,
        )
        .expect("spec GLM valide");
        spec.layer_id = Some(index);
        v.push(spec);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::{glm_attention_specs, LayerKind, LayerPolicy, LayerSpec};
    use crate::tensor_spec::TensorSpec;
    use pmg_core::{DType, Shape, TensorRole};

    fn layer_tensor(index: u64, name: &str, role: TensorRole) -> TensorSpec {
        let mut spec =
            TensorSpec::new(name, Shape::new(vec![4, 4]).unwrap(), DType::Bf16, role).unwrap();
        spec.layer_id = Some(index);
        spec
    }

    #[test]
    fn empty_layer_validate_ok() {
        let layer = LayerSpec::new(0, LayerKind::Dense);
        assert!(layer.validate().is_ok());
        assert_eq!(layer.tensor_count(), 0);
    }

    #[test]
    fn layer_policy_is_deterministic_and_non_zero() {
        let p0 = LayerPolicy::default_for(0);
        let p0b = LayerPolicy::default_for(0);
        let p1 = LayerPolicy::default_for(1);
        assert_eq!(p0, p0b);
        assert_ne!(p0, p1);
        assert!(p0.layer_seed_shift >= 1);
    }

    #[test]
    fn layer_validate_detects_wrong_layer_id() {
        let mut layer = LayerSpec::new(3, LayerKind::Dense);
        layer
            .attention
            .push(layer_tensor(2, "x", TensorRole::Other));
        assert!(layer.validate().is_err());
    }

    #[test]
    fn glm_attention_specs_match_real_patterns() {
        let specs = glm_attention_specs(0, 6144, "");
        let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "model.layers.0.self_attn.q_a_proj.weight",
                "model.layers.0.self_attn.q_b_proj.weight",
                "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                "model.layers.0.self_attn.kv_b_proj.weight",
                "model.layers.0.self_attn.o_proj.weight",
            ]
        );
        for spec in &specs {
            assert_eq!(spec.layer_id, Some(0));
        }
    }

    #[test]
    fn serde_roundtrip() {
        let layer = LayerSpec::new(5, LayerKind::MoE);
        let json = serde_json::to_string(&layer).unwrap();
        assert_eq!(serde_json::from_str::<LayerSpec>(&json).unwrap(), layer);
    }
}
