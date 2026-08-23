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

//! Planification des tenseurs d'un modèle à partir du blueprint.
//!
//! Le planner construit la liste **déterministe et ordonnée** de toutes les
//! `TensorSpec` du modèle (couches et non-couches : embeddings, norme finale,
//! LM head, experts MoE, hyper-connections…) sans aucune donnée numérique.
//!
//! Invariants (§4.3) : ordre stable (indépendant des HashMap), chaque tenseur
//! appartient à exactement un groupe, les noms sont uniques.

use pmg_core::{DType, Shape, TensorRole};

use crate::blueprint::ModelBlueprint;
use crate::error::{BlueprintError, BlueprintResult};
use crate::layer::LayerKind;
use crate::moe::ExpertSpec;
use crate::naming::ExpertProj;
use crate::tensor_spec::TensorSpec;

/// Résultat de la planification : liste ordonnée + statistiques.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    /// Spécifications ordonnées (ordre d'émission déterministe).
    pub tensors: Vec<TensorSpec>,
    /// Nombre total d'éléments (Σ), vérifié.
    pub parameter_count: u64,
}

/// Planifie l'ensemble des tenseurs du blueprint.
///
/// Ordre stable : embeddings → couches (attention, MLP dense, normes, hc,
/// MoE) → norme finale → LM head → extras. Chaque `TensorSpec` reçoit son
/// rôle, son `layer_id`, son `expert_id` et sa politique de distribution.
pub fn plan_blueprint(blueprint: &ModelBlueprint) -> BlueprintResult<Plan> {
    let mut tensors: Vec<TensorSpec> = Vec::new();

    // 1. Embeddings.
    tensors.extend(blueprint.embeddings.iter().cloned());

    // 2. Couches (attention, MLP dense, normes, hyper-connections, MoE).
    for layer in &blueprint.layers {
        tensors.extend(layer.attention.iter().cloned());
        tensors.extend(layer.mlp.iter().cloned());
        tensors.extend(layer.norms.iter().cloned());
        tensors.extend(layer.hyper_connections.iter().cloned());
        if let Some(moe) = &layer.moe_block {
            tensors.push(moe.router.clone());
            tensors.extend(moe.shared_experts.iter().cloned());
            for expert in &moe.routed_experts {
                tensors.push(expert.up.clone());
                tensors.push(expert.gate.clone());
                tensors.push(expert.down.clone());
            }
        }
    }

    // 3. Norme finale, LM head, extras.
    tensors.extend(blueprint.final_norm.iter().cloned());
    tensors.extend(blueprint.lm_head.iter().cloned());
    tensors.extend(blueprint.extra_tensors.iter().cloned());

    // Invariants : noms uniques.
    let mut seen = std::collections::BTreeSet::new();
    for spec in &tensors {
        if !seen.insert(spec.name.as_str()) {
            return Err(BlueprintError::PlanError(format!(
                "nom de tenseur dupliqué dans le plan : '{}'",
                spec.name
            )));
        }
    }

    // Nombre total de paramètres, vérifié (u64).
    let parameter_count = tensors.iter().try_fold(0u64, |acc, spec| {
        let n = spec.num_elements()?;
        acc.checked_add(n).ok_or_else(|| {
            BlueprintError::PlanError("dépassement u64 du nombre de paramètres du plan".into())
        })
    })?;

    Ok(Plan {
        tensors,
        parameter_count,
    })
}

/// Construit les spécifications MoE d'une couche sparse à partir d'une config.
///
/// Génère : routeur, experts partagés (3 matrices) et `n_routed_experts`
/// experts routés (up/gate/down chacun), avec les noms alignés sur les règles
/// GLM (`{up|gate|down}_proj`) ou DeepSeek (`w1/w2/w3`).
pub fn build_moe_specs(
    blueprint: &ModelBlueprint,
    layer_index: u64,
    hidden_size: u64,
    expert_intermediate_size: u64,
    n_routed_experts: u64,
    n_shared_experts: u64,
    top_k: u64,
) -> BlueprintResult<crate::moe::MoeBlockSpec> {
    let rules = &blueprint.naming_rules;

    // Routeur : [hidden, n_routed_experts] (GLM/DeepSeek) ou [hidden, total].
    let router = make_spec(
        &rules.mlp(layer_index, "gate.weight"),
        vec![hidden_size, n_routed_experts],
        DType::F32,
        TensorRole::MoeRouter,
        Some(layer_index),
        None,
    )?;

    // Experts partagés : up/gate/down (ou w1/w3/w2 pour DS).
    let mut shared_experts = Vec::with_capacity(n_shared_experts as usize * 3);
    for e in 0..n_shared_experts {
        for proj in [ExpertProj::Up, ExpertProj::Gate, ExpertProj::Down] {
            let name = if blueprint.naming_rules.prefix.is_empty() {
                // DeepSeek : `ffn.shared_experts.w1.weight`.
                format!(
                    "layers.{layer_index}.ffn.shared_experts.{}.weight",
                    proj.deepseek_suffix()
                )
            } else {
                rules.shared_expert(layer_index, proj)
            };
            shared_experts.push(make_spec(
                &name,
                vec![expert_intermediate_size, hidden_size],
                blueprint.config.dtype_declared,
                TensorRole::MoeSharedExpert,
                Some(layer_index),
                Some(e),
            )?);
        }
    }

    // Experts routés : up/gate/down par expert (w1/w3/w2 pour DS).
    let mut routed_experts = Vec::with_capacity(n_routed_experts as usize);
    for e in 0..n_routed_experts {
        let (up, gate, down) = if blueprint.naming_rules.prefix.is_empty() {
            (
                format!("layers.{layer_index}.ffn.experts.{e}.w1.weight"),
                format!("layers.{layer_index}.ffn.experts.{e}.w3.weight"),
                format!("layers.{layer_index}.ffn.experts.{e}.w2.weight"),
            )
        } else {
            (
                rules.routed_expert(layer_index, e, ExpertProj::Up),
                rules.routed_expert(layer_index, e, ExpertProj::Gate),
                rules.routed_expert(layer_index, e, ExpertProj::Down),
            )
        };
        routed_experts.push(ExpertSpec {
            index: e,
            up: make_spec(
                &up,
                vec![expert_intermediate_size, hidden_size],
                blueprint.config.dtype_declared,
                TensorRole::MoeRoutedExpert,
                Some(layer_index),
                Some(e),
            )?,
            gate: make_spec(
                &gate,
                vec![expert_intermediate_size, hidden_size],
                blueprint.config.dtype_declared,
                TensorRole::MoeRoutedExpert,
                Some(layer_index),
                Some(e),
            )?,
            down: make_spec(
                &down,
                vec![expert_intermediate_size, hidden_size],
                blueprint.config.dtype_declared,
                TensorRole::MoeRoutedExpert,
                Some(layer_index),
                Some(e),
            )?,
        });
    }

    let block = crate::moe::MoeBlockSpec {
        router,
        shared_experts,
        routed_experts,
        top_k,
        layer_type: crate::moe::LayerType::Sparse,
    };
    block.validate()?;
    Ok(block)
}

/// Construit une `TensorSpec` avec les champs communs (dims, rôle, ids).
fn make_spec(
    name: &str,
    dims: Vec<u64>,
    dtype: DType,
    role: TensorRole,
    layer_id: Option<u64>,
    expert_id: Option<u64>,
) -> BlueprintResult<TensorSpec> {
    let mut spec = TensorSpec::new(name, Shape::new(dims)?, dtype, role)?;
    spec.layer_id = layer_id;
    spec.expert_id = expert_id;
    Ok(spec)
}

/// Construit la liste des `LayerKind` à partir des types de couches MoE
/// (`dense`/`sparse`, GLM : 3 denses puis sparse).
pub fn layer_kinds_from_moe_config(cfg: &pmg_core::MoeConfig) -> Vec<LayerKind> {
    let n = cfg.layer_types.len();
    let mut kinds = Vec::with_capacity(n);
    for (i, t) in cfg.layer_types.iter().enumerate() {
        let kind = if let Some(k) = cfg.first_k_dense_replace {
            // GLM : les k premières couches sont denses.
            if (i as u64) < k {
                LayerKind::Dense
            } else {
                LayerKind::MoE
            }
        } else {
            // DeepSeek : toutes les couches sont MoE.
            match t.as_str() {
                "dense" => LayerKind::Dense,
                _ => LayerKind::MoE,
            }
        };
        kinds.push(kind);
    }
    kinds
}

#[cfg(test)]
mod tests {
    use super::{build_moe_specs, layer_kinds_from_moe_config, plan_blueprint};
    use crate::architecture::ArchitectureKind;
    use crate::blueprint::ModelBlueprint;
    use crate::layer::{LayerKind, LayerSpec};
    use crate::naming::NamingRules;
    use crate::tensor_spec::TensorSpec;
    use pmg_core::model_config::{deepseek_v4_flash_test_config, glm52_test_config};
    use pmg_core::{DType, Shape, TensorRole};

    fn glm_blueprint_with_layers(num_layers: u64, moe: bool) -> ModelBlueprint {
        let mut cfg = glm52_test_config();
        cfg.num_layers = num_layers;
        if !moe {
            cfg.moe = None;
        }
        let mut bp = ModelBlueprint::new(
            "glm-test",
            ArchitectureKind::MoETransformer,
            cfg,
            NamingRules::glm52(),
        );
        bp.embeddings.push(
            TensorSpec::new(
                "model.embed_tokens.weight",
                Shape::new(vec![154880, 6144]).unwrap(),
                DType::Bf16,
                TensorRole::Embedding,
            )
            .unwrap(),
        );
        bp.final_norm.push(
            TensorSpec::new(
                "model.norm.weight",
                Shape::new(vec![6144]).unwrap(),
                DType::Bf16,
                TensorRole::Norm,
            )
            .unwrap(),
        );
        bp.lm_head.push(
            TensorSpec::new(
                "lm_head.weight",
                Shape::new(vec![154880, 6144]).unwrap(),
                DType::Bf16,
                TensorRole::LmHead,
            )
            .unwrap(),
        );
        for i in 0..num_layers {
            bp.layers.push(LayerSpec::new(i, LayerKind::Dense));
        }
        bp
    }

    #[test]
    fn plan_is_deterministic_and_ordered() {
        let bp = glm_blueprint_with_layers(2, false);
        let p1 = plan_blueprint(&bp).unwrap();
        let p2 = plan_blueprint(&bp).unwrap();
        // Déterminisme strict : mêmes noms dans le même ordre.
        let names1: Vec<&str> = p1.tensors.iter().map(|t| t.name.as_str()).collect();
        let names2: Vec<&str> = p2.tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names1, names2);
        // Ordre : embed d'abord, puis couches, puis norme finale, puis lm_head.
        assert_eq!(p1.tensors[0].name, "model.embed_tokens.weight");
        assert_eq!(p1.tensors.last().unwrap().name, "lm_head.weight");
        // Paramètres : embed + lm_head (2×154880×6144) + 1 norme finale [6144].
        assert_eq!(p1.parameter_count, 2 * 154880 * 6144 + 6144);
    }

    #[test]
    fn plan_with_moe_block_counts_experts() {
        let mut bp = glm_blueprint_with_layers(1, true);
        // Ajoute un bloc MoE à la couche 0 (8 experts pour le test).
        let moe = build_moe_specs(&bp, 0, 6144, 2048, 8, 1, 8).unwrap();
        bp.layers[0].kind = LayerKind::MoE;
        bp.layers[0].moe_block = Some(moe);
        let plan = plan_blueprint(&bp).unwrap();
        // embed(1) + couche(attention 0 + mlp 0 + norms 0 + router 1 + shared 3 + experts 8×3=24) + norm finale(1) + lm_head(1).
        assert_eq!(plan.tensors.len(), 1 + 1 + 3 + 24 + 1 + 1);
    }

    #[test]
    fn duplicate_name_in_plan_is_rejected() {
        let mut bp = glm_blueprint_with_layers(1, false);
        bp.lm_head.push(bp.embeddings[0].clone()); // duplicat
        assert!(plan_blueprint(&bp).is_err());
    }

    #[test]
    fn moe_specs_match_glm_naming() {
        let bp = glm_blueprint_with_layers(1, true);
        let moe = build_moe_specs(&bp, 0, 6144, 2048, 4, 1, 4).unwrap();
        assert_eq!(moe.router.name, "model.layers.0.mlp.gate.weight");
        assert_eq!(moe.router.role, TensorRole::MoeRouter);
        assert_eq!(
            moe.shared_experts[0].name,
            "model.layers.0.mlp.shared_experts.up_proj.weight"
        );
        assert_eq!(
            moe.routed_experts[0].up.name,
            "model.layers.0.mlp.experts.0.up_proj.weight"
        );
        assert_eq!(
            moe.routed_experts[3].down.name,
            "model.layers.0.mlp.experts.3.down_proj.weight"
        );
    }

    #[test]
    fn moe_specs_match_deepseek_naming() {
        let cfg = deepseek_v4_flash_test_config();
        let bp = ModelBlueprint::new(
            "ds-test",
            ArchitectureKind::HybridAttentionTransformer,
            cfg,
            NamingRules::deepseek_v4_flash(),
        );
        let moe = build_moe_specs(&bp, 0, 4096, 2048, 4, 1, 4).unwrap();
        assert_eq!(moe.router.name, "layers.0.ffn.gate.weight");
        assert_eq!(
            moe.shared_experts[0].name,
            "layers.0.ffn.shared_experts.w1.weight"
        );
        assert_eq!(
            moe.routed_experts[0].up.name,
            "layers.0.ffn.experts.0.w1.weight"
        );
        assert_eq!(
            moe.routed_experts[2].gate.name,
            "layers.0.ffn.experts.2.w3.weight"
        );
    }

    #[test]
    fn layer_kinds_follow_first_k_dense() {
        // GLM : 3 denses puis sparse.
        let cfg = glm52_test_config();
        let kinds = layer_kinds_from_moe_config(cfg.moe.as_ref().unwrap());
        assert_eq!(kinds.len(), 78);
        assert!(kinds[..3].iter().all(|k| *k == LayerKind::Dense));
        assert!(kinds[3..].iter().all(|k| *k == LayerKind::MoE));
    }
}
