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

//! Parsing spécifique à l'architecture MoE (Mixture of Experts).
//!
//! Ce module contient les fonctions d'extraction et de conversion
//! des champs spécifiques aux modèles MoE depuis un fichier JSON.

use pmg_core::model_config::AttentionKind;
use pmg_core::DType;

/// Extrait la configuration MoE à partir d'un objet JSON.
///
/// # Arguments
///
/// * `moe_json` - Objet JSON contenant la configuration MoE.
///
/// # Retour
///
/// La configuration MoE normalisée si les champs essentiels sont présents.
pub(crate) fn parse_moe_config(moe_json: &serde_json::Value) -> Option<pmg_core::MoeConfig> {
    let n_routed_experts = moe_json
        .get("num_experts")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let experts_per_tok = moe_json.get("top_k").and_then(|v| v.as_u64()).unwrap_or(0);

    // Extraction des autres champs MoE avec des valeurs par défaut
    let n_shared_experts = moe_json
        .get("n_shared_experts")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let router_dtype = moe_json
        .get("router_dtype")
        .and_then(|v| v.as_str())
        .map(parse_dtype_from_str)
        .unwrap_or(DType::F32);

    let routed_scaling_factor = moe_json
        .get("routed_scaling_factor")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);

    let norm_topk_prob = moe_json
        .get("norm_topk_prob")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let topk_method = moe_json
        .get("topk_method")
        .and_then(|v| v.as_str())
        .unwrap_or("noaux_tc")
        .to_string();

    let first_k_dense_replace = moe_json
        .get("first_k_dense_replace")
        .and_then(|v| v.as_u64());

    let layer_types = moe_json
        .get("layer_types")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let expert_dtype = moe_json
        .get("expert_dtype")
        .and_then(|v| v.as_str())
        .map(parse_dtype_from_str);

    Some(pmg_core::MoeConfig {
        n_routed_experts,
        n_shared_experts,
        experts_per_tok,
        router_dtype,
        routed_scaling_factor,
        norm_topk_prob,
        topk_method,
        first_k_dense_replace,
        layer_types,
        expert_dtype,
    })
}

/// Détecte le type d'attention en fonction de la configuration et des architectures.
///
/// # Arguments
///
/// * `json` - Objet JSON complet de la configuration.
/// * `architectures` - Liste des architectures déclarées.
///
/// # Retour
///
/// Le type d'attention détecté.
pub(crate) fn detect_attention_type(
    json: &serde_json::Value,
    architectures: &[String],
) -> AttentionKind {
    // Vérification des indices spécifiques aux architectures
    for arch in architectures {
        match arch.as_str() {
            "GlmMoeDsaForCausalLM" | "GlmMoeDsaModel" => return AttentionKind::Dsa,
            "DeepseekV4ForCausalLM" | "DeepseekV4Model" => return AttentionKind::Mla,
            _ => {},
        }
    }

    // Vérification des champs de configuration
    if json.get("qk_head_dim").is_some() || json.get("v_head_dim").is_some() {
        // Présence de dimensions spécifiques Q/K/V → probablement DSA ou MLA
        if json.get("qk_head_dim").is_some() && json.get("v_head_dim").is_some() {
            return AttentionKind::Dsa;
        } else if json.get("latent_dim").is_some() || json.get("compress_dim").is_some() {
            return AttentionKind::Mla;
        }
    }

    // Vérification de la présence de MoE avec attention sparse
    if let Some(moe) = json.get("moe") {
        if let Some(sparse_attention) = moe.get("sparse_attention") {
            if sparse_attention.as_bool().unwrap_or(false) {
                return AttentionKind::Hybrid;
            }
        }
    }

    // Détection basée sur num_key_value_heads vs num_attention_heads
    let num_heads = json
        .get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let num_kv_heads = json
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(num_heads);

    if num_kv_heads == num_heads {
        AttentionKind::Dense
    } else {
        AttentionKind::Gqa
    }
}

/// Parse une chaîne de caractères en DType.
///
/// # Arguments
///
/// * `s` - Chaîne représentant le type de données.
///
/// # Retour
///
/// Le DType correspondant, ou F32 par défaut.
pub fn parse_dtype_from_str(s: &str) -> DType {
    match s.to_lowercase().as_str() {
        "float32" | "f32" => DType::F32,
        "float16" | "f16" => DType::F16,
        "bfloat16" | "bf16" => DType::Bf16,
        "int8" => DType::I8,
        "int16" => DType::I16,
        "int32" => DType::I32,
        "int64" => DType::I64,
        "uint8" => DType::U8,
        "uint16" => DType::U16,
        "uint32" => DType::U32,
        "uint64" => DType::U64,
        "bool" => DType::Bool,
        "float8_e4m3fn" => DType::F8E4M3,
        "float8_e5m2" => DType::F8E5M2,
        "float8_e8m0fnu" => DType::F8E8M0,
        "float8_e4m3fnuz" => DType::F8E4M3,
        "float8_e5m2fnuz" => DType::F8E5M2,
        "float4_e2m1" => DType::F4,
        "float4_e2m1fn" => DType::F4,
        "float4_e3m0" => DType::F4,
        "float4" | "f4" => DType::F4,
        _ => DType::F32,
    }
}
