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

//! Rôle fonctionnel d'un tenseur (`TensorRole`).
//!
//! Le rôle pilote la distribution statistique, la politique de dtype,
//! l'injection et les cibles d'entropie/sparsité : jamais un seul générateur
//! pour tous les tenseurs (spécification `docs/architecture/03-modeles-de-donnees.md` §2.4).
//!
//! Ce module fournit aussi un mapping de nommage : à partir d'un nom de
//! tenseur d'un index Safetensors, on déduit le rôle (utilisé par
//! `pmg-blueprint::naming` et l'inventaire).

use serde::{Deserialize, Serialize};

/// Rôle fonctionnel d'un tenseur dans l'architecture d'un modèle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TensorRole {
    /// Table d'embeddings (tokens) — `embed_tokens.weight` / `embed.weight`.
    Embedding,
    /// Projection Q d'attention — `q_a_proj` / `wq_a` / `q_proj`.
    AttentionQuery,
    /// Projection K d'attention — `kv_a_proj` / `wkv` / `k_proj`.
    AttentionKey,
    /// Projection V d'attention — `v_proj`.
    AttentionValue,
    /// Projection de sortie d'attention — `o_proj` / `wo_a` / `wo_b`.
    AttentionOutput,
    /// Projection `up` du MLP — `up_proj` / `w1`.
    MlpUp,
    /// Projection `gate` du MLP (ou `gate_proj` d'un expert) — `w3`.
    MlpGate,
    /// Projection `down` du MLP — `down_proj` / `w2`.
    MlpDown,
    /// Routeur MoE — `mlp.gate` / `ffn.gate` (logits par expert).
    MoeRouter,
    /// Expert partagé MoE (utilisé par toutes les couches).
    MoeSharedExpert,
    /// Expert routé MoE (choisi par le top-k).
    MoeRoutedExpert,
    /// Normalisation — `input_layernorm`, `post_attention_layernorm`, `norm`.
    Norm,
    /// Tête de langage (LM head) — `lm_head.weight` / `head.weight`.
    LmHead,
    /// Module Multi-Token Prediction — `mtp.*`.
    Mtp,
    /// Hyper-connexion — `hc_attn_base`, `hc_ffn_*`.
    HyperConnection,
    /// Indexeur d'attention (DSA/MLA GLM/DeepSeek) — `self_attn.indexer.*`.
    AttentionIndexer,
    /// Autre rôle non catégorisé.
    Other,
}

impl TensorRole {
    /// Libellé français court pour l'affichage (espec, rapports).
    pub fn label_fr(self) -> &'static str {
        match self {
            TensorRole::Embedding => "embeddings",
            TensorRole::AttentionQuery => "attention Q",
            TensorRole::AttentionKey => "attention K",
            TensorRole::AttentionValue => "attention V",
            TensorRole::AttentionOutput => "attention O",
            TensorRole::MlpUp => "MLP up",
            TensorRole::MlpGate => "MLP gate",
            TensorRole::MlpDown => "MLP down",
            TensorRole::MoeRouter => "routeur MoE",
            TensorRole::MoeSharedExpert => "expert partagé",
            TensorRole::MoeRoutedExpert => "expert routé",
            TensorRole::Norm => "normalisation",
            TensorRole::LmHead => "tête de langage",
            TensorRole::Mtp => "MTP",
            TensorRole::HyperConnection => "hyper-connexion",
            TensorRole::AttentionIndexer => "indexeur d'attention",
            TensorRole::Other => "autre",
        }
    }

    /// Déduit le rôle d'un tenseur à partir de son nom (motifs d'index réels).
    ///
    /// L'ordre des motifs est important : les noms les plus spécifiques
    /// (`indexer`, `shared_experts`, `experts.`) sont testés avant les motifs
    /// génériques (`gate`, `up`, `down`).
    pub fn from_name(name: &str) -> TensorRole {
        let lower = name.to_ascii_lowercase();
        // Non-couches.
        if lower.contains("embed") {
            return TensorRole::Embedding;
        }
        if lower.contains("lm_head") || lower == "head.weight" || lower.ends_with(".head.weight") {
            return TensorRole::LmHead;
        }
        if lower.starts_with("mtp.") || lower.contains(".mtp.") {
            return TensorRole::Mtp;
        }
        if lower.contains("hc_") {
            return TensorRole::HyperConnection;
        }
        // Normalisations : `layernorm`, `rms_norm`, ou tout nom finissant par
        // `norm.weight` (`model.norm.weight`, `layers.0.attn_norm.weight`,
        // `layers.0.ffn_norm.weight`…). NB : `mtp.*` est traité plus haut.
        if lower.contains("layernorm")
            || lower.contains("rms_norm")
            || lower.ends_with("norm.weight")
        {
            return TensorRole::Norm;
        }
        // Indexeur d'attention (DSA/MLA).
        if lower.contains("indexer") || lower.contains("weights_proj") || lower.contains("wk.") {
            return TensorRole::AttentionIndexer;
        }
        // MoE : router, shared et routed experts (avant les projections MLP).
        // Motifs réels : `mlp.gate.weight`, `ffn.gate.weight`, `ffn.gate.bias`,
        // `ffn.gate.tid2eid`, ou tout nom contenant « router ».
        if lower.contains("mlp.gate") || lower.contains("ffn.gate") || lower.contains("router") {
            return TensorRole::MoeRouter;
        }
        if lower.contains("shared_experts") {
            return TensorRole::MoeSharedExpert;
        }
        if lower.contains("experts.") {
            return TensorRole::MoeRoutedExpert;
        }
        // Attention (avant MLP : `kv_a_proj` contient "a_proj" générique).
        if lower.contains("q_a_proj")
            || lower.contains("q_b_proj")
            || lower.contains("wq_a")
            || lower.contains("wq_b")
            || lower.contains("q_proj")
        {
            return TensorRole::AttentionQuery;
        }
        if lower.contains("kv_a_proj")
            || lower.contains("kv_b_proj")
            || lower.contains("wkv")
            || lower.contains("k_proj")
        {
            return TensorRole::AttentionKey;
        }
        if lower.contains("v_proj") && !lower.contains("kv_") {
            return TensorRole::AttentionValue;
        }
        if lower.contains("o_proj") || lower.contains("wo_a") || lower.contains("wo_b") {
            return TensorRole::AttentionOutput;
        }
        // MLP dense.
        if lower.contains("gate_proj") {
            return TensorRole::MlpGate;
        }
        if lower.contains("up_proj") {
            return TensorRole::MlpUp;
        }
        if lower.contains("down_proj") {
            return TensorRole::MlpDown;
        }
        if lower.ends_with(".w1") || lower.ends_with("w1.weight") {
            return TensorRole::MlpUp;
        }
        if lower.ends_with(".w2") || lower.ends_with("w2.weight") {
            return TensorRole::MlpDown;
        }
        if lower.ends_with(".w3") || lower.ends_with("w3.weight") {
            return TensorRole::MlpGate;
        }
        TensorRole::Other
    }
}

#[cfg(test)]
mod tests {
    use super::TensorRole;

    /// Échantillon de noms réels extraits des index GLM-5.2 et DeepSeek-V4-Flash.
    #[test]
    fn glm_real_names_map_to_roles() {
        // GLM-5.2 : préfixe "model.".
        let cases = [
            ("model.embed_tokens.weight", TensorRole::Embedding),
            ("lm_head.weight", TensorRole::LmHead),
            ("model.norm.weight", TensorRole::Norm),
            ("model.layers.0.input_layernorm.weight", TensorRole::Norm),
            (
                "model.layers.0.post_attention_layernorm.weight",
                TensorRole::Norm,
            ),
            (
                "model.layers.0.self_attn.q_a_proj.weight",
                TensorRole::AttentionQuery,
            ),
            (
                "model.layers.0.self_attn.q_b_proj.weight",
                TensorRole::AttentionQuery,
            ),
            (
                "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
                TensorRole::AttentionKey,
            ),
            (
                "model.layers.0.self_attn.kv_b_proj.weight",
                TensorRole::AttentionKey,
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                TensorRole::AttentionOutput,
            ),
            (
                "model.layers.0.self_attn.indexer.wk.weight",
                TensorRole::AttentionIndexer,
            ),
            ("model.layers.0.mlp.gate.weight", TensorRole::MoeRouter),
            ("model.layers.0.mlp.up_proj.weight", TensorRole::MlpUp),
            ("model.layers.0.mlp.down_proj.weight", TensorRole::MlpDown),
            (
                "model.layers.0.mlp.shared_experts.up_proj.weight",
                TensorRole::MoeSharedExpert,
            ),
            (
                "model.layers.10.mlp.experts.5.up_proj.weight",
                TensorRole::MoeRoutedExpert,
            ),
            (
                "model.layers.10.mlp.experts.5.gate_proj.weight",
                TensorRole::MoeRoutedExpert,
            ),
            (
                "model.layers.10.mlp.experts.5.down_proj.weight",
                TensorRole::MoeRoutedExpert,
            ),
        ];
        for (name, expected) in cases {
            assert_eq!(TensorRole::from_name(name), expected, "nom {name}");
        }
    }

    #[test]
    fn deepseek_real_names_map_to_roles() {
        // DeepSeek-V4-Flash : sans préfixe "model.", suffixes ".scale" compris.
        let cases = [
            ("embed.weight", TensorRole::Embedding),
            ("head.weight", TensorRole::LmHead),
            ("norm.weight", TensorRole::Norm),
            ("layers.0.attn_norm.weight", TensorRole::Norm),
            ("layers.0.ffn_norm.weight", TensorRole::Norm),
            ("layers.0.hc_attn_base", TensorRole::HyperConnection),
            ("layers.0.hc_ffn_base", TensorRole::HyperConnection),
            ("layers.0.hc_attn_scale", TensorRole::HyperConnection),
            ("layers.0.attn.wq_a.weight", TensorRole::AttentionQuery),
            ("layers.0.attn.wq_b.weight", TensorRole::AttentionQuery),
            ("layers.0.attn.wkv.weight", TensorRole::AttentionKey),
            ("layers.0.attn.wo_a.weight", TensorRole::AttentionOutput),
            ("layers.0.attn.wo_b.weight", TensorRole::AttentionOutput),
            ("layers.0.ffn.gate.weight", TensorRole::MoeRouter),
            ("layers.0.ffn.gate.tid2eid", TensorRole::MoeRouter),
            (
                "layers.0.ffn.shared_experts.w1.weight",
                TensorRole::MoeSharedExpert,
            ),
            (
                "layers.0.ffn.shared_experts.w2.weight",
                TensorRole::MoeSharedExpert,
            ),
            (
                "layers.0.ffn.shared_experts.w3.weight",
                TensorRole::MoeSharedExpert,
            ),
            (
                "layers.0.ffn.experts.0.w1.weight",
                TensorRole::MoeRoutedExpert,
            ),
            (
                "layers.0.ffn.experts.0.w2.weight",
                TensorRole::MoeRoutedExpert,
            ),
            (
                "layers.0.ffn.experts.0.w3.weight",
                TensorRole::MoeRoutedExpert,
            ),
            (
                "layers.0.ffn.experts.0.w1.scale",
                TensorRole::MoeRoutedExpert,
            ),
            ("mtp.2.ffn.gate.weight", TensorRole::Mtp),
            ("mtp.2.norm.weight", TensorRole::Mtp),
        ];
        for (name, expected) in cases {
            assert_eq!(TensorRole::from_name(name), expected, "nom {name}");
        }
    }

    #[test]
    fn unknown_names_fall_back_to_other() {
        assert_eq!(
            TensorRole::from_name("some.unknown.tensor"),
            TensorRole::Other
        );
        assert_eq!(TensorRole::from_name(""), TensorRole::Other);
    }

    #[test]
    fn labels_are_french() {
        assert_eq!(TensorRole::Embedding.label_fr(), "embeddings");
        assert_eq!(TensorRole::MoeRouter.label_fr(), "routeur MoE");
        assert_eq!(TensorRole::Other.label_fr(), "autre");
    }
}
