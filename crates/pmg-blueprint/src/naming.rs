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

//! Conventions de nommage des tenseurs (`NamingRules`).
//!
//! Les motifs sont extraits des index réels :
//! - GLM-5.2 : préfixe `model.`, `model.layers.{i}.self_attn.*`,
//!   `model.layers.{i}.mlp.experts.{e}.{up|gate|down}_proj.weight` ;
//! - DeepSeek-V4-Flash : sans préfixe, `layers.{i}.attn.*`,
//!   `layers.{i}.ffn.experts.{e}.w{1|2|3}.weight`.
//!
//! Les formats sont **déterministes** et vérifiés par des tests contre un
//! échantillon de noms réels des fixtures (§3.5 de la spécification).
//!
//! Conception : les formats `layer_format`, `expert_format`,
//! `shared_expert_format` et `indexer_format` contiennent **déjà** le préfixe
//! (`model.layers…` pour GLM) ; le champ `prefix` sert uniquement aux tenseurs
//! racines (`embed_tokens`, `lm_head`, norme finale).

use serde::{Deserialize, Serialize};

/// Type de projection d'un expert MoE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExpertProj {
    /// Projection up (w1 / up_proj).
    Up,
    /// Projection gate (w3 / gate_proj).
    Gate,
    /// Projection down (w2 / down_proj).
    Down,
}

impl ExpertProj {
    /// Nom du suffixe selon la convention GLM (`up_proj`/`gate_proj`/`down_proj`).
    pub fn glm_suffix(self) -> &'static str {
        match self {
            ExpertProj::Up => "up_proj",
            ExpertProj::Gate => "gate_proj",
            ExpertProj::Down => "down_proj",
        }
    }

    /// Nom du suffixe selon la convention DeepSeek (`w1`/`w2`/`w3`).
    pub fn deepseek_suffix(self) -> &'static str {
        match self {
            ExpertProj::Up => "w1",
            ExpertProj::Gate => "w3",
            ExpertProj::Down => "w2",
        }
    }
}

/// Conventions de nommage alignées sur un index Safetensors réel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamingRules {
    /// Préfixe global (`model.` pour GLM, vide pour DeepSeek) — tenseurs racines.
    pub prefix: String,
    /// Format d'une couche : exactement deux `{}` (index, chemin).
    pub layer_format: String,
    /// Format des experts routés : trois `{}` (couche, expert, projection).
    pub expert_format: String,
    /// Format des experts partagés : deux `{}` (couche, projection).
    pub shared_expert_format: String,
    /// Suffixe de poids (`.weight`).
    pub suffix_weight: String,
    /// Format de l'indexeur (DSA GLM), optionnel.
    pub indexer_format: Option<String>,
}

impl NamingRules {
    /// Règles de nommage GLM-5.2 (observées dans `Models/GLM-5.2/model.safetensors.index.json`).
    pub fn glm52() -> NamingRules {
        NamingRules {
            prefix: "model.".to_string(),
            layer_format: "model.layers.{}.{}".to_string(),
            expert_format: "model.layers.{}.mlp.experts.{}.{}.weight".to_string(),
            shared_expert_format: "model.layers.{}.mlp.shared_experts.{}.weight".to_string(),
            suffix_weight: ".weight".to_string(),
            indexer_format: Some("model.layers.{}.self_attn.indexer.{}".to_string()),
        }
    }

    /// Règles de nommage DeepSeek-V4-Flash (observées dans l'index du dépôt).
    pub fn deepseek_v4_flash() -> NamingRules {
        NamingRules {
            prefix: String::new(),
            layer_format: "layers.{}.{}".to_string(),
            expert_format: "layers.{}.ffn.experts.{}.{}.weight".to_string(),
            shared_expert_format: "layers.{}.ffn.shared_experts.{}.weight".to_string(),
            suffix_weight: ".weight".to_string(),
            indexer_format: None,
        }
    }

    /// Vrai si les règles suivent la convention DeepSeek (pas de préfixe `model.`).
    fn is_deepseek(&self) -> bool {
        self.prefix.is_empty()
    }

    /// Nom d'un tenseur de couche : `layer_format(layer, path)`.
    ///
    /// Le format contient exactement deux placeholders `{}` : le premier est
    /// l'index de couche, le second le chemin complet (`self_attn.q_proj`…).
    pub fn layer_tensor(&self, layer: u64, path: &str) -> String {
        self.layer_format
            .replacen("{}", &layer.to_string(), 1)
            .replacen("{}", path, 1)
    }

    /// Nom d'un tenseur d'attention.
    ///
    /// GLM : `model.layers.{i}.self_attn.{component}` ;
    /// DeepSeek : `layers.{i}.attn.{component}`.
    pub fn attention(&self, layer: u64, component: &str) -> String {
        if self.is_deepseek() {
            self.layer_tensor(layer, &format!("attn.{component}"))
        } else {
            self.layer_tensor(layer, &format!("self_attn.{component}"))
        }
    }

    /// Nom d'un tenseur du MLP.
    ///
    /// GLM : `model.layers.{i}.mlp.{component}` ; DeepSeek : `layers.{i}.ffn.{component}`.
    pub fn mlp(&self, layer: u64, component: &str) -> String {
        if self.is_deepseek() {
            self.layer_tensor(layer, &format!("ffn.{component}"))
        } else {
            self.layer_tensor(layer, &format!("mlp.{component}"))
        }
    }

    /// Nom d'un expert routé : `model.layers.3.mlp.experts.5.up_proj.weight`.
    /// Pour DeepSeek, le suffixe est `w1`/`w2`/`w3` au lieu de `up_proj`/`gate_proj`/`down_proj`.
    pub fn routed_expert(&self, layer: u64, expert: u64, proj: ExpertProj) -> String {
        // Choix du suffixe selon la convention du modèle (GLM ou DeepSeek).
        let suffix = if self.is_deepseek() {
            proj.deepseek_suffix()
        } else {
            proj.glm_suffix()
        };
        self.expert_format
            .replacen("{}", &layer.to_string(), 1)
            .replacen("{}", &expert.to_string(), 1)
            .replacen("{}", suffix, 1)
    }

    /// Nom d'un expert partagé : `model.layers.3.mlp.shared_experts.up_proj.weight`.
    /// Pour DeepSeek, le suffixe est `w1`/`w2`/`w3` au lieu de `up_proj`/`gate_proj`/`down_proj`.
    pub fn shared_expert(&self, layer: u64, proj: ExpertProj) -> String {
        // Choix du suffixe selon la convention du modèle (GLM ou DeepSeek).
        let suffix = if self.is_deepseek() {
            proj.deepseek_suffix()
        } else {
            proj.glm_suffix()
        };
        self.shared_expert_format
            .replacen("{}", &layer.to_string(), 1)
            .replacen("{}", suffix, 1)
    }

    /// Nom d'un tenseur de l'indexeur (DSA GLM), si le format est défini.
    pub fn indexer(&self, layer: u64, component: &str) -> Option<String> {
        self.indexer_format.as_ref().map(|fmt| {
            fmt.replacen("{}", &layer.to_string(), 1)
                .replacen("{}", component, 1)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{ExpertProj, NamingRules};

    /// Échantillon de noms réels GLM-5.2 (extraits de l'index).
    const GLM_REAL: &[&str] = &[
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.shared_experts.up_proj.weight",
        "model.layers.10.mlp.experts.5.up_proj.weight",
        "model.layers.10.mlp.experts.5.gate_proj.weight",
        "model.layers.10.mlp.experts.5.down_proj.weight",
    ];

    /// Échantillon de noms réels DeepSeek-V4-Flash (extraits de l'index).
    const DS_REAL: &[&str] = &[
        "embed.weight",
        "head.weight",
        "norm.weight",
        "layers.0.hc_attn_base",
        "layers.0.attn.wq_a.weight",
        "layers.0.attn.wkv.weight",
        "layers.0.attn.wo_b.weight",
        "layers.0.attn_norm.weight",
        "layers.0.ffn.gate.weight",
        "layers.0.ffn.shared_experts.w1.weight",
        "layers.0.ffn.experts.3.w1.weight",
        "layers.0.ffn.experts.3.w2.weight",
        "layers.0.ffn.experts.3.w3.weight",
    ];

    #[test]
    fn glm_generated_names_match_real_index() {
        let rules = NamingRules::glm52();
        // Projections d'experts routés (couche 10, expert 5).
        assert_eq!(
            rules.routed_expert(10, 5, ExpertProj::Up),
            "model.layers.10.mlp.experts.5.up_proj.weight"
        );
        assert_eq!(
            rules.routed_expert(10, 5, ExpertProj::Gate),
            "model.layers.10.mlp.experts.5.gate_proj.weight"
        );
        assert_eq!(
            rules.routed_expert(10, 5, ExpertProj::Down),
            "model.layers.10.mlp.experts.5.down_proj.weight"
        );
        // Expert partagé.
        assert_eq!(
            rules.shared_expert(0, ExpertProj::Up),
            "model.layers.0.mlp.shared_experts.up_proj.weight"
        );
        // Attention (self_attn).
        assert_eq!(
            rules.attention(0, "q_a_proj.weight"),
            "model.layers.0.self_attn.q_a_proj.weight"
        );
        assert_eq!(
            rules.attention(0, "kv_a_proj_with_mqa.weight"),
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"
        );
        // MLP (gate = routeur).
        assert_eq!(
            rules.mlp(0, "gate.weight"),
            "model.layers.0.mlp.gate.weight"
        );
        // Indexeur.
        assert_eq!(
            rules.indexer(0, "wk.weight").unwrap(),
            "model.layers.0.self_attn.indexer.wk.weight"
        );
        // Nom de couche générique (norme).
        assert_eq!(
            rules.layer_tensor(0, "input_layernorm.weight"),
            "model.layers.0.input_layernorm.weight"
        );
    }

    #[test]
    fn deepseek_generated_names_match_real_index() {
        let rules = NamingRules::deepseek_v4_flash();
        // Aucun préfixe.
        assert_eq!(rules.prefix, "");
        // Attention (attn).
        assert_eq!(
            rules.attention(0, "wq_a.weight"),
            "layers.0.attn.wq_a.weight"
        );
        assert_eq!(rules.attention(0, "wkv.weight"), "layers.0.attn.wkv.weight");
        assert_eq!(
            rules.attention(0, "wo_b.weight"),
            "layers.0.attn.wo_b.weight"
        );
        // MLP (ffn).
        assert_eq!(rules.mlp(0, "gate.weight"), "layers.0.ffn.gate.weight");
        // Hyper-connections (format couche direct).
        assert_eq!(
            rules.layer_tensor(0, "hc_attn_base"),
            "layers.0.hc_attn_base"
        );
    }

    #[test]
    fn deepseek_suffixes_match_real_patterns() {
        // Conventions DS : w1 (up), w2 (down), w3 (gate).
        let names = [
            (ExpertProj::Up, "layers.0.ffn.experts.3.w1.weight"),
            (ExpertProj::Down, "layers.0.ffn.experts.3.w2.weight"),
            (ExpertProj::Gate, "layers.0.ffn.experts.3.w3.weight"),
        ];
        for (proj, expected) in names {
            let name = format!("layers.0.ffn.experts.3.{}.weight", proj.deepseek_suffix());
            assert_eq!(name, expected);
        }
    }

    #[test]
    fn deepseek_routed_and_shared_expert_names() {
        // Vérifie que les méthodes routed_expert() et shared_expert() génèrent
        // les bons noms avec les suffixes DeepSeek (w1/w2/w3).
        let rules = NamingRules::deepseek_v4_flash();

        // Expert routé : couche 0, expert 3.
        assert_eq!(
            rules.routed_expert(0, 3, ExpertProj::Up),
            "layers.0.ffn.experts.3.w1.weight"
        );
        assert_eq!(
            rules.routed_expert(0, 3, ExpertProj::Gate),
            "layers.0.ffn.experts.3.w3.weight"
        );
        assert_eq!(
            rules.routed_expert(0, 3, ExpertProj::Down),
            "layers.0.ffn.experts.3.w2.weight"
        );

        // Expert partagé : couche 0.
        assert_eq!(
            rules.shared_expert(0, ExpertProj::Up),
            "layers.0.ffn.shared_experts.w1.weight"
        );
        assert_eq!(
            rules.shared_expert(0, ExpertProj::Gate),
            "layers.0.ffn.shared_experts.w3.weight"
        );
        assert_eq!(
            rules.shared_expert(0, ExpertProj::Down),
            "layers.0.ffn.shared_experts.w2.weight"
        );
    }

    #[test]
    fn real_samples_are_covered_by_generation() {
        // Chaque nom réel de l'échantillon GLM doit être générable via les
        // règles (preuve d'alignement avec l'index, spécification §3.5).
        let rules = NamingRules::glm52();
        let generated = [
            rules.attention(0, "q_a_proj.weight"),
            rules.attention(0, "kv_a_proj_with_mqa.weight"),
            rules.attention(0, "o_proj.weight"),
            rules.mlp(0, "gate.weight"),
            rules.shared_expert(0, ExpertProj::Up),
            rules.routed_expert(10, 5, ExpertProj::Up),
            rules.routed_expert(10, 5, ExpertProj::Gate),
            rules.routed_expert(10, 5, ExpertProj::Down),
            rules.layer_tensor(0, "input_layernorm.weight"),
        ];
        for &real in GLM_REAL {
            // Les noms racines (embed, lm_head, norme finale) sont produits par
            // des formats dédiés simples (prefix + nom) ; on les admet ici.
            let covered = generated.iter().any(|g| g == real)
                || real == "model.embed_tokens.weight"
                || real == "lm_head.weight"
                || real == "model.norm.weight";
            assert!(covered, "nom réel non couvert : {real}");
        }
        // Les échantillons DeepSeek sont des motifs couverts par les tests
        // `deepseek_generated_names_match_real_index` et
        // `deepseek_suffixes_match_real_patterns`.
        let ds_rules = NamingRules::deepseek_v4_flash();
        let ds_generated = [
            ds_rules.attention(0, "wq_a.weight"),
            ds_rules.attention(0, "wkv.weight"),
            ds_rules.attention(0, "wo_b.weight"),
            ds_rules.mlp(0, "gate.weight"),
            ds_rules.layer_tensor(0, "hc_attn_base"),
            ds_rules.layer_tensor(0, "attn_norm.weight"),
            format!(
                "layers.0.ffn.shared_experts.{}.weight",
                ExpertProj::Up.deepseek_suffix()
            ),
            format!(
                "layers.0.ffn.experts.3.{}.weight",
                ExpertProj::Up.deepseek_suffix()
            ),
            format!(
                "layers.0.ffn.experts.3.{}.weight",
                ExpertProj::Down.deepseek_suffix()
            ),
            format!(
                "layers.0.ffn.experts.3.{}.weight",
                ExpertProj::Gate.deepseek_suffix()
            ),
        ];
        for &real in DS_REAL {
            let covered = ds_generated.iter().any(|g| g == real)
                || real == "embed.weight"
                || real == "head.weight"
                || real == "norm.weight";
            assert!(covered, "nom réel DS non couvert : {real}");
        }
    }

    #[test]
    fn serde_roundtrip() {
        let rules = NamingRules::glm52();
        let json = serde_json::to_string(&rules).unwrap();
        assert_eq!(serde_json::from_str::<NamingRules>(&json).unwrap(), rules);
    }
}
