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

//! Résumé architectural du modèle.
//!
//! Ce module génère une représentation humaine de l'architecture du modèle
//! à partir de la configuration et des statistiques structurelles.
//!
//! # Exemple
//!
//! ```rust
//! use pmg_inspect::architecture::summarize_architecture;
//!
//! // Génération du résumé (données fictives)
//! // let summary = summarize_architecture(&config, &structural);
//! // println!("{}", summary);
//! ```

use crate::config_inspector::ConfigInspection;
use crate::structural_stats::StructuralStats;

/// Résumé architectural du modèle.
#[derive(Debug, Clone, Default)]
pub struct ArchitectureSummary {
    /// Type d'architecture (ex: "Transformer", "Mixture of Experts").
    pub architecture_type: String,
    /// Type d'attention (ex: "Dense", "GQA", "DSA", "MLA").
    pub attention_type: String,
    /// Nombre de couches.
    pub num_layers: u64,
    /// Taille cachée (hidden_size).
    pub hidden_size: u64,
    /// Nombre de têtes d'attention.
    pub num_attention_heads: u64,
    /// Nombre de têtes K/V.
    pub num_key_value_heads: u64,
    /// Taille intermédiaire (si disponible).
    pub intermediate_size: Option<u64>,
    /// Taille du vocabulaire.
    pub vocab_size: u64,
    /// Type de données principal.
    pub primary_dtype: String,
    /// Nombre total de paramètres.
    pub total_parameters: u64,
    /// Présence de MoE (Mixture of Experts).
    pub has_moe: bool,
    /// Nombre d'experts (si MoE).
    pub num_experts: Option<u64>,
    /// Taille par tête (hidden_size / num_attention_heads).
    pub head_dim: Option<u64>,
}

impl std::fmt::Display for ArchitectureSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Architecture du modèle ===")?;
        writeln!(f, "Type : {}", self.architecture_type)?;
        writeln!(f, "Attention : {}", self.attention_type)?;
        writeln!(f, "Couches : {}", self.num_layers)?;
        writeln!(f, "Hidden size : {}", self.hidden_size)?;
        writeln!(f, "Têtes d'attention : {}", self.num_attention_heads)?;
        writeln!(f, "Têtes K/V : {}", self.num_key_value_heads)?;
        if let Some(inter) = self.intermediate_size {
            writeln!(f, "Taille intermédiaire : {}", inter)?;
        }
        writeln!(f, "Vocabulaire : {}", self.vocab_size)?;
        writeln!(f, "dtype : {}", self.primary_dtype)?;
        writeln!(f, "Paramètres : {}", format_number(self.total_parameters))?;
        if self.has_moe {
            writeln!(f, "MoE : oui")?;
            if let Some(experts) = self.num_experts {
                writeln!(f, "  Experts : {}", experts)?;
            }
        } else {
            writeln!(f, "MoE : non")?;
        }
        if let Some(head_dim) = self.head_dim {
            writeln!(f, "Taille par tête : {}", head_dim)?;
        }
        Ok(())
    }
}

/// Génère un résumé architectural à partir de la configuration et des statistiques.
///
/// # Paramètres
/// - `config` : inspection de la configuration (peut être None).
/// - `structural` : statistiques structurelles.
pub fn summarize_architecture(
    config: &Option<ConfigInspection>,
    structural: &StructuralStats,
) -> ArchitectureSummary {
    let mut summary = ArchitectureSummary::default();

    if let Some(config) = config {
        // Informations de la configuration
        summary.architecture_type = determine_architecture_type(&config.architectures);
        summary.attention_type = format!("{:?}", config.attention_type);
        summary.num_layers = config.num_layers;
        summary.hidden_size = config.hidden_size;
        summary.num_attention_heads = config.num_attention_heads;
        summary.num_key_value_heads = config.num_key_value_heads;
        summary.intermediate_size = config.intermediate_size;
        summary.vocab_size = config.vocab_size;
        summary.primary_dtype = format!("{:?}", config.dtype);
        summary.has_moe = config.moe.is_some();
        if let Some(ref moe) = config.moe {
            summary.num_experts = Some(moe.n_routed_experts);
        }

        // Calcul de la taille par tête
        if summary.hidden_size > 0 && summary.num_attention_heads > 0 {
            summary.head_dim = Some(summary.hidden_size / summary.num_attention_heads);
        }
    }

    // Utilisation des statistiques structurelles pour le nombre de paramètres
    summary.total_parameters = structural.total_parameters;

    // Si la configuration n'est pas disponible, on déduit à partir des stats
    if summary.architecture_type.is_empty() {
        summary.architecture_type = "Transformer".to_string();
    }
    if summary.attention_type.is_empty() {
        summary.attention_type = "Dense".to_string();
    }
    if summary.primary_dtype.is_empty() {
        summary.primary_dtype = "Inconnu".to_string();
    }

    summary
}

/// Détermine le type d'architecture à partir des architectures déclarées.
fn determine_architecture_type(architectures: &[String]) -> String {
    for arch in architectures {
        match arch.as_str() {
            "GlmMoeDsaForCausalLM" | "GlmMoeDsaModel" => {
                return "Transformer + MoE + DSA".to_string();
            },
            "DeepseekV4ForCausalLM" | "DeepseekV4Model" => {
                return "Transformer + MoE + MLA".to_string();
            },
            "LlamaForCausalLM" | "LlamaModel" => {
                return "Transformer + LLaMA".to_string();
            },
            "MistralForCausalLM" | "MistralModel" => {
                return "Transformer + Mistral".to_string();
            },
            "GPT2LMHeadModel" | "GPT2Model" => {
                return "Transformer + GPT-2".to_string();
            },
            "BertForMaskedLM" | "BertModel" => {
                return "Transformer + BERT".to_string();
            },
            _ => {},
        }
    }
    "Transformer".to_string()
}

/// Formate un nombre avec des séparateurs de milliers.
fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push('_');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config_inspector::ConfigInspection;
    use crate::structural_stats::StructuralStats;
    use pmg_core::dtype::DType;
    use pmg_core::model_config::AttentionKind;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    fn create_test_config() -> ConfigInspection {
        ConfigInspection {
            config_path: PathBuf::from("/fake/config.json"),
            model_type: "glm_moe_dsa".to_string(),
            architectures: vec!["GlmMoeDsaForCausalLM".to_string()],
            hidden_size: 6144,
            num_layers: 78,
            num_attention_heads: 64,
            num_key_value_heads: 64,
            intermediate_size: Some(12288),
            vocab_size: 65536,
            dtype: DType::Bf16,
            attention_type: AttentionKind::Dsa,
            max_position_embeddings: 131072,
            moe: None,
            provenance: BTreeMap::new(),
        }
    }

    #[test]
    fn test_summarize_architecture_with_config() {
        let config = Some(create_test_config());
        let structural = StructuralStats {
            total_parameters: 7000000000,
            ..Default::default()
        };

        let summary = summarize_architecture(&config, &structural);

        assert_eq!(summary.architecture_type, "Transformer + MoE + DSA");
        assert_eq!(summary.attention_type, "Dsa");
        assert_eq!(summary.num_layers, 78);
        assert_eq!(summary.hidden_size, 6144);
        assert_eq!(summary.num_attention_heads, 64);
        assert_eq!(summary.num_key_value_heads, 64);
        assert_eq!(summary.intermediate_size, Some(12288));
        assert_eq!(summary.vocab_size, 65536);
        assert_eq!(summary.primary_dtype, "Bf16");
        assert_eq!(summary.total_parameters, 7000000000);
        assert!(!summary.has_moe);
        assert_eq!(summary.head_dim, Some(6144 / 64));
    }

    #[test]
    fn test_summarize_architecture_without_config() {
        let structural = StructuralStats {
            total_parameters: 100000000,
            ..Default::default()
        };

        let summary = summarize_architecture(&None, &structural);

        assert_eq!(summary.architecture_type, "Transformer");
        assert_eq!(summary.attention_type, "Dense");
        assert_eq!(summary.num_layers, 0);
        assert_eq!(summary.hidden_size, 0);
        assert_eq!(summary.primary_dtype, "Inconnu");
        assert_eq!(summary.total_parameters, 100000000);
    }

    #[test]
    fn test_determine_architecture_type() {
        assert_eq!(
            determine_architecture_type(&["GlmMoeDsaForCausalLM".to_string()]),
            "Transformer + MoE + DSA"
        );
        assert_eq!(
            determine_architecture_type(&["DeepseekV4ForCausalLM".to_string()]),
            "Transformer + MoE + MLA"
        );
        assert_eq!(
            determine_architecture_type(&["LlamaForCausalLM".to_string()]),
            "Transformer + LLaMA"
        );
        assert_eq!(
            determine_architecture_type(&["UnknownModel".to_string()]),
            "Transformer"
        );
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(123), "123");
        assert_eq!(format_number(1234), "1_234");
        assert_eq!(format_number(1234567), "1_234_567");
    }
}
