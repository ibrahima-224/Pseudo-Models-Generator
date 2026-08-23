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

//! Inspection de la configuration du modèle.
//!
//! Ce module lit le fichier `config.json` d'un modèle et extrait les informations
//! de configuration normalisées (architecture, hidden_size, num_layers, etc.)
//! sans charger les poids.
//!
//! # Exemple
//!
//! ```rust
//! use pmg_inspect::config_inspector::inspect_config;
//!
//! // Inspection d'un modèle réel (chemin fictif)
//! // let config = inspect_config("path/to/model").unwrap();
//! // println!("Architecture : {:?}", config.architecture);
//! ```

use std::path::Path;

use pmg_core::model_config::AttentionKind;
use pmg_core::DType;

use crate::config_moe_parser::{detect_attention_type, parse_dtype_from_str, parse_moe_config};
use crate::InspectError;

/// Résultat de l'inspection de la configuration.
#[derive(Debug, Clone)]
pub struct ConfigInspection {
    /// Chemin vers le fichier config.json analysé.
    pub config_path: std::path::PathBuf,
    /// Type de modèle déclaré (ex: "glm_moe_dsa", "deepseek_v4").
    pub model_type: String,
    /// Architectures déclarées (ex: ["GlmMoeDsaForCausalLM"]).
    pub architectures: Vec<String>,
    /// Taille cachée (hidden_size).
    pub hidden_size: u64,
    /// Nombre de couches (num_hidden_layers).
    pub num_layers: u64,
    /// Nombre de têtes d'attention (num_attention_heads).
    pub num_attention_heads: u64,
    /// Nombre de têtes K/V (num_key_value_heads).
    pub num_key_value_heads: u64,
    /// Taille intermédiaire (intermediate_size), si disponible.
    pub intermediate_size: Option<u64>,
    /// Taille du vocabulaire (vocab_size).
    pub vocab_size: u64,
    /// Type de données déclaré (dtype).
    pub dtype: DType,
    /// Type d'attention détecté.
    pub attention_type: AttentionKind,
    /// Longueur maximale de contexte (max_position_embeddings).
    pub max_position_embeddings: u64,
    /// Configuration MoE (si applicable).
    pub moe: Option<pmg_core::MoeConfig>,
    /// Provenance des champs (normalisés).
    pub provenance: std::collections::BTreeMap<String, pmg_core::Origin>,
}

impl std::fmt::Display for ConfigInspection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Fichier : {}", self.config_path.display())?;
        writeln!(f, "Type de modèle : {}", self.model_type)?;
        writeln!(f, "Architectures : {:?}", self.architectures)?;
        writeln!(f, "Hidden size : {}", self.hidden_size)?;
        writeln!(f, "Nombre de couches : {}", self.num_layers)?;
        writeln!(f, "Têtes d'attention : {}", self.num_attention_heads)?;
        writeln!(f, "Têtes K/V : {}", self.num_key_value_heads)?;
        if let Some(inter) = self.intermediate_size {
            writeln!(f, "Taille intermédiaire : {}", inter)?;
        }
        writeln!(f, "Vocabulaire : {}", self.vocab_size)?;
        writeln!(f, "dtype : {:?}", self.dtype)?;
        writeln!(f, "Type d'attention : {:?}", self.attention_type)?;
        writeln!(f, "Contexte max : {}", self.max_position_embeddings)?;
        if let Some(ref moe) = self.moe {
            writeln!(
                f,
                "MoE : {} experts, top-k {}",
                moe.n_routed_experts, moe.experts_per_tok
            )?;
        }
        Ok(())
    }
}

/// Inspecte la configuration d'un modèle situé au chemin donné.
///
/// # Paramètres
/// - `model_path` : chemin vers le répertoire contenant le modèle.
///
/// # Erreurs
/// Retourne une erreur si le fichier config.json est manquant ou invalide.
pub fn inspect_config(model_path: &Path) -> Result<ConfigInspection, InspectError> {
    let config_path = model_path.join("config.json");
    if !config_path.exists() {
        return Err(InspectError::ConfigNotFound(config_path));
    }

    // Lecture du fichier JSON
    let content = std::fs::read_to_string(&config_path)
        .map_err(|e| InspectError::Io(e, config_path.clone()))?;

    // Parsing du JSON
    let json: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| InspectError::Json(e, config_path.clone()))?;

    // Extraction des champs avec gestion des différences d'architectures
    let model_type = json
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let architectures: Vec<String> = json
        .get("architectures")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let hidden_size = json
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let num_layers = json
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let num_attention_heads = json
        .get("num_attention_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let num_key_value_heads = json
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(num_attention_heads); // Par défaut = num_attention_heads

    let intermediate_size = json.get("intermediate_size").and_then(|v| v.as_u64());

    let vocab_size = json.get("vocab_size").and_then(|v| v.as_u64()).unwrap_or(0);

    // Détection du dtype déclaré
    let dtype = json
        .get("torch_dtype")
        .and_then(|v| v.as_str())
        .map(parse_dtype_from_str)
        .unwrap_or(DType::F32);

    // Détection du type d'attention
    let attention_type = detect_attention_type(&json, &architectures);

    let max_position_embeddings = json
        .get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    // Extraction de la configuration MoE (si présente)
    let moe = json.get("moe").and_then(parse_moe_config);

    // Construction de la provenance (tous les champs sont OBSERVED car lus depuis config.json)
    let mut provenance = std::collections::BTreeMap::new();
    for field in &[
        "model_type",
        "architectures",
        "hidden_size",
        "num_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "vocab_size",
        "dtype",
        "attention_type",
        "max_position_embeddings",
        "moe",
    ] {
        provenance.insert(field.to_string(), pmg_core::Origin::Observed);
    }

    Ok(ConfigInspection {
        config_path,
        model_type,
        architectures,
        hidden_size,
        num_layers,
        num_attention_heads,
        num_key_value_heads,
        intermediate_size,
        vocab_size,
        dtype,
        attention_type,
        max_position_embeddings,
        moe,
        provenance,
    })
}

/// Charge une `ModelConfig` à partir d'une inspection (pour compatibilité).
impl ConfigInspection {
    /// Convertit l'inspection en `ModelConfig` normalisée.
    pub fn to_model_config(&self) -> pmg_core::ModelConfig {
        let mut config = pmg_core::ModelConfig {
            model_type: self.model_type.clone(),
            architectures: self.architectures.clone(),
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: None, // Non extrait directement
            qk_head_dim: None,
            v_head_dim: None,
            intermediate_size: self.intermediate_size,
            moe_intermediate_size: None,
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings,
            rms_norm_eps: 1e-5,  // Valeur par défaut typique
            rope_theta: 10000.0, // Valeur par défaut typique
            tie_word_embeddings: false,
            moe: self.moe.clone(),
            attention_type: self.attention_type,
            hyper_connections: false,
            dtype_declared: self.dtype,
            extras: std::collections::BTreeMap::new(),
            provenance: self.provenance.clone(),
        };
        // Marquer tous les champs comme observés
        for field in &[
            "model_type",
            "architectures",
            "hidden_size",
            "num_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "intermediate_size",
            "vocab_size",
            "dtype",
            "attention_type",
            "max_position_embeddings",
            "moe",
        ] {
            config.mark_observed(field);
        }
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_config_inspection_missing_file() {
        let temp_dir = std::env::temp_dir().join("pmg_test_config_missing");
        let result = inspect_config(&temp_dir);
        assert!(result.is_err());
        match result.unwrap_err() {
            InspectError::ConfigNotFound(_) => {}, // OK
            e => panic!("Erreur inattendue : {:?}", e),
        }
    }

    #[test]
    fn test_config_inspection_invalid_json() {
        let temp_dir = std::env::temp_dir().join("pmg_test_config_invalid");
        std::fs::create_dir_all(&temp_dir).unwrap();
        let config_path = temp_dir.join("config.json");
        std::fs::write(&config_path, "invalid json").unwrap();

        let result = inspect_config(&temp_dir);
        assert!(result.is_err());
        match result.unwrap_err() {
            InspectError::Json(_, _) => {}, // OK
            e => panic!("Erreur inattendue : {:?}", e),
        }

        // Nettoyage
        std::fs::remove_file(config_path).unwrap();
        std::fs::remove_dir(temp_dir).unwrap();
    }

    #[test]
    fn test_config_inspection_valid() {
        let temp_dir = std::env::temp_dir().join("pmg_test_config_valid");
        std::fs::create_dir_all(&temp_dir).unwrap();
        let config_path = temp_dir.join("config.json");
        let config_json = r#"{
            "model_type": "glm_moe_dsa",
            "architectures": ["GlmMoeDsaForCausalLM"],
            "hidden_size": 6144,
            "num_hidden_layers": 78,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "intermediate_size": 12288,
            "vocab_size": 65536,
            "torch_dtype": "bfloat16",
            "max_position_embeddings": 131072,
            "moe": {
                "num_experts": 8,
                "top_k": 2,
                "expert_ids": [0, 1, 2, 3, 4, 5, 6, 7],
                "dense_layer_ids": []
            }
        }"#;
        std::fs::write(&config_path, config_json).unwrap();

        let result = inspect_config(&temp_dir);
        assert!(result.is_ok());

        let inspection = result.unwrap();
        assert_eq!(inspection.model_type, "glm_moe_dsa");
        assert_eq!(inspection.architectures, vec!["GlmMoeDsaForCausalLM"]);
        assert_eq!(inspection.hidden_size, 6144);
        assert_eq!(inspection.num_layers, 78);
        assert_eq!(inspection.num_attention_heads, 64);
        assert_eq!(inspection.num_key_value_heads, 64);
        assert_eq!(inspection.intermediate_size, Some(12288));
        assert_eq!(inspection.vocab_size, 65536);
        assert_eq!(inspection.dtype, DType::Bf16);
        assert_eq!(inspection.attention_type, AttentionKind::Dsa);
        assert_eq!(inspection.max_position_embeddings, 131072);
        assert!(inspection.moe.is_some());
        let moe = inspection.moe.unwrap();
        assert_eq!(moe.n_routed_experts, 8);
        assert_eq!(moe.experts_per_tok, 2);

        // Nettoyage
        std::fs::remove_file(config_path).unwrap();
        std::fs::remove_dir(temp_dir).unwrap();
    }

    #[test]
    fn test_to_model_config() {
        // Crée une inspection fictive
        let inspection = ConfigInspection {
            config_path: PathBuf::from("/fake/config.json"),
            model_type: "test".to_string(),
            architectures: vec!["TestModel".to_string()],
            hidden_size: 1024,
            num_layers: 12,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            intermediate_size: None,
            vocab_size: 30000,
            dtype: DType::F32,
            attention_type: AttentionKind::Dense,
            max_position_embeddings: 2048,
            moe: None,
            provenance: std::collections::BTreeMap::new(),
        };

        let config = inspection.to_model_config();
        assert_eq!(config.model_type, "test");
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 16);
        assert_eq!(config.vocab_size, 30000);
        assert_eq!(config.dtype_declared, DType::F32);
        assert_eq!(config.attention_type, AttentionKind::Dense);
        assert_eq!(config.max_position_embeddings, 2048);
    }
}
