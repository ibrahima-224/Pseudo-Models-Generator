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

//! Définition du trait `ModelProfile` et des types associés.

use std::path::Path;

use crate::error::{ModelProfileError, Result};
use crate::policies::{
    CorrelationPolicy, DtypePolicy, GenerationPolicy, LayerPolicyGlm, LowRankPolicy, OutlierPolicy,
    SerializationPolicy, TensorRule,
};

/// Origine d'une propriété du modèle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum MetadataSource {
    /// Valeur exacte extraite du fichier de configuration original.
    Exact,
    /// Valeur dérivée par calcul à partir d'autres propriétés.
    Derived,
    /// Valeur estimée à partir d'observations ou d'approximations.
    Estimated,
    /// Valeur synthétisée pour les besoins du profil.
    Synthetic,
    /// Origine inconnue.
    Unknown,
}

impl std::fmt::Display for MetadataSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetadataSource::Exact => write!(f, "EXACT"),
            MetadataSource::Derived => write!(f, "DERIVED"),
            MetadataSource::Estimated => write!(f, "ESTIMATED"),
            MetadataSource::Synthetic => write!(f, "SYNTHETIC"),
            MetadataSource::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

/// Trait décrivant les propriétés d'un profil de modèle cible.
///
/// Chaque implémentation fournit les informations structurelles et
/// comportementales d'un modèle spécifique, permettant la génération
/// de pseudo-modèles réalistes.
///
/// # Cas limites
/// - `hidden_size` : Doit être > 0 et divisible par `num_attention_heads`
/// - `expert_capacity` : Si présent, doit être ≥ `num_experts`
/// - `head_dim` : Si présent, doit être égal à `hidden_size / num_attention_heads`
/// - `kv_lora_rank` : Si présent, doit être > 0
///
/// # Exemple
///
/// ```rust
/// use pmg_models::{Glm52Profile, ModelProfile};
///
/// // Utilisation d'un profil concret
/// let profile = Glm52Profile::default_profile();
/// println!("Famille : {}", profile.model_family());
/// println!("Couches : {}", profile.num_layers());
///
/// // Validation du profil
/// assert!(profile.validate().is_ok());
/// ```
pub trait ModelProfile {
    /// Nom de la famille du modèle (ex: "GLM", "DeepSeek").
    fn model_family(&self) -> &str;

    /// Architecture spécifique (ex: "GlmMoeDsaForCausalLM", "DeepseekV4ForCausalLM").
    fn architecture(&self) -> &str;

    /// Nombre de couches cachées.
    fn num_layers(&self) -> u32;

    /// Dimension de la couche cachée.
    fn hidden_size(&self) -> u32;

    /// Nombre de têtes d'attention.
    fn num_attention_heads(&self) -> u32;

    /// Nombre total d'experts (si MoE), sinon `None`.
    fn num_experts(&self) -> Option<u32>;

    /// Capacité par expert (nombre de tokens par expert), sinon `None`.
    fn expert_capacity(&self) -> Option<u32>;

    /// Taille du vocabulaire.
    fn vocab_size(&self) -> u32;

    /// Nombre maximal de positions (contexte).
    fn max_position_embeddings(&self) -> u32;

    /// Dimension de tête d'attention (optionnel, utilisé pour MLA), sinon `None`.
    fn head_dim(&self) -> Option<u32> {
        None
    }

    /// Rang de compression KV pour MLA (optionnel), sinon `None`.
    fn kv_lora_rank(&self) -> Option<u32> {
        None
    }

    /// Noms des tenseurs attendus pour ce modèle.
    fn tensor_names(&self) -> Vec<String>;

    /// Source des métadonnées pour ce profil.
    fn metadata_source(&self) -> MetadataSource;

    /// Politique de génération globale pour ce modèle.
    fn generation_policy(&self) -> &GenerationPolicy;

    /// Politique de types de données pour ce modèle.
    fn dtype_policy(&self) -> &DtypePolicy;

    /// Politique de génération par couche pour ce modèle.
    fn layer_policy(&self) -> &LayerPolicyGlm;

    /// Politique des outliers (super-poids) pour ce modèle.
    fn outlier_policy(&self) -> &OutlierPolicy;

    /// Politique de corrélation entre colonnes pour ce modèle.
    fn correlation_policy(&self) -> &CorrelationPolicy;

    /// Politique de décomposition bas-rang pour ce modèle.
    fn low_rank_policy(&self) -> &LowRankPolicy;

    /// Politique de sérialisation pour ce modèle.
    fn serialization_policy(&self) -> &SerializationPolicy;

    /// Règles de mapping pattern → rôle + politiques pour ce modèle.
    fn tensor_rules(&self) -> &[TensorRule];
}

/// Valide les propriétés de base d'un profil.
///
/// # Arguments
///
/// * `profile` - Référence vers le profil à valider.
///
/// # Erreurs
///
/// Retourne une erreur si le profil contient des valeurs incohérentes,
/// des champs manquants ou des valeurs hors plage.
///
/// # Exemple
///
/// ```rust
/// use pmg_models::{Glm52Profile, ModelProfile};
///
/// let profile = Glm52Profile::default_profile();
/// match profile.validate() {
///     Ok(()) => println!("Profil valide"),
///     Err(e) => println!("Erreur de validation : {}", e),
/// }
/// ```
pub fn validate_profile(profile: &dyn ModelProfile) -> Result<()> {
    // Validation par défaut : vérifications de base
    if profile.num_layers() == 0 {
        return Err(ModelProfileError::InvalidValue {
            field: "num_layers".to_string(),
            message: "le nombre de couches doit être supérieur à 0".to_string(),
        });
    }
    if profile.hidden_size() == 0 {
        return Err(ModelProfileError::InvalidValue {
            field: "hidden_size".to_string(),
            message: "la dimension cachée doit être supérieure à 0".to_string(),
        });
    }
    if profile.num_attention_heads() == 0 {
        return Err(ModelProfileError::InvalidValue {
            field: "num_attention_heads".to_string(),
            message: "le nombre de têtes d'attention doit être supérieur à 0".to_string(),
        });
    }
    if profile.vocab_size() == 0 {
        return Err(ModelProfileError::InvalidValue {
            field: "vocab_size".to_string(),
            message: "la taille du vocabulaire doit être supérieure à 0".to_string(),
        });
    }
    if profile.max_position_embeddings() == 0 {
        return Err(ModelProfileError::InvalidValue {
            field: "max_position_embeddings".to_string(),
            message: "le nombre maximal de positions doit être supérieur à 0".to_string(),
        });
    }
    // Vérification de cohérence : hidden_size doit être divisible par num_attention_heads
    if profile.hidden_size() % profile.num_attention_heads() != 0 {
        return Err(ModelProfileError::InconsistentArchitecture {
            message: format!(
                "hidden_size ({}) doit être divisible par num_attention_heads ({})",
                profile.hidden_size(),
                profile.num_attention_heads()
            ),
        });
    }
    // Vérification des experts si présents
    if let Some(experts) = profile.num_experts() {
        if experts == 0 {
            return Err(ModelProfileError::InvalidValue {
                field: "n_routed_experts".to_string(),
                message: "le nombre d'experts doit être supérieur à 0".to_string(),
            });
        }
        if let Some(capacity) = profile.expert_capacity() {
            if capacity == 0 {
                return Err(ModelProfileError::InvalidValue {
                    field: "expert_capacity".to_string(),
                    message: "la capacité par expert doit être supérieure à 0".to_string(),
                });
            }
            if capacity > experts {
                return Err(ModelProfileError::InconsistentArchitecture {
                    message: format!(
                        "expert_capacity ({}) ne peut pas dépasser le nombre d'experts ({})",
                        capacity, experts
                    ),
                });
            }
        }
    }
    // Validation de head_dim si présent
    // La contrainte head_dim = hidden_size / num_attention_heads ne s'applique
    // que si kv_lora_rank n'est pas utilisé (architecture attention standard)
    if let Some(head_dim) = profile.head_dim() {
        // Si kv_lora_rank est présent, c'est une architecture MLA avec head_dim libre
        if profile.kv_lora_rank().is_none() {
            let expected_head_dim = profile.hidden_size() / profile.num_attention_heads();
            if head_dim != expected_head_dim {
                return Err(ModelProfileError::InconsistentArchitecture {
                    message: format!(
                        "head_dim ({}) doit être égal à hidden_size / num_attention_heads ({})",
                        head_dim, expected_head_dim
                    ),
                });
            }
        }
    }
    // Validation de kv_lora_rank si présent
    if let Some(kv_lora_rank) = profile.kv_lora_rank() {
        if kv_lora_rank == 0 {
            return Err(ModelProfileError::InvalidValue {
                field: "kv_lora_rank".to_string(),
                message: "kv_lora_rank doit être > 0 si présent".to_string(),
            });
        }
    }
    Ok(())
}

/// Charge un profil de modèle depuis un fichier JSON.
///
/// # Arguments
///
/// * `path` - Chemin vers le fichier JSON du profil.
///
/// # Erreurs
///
/// Retourne une erreur si le fichier est introuvable, illisible ou invalide.
///
/// # Exemple
///
/// ```rust,no_run
/// use std::path::Path;
/// use pmg_models::{load_profile_from_file, validate_profile};
///
/// let path = Path::new("profiles/glm52.json");
/// match load_profile_from_file(path) {
///     Ok(profile) => {
///         println!("Modèle chargé : {}", profile.model_family());
///         // Validation du profil chargé
///         if let Err(e) = validate_profile(&*profile) {
///             eprintln!("Profil invalide : {}", e);
///         }
///     }
///     Err(e) => eprintln!("Erreur de chargement : {}", e),
/// }
/// ```
pub fn load_profile_from_file(path: &Path) -> Result<Box<dyn ModelProfile>> {
    if !path.exists() {
        return Err(ModelProfileError::ProfileNotFound {
            path: path.to_path_buf(),
        });
    }
    if !path.is_file() {
        return Err(ModelProfileError::NotAFile {
            path: path.to_path_buf(),
        });
    }

    let content = std::fs::read_to_string(path).map_err(|e| ModelProfileError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;

    let profile: ProfileData =
        serde_json::from_str(&content).map_err(|e| ModelProfileError::Json {
            path: path.to_path_buf(),
            source: e,
        })?;

    match profile.model_type.as_str() {
        "glm_moe_dsa" => Ok(Box::new(crate::Glm52Profile::from_data(profile))),
        "deepseek_v4" => Ok(Box::new(crate::DeepseekV4FlashProfile::from_data(profile))),
        other => Err(ModelProfileError::UnknownModel {
            name: other.to_string(),
        }),
    }
}

/// Structure de données intermédiaire pour le chargement JSON.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ProfileData {
    pub model_type: String,
    pub architecture: String,
    pub hidden_size: u32,
    pub num_attention_heads: u32,
    pub num_hidden_layers: u32,
    pub vocab_size: u32,
    pub max_position_embeddings: u32,
    pub n_routed_experts: Option<u32>,
    pub n_shared_experts: Option<u32>,
    pub expert_capacity: Option<u32>,
    pub tensor_patterns: Vec<String>,
    /// Dimension de tête d'attention (optionnel, utilisé pour MLA)
    #[serde(default)]
    pub head_dim: Option<u32>,
    /// Rang de compression KV pour MLA (optionnel)
    #[serde(default)]
    pub kv_lora_rank: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_data_serialization() {
        let data = ProfileData {
            model_type: "glm_moe_dsa".to_string(),
            architecture: "GlmMoeDsaForCausalLM".to_string(),
            hidden_size: 6144,
            num_attention_heads: 64,
            num_hidden_layers: 78,
            vocab_size: 154880,
            max_position_embeddings: 1048576,
            n_routed_experts: Some(256),
            n_shared_experts: Some(1),
            expert_capacity: Some(8),
            tensor_patterns: vec!["model.embed_tokens.weight".to_string()],
            head_dim: None,
            kv_lora_rank: None,
        };

        // Sérialisation en JSON
        let json = serde_json::to_string(&data).expect("La sérialisation doit réussir");
        assert!(json.contains("glm_moe_dsa"));
        assert!(json.contains("GlmMoeDsaForCausalLM"));

        // Désérialisation depuis JSON
        let deserialized: ProfileData =
            serde_json::from_str(&json).expect("La désérialisation doit réussir");
        assert_eq!(deserialized.model_type, "glm_moe_dsa");
        assert_eq!(deserialized.architecture, "GlmMoeDsaForCausalLM");
        assert_eq!(deserialized.hidden_size, 6144);
        assert_eq!(deserialized.num_attention_heads, 64);
        assert_eq!(deserialized.num_hidden_layers, 78);
        assert_eq!(deserialized.vocab_size, 154880);
        assert_eq!(deserialized.max_position_embeddings, 1048576);
        assert_eq!(deserialized.n_routed_experts, Some(256));
        assert_eq!(deserialized.n_shared_experts, Some(1));
        assert_eq!(deserialized.expert_capacity, Some(8));
        assert_eq!(
            deserialized.tensor_patterns,
            vec!["model.embed_tokens.weight".to_string()]
        );
        assert_eq!(deserialized.head_dim, None);
        assert_eq!(deserialized.kv_lora_rank, None);
    }

    #[test]
    fn test_glm52_profile_serialization() {
        let profile = crate::Glm52Profile::default_profile();
        let data = ProfileData {
            model_type: "glm_moe_dsa".to_string(),
            architecture: "GlmMoeDsaForCausalLM".to_string(),
            hidden_size: profile.hidden_size(),
            num_attention_heads: profile.num_attention_heads(),
            num_hidden_layers: profile.num_layers(),
            vocab_size: profile.vocab_size(),
            max_position_embeddings: profile.max_position_embeddings(),
            n_routed_experts: profile.num_experts(),
            n_shared_experts: None,
            expert_capacity: profile.expert_capacity(),
            tensor_patterns: profile.tensor_names(),
            head_dim: None,
            kv_lora_rank: None,
        };

        let json = serde_json::to_string(&data).expect("La sérialisation doit réussir");
        let deserialized: ProfileData =
            serde_json::from_str(&json).expect("La désérialisation doit réussir");
        assert_eq!(deserialized.model_type, "glm_moe_dsa");
        assert_eq!(deserialized.architecture, "GlmMoeDsaForCausalLM");
    }

    #[test]
    fn test_deepseek_v4_flash_profile_serialization() {
        let profile = crate::DeepseekV4FlashProfile::default_profile();
        let data = ProfileData {
            model_type: "deepseek_v4".to_string(),
            architecture: "DeepseekV4ForCausalLM".to_string(),
            hidden_size: profile.hidden_size(),
            num_attention_heads: profile.num_attention_heads(),
            num_hidden_layers: profile.num_layers(),
            vocab_size: profile.vocab_size(),
            max_position_embeddings: profile.max_position_embeddings(),
            n_routed_experts: profile.num_experts(),
            n_shared_experts: None,
            expert_capacity: profile.expert_capacity(),
            tensor_patterns: profile.tensor_names(),
            head_dim: Some(512),     // Valeur par défaut pour DeepSeek-V4
            kv_lora_rank: Some(512), // Valeur par défaut pour DeepSeek-V4
        };

        let json = serde_json::to_string(&data).expect("La sérialisation doit réussir");
        let deserialized: ProfileData =
            serde_json::from_str(&json).expect("La désérialisation doit réussir");
        assert_eq!(deserialized.model_type, "deepseek_v4");
        assert_eq!(deserialized.architecture, "DeepseekV4ForCausalLM");
        assert_eq!(deserialized.head_dim, Some(512));
        assert_eq!(deserialized.kv_lora_rank, Some(512));
    }

    #[test]
    fn test_validate_profile_function() {
        let profile = crate::Glm52Profile::default_profile();
        let result = validate_profile(&profile);
        assert!(result.is_ok());
    }

    #[test]
    fn test_metadata_source_display() {
        assert_eq!(MetadataSource::Exact.to_string(), "EXACT");
        assert_eq!(MetadataSource::Derived.to_string(), "DERIVED");
        assert_eq!(MetadataSource::Estimated.to_string(), "ESTIMATED");
        assert_eq!(MetadataSource::Synthetic.to_string(), "SYNTHETIC");
        assert_eq!(MetadataSource::Unknown.to_string(), "UNKNOWN");
    }

    #[test]
    fn test_validate_profile_head_dim_incorrect_without_mla() {
        // Test de validation head_dim incohérent sans MLA (kv_lora_rank absent)
        let data = ProfileData {
            model_type: "glm_moe_dsa".to_string(),
            architecture: "GlmMoeDsaForCausalLM".to_string(),
            hidden_size: 6144,
            num_attention_heads: 64,
            num_hidden_layers: 78,
            vocab_size: 154880,
            max_position_embeddings: 1048576,
            n_routed_experts: Some(256),
            n_shared_experts: Some(1),
            expert_capacity: Some(8),
            tensor_patterns: vec!["model.embed_tokens.weight".to_string()],
            head_dim: Some(100), // Incorrect: devrait être 6144/64 = 96
            kv_lora_rank: None,  // Pas de MLA
        };
        let profile = crate::Glm52Profile::from_data(data);
        let result = validate_profile(&profile);
        assert!(result.is_err());
        match result {
            Err(crate::error::ModelProfileError::InconsistentArchitecture { message }) => {
                assert!(message.contains("head_dim"));
                assert!(message.contains("hidden_size / num_attention_heads"));
            },
            _ => panic!("Erreur attendue: InconsistentArchitecture pour head_dim incorrect"),
        }
    }

    #[test]
    fn test_validate_profile_kv_lora_rank_zero() {
        // Test de validation kv_lora_rank == 0
        let data = ProfileData {
            model_type: "deepseek_v4".to_string(),
            architecture: "DeepseekV4ForCausalLM".to_string(),
            hidden_size: 4096,
            num_attention_heads: 64,
            num_hidden_layers: 43,
            vocab_size: 129280,
            max_position_embeddings: 1048576,
            n_routed_experts: Some(256),
            n_shared_experts: Some(1),
            expert_capacity: Some(6),
            tensor_patterns: vec!["model.embed_tokens.weight".to_string()],
            head_dim: Some(512),
            kv_lora_rank: Some(0), // Invalide: doit être > 0
        };
        let profile = crate::DeepseekV4FlashProfile::from_data(data);
        let result = validate_profile(&profile);
        assert!(result.is_err());
        match result {
            Err(crate::error::ModelProfileError::InvalidValue { field, message }) => {
                assert_eq!(field, "kv_lora_rank");
                assert!(message.contains("kv_lora_rank doit être > 0"));
            },
            _ => panic!("Erreur attendue: InvalidValue pour kv_lora_rank == 0"),
        }
    }

    #[test]
    fn test_validate_profile_head_dim_with_mla() {
        // Test de validation head_dim avec MLA (kv_lora_rank présent)
        // La contrainte head_dim = hidden_size / num_attention_heads ne s'applique pas
        let data = ProfileData {
            model_type: "deepseek_v4".to_string(),
            architecture: "DeepseekV4ForCausalLM".to_string(),
            hidden_size: 4096,
            num_attention_heads: 64,
            num_hidden_layers: 43,
            vocab_size: 129280,
            max_position_embeddings: 1048576,
            n_routed_experts: Some(256),
            n_shared_experts: Some(1),
            expert_capacity: Some(6),
            tensor_patterns: vec!["model.embed_tokens.weight".to_string()],
            head_dim: Some(512), // Différent de 4096/64 = 64, mais acceptable avec MLA
            kv_lora_rank: Some(512), // MLA activé
        };
        let profile = crate::DeepseekV4FlashProfile::from_data(data);
        let result = validate_profile(&profile);
        // La validation ne doit pas échouer pour head_dim car kv_lora_rank est présent
        assert!(result.is_ok());
    }
}
