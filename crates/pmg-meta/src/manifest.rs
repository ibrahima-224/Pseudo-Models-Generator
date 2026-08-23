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

//! Module `manifest` — Structure du manifeste canonique PMG.
//!
//! Définit la structure `PmgMetadata` pour le fichier `pmg_metadata.json`.
//! Ce module assure la sérialisation/désérialisation et la validation
//! des métadonnées de génération.
//!
//! ## Structure
//!
//! Le manifeste contient toutes les informations nécessaires pour :
//! - Identifier la génération (format, version, modèle)
//! - tracer l'origine (hash, timestamp)
//! - valider la cohérence (tailles, compteurs)
//!
//! ## Validation
//!
//! La fonction [`PmgMetadata::validate()`] vérifie :
//! - Le format et la version du schéma
//! - Les champs obligatoires non vides
//! - La cohérence des tailles (target <= estimated <= actual)
//! - Le format du hash et du timestamp
//!
//! ## Exemple
//!
//! ```rust
//! use pmg_meta::manifest::PmgMetadata;
//!
//! // Création d'un manifeste par défaut
//! let metadata = PmgMetadata::new_default();
//! assert!(metadata.validate().is_ok());
//!
//! // Affichage en français
//! println!("{}", metadata.display_french());
//!
//! // Sérialisation JSON
//! let json = metadata.to_json().unwrap();
//! let deserialized = PmgMetadata::from_json(&json).unwrap();
//! assert_eq!(metadata, deserialized);
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Erreurs spécifiques au manifeste PMG.
///
/// Ces erreurs sont levées lors de la validation ou de la
/// sérialisation/désérialisation du manifeste.
#[derive(Error, Debug)]
pub enum MetadataError {
    /// Un champ obligatoire est manquant ou vide.
    #[error("Champ obligatoire manquant: {0}")]
    MissingField(String),
    /// Le format ou la structure du manifeste est invalide.
    #[error("Format de manifeste invalide: {0}")]
    InvalidFormat(String),
    /// Erreur lors de la sérialisation JSON.
    #[error("Erreur de sérialisation JSON: {0}")]
    SerializationError(String),
    /// Erreur lors de la désérialisation JSON.
    #[error("Erreur de désérialisation JSON: {0}")]
    DeserializationError(String),
}

/// Représente le manifeste canonique `pmg_metadata.json`.
///
/// Ce structure contient toutes les métadonnées d'une génération PMG.
/// Tous les champs sont requis sauf `quantization`, `pseudo_model` et
/// `weights_are_synthetic` (pour rétrocompatibilité).
///
/// # Validations
///
/// La validation vérifie :
/// - `format` doit être `"pmg-metadata"`
/// - `format_version` doit être `1`
/// - Les champs `model`, `pmg_version`, `generation_mode`, `dtype` ne doivent pas être vides
/// - `actual_size_bytes`, `target_size_bytes`, `estimated_size_bytes` doivent être > 0
/// - `estimated_size_bytes >= target_size_bytes` et `actual_size_bytes >= estimated_size_bytes`
/// - `tensor_count` et `parameter_count` doivent être > 0
/// - `source_metadata_hash` doit commencer par `"sha256:"`
/// - `timestamp_utc` doit être au format ISO 8601 (ex: `"2026-01-01T00:00:00Z"`)
///
/// # Exemple
///
/// ```rust
/// use pmg_meta::manifest::PmgMetadata;
///
/// let metadata = PmgMetadata::new_default();
/// assert!(metadata.validate().is_ok());
///
/// // Modification et validation
/// let mut custom = PmgMetadata::new_default();
/// custom.model = "deepseek-v4-flash".to_string();
/// custom.seed = 12345;
/// assert!(custom.validate().is_ok());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PmgMetadata {
    /// Identifiant de format, doit être `"pmg-metadata"`.
    pub format: String,
    /// Version du schéma (actuellement 1).
    pub format_version: u32,
    /// Version du générateur PMG.
    pub pmg_version: String,
    /// Version du générateur spécifique.
    pub generator_version: String,
    /// Version du profil du modèle.
    pub profile_version: String,
    /// Nom du modèle source.
    pub model: String,
    /// Champ obligatoire: `true` si les poids sont synthétiques.
    pub synthetic: bool,
    /// Graine aléatoire utilisée pour la génération.
    pub seed: u64,
    /// Mode de génération (ex: "size-constrained", "full-structural").
    pub generation_mode: String,
    /// Taille cible en octets.
    pub target_size_bytes: u64,
    /// Taille estimée en octets.
    pub estimated_size_bytes: u64,
    /// Taille réelle en octets.
    pub actual_size_bytes: u64,
    /// Nombre de tenseurs.
    pub tensor_count: u64,
    /// Nombre total de paramètres.
    pub parameter_count: u64,
    /// Type de données (ex: "bf16", "f32").
    pub dtype: String,
    /// Schéma de quantification (optionnel).
    pub quantization: Option<String>,
    /// Profil statistique (ex: "realistic").
    pub statistical_profile: String,
    /// Hash SHA-256 des métadonnées source.
    pub source_metadata_hash: String,
    /// Éléments par chunk.
    pub chunk_elements: u64,
    /// Nombre de shards.
    pub shards: u32,
    /// Horodatage UTC au format ISO 8601.
    pub timestamp_utc: String,
    /// (Rétrocompatibilité) Ancien champ pseudo_model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pseudo_model: Option<String>,
    /// (Rétrocompatibilité) Ancien champ weights_are_synthetic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weights_are_synthetic: Option<bool>,
}

impl PmgMetadata {
    /// Crée un nouveau manifeste avec des valeurs par défaut pour le débogage.
    ///
    /// Les valeurs par défaut sont arbitraires et destinées aux tests.
    pub fn new_default() -> Self {
        Self {
            format: "pmg-metadata".to_string(),
            format_version: 1,
            pmg_version: "1.0.0".to_string(),
            generator_version: "1.0.0".to_string(),
            profile_version: "glm52-v1".to_string(),
            model: "glm-5.2".to_string(),
            synthetic: true,
            seed: 42,
            generation_mode: "size-constrained".to_string(),
            target_size_bytes: 1073741824,
            estimated_size_bytes: 1073741824,
            actual_size_bytes: 1074000000,
            tensor_count: 1240,
            parameter_count: 753000000000,
            dtype: "bf16".to_string(),
            quantization: None,
            statistical_profile: "realistic".to_string(),
            source_metadata_hash:
                "sha256:0000000000000000000000000000000000000000000000000000000000000000"
                    .to_string(),
            chunk_elements: 1048576,
            shards: 4,
            timestamp_utc: "2026-01-01T00:00:00Z".to_string(),
            pseudo_model: None,
            weights_are_synthetic: None,
        }
    }

    /// Valide le manifeste selon les règles canoniques.
    ///
    /// Retourne Ok(()) si valide, ou une erreur descriptive.
    pub fn validate(&self) -> Result<(), MetadataError> {
        // Vérification du champ obligatoire synthetic
        // (toujours vrai car bool, mais on vérifie la cohérence)
        if !self.synthetic && self.pseudo_model.is_none() && self.weights_are_synthetic.is_none() {
            return Err(MetadataError::MissingField(
                "Le champ 'synthetic' doit être true pour un manifeste canonique".to_string(),
            ));
        }

        // Vérification du format
        if self.format != "pmg-metadata" {
            return Err(MetadataError::InvalidFormat(format!(
                "Format attendu 'pmg-metadata', obtenu '{}'",
                self.format
            )));
        }

        // Vérification de la version du schéma
        if self.format_version != 1 {
            return Err(MetadataError::InvalidFormat(format!(
                "Version de schéma attendue 1, obtenue {}",
                self.format_version
            )));
        }

        // Vérification des champs obligatoires non vides
        if self.model.is_empty() {
            return Err(MetadataError::MissingField(
                "Le champ 'model' ne peut pas être vide".to_string(),
            ));
        }

        if self.pmg_version.is_empty() {
            return Err(MetadataError::MissingField(
                "Le champ 'pmg_version' ne peut pas être vide".to_string(),
            ));
        }

        if self.generation_mode.is_empty() {
            return Err(MetadataError::MissingField(
                "Le champ 'generation_mode' ne peut pas être vide".to_string(),
            ));
        }

        if self.dtype.is_empty() {
            return Err(MetadataError::MissingField(
                "Le champ 'dtype' ne peut pas être vide".to_string(),
            ));
        }

        // Vérification de la cohérence des tailles
        if self.actual_size_bytes == 0 {
            return Err(MetadataError::InvalidFormat(
                "La taille réelle ne peut pas être nulle".to_string(),
            ));
        }

        if self.target_size_bytes == 0 {
            return Err(MetadataError::InvalidFormat(
                "La taille cible ne peut pas être nulle".to_string(),
            ));
        }

        if self.estimated_size_bytes == 0 {
            return Err(MetadataError::InvalidFormat(
                "La taille estimée ne peut pas être nulle".to_string(),
            ));
        }

        // Vérification de la cohérence target <= estimated <= actual (avec marge de 20%)
        if self.estimated_size_bytes < self.target_size_bytes {
            return Err(MetadataError::InvalidFormat(
                "La taille estimée ne peut pas être inférieure à la taille cible".to_string(),
            ));
        }

        if self.actual_size_bytes < self.estimated_size_bytes {
            return Err(MetadataError::InvalidFormat(
                "La taille réelle ne peut pas être inférieure à la taille estimée".to_string(),
            ));
        }

        // Vérification des compteurs positifs
        if self.tensor_count == 0 {
            return Err(MetadataError::InvalidFormat(
                "Le nombre de tenseurs ne peut pas être nul".to_string(),
            ));
        }

        if self.parameter_count == 0 {
            return Err(MetadataError::InvalidFormat(
                "Le nombre de paramètres ne peut pas être nul".to_string(),
            ));
        }

        // Vérification du hash (SHA-256 complet)
        if !self.validate_hash(&self.source_metadata_hash) {
            return Err(MetadataError::InvalidFormat(
                "Le hash doit être au format 'sha256:' suivi de 64 caractères hexadécimaux"
                    .to_string(),
            ));
        }

        // Vérification du format timestamp_utc (ISO 8601 strict)
        if self.timestamp_utc.is_empty() {
            return Err(MetadataError::MissingField(
                "Le champ 'timestamp_utc' ne peut pas être vide".to_string(),
            ));
        }

        if !self.validate_timestamp(&self.timestamp_utc) {
            return Err(MetadataError::InvalidFormat(
                "Le timestamp_utc doit être au format ISO 8601 (ex: 2026-01-01T00:00:00Z ou 2026-01-01T00:00:00.123Z)"
                    .to_string(),
            ));
        }

        Ok(())
    }

    /// Sérialise le manifeste en JSON UTF-8.
    pub fn to_json(&self) -> Result<String, MetadataError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| MetadataError::SerializationError(e.to_string()))
    }

    /// Désérialise un manifeste depuis du JSON UTF-8.
    pub fn from_json(json: &str) -> Result<Self, MetadataError> {
        serde_json::from_str(json).map_err(|e| MetadataError::DeserializationError(e.to_string()))
    }

    /// Retourne une représentation textuelle en français du manifeste.
    pub fn display_french(&self) -> String {
        format!(
            "Manifeste PMG v{} pour le modèle {} (profil {})\n\
             Généré par PMG {} en mode {}\n\
             Taille cible: {} octets, estimée: {} octets, réelle: {} octets\n\
             Tenseurs: {}, paramètres: {}, dtype: {}\n\
             Statistiques: {}, shards: {}, chunk: {} éléments\n\
             Horodatage: {}",
            self.format_version,
            self.model,
            self.profile_version,
            self.generator_version,
            self.generation_mode,
            self.target_size_bytes,
            self.estimated_size_bytes,
            self.actual_size_bytes,
            self.tensor_count,
            self.parameter_count,
            self.dtype,
            self.statistical_profile,
            self.shards,
            self.chunk_elements,
            self.timestamp_utc
        )
    }

    /// Valide qu'un hash est au format SHA-256 correct.
    ///
    /// # Paramètres
    /// - `hash` : chaîne à valider.
    ///
    /// # Retour
    /// `true` si le hash est valide, `false` sinon.
    fn validate_hash(&self, hash: &str) -> bool {
        if let Some(hex) = hash.strip_prefix("sha256:") {
            // Vérifier que le hash fait exactement 64 caractères hexadécimaux
            hex.len() == 64 && hex.chars().all(|c| c.is_ascii_hexdigit())
        } else {
            false
        }
    }

    /// Valide qu'un timestamp est au format ISO 8601 strict.
    ///
    /// # Paramètres
    /// - `ts` : chaîne à valider.
    ///
    /// # Retour
    /// `true` si le timestamp est valide, `false` sinon.
    fn validate_timestamp(&self, ts: &str) -> bool {
        // Format ISO 8601 basique: YYYY-MM-DDTHH:MM:SSZ ou YYYY-MM-DDTHH:MM:SS.nnnZ
        // Validation manuelle stricte sans dépendance regex
        let chars: Vec<char> = ts.chars().collect();
        let len = chars.len();

        // Longueur minimale: 20 caractères (2026-01-01T00:00:00Z)
        if len < 20 {
            return false;
        }

        // Vérifier le pattern: 4 chiffres - 2 chiffres - 2 chiffres T 2 chiffres : 2 chiffres : 2 chiffres Z
        // ou avec millisecondes: ... . 3 chiffres Z
        if len == 20 {
            // Format sans millisecondes: YYYY-MM-DDTHH:MM:SSZ
            return chars[4] == '-'
                && chars[7] == '-'
                && chars[10] == 'T'
                && chars[13] == ':'
                && chars[16] == ':'
                && chars[19] == 'Z'
                && chars[0..4].iter().all(|c| c.is_ascii_digit())
                && chars[5..7].iter().all(|c| c.is_ascii_digit())
                && chars[8..10].iter().all(|c| c.is_ascii_digit())
                && chars[11..13].iter().all(|c| c.is_ascii_digit())
                && chars[14..16].iter().all(|c| c.is_ascii_digit())
                && chars[17..19].iter().all(|c| c.is_ascii_digit());
        } else if len > 20 {
            // Format avec millisecondes: YYYY-MM-DDTHH:MM:SS.nnnZ
            // Vérifier la présence du point et du Z final
            if chars[19] != '.' || chars[len - 1] != 'Z' {
                return false;
            }

            // Vérifier que entre le point et Z il y a exactement 3 chiffres
            let ms_part = &chars[20..len - 1];
            if ms_part.len() != 3 {
                return false;
            }

            // Vérifier les parties de base
            return chars[4] == '-'
                && chars[7] == '-'
                && chars[10] == 'T'
                && chars[13] == ':'
                && chars[16] == ':'
                && chars[0..4].iter().all(|c| c.is_ascii_digit())
                && chars[5..7].iter().all(|c| c.is_ascii_digit())
                && chars[8..10].iter().all(|c| c.is_ascii_digit())
                && chars[11..13].iter().all(|c| c.is_ascii_digit())
                && chars[14..16].iter().all(|c| c.is_ascii_digit())
                && chars[17..19].iter().all(|c| c.is_ascii_digit())
                && ms_part.iter().all(|c| c.is_ascii_digit());
        }

        false
    }
}

impl std::fmt::Display for PmgMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_french())
    }
}

#[cfg(test)]
#[path = "manifest_tests.rs"]
mod tests;
