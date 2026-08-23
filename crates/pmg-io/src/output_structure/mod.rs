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

//! Création de la structure de dossier de sortie pour les pseudo-modèles générés.
//!
//! Ce module implémente la création de la structure canonique de sortie définie
//! dans la décision architecturale D4. Il gère la copie des fichiers de
//! configuration depuis la source, l'écriture du manifeste et des artefacts
//! d'analyse, ainsi que l'atomicité des écritures.
//!
//! ## Structure canonique
//!
//! ```text
//! <output>/
//! ├── config.json
//! ├── generation_config.json
//! ├── tokenizer.json
//! ├── tokenizer_config.json
//! ├── special_tokens_map.json        (si présent dans la source)
//! ├── chat_template.jinja           (si présent dans la source)
//! ├── model.safetensors.index.json
//! ├── model-00001-of-XXXXX.safetensors
//! ├── ...
//! ├── pmg_metadata.json              ← manifeste racine (fichier canonique)
//! └── pmg/
//!     ├── statistics.json            ← statistiques GÉNÉRÉES par tenseur
//!     └── provenance.json            ← traçabilité OBSERVÉ/ESTIMÉ/GÉNÉRÉ/INCONNU
//! ```
//!
//! ## Contraintes
//!
//! - **Atomicité** : écriture dans dossier temporaire puis renommage
//! - **Mémoire bornée** : ne pas charger tous les fichiers en mémoire
//! - **Gestion des erreurs robuste** : erreurs typées avec contexte
//! - **Documentation en français**

mod config;
mod copy;
mod metadata;
mod utils;

// Réexportations des types et fonctions publiques
pub use config::{OutputConfig, SourceModel};
pub use copy::copy_config_files;
pub use metadata::{write_pmg_metadata, write_pmg_provenance, write_pmg_statistics};
pub use utils::{atomic_write, create_pmg_subdirectory};

use pmg_core::error::CoreResult;
use std::path::Path;

/// Crée la structure de dossier de sortie complète.
///
/// Cette fonction implémente le processus atomique de création de la structure
/// de sortie pour un pseudo-modèle généré. Elle crée d'abord un répertoire
/// temporaire, y écrit tous les fichiers, puis renomme atomiquement vers le
/// répertoire final.
///
/// # Paramètres
/// - `config` : configuration de la sortie ;
/// - `tensors_metadata` : métadonnées des tenseurs générés (pour le calcul des statistiques).
///
/// # Retourne
/// `Ok(())` si la structure a été créée avec succès.
///
/// # Erreurs
/// Retourne une erreur si :
/// - La création du répertoire temporaire échoue
/// - La copie des fichiers de configuration échoue
/// - L'écriture des métadonnées échoue
/// - Le renommage atomique échoue
///
/// # Exemple
///
/// ```rust,ignore
/// use pmg_io::output_structure::{OutputConfig, SourceModel, create_output_structure};
/// use pmg_core::tensor_metadata::TensorMetadata;
/// use pmg_core::shape::Shape;
/// use pmg_core::dtype::DType;
/// use std::path::PathBuf;
///
/// let config = OutputConfig {
///     output_dir: PathBuf::from("/tmp/my_model"),
///     source_dir: PathBuf::from("Models/GLM-5.2"),
///     source_model: SourceModel::Glm52,
///     seed: 42,
///     generator_version: "1.0.0".to_string(),
///     timestamp_utc: "2026-01-01T00:00:00Z".to_string(),
///     parameter_count: 1506659919872,
///     tensor_count: 1240,
///     shards: 4,
///     target_size_bytes: 1073741824,
///     estimated_size_bytes: 1073741824,
///     actual_size_bytes: 1074000000,
///     dtype: "bf16".to_string(),
///     generation_mode: "size-constrained".to_string(),
/// };
///
/// let tensors = vec![
///     TensorMetadata::new(
///         "model.embed_tokens.weight",
///         Shape::new(vec![100, 64]).unwrap(),
///         DType::F32,
///     ).unwrap()
/// ];
///
/// create_output_structure(&config, &tensors).unwrap();
/// ```
pub fn create_output_structure(
    config: &OutputConfig,
    tensors_metadata: &[pmg_core::TensorMetadata],
) -> CoreResult<()> {
    // Crée le dossier temporaire pour l'écriture atomique
    let temp_dir = utils::create_temp_dir(&config.output_dir)?;

    // Crée la structure dans le dossier temporaire
    create_structure_in_dir(&temp_dir, config, tensors_metadata)?;

    // Renomme le dossier temporaire en dossier final (atomicité)
    utils::atomic_rename(&temp_dir, &config.output_dir)?;

    Ok(())
}

/// Crée la structure de dossier dans un répertoire spécifié.
///
/// Cette fonction est exposée pour les tests et pour permettre la création
/// dans un répertoire temporaire.
pub fn create_structure_in_dir(
    dir: &Path,
    config: &OutputConfig,
    tensors_metadata: &[pmg_core::TensorMetadata],
) -> CoreResult<()> {
    // Crée le dossier pmg/
    create_pmg_subdirectory(dir)?;

    // Copie les fichiers de configuration depuis la source
    copy_config_files(dir, &config.source_dir, &config.source_model)?;

    // Écrit le manifeste pmg_metadata.json
    write_pmg_metadata(dir, config)?;

    // Écrit les statistiques et la provenance dans pmg/
    write_pmg_statistics(dir, tensors_metadata)?;
    write_pmg_provenance(dir, config)?;

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use pmg_core::dtype::DType;
    use pmg_core::shape::Shape;
    use pmg_core::tensor_metadata::TensorMetadata;
    use std::path::PathBuf;

    /// Test de création de structure de dossier de sortie.
    #[test]
    fn test_create_output_structure() {
        let temp_dir = std::env::temp_dir().join("pmg_test_output_structure");
        let _ = std::fs::remove_dir_all(&temp_dir);

        // Crée un répertoire source temporaire avec les fichiers de configuration minimal
        let source_dir = temp_dir.join("source");
        std::fs::create_dir_all(&source_dir).unwrap();

        // Fichiers de configuration pour GLM-5.2
        let config_json = r#"{"model_type": "pseudo_model"}"#;
        let generation_config_json = r#"{"do_sample": true}"#;
        let tokenizer_json = r#"{"model_max_length": 1024}"#;
        let tokenizer_config_json = r#"{"pad_token": "[PAD]"}"#;
        let special_tokens_map_json = r#"{"eos_token": "</s>"}"#;
        let chat_template_jinja =
            r#"{% for message in messages %}{{ message.content }}{% endfor %}"#;

        std::fs::write(source_dir.join("config.json"), config_json).unwrap();
        std::fs::write(
            source_dir.join("generation_config.json"),
            generation_config_json,
        )
        .unwrap();
        std::fs::write(source_dir.join("tokenizer.json"), tokenizer_json).unwrap();
        std::fs::write(
            source_dir.join("tokenizer_config.json"),
            tokenizer_config_json,
        )
        .unwrap();
        std::fs::write(
            source_dir.join("special_tokens_map.json"),
            special_tokens_map_json,
        )
        .unwrap();
        std::fs::write(source_dir.join("chat_template.jinja"), chat_template_jinja).unwrap();

        let config = OutputConfig {
            output_dir: temp_dir.join("output"),
            source_dir: source_dir.clone(),
            source_model: SourceModel::Glm52,
            seed: 42,
            generator_version: "1.0.0".to_string(),
            timestamp_utc: "2026-01-01T00:00:00Z".to_string(),
            parameter_count: 1506659919872,
            tensor_count: 1240,
            shards: 4,
            target_size_bytes: 1073741824,
            estimated_size_bytes: 1073741824,
            actual_size_bytes: 1074000000,
            dtype: "bf16".to_string(),
            generation_mode: "size-constrained".to_string(),
        };

        let tensors = vec![TensorMetadata::new(
            "model.embed_tokens.weight",
            Shape::new(vec![100, 64]).unwrap(),
            DType::F32,
        )
        .unwrap()];

        let result = create_output_structure(&config, &tensors);
        assert!(
            result.is_ok(),
            "create_output_structure a échoué: {:?}",
            result.err()
        );

        // Vérifie que la structure a été créée
        let output_dir = &config.output_dir;
        assert!(output_dir.exists());
        assert!(output_dir.join("pmg").exists());
        assert!(output_dir.join("pmg_metadata.json").exists());
        assert!(output_dir.join("pmg").join("statistics.json").exists());
        assert!(output_dir.join("pmg").join("provenance.json").exists());

        // Vérifie que les fichiers de configuration ont été copiés
        assert!(output_dir.join("config.json").exists());
        assert!(output_dir.join("generation_config.json").exists());
        assert!(output_dir.join("tokenizer.json").exists());
        assert!(output_dir.join("tokenizer_config.json").exists());
        assert!(output_dir.join("special_tokens_map.json").exists());
        assert!(output_dir.join("chat_template.jinja").exists());

        // Nettoyage
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    /// Test de création de structure pour DeepSeek-V4-Flash.
    #[test]
    fn test_create_output_structure_deepseek() {
        let temp_dir = std::env::temp_dir().join("pmg_test_output_deepseek");
        let _ = std::fs::remove_dir_all(&temp_dir);

        // Crée un répertoire source temporaire avec les fichiers de configuration minimal
        let source_dir = temp_dir.join("source");
        std::fs::create_dir_all(&source_dir).unwrap();

        // Fichiers de configuration pour DeepSeek-V4-Flash (sans special_tokens_map et chat_template)
        let config_json = r#"{"model_type": "pseudo_model"}"#;
        let generation_config_json = r#"{"do_sample": true}"#;
        let tokenizer_json = r#"{"model_max_length": 2048}"#;
        let tokenizer_config_json = r#"{"pad_token": "<pad>"}"#;

        std::fs::write(source_dir.join("config.json"), config_json).unwrap();
        std::fs::write(
            source_dir.join("generation_config.json"),
            generation_config_json,
        )
        .unwrap();
        std::fs::write(source_dir.join("tokenizer.json"), tokenizer_json).unwrap();
        std::fs::write(
            source_dir.join("tokenizer_config.json"),
            tokenizer_config_json,
        )
        .unwrap();

        let config = OutputConfig {
            output_dir: temp_dir.join("output"),
            source_dir: source_dir.clone(),
            source_model: SourceModel::DeepSeekV4Flash,
            seed: 123,
            generator_version: "2.0.0".to_string(),
            timestamp_utc: "2026-06-15T12:00:00Z".to_string(),
            parameter_count: 7000000000,
            tensor_count: 500,
            shards: 2,
            target_size_bytes: 14000000000,
            estimated_size_bytes: 14000000000,
            actual_size_bytes: 14100000000,
            dtype: "fp16".to_string(),
            generation_mode: "full-structural".to_string(),
        };

        let tensors = vec![TensorMetadata::new(
            "model.embed_tokens.weight",
            Shape::new(vec![200, 64]).unwrap(),
            DType::F16,
        )
        .unwrap()];

        let result = create_output_structure(&config, &tensors);
        assert!(
            result.is_ok(),
            "create_output_structure a échoué pour DeepSeek: {:?}",
            result.err()
        );

        // Vérifie que la structure a été créée
        let output_dir = &config.output_dir;
        assert!(output_dir.exists());
        assert!(output_dir.join("pmg").exists());
        assert!(output_dir.join("pmg_metadata.json").exists());

        // Vérifie que les fichiers de configuration ont été copiés
        assert!(output_dir.join("config.json").exists());
        assert!(output_dir.join("generation_config.json").exists());
        assert!(output_dir.join("tokenizer.json").exists());
        assert!(output_dir.join("tokenizer_config.json").exists());

        // Vérifie que special_tokens_map et chat_template ne sont PAS présents
        assert!(!output_dir.join("special_tokens_map.json").exists());
        assert!(!output_dir.join("chat_template.jinja").exists());

        // Vérifie le contenu du manifeste
        let manifest_content =
            std::fs::read_to_string(output_dir.join("pmg_metadata.json")).unwrap();
        assert!(manifest_content.contains("\"model\": \"deepseek-v4-flash\""));
        assert!(manifest_content.contains("\"seed\": 123"));
        assert!(manifest_content.contains("\"generation_mode\": \"full-structural\""));

        // Nettoyage
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    /// Test de validation du contenu du manifeste pmg_metadata.json.
    #[test]
    fn test_validate_pmg_metadata_content() {
        let temp_dir = std::env::temp_dir().join("pmg_test_validate_metadata");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        let config = OutputConfig {
            output_dir: temp_dir.clone(),
            source_dir: PathBuf::from("Models/GLM-5.2"),
            source_model: SourceModel::Glm52,
            seed: 999,
            generator_version: "3.0.0".to_string(),
            timestamp_utc: "2026-12-31T23:59:59Z".to_string(),
            parameter_count: 1000000000000,
            tensor_count: 2000,
            shards: 8,
            target_size_bytes: 2000000000000,
            estimated_size_bytes: 2000000000000,
            actual_size_bytes: 2010000000000,
            dtype: "bf16".to_string(),
            generation_mode: "custom".to_string(),
        };

        let result = write_pmg_metadata(&temp_dir, &config);
        assert!(result.is_ok());

        // Lit et parse le manifeste
        let path = temp_dir.join("pmg_metadata.json");
        let content = std::fs::read_to_string(&path).unwrap();
        let manifest: serde_json::Value = serde_json::from_str(&content).unwrap();

        // Vérifie les champs obligatoires
        assert_eq!(manifest["format"], "pmg-metadata");
        assert_eq!(manifest["format_version"], 1);
        assert_eq!(manifest["pmg_version"], "1.0.0");
        assert_eq!(manifest["generator_version"], "3.0.0");
        assert_eq!(manifest["profile_version"], "glm52-v1");
        assert_eq!(manifest["model"], "glm-5.2");
        assert_eq!(manifest["synthetic"], true);
        assert_eq!(manifest["seed"], 999);
        assert_eq!(manifest["generation_mode"], "custom");
        assert_eq!(manifest["target_size_bytes"], 2_000_000_000_000_i64);
        assert_eq!(manifest["estimated_size_bytes"], 2_000_000_000_000_i64);
        assert_eq!(manifest["actual_size_bytes"], 2_010_000_000_000_i64);
        assert_eq!(manifest["tensor_count"], 2000);
        assert_eq!(manifest["parameter_count"], 1_000_000_000_000_i64);
        assert_eq!(manifest["dtype"], "bf16");
        assert_eq!(manifest["quantization"], serde_json::Value::Null);
        assert_eq!(manifest["statistical_profile"], "realistic");
        assert_eq!(manifest["source_metadata_hash"], serde_json::Value::Null);
        assert_eq!(manifest["chunk_elements"], 1048576);
        assert_eq!(manifest["shards"], 8);
        assert_eq!(manifest["timestamp_utc"], "2026-12-31T23:59:59Z");

        // Nettoyage
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
