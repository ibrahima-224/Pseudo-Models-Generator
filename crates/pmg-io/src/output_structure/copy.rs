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

//! Copie des fichiers de configuration depuis la source vers la sortie.

use std::path::Path;

use pmg_core::error::{CoreError, CoreResult};

use super::config::SourceModel;

/// Copie les fichiers de configuration depuis la source vers la sortie.
///
/// Selon le modèle source, différents fichiers sont copiés :
/// - **GLM-5.2** : config.json, generation_config.json, tokenizer.json,
///   tokenizer_config.json, special_tokens_map.json, chat_template.jinja
/// - **DeepSeek-V4-Flash** : config.json, generation_config.json, tokenizer.json,
///   tokenizer_config.json
///
/// Les fichiers requis (config.json, generation_config.json, tokenizer.json,
/// tokenizer_config.json) génèrent une erreur s'ils sont absents de la source.
/// Les fichiers optionnels (special_tokens_map.json, chat_template.jinja) sont
/// ignorés silencieusement s'ils sont absents.
///
/// # Paramètres
/// - `output_dir` : répertoire de sortie ;
/// - `source_dir` : répertoire source contenant les fichiers de configuration ;
/// - `source_model` : modèle source (détermine quels fichiers sont présents).
///
/// # Erreurs
/// Retourne une erreur si :
/// - Un fichier requis est absent de la source
/// - La copie d'un fichier échoue
///
/// # Exemple
///
/// ```rust,ignore
/// use pmg_io::output_structure::{copy_config_files, SourceModel};
/// use std::path::PathBuf;
///
/// let output_dir = PathBuf::from("/tmp/my_model");
/// let source_dir = PathBuf::from("Models/GLM-5.2");
///
/// copy_config_files(&output_dir, &source_dir, &SourceModel::Glm52).unwrap();
/// ```
pub fn copy_config_files(
    output_dir: &Path,
    source_dir: &Path,
    source_model: &SourceModel,
) -> CoreResult<()> {
    // Liste des fichiers de configuration à copier
    let config_files = get_config_files_to_copy(source_model);

    for filename in &config_files {
        let src = source_dir.join(filename);
        let dst = output_dir.join(filename);

        if src.exists() {
            // Copie avec gestion d'erreur robuste
            std::fs::copy(&src, &dst).map_err(|e| {
                CoreError::Internal(format!("échec copie {} : {}", src.display(), e))
            })?;
        } else if is_required_config_file(filename) {
            // Fichier requis mais absent → erreur
            return Err(CoreError::Internal(format!(
                "fichier requis {} absent dans la source",
                filename
            )));
        }
        // Fichiers optionnels absents : silencieux (conforme à la décision D4)
    }

    Ok(())
}

/// Retourne la liste des fichiers de configuration à copier selon le modèle.
pub fn get_config_files_to_copy(source_model: &SourceModel) -> Vec<&'static str> {
    let mut files = vec![
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ];

    match source_model {
        SourceModel::Glm52 => {
            // GLM-5.2 a special_tokens_map.json et chat_template.jinja
            files.push("special_tokens_map.json");
            files.push("chat_template.jinja");
        },
        SourceModel::DeepSeekV4Flash => {
            // DeepSeek-V4-Flash n'a pas special_tokens_map.json ni chat_template.jinja
        },
    }

    files
}

/// Indique si un fichier de configuration est requis (absence = erreur).
pub fn is_required_config_file(filename: &str) -> bool {
    matches!(
        filename,
        "config.json" | "generation_config.json" | "tokenizer.json" | "tokenizer_config.json"
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test de copie des fichiers de configuration.
    #[test]
    fn test_copy_config_files() {
        let temp_dir = std::env::temp_dir().join("pmg_test_copy_config");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        // Crée un répertoire source temporaire avec des fichiers de test
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

        let output_dir = temp_dir.join("output");
        std::fs::create_dir_all(&output_dir).unwrap();

        let result = copy_config_files(&output_dir, &source_dir, &SourceModel::Glm52);
        assert!(
            result.is_ok(),
            "copy_config_files a échoué: {:?}",
            result.err()
        );

        // Vérifie que les fichiers ont été copiés
        assert!(output_dir.join("config.json").exists());
        assert!(output_dir.join("generation_config.json").exists());
        assert!(output_dir.join("tokenizer.json").exists());
        assert!(output_dir.join("tokenizer_config.json").exists());
        assert!(output_dir.join("special_tokens_map.json").exists());
        assert!(output_dir.join("chat_template.jinja").exists());

        // Nettoyage
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    /// Test d'erreur quand un fichier requis est manquant.
    #[test]
    fn test_copy_config_files_missing_required() {
        let temp_dir = std::env::temp_dir().join("pmg_test_missing_required");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        // Crée un répertoire source temporaire avec seulement config.json
        let source_dir = temp_dir.join("source");
        std::fs::create_dir_all(&source_dir).unwrap();

        let config_json = r#"{"model_type": "pseudo_model"}"#;
        std::fs::write(source_dir.join("config.json"), config_json).unwrap();
        // manque generation_config.json, tokenizer.json, tokenizer_config.json

        let output_dir = temp_dir.join("output");
        std::fs::create_dir_all(&output_dir).unwrap();

        let result = copy_config_files(&output_dir, &source_dir, &SourceModel::Glm52);
        assert!(
            result.is_err(),
            "devrait échouer en raison de fichiers manquants"
        );

        // Vérifie que l'erreur mentionne le fichier manquant
        let error_msg = format!("{}", result.err().unwrap());
        assert!(error_msg.contains("fichier requis"));

        // Nettoyage
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    /// Test de la liste des fichiers de configuration.
    #[test]
    fn test_get_config_files_to_copy() {
        let glm_files = get_config_files_to_copy(&SourceModel::Glm52);
        assert!(glm_files.contains(&"config.json"));
        assert!(glm_files.contains(&"generation_config.json"));
        assert!(glm_files.contains(&"tokenizer.json"));
        assert!(glm_files.contains(&"tokenizer_config.json"));
        assert!(glm_files.contains(&"special_tokens_map.json"));
        assert!(glm_files.contains(&"chat_template.jinja"));

        let deepseek_files = get_config_files_to_copy(&SourceModel::DeepSeekV4Flash);
        assert!(deepseek_files.contains(&"config.json"));
        assert!(deepseek_files.contains(&"generation_config.json"));
        assert!(deepseek_files.contains(&"tokenizer.json"));
        assert!(deepseek_files.contains(&"tokenizer_config.json"));
        assert!(!deepseek_files.contains(&"special_tokens_map.json"));
        assert!(!deepseek_files.contains(&"chat_template.jinja"));
    }
}
