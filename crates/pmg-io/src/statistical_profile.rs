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

//! Module de chargement des profils statistiques depuis des fichiers JSON.
//!
//! Ce module fournit la fonction [`load_from_file`] qui lit un fichier JSON
//! contenant un profil statistique, le désérialise et valide son contenu.
//! La fonction est séparée du type [`StatisticalProfile`] pour respecter
//! la séparation des responsabilités : le socle (`pmg-core`) ne contient
//! pas d'I/O, tandis que `pmg-io` gère toutes les opérations d'entrée-sortie.

use std::path::Path;

use pmg_core::statistical_profile::StatisticalProfile;

/// Erreurs spécifiques au chargement des profils statistiques.
#[derive(Debug, thiserror::Error)]
pub enum StatisticalProfileError {
    /// Le fichier spécifié est introuvable.
    #[error("fichier de profil statistique introuvable : {0}")]
    FileNotFound(String),

    /// Le chemin ne pointe pas vers un fichier régulier.
    #[error("le chemin n'est pas un fichier : {0}")]
    NotAFile(String),

    /// Erreur de lecture du fichier (permissions, I/O, etc.).
    #[error("erreur de lecture du fichier {0} : {1}")]
    ReadError(String, std::io::Error),

    /// Erreur de décodage du contenu JSON.
    #[error("erreur de décodage JSON du fichier {0} : {1}")]
    JsonError(String, serde_json::Error),

    /// Erreur de validation du profil (contenu invalide).
    #[error("erreur de validation du profil : {0}")]
    ValidationError(String),
}

/// Charge un profil statistique depuis un fichier JSON.
///
/// # Arguments
///
/// * `path` - Chemin vers le fichier JSON du profil.
///
/// # Erreurs
///
/// Retourne une [`StatisticalProfileError`] si :
/// - Le fichier est introuvable
/// - Le chemin ne pointe pas vers un fichier
/// - Le fichier est illisible
/// - Le contenu n'est pas du JSON valide
/// - Le profil ne passe pas la validation
///
/// # Exemple
///
/// ```rust,no_run
/// use pmg_io::statistical_profile::load_from_file;
/// use std::path::Path;
///
/// let profile = load_from_file(Path::new("statistical_profiles/glm52.json"))
///     .expect("profil valide");
/// ```
pub fn load_from_file(path: &Path) -> Result<StatisticalProfile, StatisticalProfileError> {
    // Vérification de l'existence du fichier
    if !path.exists() {
        return Err(StatisticalProfileError::FileNotFound(
            path.display().to_string(),
        ));
    }

    // Vérification que le chemin pointe vers un fichier régulier
    if !path.is_file() {
        return Err(StatisticalProfileError::NotAFile(
            path.display().to_string(),
        ));
    }

    // Lecture du contenu du fichier
    let content = std::fs::read_to_string(path)
        .map_err(|e| StatisticalProfileError::ReadError(path.display().to_string(), e))?;

    // Désérialisation du JSON en StatisticalProfile
    let profile: StatisticalProfile = serde_json::from_str(&content)
        .map_err(|e| StatisticalProfileError::JsonError(path.display().to_string(), e))?;

    // Validation du profil
    profile
        .validate()
        .map_err(|e| StatisticalProfileError::ValidationError(e.to_string()))?;

    Ok(profile)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Retourne le chemin de base du projet (racine du workspace).
    fn project_root() -> PathBuf {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        PathBuf::from(&manifest_dir)
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf()
    }

    #[test]
    fn test_load_from_file_success() {
        let root = project_root();
        let path = root.join("statistical_profiles/glm52.json");
        let result = load_from_file(&path);
        assert!(result.is_ok());

        let profile = result.unwrap();
        assert_eq!(profile.name, "glm52_statistical_profile");
        assert_eq!(profile.version, "1.0.0");
    }

    #[test]
    fn test_load_from_file_nonexistent() {
        let path = Path::new("/chemin/inexistant/profil.json");
        let result = load_from_file(path);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, StatisticalProfileError::FileNotFound(_)));
    }

    #[test]
    fn test_load_from_file_not_a_file() {
        let path = Path::new("/dev/null"); // Device file, pas un fichier régulier sous Linux
        if path.exists() {
            let result = load_from_file(path);
            assert!(result.is_err());
            let err = result.unwrap_err();
            assert!(matches!(err, StatisticalProfileError::NotAFile(_)));
        }
    }

    #[test]
    fn test_load_from_file_invalid_json() {
        let root = project_root();
        let temp_dir = root.join("target");
        let _ = std::fs::create_dir_all(&temp_dir);

        let temp_file = temp_dir.join("invalid_statistical_profile.json");
        std::fs::write(&temp_file, "{ invalid json }").unwrap();

        let result = load_from_file(&temp_file);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(matches!(err, StatisticalProfileError::JsonError(_, _)));

        // Nettoyage
        let _ = std::fs::remove_file(&temp_file);
    }
}
