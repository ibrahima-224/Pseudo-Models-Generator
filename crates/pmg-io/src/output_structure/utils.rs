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

//! Fonctions utilitaires pour la création de la structure de sortie.

use std::path::{Path, PathBuf};

use pmg_core::error::{CoreError, CoreResult};

/// Crée le sous-répertoire `pmg/` dans le répertoire de sortie.
///
/// Ce sous-répertoire contient les artefacts d'analyse :
/// - `statistics.json` : statistiques générées par tenseur
/// - `provenance.json` : traçabilité des sources (OBSERVÉ/ESTIMÉ/GÉNÉRÉ/INCONNU)
///
/// # Paramètres
/// - `output_dir` : répertoire de sortie.
///
/// # Erreurs
/// Retourne une erreur si la création échoue.
///
/// # Exemple
///
/// ```rust,ignore
/// use pmg_io::output_structure::create_pmg_subdirectory;
/// use std::path::PathBuf;
///
/// let output_dir = PathBuf::from("/tmp/my_model");
/// create_pmg_subdirectory(&output_dir).unwrap();
///
/// // Le répertoire /tmp/my_model/pmg/ a été créé
/// ```
pub fn create_pmg_subdirectory(output_dir: &Path) -> CoreResult<()> {
    let pmg_dir = output_dir.join("pmg");
    std::fs::create_dir_all(&pmg_dir)
        .map_err(|e| CoreError::Internal(format!("échec création dossier pmg/ : {}", e)))?;

    Ok(())
}

/// Écriture atomique via dossier temporaire.
///
/// Écrit le contenu dans un fichier temporaire (avec extension `.tmp`) puis
/// renomme atomiquement vers le chemin final. En cas d'échec du renommage,
/// le fichier temporaire est supprimé.
///
/// Cette approche garantit que le fichier final n'est jamais dans un état
/// partiellement écrit.
///
/// # Paramètres
/// - `path` : chemin du fichier à écrire ;
/// - `data` : données à écrire.
///
/// # Erreurs
/// Retourne une erreur si l'écriture ou le renommage échoue.
///
/// # Exemple
///
/// ```rust,ignore
/// use pmg_io::output_structure::atomic_write;
/// use std::path::PathBuf;
///
/// let path = PathBuf::from("/tmp/test.json");
/// let data = b"{\"key\": \"value\"}";
///
/// atomic_write(&path, data).unwrap();
/// assert!(path.exists());
/// ```
pub fn atomic_write(path: &Path, data: &[u8]) -> CoreResult<()> {
    let temp_path = path.with_extension("tmp");

    // Écrit dans le fichier temporaire
    std::fs::write(&temp_path, data).map_err(|e| {
        CoreError::Internal(format!(
            "échec écriture atomique {} : {}",
            path.display(),
            e
        ))
    })?;

    // Renomme atomiquement
    std::fs::rename(&temp_path, path).map_err(|e| {
        // Nettoyage en cas d'échec
        let _ = std::fs::remove_file(&temp_path);
        CoreError::Internal(format!(
            "échec renommage atomique {} : {}",
            path.display(),
            e
        ))
    })?;

    Ok(())
}

/// Crée un répertoire temporaire pour l'écriture atomique.
pub fn create_temp_dir(output_dir: &Path) -> CoreResult<PathBuf> {
    // Utilise le PID pour éviter les collisions
    let pid = std::process::id();
    let temp_name = format!("{}.tmp-{}", output_dir.display(), pid);
    let temp_path = PathBuf::from(temp_name);

    // Si le dossier existe déjà, le supprime (propre après un crash précédent)
    if temp_path.exists() {
        std::fs::remove_dir_all(&temp_path).map_err(|e| {
            CoreError::Internal(format!(
                "échec suppression ancien dossier temporaire : {}",
                e
            ))
        })?;
    }

    std::fs::create_dir_all(&temp_path).map_err(|e| {
        CoreError::Internal(format!(
            "échec création dossier temporaire {} : {}",
            temp_path.display(),
            e
        ))
    })?;

    Ok(temp_path)
}

/// Renomme atomiquement un répertoire temporaire en répertoire final.
pub fn atomic_rename(temp_dir: &Path, final_dir: &Path) -> CoreResult<()> {
    // Si le dossier final existe déjà, le supprime
    if final_dir.exists() {
        std::fs::remove_dir_all(final_dir).map_err(|e| {
            CoreError::Internal(format!("échec suppression ancien dossier final : {}", e))
        })?;
    }

    std::fs::rename(temp_dir, final_dir).map_err(|e| {
        // Nettoyage en cas d'échec
        let _ = std::fs::remove_dir_all(temp_dir);
        CoreError::Internal(format!(
            "échec renommage atomique {} → {} : {}",
            temp_dir.display(),
            final_dir.display(),
            e
        ))
    })?;

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Test d'écriture atomique.
    #[test]
    fn test_atomic_write() {
        let temp_dir = std::env::temp_dir().join("pmg_test_atomic_write");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        let path = temp_dir.join("test.json");
        let data = b"{\"test\": true}";

        let result = atomic_write(&path, data);
        assert!(result.is_ok());

        // Vérifie que le fichier a été créé
        assert!(path.exists());

        // Vérifie le contenu
        let content = std::fs::read(&path).unwrap();
        assert_eq!(content, data);

        // Nettoyage
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    /// Test de création du sous-répertoire pmg/.
    #[test]
    fn test_create_pmg_subdirectory() {
        let temp_dir = std::env::temp_dir().join("pmg_test_pmg_dir");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        let result = create_pmg_subdirectory(&temp_dir);
        assert!(result.is_ok());

        // Vérifie que le dossier pmg/ a été créé
        assert!(temp_dir.join("pmg").exists());

        // Nettoyage
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
