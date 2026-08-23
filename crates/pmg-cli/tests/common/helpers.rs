//! Fonctions d'aide pour les tests de la CLI PMG.
//!
//! Ce module fournit des utilitaires pour :
//! - Exécuter des commandes PMG avec des arguments spécifiques
//! - Créer des répertoires et fichiers temporaires pour les tests
//! - Générer des fixtures de test pour les modèles

use assert_cmd::Command;
use std::path::{Path, PathBuf};
use tempfile::{tempdir, TempDir};

/// Obtient le chemin vers le binaire PMG compilé.
///
/// # Retourne
/// Une instance de `Command` prête à exécuter le binaire PMG.
pub fn pmg_command() -> Command {
    Command::cargo_bin("pmg").expect("Binaire PMG non trouvé. Avez-vous exécuté `cargo build` ?")
}

/// Obtient le chemin vers la racine du projet.
///
/// # Retourne
/// Le chemin absolu vers la racine du projet PMG.
pub fn project_root() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir)
        .parent()
        .expect("Répertoire parent introuvable")
        .to_path_buf()
}

/// Crée un répertoire temporaire pour un test.
///
/// # Retourne
/// Une paire (TempDir, PathBuf) où le second élément est le chemin du répertoire.
/// Le répertoire sera automatiquement supprimé lorsque TempDir sera détruit.
#[allow(dead_code)]
pub fn create_temp_dir() -> (TempDir, PathBuf) {
    let dir = tempdir().expect("Impossible de créer un répertoire temporaire");
    let path = dir.path().to_path_buf();
    (dir, path)
}

/// Crée un fichier de modèle minimal pour les tests.
///
/// # Paramètres
/// * `dir` - Répertoire où créer le fichier
/// * `name` - Nom du fichier (sans extension)
///
/// # Retourne
/// Le chemin du fichier créé.
#[allow(dead_code)]
pub fn create_test_model_file(dir: &Path, name: &str) -> PathBuf {
    let file_path = dir.join(format!("{}.safetensors", name));

    // Créer un fichier safetensors minimal valide
    let content = serde_json::json!({
        "model": {
            "layers": 2,
            "hidden_size": 64,
            "type": "glm52"
        }
    });

    std::fs::write(&file_path, content.to_string())
        .expect("Impossible de créer le fichier de modèle de test");

    file_path
}

/// Crée un fichier JSON valide pour les tests.
///
/// # Paramètres
/// * `dir` - Répertoire où créer le fichier
/// * `name` - Nom du fichier (sans extension)
/// * `content` - Contenu JSON à écrire
///
/// # Retourne
/// Le chemin du fichier créé.
#[allow(dead_code)]
pub fn create_test_json_file(dir: &Path, name: &str, content: &serde_json::Value) -> PathBuf {
    let file_path = dir.join(format!("{}.json", name));

    std::fs::write(&file_path, content.to_string())
        .expect("Impossible de créer le fichier JSON de test");

    file_path
}

/// Crée un fichier texte pour les tests.
///
/// # Paramètres
/// * `dir` - Répertoire où créer le fichier
/// * `name` - Nom du fichier (sans extension)
/// * `content` - Contenu du fichier
///
/// # Retourne
/// Le chemin du fichier créé.
#[allow(dead_code)]
pub fn create_test_text_file(dir: &Path, name: &str, content: &str) -> PathBuf {
    let file_path = dir.join(format!("{}.txt", name));

    std::fs::write(&file_path, content).expect("Impossible de créer le fichier texte de test");

    file_path
}

/// Vérifie qu'un fichier existe et est lisible.
///
/// # Paramètres
/// * `path` - Chemin du fichier à vérifier
///
/// # Panique
/// Panique si le fichier n'existe pas ou n'est pas lisible.
#[allow(dead_code)]
pub fn assert_file_exists(path: &Path) {
    assert!(path.exists(), "Le fichier {} n'existe pas", path.display());
    assert!(
        path.is_file(),
        "Le chemin {} n'est pas un fichier",
        path.display()
    );
}

/// Vérifie qu'un répertoire existe et est accessible en écriture.
///
/// # Paramètres
/// * `path` - Chemin du répertoire à vérifier
///
/// # Panique
/// Panique si le répertoire n'existe pas ou n'est pas accessible en écriture.
#[allow(dead_code)]
pub fn assert_dir_writable(path: &Path) {
    assert!(
        path.exists(),
        "Le répertoire {} n'existe pas",
        path.display()
    );
    assert!(
        path.is_dir(),
        "Le chemin {} n'est pas un répertoire",
        path.display()
    );
    assert!(
        !std::fs::metadata(path)
            .expect("Impossible de lire les métadonnées")
            .permissions()
            .readonly(),
        "Le répertoire {} n'est pas accessible en écriture",
        path.display()
    );
}

/// Crée un répertoire avec des permissions restreintes pour les tests.
///
/// # Paramètres
/// * `dir` - Répertoire parent
/// * `name` - Nom du répertoire à créer
///
/// # Retourne
/// Le chemin du répertoire créé.
#[cfg(unix)]
#[allow(dead_code)]
pub fn create_readonly_dir(dir: &Path, name: &str) -> PathBuf {
    let readonly_dir = dir.join(name);
    std::fs::create_dir(&readonly_dir).expect("Impossible de créer le répertoire");

    // Retirer les permissions d'écriture
    let mut perms = std::fs::metadata(&readonly_dir)
        .expect("Impossible de lire les métadonnées")
        .permissions();
    perms.set_readonly(true);
    std::fs::set_permissions(&readonly_dir, perms).expect("Impossible de définir les permissions");

    readonly_dir
}
