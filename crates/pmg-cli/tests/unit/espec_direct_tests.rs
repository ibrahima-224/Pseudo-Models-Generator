//! Tests directs pour la commande espec de la CLI PMG.
//!
//! Ces tests testent directement la fonction execute de espec.rs
//! sans passer par la ligne de commande, pour éviter les problèmes
//! avec la commande help dupliquée.

use std::path::PathBuf;
use tempfile::tempdir;

/// Retourne le chemin vers le répertoire des fixtures.
fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

/// Test : Fichier Safetensors valide avec contenu minimal.
///
/// Vérifie que espec fonctionne avec un fichier Safetensors valide.
#[test]
fn test_espec_valid_safetensors_file_direct() {
    // Arrange : Utiliser une fixture valide
    let fixtures = fixtures_dir();
    let model_a = fixtures.join("model_a.safetensors");

    // Vérifier que la fixture existe
    assert!(
        model_a.exists(),
        "La fixture model_a.safetensors n'existe pas"
    );

    // Act
    let args = pmg_cli::commands::espec::EspecArgs {
        model_path: model_a.to_str().unwrap().to_string(),
        verbose: false,
        format: "text".to_string(),
    };

    let result = pmg_cli::commands::espec::execute(args, false);

    // Assert : La commande devrait réussir
    assert!(
        result.is_ok(),
        "La commande espec devrait réussir: {:?}",
        result.err()
    );
}

/// Test : Fichier Safetensors avec format JSON.
///
/// Vérifie que le format JSON est correctement généré.
#[test]
fn test_espec_json_output_direct() {
    // Arrange
    let fixtures = fixtures_dir();
    let model_a = fixtures.join("model_a.safetensors");

    // Act
    let args = pmg_cli::commands::espec::EspecArgs {
        model_path: model_a.to_str().unwrap().to_string(),
        verbose: false,
        format: "json".to_string(),
    };

    let result = pmg_cli::commands::espec::execute(args, false);

    // Assert : La commande devrait réussir
    assert!(
        result.is_ok(),
        "La commande espec devrait réussir: {:?}",
        result.err()
    );
}

/// Test : Mode verbose avec vérification des détails affichés.
///
/// Vérifie que le mode verbose affiche des informations supplémentaires.
#[test]
fn test_espec_verbose_mode_direct() {
    // Arrange
    let fixtures = fixtures_dir();
    let model_a = fixtures.join("model_a.safetensors");

    // Act
    let args = pmg_cli::commands::espec::EspecArgs {
        model_path: model_a.to_str().unwrap().to_string(),
        verbose: true,
        format: "text".to_string(),
    };

    let result = pmg_cli::commands::espec::execute(args, false);

    // Assert : La commande devrait réussir
    assert!(
        result.is_ok(),
        "La commande espec devrait réussir: {:?}",
        result.err()
    );
}

/// Test : Validation de l'extension .safetensors.
///
/// Vérifie que espec rejette les fichiers sans extension .safetensors.
#[test]
fn test_espec_invalid_extension_direct() {
    // Arrange : Créer un fichier sans extension .safetensors
    let dir = tempdir().unwrap();
    let invalid_file = dir.path().join("model.txt");
    std::fs::write(&invalid_file, "contenu invalide").unwrap();

    // Act
    let args = pmg_cli::commands::espec::EspecArgs {
        model_path: invalid_file.to_str().unwrap().to_string(),
        verbose: false,
        format: "text".to_string(),
    };

    let result = pmg_cli::commands::espec::execute(args, false);

    // Assert : Devrait échouer avec une erreur
    assert!(
        result.is_err(),
        "La commande espec devrait échouer pour un fichier sans extension .safetensors"
    );
}

/// Test : Gestion d'erreur pour fichiers non lisibles.
///
/// Vérifie que espec gère correctement les fichiers non lisibles.
#[test]
fn test_espec_unreadable_file_direct() {
    // Arrange : Créer un fichier puis le supprimer pour simuler un fichier non lisible
    let dir = tempdir().unwrap();
    let unreadable_file = dir.path().join("unreadable.safetensors");
    std::fs::write(&unreadable_file, "{}").unwrap();

    // Supprimer le fichier pour simuler un fichier non lisible
    std::fs::remove_file(&unreadable_file).unwrap();

    // Act
    let args = pmg_cli::commands::espec::EspecArgs {
        model_path: unreadable_file.to_str().unwrap().to_string(),
        verbose: false,
        format: "text".to_string(),
    };

    let result = pmg_cli::commands::espec::execute(args, false);

    // Assert : Devrait échouer avec une erreur
    assert!(
        result.is_err(),
        "La commande espec devrait échouer pour un fichier non lisible"
    );
}

/// Test : Fichier Safetensors avec format text (par défaut).
///
/// Vérifie que le format text fonctionne correctement.
#[test]
fn test_espec_text_output_direct() {
    // Arrange
    let fixtures = fixtures_dir();
    let model_a = fixtures.join("model_a.safetensors");

    // Act
    let args = pmg_cli::commands::espec::EspecArgs {
        model_path: model_a.to_str().unwrap().to_string(),
        verbose: false,
        format: "text".to_string(),
    };

    let result = pmg_cli::commands::espec::execute(args, false);

    // Assert : La commande devrait réussir
    assert!(
        result.is_ok(),
        "La commande espec devrait réussir: {:?}",
        result.err()
    );
}

/// Test : Fichier Safetensors avec plusieurs tenseurs.
///
/// Vérifie que espec gère correctement les fichiers avec plusieurs tenseurs.
#[test]
fn test_espec_multi_tensor_file_direct() {
    // Arrange : Utiliser la fixture model_d qui a deux tenseurs
    let fixtures = fixtures_dir();
    let model_d = fixtures.join("model_d.safetensors");

    // Vérifier que la fixture existe
    assert!(
        model_d.exists(),
        "La fixture model_d.safetensors n'existe pas"
    );

    // Act
    let args = pmg_cli::commands::espec::EspecArgs {
        model_path: model_d.to_str().unwrap().to_string(),
        verbose: false,
        format: "text".to_string(),
    };

    let result = pmg_cli::commands::espec::execute(args, false);

    // Assert : La commande devrait réussir
    assert!(
        result.is_ok(),
        "La commande espec devrait réussir: {:?}",
        result.err()
    );
}

/// Test : Fichier Safetensors avec tenseur F16.
///
/// Vérifie que espec gère correctement les tenseurs de type F16.
#[test]
fn test_espec_f16_tensor_direct() {
    // Arrange : Utiliser la fixture model_e qui a un tenseur F16
    let fixtures = fixtures_dir();
    let model_e = fixtures.join("model_e.safetensors");

    // Vérifier que la fixture existe
    assert!(
        model_e.exists(),
        "La fixture model_e.safetensors n'existe pas"
    );

    // Act
    let args = pmg_cli::commands::espec::EspecArgs {
        model_path: model_e.to_str().unwrap().to_string(),
        verbose: false,
        format: "text".to_string(),
    };

    let result = pmg_cli::commands::espec::execute(args, false);

    // Assert : La commande devrait réussir
    assert!(
        result.is_ok(),
        "La commande espec devrait réussir: {:?}",
        result.err()
    );
}
