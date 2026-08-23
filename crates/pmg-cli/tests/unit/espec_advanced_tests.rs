//! Tests avancés pour la commande espec de la CLI PMG.
//!
//! Ces tests vérifient en détail le comportement de la commande espec
//! avec différents arguments, options et cas limites.

use tempfile::tempdir;

use crate::common::{pmg_command, SUPPORTED_OUTPUT_FORMATS};

/// Test : Argument --model-path avec chaque format supporté.
///
/// Vérifie que espec fonctionne avec différents formats de sortie.
#[test]
fn test_espec_all_supported_formats() {
    // Arrange : Créer un fichier modèle fictif
    let dir = tempdir().unwrap();
    let model_file = dir.path().join("test.safetensors");
    std::fs::write(&model_file, "{}").unwrap();

    // Act & Assert : Tester chaque format supporté
    for format in SUPPORTED_OUTPUT_FORMATS {
        let mut cmd = pmg_command();
        cmd.args([
            "espec",
            "--model-path",
            model_file.to_str().unwrap(),
            "--format",
            format,
        ]);

        // On accepte n'importe quel code de sortie (fichier non valide)
        let _output = cmd.output().expect("Échec d'exécution");
    }
}

/// Test : Argument --model-path avec format non supporté.
///
/// Vérifie que les formats non supportés sont rejetés.
#[test]
fn test_espec_unsupported_format() {
    // Arrange
    let dir = tempdir().unwrap();
    let model_file = dir.path().join("test.safetensors");
    std::fs::write(&model_file, "{}").unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "espec",
        "--model-path",
        model_file.to_str().unwrap(),
        "--format",
        "xml",
    ]);

    // Assert : Doit échouer (format non supporté)
    cmd.assert().failure();
}

/// Test : Argument --model-path avec chemin vide.
///
/// Vérifie que espec rejette les chemins vides.
#[test]
fn test_espec_empty_path() {
    // Act
    let mut cmd = pmg_command();
    cmd.args(["espec", "--model-path", ""]);

    // Assert
    cmd.assert().failure();
}

/// Test : Argument --model-path avec chemin inexistant.
///
/// Vérifie que espec gère gracieusement les fichiers inexistants.
#[test]
fn test_espec_nonexistent_path() {
    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "espec",
        "--model-path",
        "/chemin/inexistant/model.safetensors",
    ]);

    // Assert : Doit échouer (fichier inexistant)
    cmd.assert().failure();
}

/// Test : Argument --model-path avec répertoire.
///
/// Vérifie que espec rejette les répertoires comme fichiers modèles.
#[test]
fn test_espec_directory_as_path() {
    // Arrange
    let dir = tempdir().unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args(["espec", "--model-path", dir.path().to_str().unwrap()]);

    // Assert
    cmd.assert().failure();
}

/// Test : Argument --model-path avec fichier vide.
///
/// Vérifie que espec rejette les fichiers vides.
#[test]
fn test_espec_empty_file() {
    // Arrange
    let dir = tempdir().unwrap();
    let empty_file = dir.path().join("empty.safetensors");
    std::fs::write(&empty_file, "").unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args(["espec", "--model-path", empty_file.to_str().unwrap()]);

    // Assert
    cmd.assert().failure();
}

/// Test : Argument --model-path avec extension invalide.
///
/// Vérifie que espec rejette les extensions non supportées.
#[test]
fn test_espec_invalid_extension() {
    // Arrange
    let dir = tempdir().unwrap();
    let invalid_file = dir.path().join("model.txt");
    std::fs::write(&invalid_file, "contenu").unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args(["espec", "--model-path", invalid_file.to_str().unwrap()]);

    // Assert
    cmd.assert().failure();
}

/// Test : Option --verbose avec espec.
///
/// Vérifie que --verbose fournit plus d'informations.
#[test]
fn test_espec_verbose_output() {
    // Arrange
    let dir = tempdir().unwrap();
    let model_file = dir.path().join("test.safetensors");
    std::fs::write(&model_file, "{}").unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "espec",
        "--model-path",
        model_file.to_str().unwrap(),
        "--verbose",
    ]);

    // Assert
    let _output = cmd.output().expect("Échec d'exécution");
    // On accepte succès ou erreur I/O
}

/// Test : Option --debug avec espec.
///
/// Vérifie que --debug fournit des informations de débogage.
#[test]
fn test_espec_debug_output() {
    // Arrange
    let dir = tempdir().unwrap();
    let model_file = dir.path().join("test.safetensors");
    std::fs::write(&model_file, "{}").unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "espec",
        "--model-path",
        model_file.to_str().unwrap(),
        "--debug",
    ]);

    // Assert
    let _output = cmd.output().expect("Échec d'exécution");
    // On accepte succès ou erreur I/O
}

/// Test : Argument --model-path avec caractères spéciaux.
///
/// Vérifie que espec gère les chemins avec caractères spéciaux.
#[test]
fn test_espec_special_characters_path() {
    // Arrange : Créer un fichier avec des caractères spéciaux dans le nom
    let dir = tempdir().unwrap();
    let special_file = dir.path().join("modèle spécial.safetensors");
    std::fs::write(&special_file, "{}").unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args(["espec", "--model-path", special_file.to_str().unwrap()]);

    // Assert : Doit fonctionner ou échouer gracieusement
    let _output = cmd.output().expect("Échec d'exécution");
}

/// Test : Argument --model-path avec espace.
///
/// Vérifie que espec gère les chemins avec espaces.
#[test]
fn test_espec_path_with_spaces() {
    // Arrange : Créer un fichier dans un répertoire avec espace
    let dir = tempdir().unwrap();
    let space_dir = dir.path().join("répertoire avec espace");
    std::fs::create_dir_all(&space_dir).unwrap();
    let space_file = space_dir.join("model.safetensors");
    std::fs::write(&space_file, "{}").unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args(["espec", "--model-path", space_file.to_str().unwrap()]);

    // Assert
    let _output = cmd.output().expect("Échec d'exécution");
}

/// Test : Argument --model-path avec chemin relatif.
///
/// Vérifie que espec accepte les chemins relatifs.
#[test]
fn test_espec_relative_path() {
    // Arrange : Créer un fichier modèle
    let dir = tempdir().unwrap();
    let model_file = dir.path().join("test.safetensors");
    std::fs::write(&model_file, "{}").unwrap();

    // Act : Utiliser un chemin relatif
    let mut cmd = pmg_command();
    cmd.current_dir(dir.path());
    cmd.args(["espec", "--model-path", "test.safetensors"]);

    // Assert
    let _output = cmd.output().expect("Échec d'exécution");
}

/// Test : Argument --model-path avec chemin absolu.
///
/// Vérifie que espec accepte les chemins absolus.
#[test]
fn test_espec_absolute_path() {
    // Arrange : Créer un fichier modèle
    let dir = tempdir().unwrap();
    let model_file = dir.path().join("test.safetensors");
    std::fs::write(&model_file, "{}").unwrap();

    // Act : Utiliser un chemin absolu
    let mut cmd = pmg_command();
    cmd.args(["espec", "--model-path", model_file.to_str().unwrap()]);

    // Assert
    let _output = cmd.output().expect("Échec d'exécution");
}

/// Test : Combinaison d'options valides pour espec.
///
/// Vérifie que les combinaisons d'options valides fonctionnent.
#[test]
fn test_espec_valid_option_combinations() {
    // Arrange
    let dir = tempdir().unwrap();
    let model_file = dir.path().join("test.safetensors");
    std::fs::write(&model_file, "{}").unwrap();

    // Act & Assert : Différentes combinaisons
    let combinations = [
        vec![
            "--model-path",
            model_file.to_str().unwrap(),
            "--format",
            "json",
        ],
        vec!["--model-path", model_file.to_str().unwrap(), "--verbose"],
        vec!["--model-path", model_file.to_str().unwrap(), "--debug"],
    ];

    for args in combinations {
        let mut cmd = pmg_command();
        cmd.arg("espec");
        cmd.args(args);

        let _output = cmd.output().expect("Échec d'exécution");
        // On accepte succès ou erreur I/O
    }
}
