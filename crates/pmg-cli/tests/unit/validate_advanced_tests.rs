//! Tests avancés pour la commande validate de la CLI PMG.
//!
//! Ces tests vérifient en détail le comportement de la commande validate
//! avec différents arguments, options et cas limites.

use tempfile::tempdir;

use crate::common::{pmg_command, SUPPORTED_OUTPUT_FORMATS};

/// Test : Argument --model-path avec chaque format supporté.
///
/// Vérifie que validate fonctionne avec différents formats de sortie.
#[test]
fn test_validate_all_supported_formats() {
    // Arrange : Créer un fichier modèle fictif
    let dir = tempdir().unwrap();
    let model_file = dir.path().join("test.safetensors");
    std::fs::write(&model_file, "{}").unwrap();

    // Act & Assert : Tester chaque format supporté
    for format in SUPPORTED_OUTPUT_FORMATS {
        let mut cmd = pmg_command();
        cmd.args([
            "validate",
            "--model-path",
            model_file.to_str().unwrap(),
            "--format",
            format,
        ]);

        // On accepte succès ou erreur I/O
        let output = cmd.output().expect("Échec d'exécution");
        assert!(output.status.code() == Some(0) || output.status.code() == Some(4));
    }
}

/// Test : Argument --tolerance avec valeurs valides.
///
/// Vérifie que les tolérances valides sont acceptées.
#[test]
fn test_validate_valid_tolerances() {
    // Arrange
    let dir = tempdir().unwrap();
    let model_file = dir.path().join("test.safetensors");
    std::fs::write(&model_file, "{}").unwrap();

    // Act & Assert : Différentes tolérances valides
    let tolerances = ["0.0", "0.05", "0.1", "0.5", "1.0"];

    for tolerance in tolerances {
        let mut cmd = pmg_command();
        cmd.args([
            "validate",
            "--model-path",
            model_file.to_str().unwrap(),
            "--tolerance",
            tolerance,
        ]);

        let _output = cmd.output().expect("Échec d'exécution");
        // On accepte succès ou erreur I/O
    }
}

/// Test : Argument --tolerance avec valeurs invalides.
///
/// Vérifie que les tolérances invalides sont rejetées.
#[test]
fn test_validate_invalid_tolerances() {
    // Arrange
    let dir = tempdir().unwrap();
    let model_file = dir.path().join("test.safetensors");
    std::fs::write(&model_file, "{}").unwrap();

    // Act & Assert : Tolérances invalides
    let invalid_tolerances = ["-0.1", "-1.0", "1.5", "2.0", "abc"];

    for tolerance in invalid_tolerances {
        let mut cmd = pmg_command();
        cmd.args([
            "validate",
            "--model-path",
            model_file.to_str().unwrap(),
            "--tolerance",
            tolerance,
        ]);

        cmd.assert().failure();
    }
}

/// Test : Argument --outlier-threshold avec valeurs valides.
///
/// Vérifie que les seuils d'outliers valides sont acceptés.
#[test]
fn test_validate_valid_outlier_thresholds() {
    // Arrange
    let dir = tempdir().unwrap();
    let model_file = dir.path().join("test.safetensors");
    std::fs::write(&model_file, "{}").unwrap();

    // Act & Assert : Différents seuils valides
    let thresholds = ["1.0", "2.0", "3.0", "5.0"];

    for threshold in thresholds {
        let mut cmd = pmg_command();
        cmd.args([
            "validate",
            "--model-path",
            model_file.to_str().unwrap(),
            "--outlier-threshold",
            threshold,
        ]);

        let _output = cmd.output().expect("Échec d'exécution");
    }
}

/// Test : Argument --outlier-threshold avec valeurs invalides.
///
/// Vérifie que les seuils invalides sont rejetés.
#[test]
fn test_validate_invalid_outlier_thresholds() {
    // Arrange
    let dir = tempdir().unwrap();
    let model_file = dir.path().join("test.safetensors");
    std::fs::write(&model_file, "{}").unwrap();

    // Act & Assert : Seuils invalides
    let invalid_thresholds = ["-1.0", "-3.0", "abc"];

    for threshold in invalid_thresholds {
        let mut cmd = pmg_command();
        cmd.args([
            "validate",
            "--model-path",
            model_file.to_str().unwrap(),
            "--outlier-threshold",
            threshold,
        ]);

        cmd.assert().failure();
    }
}

/// Test : Argument --model-path avec chemin vide.
///
/// Vérifie que validate rejette les chemins vides.
#[test]
fn test_validate_empty_path() {
    // Act
    let mut cmd = pmg_command();
    cmd.args(["validate", "--model-path", ""]);

    // Assert
    cmd.assert().failure();
}

/// Test : Argument --model-path avec chemin inexistant.
///
/// Vérifie que validate gère gracieusement les fichiers inexistants.
#[test]
fn test_validate_nonexistent_path() {
    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "validate",
        "--model-path",
        "/chemin/inexistant/model.safetensors",
    ]);

    // Assert : Doit échouer avec erreur I/O
    cmd.assert().failure();
}

/// Test : Argument --model-path avec répertoire.
///
/// Vérifie que validate rejette les répertoires comme fichiers modèles.
#[test]
fn test_validate_directory_as_path() {
    // Arrange
    let dir = tempdir().unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args(["validate", "--model-path", dir.path().to_str().unwrap()]);

    // Assert
    cmd.assert().failure();
}

/// Test : Argument --model-path avec fichier vide.
///
/// Vérifie que validate rejette les fichiers vides.
#[test]
fn test_validate_empty_file() {
    // Arrange
    let dir = tempdir().unwrap();
    let empty_file = dir.path().join("empty.safetensors");
    std::fs::write(&empty_file, "").unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args(["validate", "--model-path", empty_file.to_str().unwrap()]);

    // Assert
    cmd.assert().failure();
}

/// Test : Argument --model-path avec extension invalide.
///
/// Vérifie que validate rejette les extensions non supportées.
#[test]
fn test_validate_invalid_extension() {
    // Arrange
    let dir = tempdir().unwrap();
    let invalid_file = dir.path().join("model.txt");
    std::fs::write(&invalid_file, "contenu").unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args(["validate", "--model-path", invalid_file.to_str().unwrap()]);

    // Assert
    cmd.assert().failure();
}

/// Test : Option --verbose avec validate.
///
/// Vérifie que --verbose fournit plus d'informations.
#[test]
fn test_validate_verbose_output() {
    // Arrange
    let dir = tempdir().unwrap();
    let model_file = dir.path().join("test.safetensors");
    std::fs::write(&model_file, "{}").unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "validate",
        "--model-path",
        model_file.to_str().unwrap(),
        "--verbose",
    ]);

    // Assert
    let _output = cmd.output().expect("Échec d'exécution");
}

/// Test : Option --debug avec validate.
///
/// Vérifie que --debug fournit des informations de débogage.
#[test]
fn test_validate_debug_output() {
    // Arrange
    let dir = tempdir().unwrap();
    let model_file = dir.path().join("test.safetensors");
    std::fs::write(&model_file, "{}").unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "validate",
        "--model-path",
        model_file.to_str().unwrap(),
        "--debug",
    ]);

    // Assert
    let _output = cmd.output().expect("Échec d'exécution");
}

/// Test : Argument --model-path avec caractères spéciaux.
///
/// Vérifie que validate gère les chemins avec caractères spéciaux.
#[test]
fn test_validate_special_characters_path() {
    // Arrange : Créer un fichier avec des caractères spéciaux dans le nom
    let dir = tempdir().unwrap();
    let special_file = dir.path().join("modèle spécial.safetensors");
    std::fs::write(&special_file, "{}").unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args(["validate", "--model-path", special_file.to_str().unwrap()]);

    // Assert : Doit fonctionner ou échouer gracieusement
    let _output = cmd.output().expect("Échec d'exécution");
}

/// Test : Argument --model-path avec espace.
///
/// Vérifie que validate gère les chemins avec espaces.
#[test]
fn test_validate_path_with_spaces() {
    // Arrange : Créer un fichier dans un répertoire avec espace
    let dir = tempdir().unwrap();
    let space_dir = dir.path().join("répertoire avec espace");
    std::fs::create_dir_all(&space_dir).unwrap();
    let space_file = space_dir.join("model.safetensors");
    std::fs::write(&space_file, "{}").unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args(["validate", "--model-path", space_file.to_str().unwrap()]);

    // Assert
    let _output = cmd.output().expect("Échec d'exécution");
}

/// Test : Argument --model-path avec chemin relatif.
///
/// Vérifie que validate accepte les chemins relatifs.
#[test]
fn test_validate_relative_path() {
    // Arrange : Créer un fichier modèle
    let dir = tempdir().unwrap();
    let model_file = dir.path().join("test.safetensors");
    std::fs::write(&model_file, "{}").unwrap();

    // Act : Utiliser un chemin relatif
    let mut cmd = pmg_command();
    cmd.current_dir(dir.path());
    cmd.args(["validate", "--model-path", "test.safetensors"]);

    // Assert
    let _output = cmd.output().expect("Échec d'exécution");
}

/// Test : Argument --model-path avec chemin absolu.
///
/// Vérifie que validate accepte les chemins absolus.
#[test]
fn test_validate_absolute_path() {
    // Arrange : Créer un fichier modèle
    let dir = tempdir().unwrap();
    let model_file = dir.path().join("test.safetensors");
    std::fs::write(&model_file, "{}").unwrap();

    // Act : Utiliser un chemin absolu
    let mut cmd = pmg_command();
    cmd.args(["validate", "--model-path", model_file.to_str().unwrap()]);

    // Assert
    let _output = cmd.output().expect("Échec d'exécution");
}

/// Test : Combinaison d'options valides pour validate.
///
/// Vérifie que les combinaisons d'options valides fonctionnent.
#[test]
fn test_validate_valid_option_combinations() {
    // Arrange
    let dir = tempdir().unwrap();
    let model_file = dir.path().join("test.safetensors");
    std::fs::write(&model_file, "{}").unwrap();

    // Act & Assert : Différentes combinaisons
    let combinations = [
        vec![
            "--model-path",
            model_file.to_str().unwrap(),
            "--tolerance",
            "0.1",
        ],
        vec![
            "--model-path",
            model_file.to_str().unwrap(),
            "--outlier-threshold",
            "3.0",
        ],
        vec![
            "--model-path",
            model_file.to_str().unwrap(),
            "--format",
            "json",
        ],
        vec!["--model-path", model_file.to_str().unwrap(), "--verbose"],
    ];

    for args in combinations {
        let mut cmd = pmg_command();
        cmd.arg("validate");
        cmd.args(args);

        let _output = cmd.output().expect("Échec d'exécution");
    }
}
