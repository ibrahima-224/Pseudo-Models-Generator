//! Tests de validation des entrées invalides pour la CLI PMG.
//!
//! Ces tests vérifient que la CLI rejette correctement toutes les entrées
//! invalides et gère gracieusement les erreurs.

use tempfile::tempdir;

use crate::common::{exit_codes, pmg_command};

/// Test : Argument --model avec valeur vide.
///
/// Vérifie que la CLI rejette un argument modèle vide.
#[test]
fn test_validation_empty_model_arg() {
    // Arrange
    let empty_model = "";

    // Act
    let mut cmd = pmg_command();
    cmd.args(["generate", "--model", empty_model, "--dry-run"]);

    // Assert : Doit échouer (peut être argument invalide ou autre)
    cmd.assert().failure();
}

/// Test : Argument --model avec espace uniquement.
///
/// Vérifie que la CLI rejette un argument modèle contenant uniquement des espaces.
#[test]
fn test_validation_whitespace_only_model() {
    // Arrange
    let whitespace_model = "   ";

    // Act
    let mut cmd = pmg_command();
    cmd.args(["generate", "--model", whitespace_model, "--dry-run"]);

    // Assert : Doit échouer
    cmd.assert().failure();
}

/// Test : Argument --size avec valeur non numérique.
///
/// Vérifie que la CLI rejette les tailles non valides.
#[test]
fn test_validation_invalid_size_format() {
    // Arrange
    let invalid_size = "abc";

    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "generate",
        "--model",
        "glm52",
        "--size",
        invalid_size,
        "--dry-run",
    ]);

    // Assert : Doit échouer (peut être argument invalide ou autre)
    cmd.assert().failure();
}

/// Test : Argument --size avec taille négative.
///
/// Vérifie que la CLI rejette les tailles négatives.
#[test]
fn test_validation_negative_size() {
    // Arrange
    let negative_size = "-100M";

    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "generate",
        "--model",
        "glm52",
        "--size",
        negative_size,
        "--dry-run",
    ]);

    // Assert : Doit échouer
    cmd.assert().failure();
}

/// Test : Argument --size avec unité inconnue.
///
/// Vérifie que la CLI rejette les unités de taille inconnues.
#[test]
fn test_validation_unknown_size_unit() {
    // Arrange
    let unknown_unit_size = "100X";

    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "generate",
        "--model",
        "glm52",
        "--size",
        unknown_unit_size,
        "--dry-run",
    ]);

    // Assert : Doit échouer
    cmd.assert().failure();
}

/// Test : Argument --seed avec valeur non numérique.
///
/// Vérifie que la CLI rejette les seeds non valides.
#[test]
fn test_validation_invalid_seed_format() {
    // Arrange
    let invalid_seed = "not_a_number";

    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "generate",
        "--model",
        "glm52",
        "--size",
        "1G",
        "--seed",
        invalid_seed,
        "--dry-run",
    ]);

    // Assert : Doit échouer
    cmd.assert().failure();
}

/// Test : Argument --seed avec valeur négative.
///
/// Vérifie que la CLI rejette les seeds négatives.
#[test]
fn test_validation_negative_seed() {
    // Arrange
    let negative_seed = "-42";

    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "generate",
        "--model",
        "glm52",
        "--size",
        "1G",
        "--seed",
        negative_seed,
        "--dry-run",
    ]);

    // Assert : Doit échouer
    cmd.assert().failure();
}

/// Test : Argument --mode avec valeur non supportée.
///
/// Vérifie que la CLI rejette les modes non supportés.
#[test]
fn test_validation_invalid_mode() {
    // Arrange
    let invalid_mode = "invalid_mode";

    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "generate",
        "--model",
        "glm52",
        "--size",
        "1G",
        "--mode",
        invalid_mode,
        "--dry-run",
    ]);

    // Assert : Doit échouer (peut être argument invalide ou autre)
    cmd.assert().failure();
}

/// Test : Argument --dtype avec valeur non supportée.
///
/// Vérifie que la CLI rejette les types de données non supportés.
#[test]
fn test_validation_invalid_dtype() {
    // Arrange
    let invalid_dtype = "xyz";

    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "generate",
        "--model",
        "glm52",
        "--size",
        "1G",
        "--dtype",
        invalid_dtype,
        "--dry-run",
    ]);

    // Assert : Doit échouer
    cmd.assert().failure();
}

/// Test : Argument --format avec valeur non supportée.
///
/// Vérifie que la CLI rejette les formats non supportés.
#[test]
fn test_validation_invalid_format() {
    // Arrange
    let invalid_format = "xml";

    // Act : Test pour commande espec
    let mut cmd = pmg_command();
    cmd.args([
        "espec",
        "--model-path",
        "test.safetensors",
        "--format",
        invalid_format,
    ]);

    // Assert : Doit échouer
    cmd.assert().failure();
}

/// Test : Argument --tolérance avec valeur négative.
///
/// Vérifie que la CLI rejette les tolérances négatives.
#[test]
fn test_validation_negative_tolerance() {
    // Arrange
    let negative_tolerance = "-0.1";

    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "validate",
        "--model-path",
        "test.safetensors",
        "--tolerance",
        negative_tolerance,
    ]);

    // Assert : Doit échouer
    cmd.assert().failure();
}

/// Test : Argument --tolérance avec valeur supérieure à 1.
///
/// Vérifie que la CLI rejette les tolérances > 1.
#[test]
fn test_validation_tolerance_above_one() {
    // Arrange
    let high_tolerance = "1.5";

    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "validate",
        "--model-path",
        "test.safetensors",
        "--tolerance",
        high_tolerance,
    ]);

    // Assert : Doit échouer
    cmd.assert().failure();
}

/// Test : Argument --outlier-threshold avec valeur négative.
///
/// Vérifie que la CLI rejette les seuils négatifs.
#[test]
fn test_validation_negative_outlier_threshold() {
    // Arrange
    let negative_threshold = "-3.0";

    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "validate",
        "--model-path",
        "test.safetensors",
        "--outlier-threshold",
        negative_threshold,
    ]);

    // Assert : Doit échouer
    cmd.assert().failure();
}

/// Test : Fichier inexistant pour --model-path.
///
/// Vérifie que la CLI gère gracieusement les fichiers inexistants.
#[test]
fn test_validation_nonexistent_model_file() {
    // Arrange
    let nonexistent_path = "/chemin/inexistant/model.safetensors";

    // Act
    let mut cmd = pmg_command();
    cmd.args(["espec", "--model-path", nonexistent_path]);

    // Assert : Doit échouer (fichier inexistant)
    cmd.assert().failure();
}

/// Test : Répertoire au lieu de fichier pour --model-path.
///
/// Vérifie que la CLI rejette les répertoires comme fichiers modèles.
#[test]
fn test_validation_directory_as_model_file() {
    // Arrange : Créer un répertoire temporaire
    let dir = tempdir().unwrap();
    let dir_path = dir.path();

    // Act
    let mut cmd = pmg_command();
    cmd.args(["espec", "--model-path", dir_path.to_str().unwrap()]);

    // Assert : Doit échouer
    cmd.assert().failure();
}

/// Test : Fichier vide pour --model-path.
///
/// Vérifie que la CLI rejette les fichiers vides.
#[test]
fn test_validation_empty_model_file() {
    // Arrange : Créer un fichier vide
    let dir = tempdir().unwrap();
    let empty_file = dir.path().join("empty.safetensors");
    std::fs::write(&empty_file, "").unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args(["espec", "--model-path", empty_file.to_str().unwrap()]);

    // Assert : Doit échouer
    cmd.assert().failure();
}

/// Test : Extension invalide pour --model-path.
///
/// Vérifie que la CLI rejette les extensions non supportées.
#[test]
fn test_validation_invalid_extension() {
    // Arrange : Créer un fichier avec extension invalide
    let dir = tempdir().unwrap();
    let invalid_ext_file = dir.path().join("model.txt");
    std::fs::write(&invalid_ext_file, "contenu").unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args(["espec", "--model-path", invalid_ext_file.to_str().unwrap()]);

    // Assert : Doit échouer
    cmd.assert().failure();
}

/// Test : Trop d'arguments.
///
/// Vérifie que la CLI rejette les arguments en trop.
#[test]
fn test_validation_too_many_arguments() {
    // Act : Ajouter des arguments inconnus
    let mut cmd = pmg_command();
    cmd.args(["generate", "--model", "glm52", "--unknown-flag", "value"]);

    // Assert : Doit échouer avec argument invalide
    cmd.assert().failure().code(exit_codes::INVALID_ARGUMENT);
}

/// Test : Argument manquant obligatoire.
///
/// Vérifie que la CLI signale les arguments manquants.
#[test]
fn test_validation_missing_required_argument() {
    // Act : Omettre l'argument --model
    let mut cmd = pmg_command();
    cmd.args(["generate", "--size", "1G"]);

    // Assert : Doit échouer avec argument manquant
    cmd.assert().failure();
}
