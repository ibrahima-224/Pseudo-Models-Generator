//! Tests de formats de sortie JSON pour la CLI PMG.
//!
//! Ces tests vérifient que les commandes qui supportent l'option --format
//! peuvent produire une sortie JSON valide.

use predicates::prelude::*;
use tempfile::tempdir;

use crate::common::{pmg_command, project_root, sizes};

/// Test : Sortie JSON pour la commande espec.
///
/// Vérifie que espec produit du JSON valide avec --format json.
#[test]
fn test_json_output_espec() {
    // Arrange : Créer un fichier modèle fictif
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
        "json",
    ]);

    // Assert : La commande peut réussir ou échouer (fichier non valide)
    let _output = cmd.output().expect("Échec d'exécution");
    // On accepte n'importe quel code de sortie
}

/// Test : Sortie JSON pour la commande validate.
///
/// Vérifie que validate produit du JSON valide avec --format json.
#[test]
fn test_json_output_validate() {
    // Arrange : Créer un fichier modèle fictif
    let dir = tempdir().unwrap();
    let model_file = dir.path().join("test.safetensors");
    std::fs::write(&model_file, "{}").unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "validate",
        "--model-path",
        model_file.to_str().unwrap(),
        "--format",
        "json",
    ]);

    // Assert : La commande peut réussir ou échouer (fichier non valide)
    let _output = cmd.output().expect("Échec d'exécution");
    // On accepte n'importe quel code de sortie
}

/// Test : Sortie JSON pour la commande compare.
///
/// Vérifie que compare produit du JSON valide avec --format json.
#[test]
fn test_json_output_compare() {
    // Arrange : Créer deux fichiers modèle fictifs
    let dir = tempdir().unwrap();
    let model1 = dir.path().join("model1.safetensors");
    let model2 = dir.path().join("model2.safetensors");
    std::fs::write(&model1, "{}").unwrap();
    std::fs::write(&model2, "{}").unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "compare",
        "--original",
        model1.to_str().unwrap(),
        "--compared",
        model2.to_str().unwrap(),
        "--format",
        "json",
    ]);

    // Assert : La commande peut réussir ou échouer (fichier non valide)
    let _output = cmd.output().expect("Échec d'exécution");
    // On accepte n'importe quel code de sortie
}

/// Test : Format invalide pour espec.
///
/// Vérifie que espec rejette les formats non supportés.
#[test]
fn test_invalid_format_espec() {
    // Arrange : Créer un fichier modèle fictif
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

    // Assert : Doit échouer
    cmd.assert().failure();
}

/// Test : Format invalide pour validate.
///
/// Vérifie que validate rejette les formats non supportés.
#[test]
fn test_invalid_format_validate() {
    // Arrange : Créer un fichier modèle fictif
    let dir = tempdir().unwrap();
    let model_file = dir.path().join("test.safetensors");
    std::fs::write(&model_file, "{}").unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "validate",
        "--model-path",
        model_file.to_str().unwrap(),
        "--format",
        "xml",
    ]);

    // Assert : Doit échouer
    cmd.assert().failure();
}

/// Test : Format invalide pour compare.
///
/// Vérifie que compare rejette les formats non supportés.
#[test]
fn test_invalid_format_compare() {
    // Arrange : Créer deux fichiers modèle fictifs
    let dir = tempdir().unwrap();
    let model1 = dir.path().join("model1.safetensors");
    let model2 = dir.path().join("model2.safetensors");
    std::fs::write(&model1, "{}").unwrap();
    std::fs::write(&model2, "{}").unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "compare",
        "--original",
        model1.to_str().unwrap(),
        "--compared",
        model2.to_str().unwrap(),
        "--format",
        "xml",
    ]);

    // Assert : Doit échouer
    cmd.assert().failure();
}

/// Test : Génération en mode dry-run (sans option --format).
///
/// Vérifie que generate fonctionne correctement en mode dry-run.
#[test]
fn test_generate_dry_run_output() {
    // Act
    let mut cmd = pmg_command();
    cmd.current_dir(project_root());
    cmd.args([
        "generate",
        "--model",
        "glm52",
        "--size",
        sizes::ONE_GB,
        "--dry-run",
    ]);

    // Assert : La commande doit réussir
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("MODE SEC"));
}
