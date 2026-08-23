//! Tests de couverture pour la commande espec
//!
//! Ces tests améliorent la couverture de code pour espec.rs

use assert_cmd::Command;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

/// Helper pour obtenir le chemin vers une fixture
fn fixture_path(name: &str) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures");
    path.push(name);
    path
}

/// Test : extension en majuscules (rejetée par la commande)
#[test]
fn test_espec_extension_uppercase() {
    let temp = TempDir::new().unwrap();
    let model = temp.path().join("model.SAFETENSORS");

    // Copier une fixture existante
    fs::copy(fixture_path("model_a.safetensors"), &model).unwrap();

    // La commande espec rejettera l'extension en majuscules
    Command::cargo_bin("pmg")
        .unwrap()
        .args(&["espec", "--model-path", model.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicates::str::contains("PMG-2"));
}

/// Test : fichier vide
#[test]
fn test_espec_empty_file() {
    let temp = TempDir::new().unwrap();
    let model = temp.path().join("empty.safetensors");

    fs::write(&model, b"").unwrap();

    Command::cargo_bin("pmg")
        .unwrap()
        .args(&["espec", "--model-path", model.to_str().unwrap()])
        .assert()
        .failure();
}

/// Test : fichier corrompu
#[test]
fn test_espec_corrupted_safetensors() {
    let temp = TempDir::new().unwrap();
    let model = temp.path().join("corrupted.safetensors");

    fs::write(&model, b"not a valid safetensors file").unwrap();

    Command::cargo_bin("pmg")
        .unwrap()
        .args(&["espec", "--model-path", model.to_str().unwrap()])
        .assert()
        .failure();
}

/// Test : format de sortie invalide
#[test]
fn test_espec_invalid_format() {
    let temp = TempDir::new().unwrap();
    let model = temp.path().join("model.safetensors");

    // Copier une fixture existante
    fs::copy(fixture_path("model_a.safetensors"), &model).unwrap();

    Command::cargo_bin("pmg")
        .unwrap()
        .args(&[
            "espec",
            "--model-path",
            model.to_str().unwrap(),
            "--format",
            "invalid",
        ])
        .assert()
        .success(); // Devrait utiliser text par défaut
}

/// Test : répertoire au lieu de fichier
#[test]
fn test_espec_directory() {
    let temp = TempDir::new().unwrap();
    let dir = temp.path().join("subdir");
    fs::create_dir(&dir).unwrap();

    Command::cargo_bin("pmg")
        .unwrap()
        .args(&["espec", "--model-path", dir.to_str().unwrap()])
        .assert()
        .failure();
}

/// Test : chemin avec espaces
#[test]
fn test_espec_spaces_in_path() {
    let temp = TempDir::new().unwrap();
    let model = temp.path().join("model with spaces.safetensors");

    // Copier une fixture existante
    fs::copy(fixture_path("model_a.safetensors"), &model).unwrap();

    Command::cargo_bin("pmg")
        .unwrap()
        .args(&["espec", "--model-path", model.to_str().unwrap()])
        .assert()
        .success();
}

/// Test : sortie JSON
#[test]
fn test_espec_json_output() {
    let temp = TempDir::new().unwrap();
    let model = temp.path().join("model.safetensors");

    // Copier une fixture existante
    fs::copy(fixture_path("model_a.safetensors"), &model).unwrap();

    Command::cargo_bin("pmg")
        .unwrap()
        .args(&[
            "espec",
            "--model-path",
            model.to_str().unwrap(),
            "--format",
            "json",
        ])
        .assert()
        .success();
}

/// Test : mode verbose
#[test]
fn test_espec_verbose() {
    let temp = TempDir::new().unwrap();
    let model = temp.path().join("model.safetensors");

    // Copier une fixture existante
    fs::copy(fixture_path("model_a.safetensors"), &model).unwrap();

    Command::cargo_bin("pmg")
        .unwrap()
        .args(&[
            "espec",
            "--model-path",
            model.to_str().unwrap(),
            "--verbose",
        ])
        .assert()
        .success();
}
