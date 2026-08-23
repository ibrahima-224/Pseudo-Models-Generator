//! Tests de couverture pour la commande compare
//!
//! Ces tests améliorent la couverture de code pour compare.rs

use assert_cmd::Command;
use predicates::prelude::*;
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

/// Test : fichier comparé inexistant
#[test]
fn test_compare_nonexistent_compared() {
    let temp = TempDir::new().unwrap();
    let model_a = temp.path().join("model_a.safetensors");
    let model_b = temp.path().join("nonexistent.safetensors");

    // Copier une fixture existante comme fichier source valide
    fs::copy(fixture_path("model_a.safetensors"), &model_a).unwrap();

    Command::cargo_bin("pmg")
        .unwrap()
        .args(&[
            "compare",
            "--original",
            model_a.to_str().unwrap(),
            "--compared",
            model_b.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("PMG-4"));
}

/// Test : les deux chemins vides
#[test]
fn test_compare_empty_both_paths() {
    Command::cargo_bin("pmg")
        .unwrap()
        .args(&["compare", "--original", "", "--compared", ""])
        .assert()
        .failure()
        .stderr(predicate::str::contains("PMG-2"));
}

/// Test : tolérance = 0.0
#[test]
fn test_compare_tolerance_zero() {
    let temp = TempDir::new().unwrap();
    let model_a = temp.path().join("model_a.safetensors");
    let model_b = temp.path().join("model_b.safetensors");

    // Utiliser des fixtures valides
    fs::copy(fixture_path("model_a.safetensors"), &model_a).unwrap();
    fs::copy(fixture_path("model_b.safetensors"), &model_b).unwrap();

    Command::cargo_bin("pmg")
        .unwrap()
        .args(&[
            "compare",
            "--original",
            model_a.to_str().unwrap(),
            "--compared",
            model_b.to_str().unwrap(),
            "--tolerance",
            "0.0",
        ])
        .assert()
        .success();
}

/// Test : tolérance négative
#[test]
fn test_compare_tolerance_negative() {
    let temp = TempDir::new().unwrap();
    let model_a = temp.path().join("model_a.safetensors");
    let model_b = temp.path().join("model_b.safetensors");

    fs::copy(fixture_path("model_a.safetensors"), &model_a).unwrap();
    fs::copy(fixture_path("model_b.safetensors"), &model_b).unwrap();

    Command::cargo_bin("pmg")
        .unwrap()
        .args(&[
            "compare",
            "--original",
            model_a.to_str().unwrap(),
            "--compared",
            model_b.to_str().unwrap(),
            "--tolerance",
            "-0.5",
        ])
        .assert()
        .failure();
}

/// Test : tolérance > 1.0
#[test]
fn test_compare_tolerance_over_one() {
    let temp = TempDir::new().unwrap();
    let model_a = temp.path().join("model_a.safetensors");
    let model_b = temp.path().join("model_b.safetensors");

    fs::copy(fixture_path("model_a.safetensors"), &model_a).unwrap();
    fs::copy(fixture_path("model_b.safetensors"), &model_b).unwrap();

    // La tolérance n'est pas validée par la commande, donc elle devrait réussir
    Command::cargo_bin("pmg")
        .unwrap()
        .args(&[
            "compare",
            "--original",
            model_a.to_str().unwrap(),
            "--compared",
            model_b.to_str().unwrap(),
            "--tolerance",
            "1.5",
        ])
        .assert()
        .success();
}

/// Test : les deux fichiers sont corrompus
#[test]
fn test_compare_both_files_corrupted() {
    let temp = TempDir::new().unwrap();
    let model_a = temp.path().join("corrupted_a.bin");
    let model_b = temp.path().join("corrupted_b.bin");

    fs::write(&model_a, b"not a safetensors").unwrap();
    fs::write(&model_b, b"also not a safetensors").unwrap();

    Command::cargo_bin("pmg")
        .unwrap()
        .args(&[
            "compare",
            "--original",
            model_a.to_str().unwrap(),
            "--compared",
            model_b.to_str().unwrap(),
        ])
        .assert()
        .failure();
}

/// Test : fichiers vides
#[test]
fn test_compare_empty_files() {
    let temp = TempDir::new().unwrap();
    let model_a = temp.path().join("empty_a.safetensors");
    let model_b = temp.path().join("empty_b.safetensors");

    fs::write(&model_a, b"").unwrap();
    fs::write(&model_b, b"").unwrap();

    Command::cargo_bin("pmg")
        .unwrap()
        .args(&[
            "compare",
            "--original",
            model_a.to_str().unwrap(),
            "--compared",
            model_b.to_str().unwrap(),
        ])
        .assert()
        .failure();
}

/// Test : format de sortie invalide
#[test]
fn test_compare_invalid_format() {
    let temp = TempDir::new().unwrap();
    let model_a = temp.path().join("model_a.safetensors");
    let model_b = temp.path().join("model_b.safetensors");

    // Utiliser des fixtures valides
    fs::copy(fixture_path("model_a.safetensors"), &model_a).unwrap();
    fs::copy(fixture_path("model_b.safetensors"), &model_b).unwrap();

    Command::cargo_bin("pmg")
        .unwrap()
        .args(&[
            "compare",
            "--original",
            model_a.to_str().unwrap(),
            "--compared",
            model_b.to_str().unwrap(),
            "--format",
            "invalid",
        ])
        .assert()
        .success(); // Devrait utiliser text par défaut
}
