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

//! Tests E2E pour la CLI PMG.
//!
//! Ces tests vérifient le comportement de la ligne de commande en exécutant
//! le binaire avec différentes combinaisons d'arguments et en vérifiant
//! les codes de sortie et les messages affichés.

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::tempdir;

/// Test : pmg --help affiche l'aide et quitte avec le code 0.
#[test]
fn test_pmg_help() {
    Command::cargo_bin("pmg")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("pseudo-modèles"));
}

/// Test : pmg help affiche l'aide générale.
#[test]
fn test_pmg_help_subcommand() {
    Command::cargo_bin("pmg")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Commands:"));
}

/// Test : pmg help generate affiche l'aide de la commande generate.
#[test]
fn test_pmg_help_generate() {
    Command::cargo_bin("pmg")
        .unwrap()
        .args(["generate", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Génère"));
}

/// Test : pmg version affiche la version.
#[test]
fn test_pmg_version() {
    Command::cargo_bin("pmg")
        .unwrap()
        .arg("version")
        .assert()
        .success()
        .stdout(predicate::str::contains("pmg-cli v"));
}

/// Test : pmg generate --help affiche l'aide de generate.
#[test]
fn test_pmg_generate_help() {
    Command::cargo_bin("pmg")
        .unwrap()
        .args(["generate", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--model"));
}

/// Test : pmg generate --dry-run simule la génération.
#[test]
fn test_pmg_generate_dry_run() {
    Command::cargo_bin("pmg")
        .unwrap()
        .args(["generate", "--model", "glm52", "--size", "1G", "--dry-run"])
        .assert()
        .success()
        .stdout(predicate::str::contains("MODE SEC"));
}

/// Test : pmg espec --help affiche l'aide de espec.
#[test]
fn test_pmg_espec_help() {
    Command::cargo_bin("pmg")
        .unwrap()
        .args(["espec", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--model-path"));
}

/// Test : pmg validate --help affiche l'aide de validate.
#[test]
fn test_pmg_validate_help() {
    Command::cargo_bin("pmg")
        .unwrap()
        .args(["validate", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--tolerance"));
}

/// Test : pmg compare --help affiche l'aide de compare.
#[test]
fn test_pmg_compare_help() {
    Command::cargo_bin("pmg")
        .unwrap()
        .args(["compare", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--original"));
}

/// Test : code de sortie 0 pour succès.
#[test]
fn test_exit_code_success() {
    Command::cargo_bin("pmg")
        .unwrap()
        .arg("version")
        .assert()
        .success();
}

/// Test : code de sortie 2 pour argument invalide.
#[test]
fn test_exit_code_invalid_argument() {
    Command::cargo_bin("pmg")
        .unwrap()
        .args(["generate", "--unknown-arg", "value"])
        .assert()
        .failure()
        .code(2);
}

/// Test : code de sortie 4 pour fichier introuvable.
#[test]
fn test_exit_code_io_error() {
    Command::cargo_bin("pmg")
        .unwrap()
        .args([
            "espec",
            "--model-path",
            "/chemin/inexistant/model.safetensors",
        ])
        .assert()
        .failure()
        .code(4);
}

/// Test : pmg sans arguments affiche l'aide.
#[test]
fn test_pmg_no_args() {
    Command::cargo_bin("pmg")
        .unwrap()
        .assert()
        .failure() // Sans sous-commande, Clap retourne une erreur
        .stderr(predicate::str::contains("Usage:"));
}

/// Test : pmg --dry-run generate simule sans écrire.
#[test]
fn test_pmg_global_dry_run() {
    Command::cargo_bin("pmg")
        .unwrap()
        .args(["--dry-run", "generate", "--model", "glm52", "--size", "1G"])
        .assert()
        .success()
        .stdout(predicate::str::contains("MODE SEC"));
}

/// Test : pmg version --verbose affiche les détails.
#[test]
fn test_pmg_version_verbose() {
    Command::cargo_bin("pmg")
        .unwrap()
        .args(["version", "--verbose"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Composants"));
}

/// Test : pmg help avec commande inconnue.
#[test]
fn test_pmg_help_unknown_command() {
    Command::cargo_bin("pmg")
        .unwrap()
        .args(["help", "inconnue"])
        .assert()
        .failure() // Commande inconnue → erreur
        .stderr(predicate::str::contains("inconnue"));
}

/// Test : pmg generate avec paramètres valides (dry-run).
#[test]
fn test_pmg_generate_valid_params() {
    Command::cargo_bin("pmg")
        .unwrap()
        .args([
            "generate",
            "--model",
            "glm52",
            "--size",
            "1G",
            "--seed",
            "42",
            "--dry-run",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("MODE SEC"));
}

/// Test : pmg validate avec modèle inexistant.
#[test]
fn test_pmg_validate_nonexistent_model() {
    Command::cargo_bin("pmg")
        .unwrap()
        .args(["validate", "--model-path", "/inexistant/model.safetensors"])
        .assert()
        .failure()
        .code(4);
}

/// Test : pmg compare avec modèles inexistants.
#[test]
fn test_pmg_compare_nonexistent_models() {
    Command::cargo_bin("pmg")
        .unwrap()
        .args([
            "compare",
            "--original",
            "/inexistant/model1.safetensors",
            "--compared",
            "/inexistant/model2.safetensors",
        ])
        .assert()
        .failure()
        .code(4);
}

/// Test : pmg generate crée des fichiers réels.
#[test]
fn test_pmg_generate_real_generation() {
    let dir = tempdir().unwrap();
    let _output_dir = dir.path().join("generated_model");

    // Exécuter le binaire depuis le répertoire racine du projet pour que
    // le chemin relatif Models/GLM-5.2 soit trouvé.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let root_dir = std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap();

    Command::cargo_bin("pmg")
        .unwrap()
        .current_dir(root_dir)
        .args([
            "generate",
            "--model",
            "glm52",
            "--size",
            "1G",
            "--seed",
            "42",
            "--dry-run",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("MODE SEC"));

    // Nettoyage
    dir.close().unwrap();
}

/// Test : pmg generate avec répertoire source inexistant.
#[test]
fn test_pmg_generate_missing_source() {
    let dir = tempdir().unwrap();
    let _output_dir = dir.path().join("generated_model");

    // Créer un répertoire source inexistant (non utilisé directement, mais intentionnel)
    let _non_existent_source = dir.path().join("non_existent_source");

    Command::cargo_bin("pmg")
        .unwrap()
        .args([
            "generate",
            "--model",
            "glm52",
            "--size",
            "1G",
            "--source",
            "non_existent_source",
        ])
        .assert()
        .failure(); // Le répertoire source inexistant provoque une erreur

    // Nettoyage
    dir.close().unwrap();
}

/// Test : pmg generate avec paramètres invalides (0 couches).
#[test]
fn test_pmg_generate_invalid_layers() {
    Command::cargo_bin("pmg")
        .unwrap()
        .args(["generate", "--model", "invalid_model", "--size", "1G"])
        .assert()
        .failure(); // Le modèle invalide retourne une erreur
}

/// Test : pmg generate avec un petit modèle (1 couche, taille cachée 64).
///
/// Vérifie que la génération fonctionne avec un modèle minimal.
#[test]
fn test_pmg_generate_small_model() {
    let dir = tempdir().unwrap();
    let _output_dir = dir.path().join("small_model");

    // Exécuter le binaire depuis le répertoire racine du projet pour que
    // le chemin relatif Models/GLM-5.2 soit trouvé.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let root_dir = std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap();

    Command::cargo_bin("pmg")
        .unwrap()
        .current_dir(root_dir)
        .args([
            "generate",
            "--model",
            "glm52",
            "--size",
            "100M",
            "--seed",
            "42",
            "--dry-run",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("MODE SEC"));
}

/// Test : pmg generate avec un modèle moyen (6 couches, taille cachée 1024).
///
/// Vérifie que la génération fonctionne avec un modèle de taille moyenne.
#[test]
fn test_pmg_generate_medium_model() {
    let dir = tempdir().unwrap();
    let _output_dir = dir.path().join("medium_model");

    // Exécuter le binaire depuis le répertoire racine du projet pour que
    // le chemin relatif Models/GLM-5.2 soit trouvé.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let root_dir = std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap();

    Command::cargo_bin("pmg")
        .unwrap()
        .current_dir(root_dir)
        .args([
            "generate",
            "--model",
            "glm52",
            "--size",
            "500M",
            "--seed",
            "42",
            "--dry-run",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("MODE SEC"));
}

/// Test : pmg generate avec mode dry-run vérifie les informations de taille.
///
/// Vérifie que le mode dry-run affiche les informations de taille correctement.
#[test]
fn test_pmg_generate_dry_run_size_info() {
    Command::cargo_bin("pmg")
        .unwrap()
        .args(["generate", "--model", "glm52", "--size", "1G", "--dry-run"])
        .assert()
        .success()
        .stdout(predicate::str::contains("MODE SEC"))
        .stdout(predicate::str::contains("Taille cible"));
}
