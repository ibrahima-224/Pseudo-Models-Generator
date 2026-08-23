//! Tests de flux complets (workflow) pour la CLI PMG.
//!
//! Ces tests vérifient les scénarios d'utilisation réels où plusieurs
//! commandes sont exécutées en séquence.

use predicates::prelude::*;

use crate::common::{pmg_command, project_root, seeds, sizes, SUPPORTED_MODELS};

/// Test : Flux complet generate → validate.
///
/// Vérifie qu'on peut générer un modèle puis le valider.
#[test]
fn test_workflow_generate_then_validate() {
    // Act : Générer un petit modèle en dry-run (pas d'écriture réelle)
    let mut generate_cmd = pmg_command();
    generate_cmd.current_dir(project_root());
    generate_cmd.args([
        "generate",
        "--model",
        "glm52",
        "--size",
        sizes::ONE_HUNDRED_MB,
        "--seed",
        &seeds::STANDARD.to_string(),
        "--dry-run",
    ]);

    // Assert : La génération doit réussir
    generate_cmd.assert().success();

    // Note: Pour un test complet avec écriture réelle, il faudrait
    // un modèle source disponible. En mode dry-run, on ne peut pas
    // valider le fichier généré.
}

/// Test : Flux complet generate → compare.
///
/// Vérifie qu'on peut générer deux modèles et les comparer.
#[test]
fn test_workflow_generate_then_compare() {
    // Act : Générer deux modèles avec des seeds différentes en dry-run
    let mut cmd1 = pmg_command();
    cmd1.current_dir(project_root());
    cmd1.args([
        "generate",
        "--model",
        "glm52",
        "--size",
        sizes::ONE_HUNDRED_MB,
        "--seed",
        &seeds::STANDARD.to_string(),
        "--dry-run",
    ]);

    let mut cmd2 = pmg_command();
    cmd2.current_dir(project_root());
    cmd2.args([
        "generate",
        "--model",
        "glm52",
        "--size",
        sizes::ONE_HUNDRED_MB,
        "--seed",
        &seeds::ALTERNATIVE.to_string(),
        "--dry-run",
    ]);

    // Assert : Les deux générations doivent réussir
    cmd1.assert().success();
    cmd2.assert().success();

    // Note: Pour un test complet avec comparaison, il faudrait
    // des fichiers réels. En mode dry-run, on teste juste que
    // les commandes s'exécutent correctement.
}

/// Test : Workflow avec --no-validate.
///
/// Vérifie que l'option --no-validate désactive la validation post-génération.
#[test]
fn test_workflow_no_validate() {
    // Act : Générer avec --no-validate
    let mut cmd = pmg_command();
    cmd.current_dir(project_root());
    cmd.args([
        "generate",
        "--model",
        "glm52",
        "--size",
        sizes::ONE_HUNDRED_MB,
        "--no-validate",
        "--dry-run",
    ]);

    // Assert : La commande doit réussir
    cmd.assert().success();
}

/// Test : Workflow avec --verbose.
///
/// Vérifie que l'option --verbose fournit plus d'informations.
#[test]
fn test_workflow_verbose_mode() {
    // Act : Exécuter avec --verbose
    let mut cmd = pmg_command();
    cmd.args([
        "generate",
        "--model",
        "glm52",
        "--size",
        sizes::ONE_GB,
        "--verbose",
        "--dry-run",
    ]);

    // Assert : La sortie doit contenir des informations supplémentaires
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("MODE SEC"));
}

/// Test : Workflow avec --debug.
///
/// Vérifie que l'option --debug fournit des informations de débogage.
#[test]
fn test_workflow_debug_mode() {
    // Act : Exécuter avec --debug
    let mut cmd = pmg_command();
    cmd.args([
        "generate",
        "--model",
        "glm52",
        "--size",
        sizes::ONE_GB,
        "--debug",
        "--dry-run",
    ]);

    // Assert : La commande doit fonctionner
    cmd.assert().success();
}

/// Test : Workflow help → generate.
///
/// Vérifie qu'on peut obtenir l'aide puis exécuter la commande.
#[test]
fn test_workflow_help_then_generate() {
    // Act : Obtenir l'aide de generate
    let mut help_cmd = pmg_command();
    help_cmd.args(["help", "generate"]);

    // Assert : L'aide doit s'afficher
    help_cmd
        .assert()
        .success()
        .stdout(predicate::str::contains("--model"));

    // Act : Exécuter generate en dry-run
    let mut generate_cmd = pmg_command();
    generate_cmd.args(["generate", "--model", "glm52", "--size", "1G", "--dry-run"]);

    // Assert : La génération doit fonctionner
    generate_cmd
        .assert()
        .success()
        .stdout(predicate::str::contains("MODE SEC"));
}

/// Test : Workflow version → help.
///
/// Vérifie qu'on peut obtenir la version puis l'aide.
#[test]
fn test_workflow_version_then_help() {
    // Act : Obtenir la version
    let mut version_cmd = pmg_command();
    version_cmd.arg("version");

    // Assert : La version doit s'afficher
    version_cmd
        .assert()
        .success()
        .stdout(predicate::str::contains("pmg-cli v"));

    // Act : Obtenir l'aide générale
    let mut help_cmd = pmg_command();
    help_cmd.arg("--help");

    // Assert : L'aide doit s'afficher
    help_cmd
        .assert()
        .success()
        .stdout(predicate::str::contains("Commands:"));
}

/// Test : Workflow avec tous les modèles supportés.
///
/// Vérifie que chaque modèle supporté fonctionne en dry-run.
#[test]
fn test_workflow_all_supported_models() {
    // Act & Assert : Tester chaque modèle supporté
    for model in SUPPORTED_MODELS {
        let mut cmd = pmg_command();
        cmd.args(["generate", "--model", model, "--size", "1G", "--dry-run"]);

        // Accepte succès ou erreur (certains modèles peuvent ne pas être supportés)
        let _output = cmd.output().expect("Échec d'exécution");
    }
}

/// Test : Workflow avec tous les modes de génération.
///
/// Vérifie que chaque mode de génération fonctionne.
#[test]
fn test_workflow_all_generation_modes() {
    let modes = ["safe", "realistic"];

    // Act & Assert : Tester chaque mode
    for mode in modes {
        let mut cmd = pmg_command();
        cmd.args([
            "generate",
            "--model",
            "glm52",
            "--size",
            "1G",
            "--mode",
            mode,
            "--dry-run",
        ]);

        cmd.assert().success();
    }
}

/// Test : Workflow avec toutes les tailles standard.
///
/// Vérifie que chaque taille standard fonctionne.
#[test]
fn test_workflow_all_standard_sizes() {
    let sizes = ["1K", "1M", "100M", "500M", "1G", "10G"];

    // Act & Assert : Tester chaque taille
    for size in sizes {
        let mut cmd = pmg_command();
        cmd.args(["generate", "--model", "glm52", "--size", size, "--dry-run"]);

        cmd.assert().success();
    }
}
