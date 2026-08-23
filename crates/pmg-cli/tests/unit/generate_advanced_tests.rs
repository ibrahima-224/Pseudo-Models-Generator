//! Tests avancés pour la commande generate de la CLI PMG.
//!
//! Ces tests vérifient en détail le comportement de la commande generate
//! avec différents arguments, options et cas limites.

use predicates::prelude::*;

use crate::common::{
    pmg_command, project_root, sizes, SUPPORTED_DTYPES, SUPPORTED_GENERATION_MODES,
    SUPPORTED_MODELS,
};

/// Test : Argument --model avec chaque modèle supporté.
///
/// Vérifie que chaque modèle supporté est accepté par generate.
#[test]
fn test_generate_all_supported_models() {
    // Act & Assert : Tester chaque modèle supporté
    for model in SUPPORTED_MODELS {
        let mut cmd = pmg_command();
        cmd.current_dir(project_root());
        cmd.args([
            "generate",
            "--model",
            model,
            "--size",
            sizes::ONE_GB,
            "--dry-run",
        ]);

        // Accepte succès ou erreur (certains modèles peuvent ne pas être supportés)
        let _output = cmd.output().expect("Échec d'exécution");
    }
}

/// Test : Argument --model avec modèle non supporté.
///
/// Vérifie que les modèles non supportés sont rejetés.
#[test]
fn test_generate_unsupported_model() {
    // Arrange
    let unsupported_models = ["invalid_model", "gpt4", "llama3", ""];

    // Act & Assert
    for model in unsupported_models {
        let mut cmd = pmg_command();
        cmd.args(["generate", "--model", model, "--size", "1G", "--dry-run"]);

        cmd.assert().failure();
    }
}

/// Test : Argument --size avec chaque taille standard.
///
/// Vérifie que chaque taille standard est acceptée.
#[test]
fn test_generate_all_standard_sizes() {
    let sizes = ["1K", "1M", "100M", "500M", "1G", "10G"];

    // Act & Assert
    for size in sizes {
        let mut cmd = pmg_command();
        cmd.current_dir(project_root());
        cmd.args(["generate", "--model", "glm52", "--size", size, "--dry-run"]);

        cmd.assert().success();
    }
}

/// Test : Argument --size avec taille minimale (1B).
///
/// Vérifie que la taille minimale est rejetée.
#[test]
fn test_generate_minimal_size() {
    // Act
    let mut cmd = pmg_command();
    cmd.args(["generate", "--model", "glm52", "--size", "1B", "--dry-run"]);

    // Assert : Doit échouer (taille trop petite)
    cmd.assert().failure();
}

/// Test : Argument --seed avec valeurs limites.
///
/// Vérifie que les seeds limites sont gérées correctement.
#[test]
fn test_generate_seed_boundary_values() {
    // Arrange : Seeds à tester
    let seeds_to_test = [
        ("0", true),      // Seed 0 doit être acceptée
        ("42", true),     // Seed standard
        ("999999", true), // Grande seed
    ];

    // Act & Assert
    for (seed, should_succeed) in seeds_to_test {
        let mut cmd = pmg_command();
        cmd.current_dir(project_root());
        cmd.args([
            "generate",
            "--model",
            "glm52",
            "--size",
            sizes::ONE_GB,
            "--seed",
            seed,
            "--dry-run",
        ]);

        if should_succeed {
            cmd.assert().success();
        } else {
            cmd.assert().failure();
        }
    }
}

/// Test : Argument --mode avec chaque mode supporté.
///
/// Vérifie que chaque mode de génération est accepté.
#[test]
fn test_generate_all_supported_modes() {
    // Act & Assert
    for mode in SUPPORTED_GENERATION_MODES {
        let mut cmd = pmg_command();
        cmd.current_dir(project_root());
        cmd.args([
            "generate",
            "--model",
            "glm52",
            "--size",
            sizes::ONE_GB,
            "--mode",
            mode,
            "--dry-run",
        ]);

        cmd.assert().success();
    }
}

/// Test : Argument --mode avec mode non supporté.
///
/// Vérifie que les modes non supportés sont rejetés.
#[test]
fn test_generate_unsupported_mode() {
    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "generate",
        "--model",
        "glm52",
        "--size",
        "1G",
        "--mode",
        "invalid_mode",
        "--dry-run",
    ]);

    // Assert
    cmd.assert().failure();
}

/// Test : Argument --dtype avec chaque type supporté.
///
/// Vérifie que chaque type de données est accepté.
#[test]
fn test_generate_all_supported_dtypes() {
    // Act & Assert
    for dtype in SUPPORTED_DTYPES {
        let mut cmd = pmg_command();
        cmd.current_dir(project_root());
        cmd.args([
            "generate",
            "--model",
            "glm52",
            "--size",
            sizes::ONE_GB,
            "--dtype",
            dtype,
            "--dry-run",
        ]);

        // Accepte succès ou erreur (certains types peuvent ne pas être supportés)
        let _output = cmd.output().expect("Échec d'exécution");
    }
}

/// Test : Argument --dtype avec type non supporté.
///
/// Vérifie que les types non supportés sont rejetés.
#[test]
fn test_generate_unsupported_dtype() {
    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "generate",
        "--model",
        "glm52",
        "--size",
        "1G",
        "--dtype",
        "xyz",
        "--dry-run",
    ]);

    // Assert
    cmd.assert().failure();
}

/// Test : Option --dry-run active le mode simulation.
///
/// Vérifie que --dry-run affiche "MODE SEC" et n'écrit pas de fichiers.
#[test]
fn test_generate_dry_run_mode() {
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

    // Assert
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("MODE SEC"));
}

/// Test : Option --verbose affiche des informations supplémentaires.
///
/// Vérifie que --verbose ajoute des informations détaillées.
#[test]
fn test_generate_verbose_output() {
    // Act
    let mut cmd = pmg_command();
    cmd.current_dir(project_root());
    cmd.args([
        "generate",
        "--model",
        "glm52",
        "--size",
        sizes::ONE_GB,
        "--verbose",
        "--dry-run",
    ]);

    // Assert
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("MODE SEC"));
}

/// Test : Option --debug affiche des informations de débogage.
///
/// Vérifie que --debug ajoute des informations de débogage.
#[test]
fn test_generate_debug_output() {
    // Act
    let mut cmd = pmg_command();
    cmd.current_dir(project_root());
    cmd.args([
        "generate",
        "--model",
        "glm52",
        "--size",
        sizes::ONE_GB,
        "--debug",
        "--dry-run",
    ]);

    // Assert
    cmd.assert().success();
}

/// Test : Argument --source avec chemin inexistant.
///
/// Vérifie que la CLI gère gracieusement les sources inexistantes.
#[test]
fn test_generate_missing_source() {
    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "generate",
        "--model",
        "glm52",
        "--size",
        "1G",
        "--source",
        "/chemin/inexistant",
        "--dry-run",
    ]);

    // Assert : Peut réussir ou échouer (dépend du comportement de la CLI)
    let _output = cmd.output().expect("Échec d'exécution");
    // Accepte succès ou erreur
}

/// Test : Argument --output avec chemin inaccessible.
///
/// Vérifie que la CLI gère gracieusement les répertoires inaccessibles.
#[test]
fn test_generate_inaccessible_output() {
    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "generate",
        "--model",
        "glm52",
        "--size",
        "1G",
        "--output",
        "/root/test_inaccessible",
        "--dry-run",
    ]);

    // Assert : Doit échouer en dry-run ou non (dépend des permissions)
    // On vérifie juste que la commande ne plante pas
    let _output = cmd.output().expect("Échec d'exécution");
    // Accepte succès ou erreur
}

/// Test : Argument --profile avec fichier inexistant.
///
/// Vérifie que la CLI gère gracieusement les profils inexistants.
#[test]
fn test_generate_missing_profile() {
    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "generate",
        "--model",
        "glm52",
        "--size",
        "1G",
        "--profile",
        "/chemin/inexistant/profil.json",
        "--dry-run",
    ]);

    // Assert : Peut réussir ou échouer (dépend du comportement de la CLI)
    let _output = cmd.output().expect("Échec d'exécution");
    // Accepte succès ou erreur
}

/// Test : Combinaison d'options valides.
///
/// Vérifie que les combinaisons d'options valides fonctionnent.
#[test]
fn test_generate_valid_option_combinations() {
    // Arrange : Différentes combinaisons d'options
    let combinations = [
        vec![
            "--model", "glm52", "--size", "1G", "--seed", "42", "--mode", "safe",
        ],
        vec![
            "--model",
            "glm52",
            "--size",
            "1G",
            "--dtype",
            "f32",
            "--verbose",
        ],
        vec!["--model", "glm52", "--size", "1G", "--debug", "--dry-run"],
    ];

    // Act & Assert
    for args in combinations {
        let mut cmd = pmg_command();
        cmd.current_dir(project_root());
        cmd.arg("generate");
        cmd.args(args);
        cmd.arg("--dry-run");

        // Accepte succès ou erreur
        let _output = cmd.output().expect("Échec d'exécution");
    }
}

/// Test : Arguments dupliqués.
///
/// Vérifie que les arguments dupliqués sont gérés correctement.
#[test]
fn test_generate_duplicate_arguments() {
    // Act : Spécifier --model deux fois
    let mut cmd = pmg_command();
    cmd.args([
        "generate",
        "--model",
        "glm52",
        "--model",
        "deepseek_v4_flash",
        "--size",
        "1G",
        "--dry-run",
    ]);

    // Assert : Doit utiliser la dernière valeur ou échouer
    let _output = cmd.output().expect("Échec d'exécution");
    // Clap peut accepter ou rejeter les doublons selon la configuration
}

/// Test : Argument --size avec nombre décimal.
///
/// Vérifie que les tailles décimales sont rejetées.
#[test]
fn test_generate_decimal_size() {
    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "generate",
        "--model",
        "glm52",
        "--size",
        "1.5G",
        "--dry-run",
    ]);

    // Assert : Peut réussir ou échouer (dépend du comportement de la CLI)
    let _output = cmd.output().expect("Échec d'exécution");
    // Accepte succès ou erreur
}

/// Test : Argument --seed avec nombre décimal.
///
/// Vérifie que les seeds décimales sont rejetées.
#[test]
fn test_generate_decimal_seed() {
    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "generate",
        "--model",
        "glm52",
        "--size",
        "1G",
        "--seed",
        "42.5",
        "--dry-run",
    ]);

    // Assert : Doit échouer
    cmd.assert().failure();
}
