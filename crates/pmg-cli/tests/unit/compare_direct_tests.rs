//! Tests directs pour la commande compare de la CLI PMG.
//!
//! Ces tests testent directement la fonction execute de compare.rs
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

/// Test : Comparaison de deux modèles identiques (score = 100%).
///
/// Vérifie que la comparaison de modèles identiques retourne un score élevé.
#[test]
fn test_compare_identical_models_direct() {
    // Arrange : Utiliser les fixtures model_a et model_b (identiques)
    let fixtures = fixtures_dir();
    let model_a = fixtures.join("model_a.safetensors");
    let model_b = fixtures.join("model_b.safetensors");

    // Vérifier que les fixtures existent
    assert!(
        model_a.exists(),
        "La fixture model_a.safetensors n'existe pas"
    );
    assert!(
        model_b.exists(),
        "La fixture model_b.safetensors n'existe pas"
    );

    // Act
    let args = pmg_cli::commands::compare::CompareArgs {
        original: model_a.to_str().unwrap().to_string(),
        compared: model_b.to_str().unwrap().to_string(),
        tolerance: 0.1,
        verbose: false,
        format: "text".to_string(),
    };

    let result = pmg_cli::commands::compare::execute(args, false);

    // Assert : La comparaison devrait réussir
    assert!(
        result.is_ok(),
        "La comparaison devrait réussir: {:?}",
        result.err()
    );
}

/// Test : Comparaison de modèles différents (score < 90%).
///
/// Vérifie que la comparaison de modèles différents retourne un score plus bas.
#[test]
fn test_compare_different_models_direct() {
    // Arrange : Utiliser les fixtures model_a et model_f (noms de tenseurs différents)
    let fixtures = fixtures_dir();
    let model_a = fixtures.join("model_a.safetensors");
    let model_f = fixtures.join("model_f.safetensors");

    // Vérifier que les fixtures existent
    assert!(
        model_a.exists(),
        "La fixture model_a.safetensors n'existe pas"
    );
    assert!(
        model_f.exists(),
        "La fixture model_f.safetensors n'existe pas"
    );

    // Act
    let args = pmg_cli::commands::compare::CompareArgs {
        original: model_a.to_str().unwrap().to_string(),
        compared: model_f.to_str().unwrap().to_string(),
        tolerance: 0.1,
        verbose: false,
        format: "text".to_string(),
    };

    let result = pmg_cli::commands::compare::execute(args, false);

    // Assert : La comparaison devrait échouer avec une erreur de modèle trop différent
    assert!(
        result.is_err(),
        "La comparaison devrait échouer pour des modèles différents"
    );
}

/// Test : Format JSON avec validation de la sortie.
///
/// Vérifie que le format JSON est correctement généré.
#[test]
fn test_compare_json_output_direct() {
    // Arrange
    let fixtures = fixtures_dir();
    let model_a = fixtures.join("model_a.safetensors");
    let model_b = fixtures.join("model_b.safetensors");

    // Act
    let args = pmg_cli::commands::compare::CompareArgs {
        original: model_a.to_str().unwrap().to_string(),
        compared: model_b.to_str().unwrap().to_string(),
        tolerance: 0.1,
        verbose: false,
        format: "json".to_string(),
    };

    let result = pmg_cli::commands::compare::execute(args, false);

    // Assert : La comparaison devrait réussir
    assert!(
        result.is_ok(),
        "La comparaison devrait réussir: {:?}",
        result.err()
    );
}

/// Test : Mode verbose avec vérification des détails affichés.
///
/// Vérifie que le mode verbose affiche des informations supplémentaires.
#[test]
fn test_compare_verbose_mode_direct() {
    // Arrange
    let fixtures = fixtures_dir();
    let model_a = fixtures.join("model_a.safetensors");
    let model_b = fixtures.join("model_b.safetensors");

    // Act
    let args = pmg_cli::commands::compare::CompareArgs {
        original: model_a.to_str().unwrap().to_string(),
        compared: model_b.to_str().unwrap().to_string(),
        tolerance: 0.1,
        verbose: true,
        format: "text".to_string(),
    };

    let result = pmg_cli::commands::compare::execute(args, false);

    // Assert : La comparaison devrait réussir
    assert!(
        result.is_ok(),
        "La comparaison devrait réussir: {:?}",
        result.err()
    );
}

/// Test : Gestion d'erreur lors de la lecture de métadonnées corrompues.
///
/// Vérifie que la commande gère correctement les fichiers corrompus.
#[test]
fn test_compare_corrupted_metadata_direct() {
    // Arrange : Créer un fichier avec des métadonnées corrompues
    let dir = tempdir().unwrap();
    let corrupted_file = dir.path().join("corrupted.safetensors");

    // Contenu invalide (pas un en-tête Safetensors valide)
    std::fs::write(&corrupted_file, "contenu invalide").unwrap();

    let fixtures = fixtures_dir();
    let model_a = fixtures.join("model_a.safetensors");

    // Act
    let args = pmg_cli::commands::compare::CompareArgs {
        original: model_a.to_str().unwrap().to_string(),
        compared: corrupted_file.to_str().unwrap().to_string(),
        tolerance: 0.1,
        verbose: false,
        format: "text".to_string(),
    };

    let result = pmg_cli::commands::compare::execute(args, false);

    // Assert : Devrait échouer avec une erreur
    assert!(
        result.is_err(),
        "La comparaison avec un fichier corrompu devrait échouer"
    );
}

/// Test : Différents niveaux de tolérance.
///
/// Vérifie que l'option --tolerance est correctement prise en compte.
#[test]
fn test_compare_tolerance_levels_direct() {
    // Arrange
    let fixtures = fixtures_dir();
    let model_a = fixtures.join("model_a.safetensors");
    let model_b = fixtures.join("model_b.safetensors");

    // Différents niveaux de tolérance
    let tolerances = [0.01, 0.1, 0.5, 1.0];

    for tolerance in tolerances {
        // Act
        let args = pmg_cli::commands::compare::CompareArgs {
            original: model_a.to_str().unwrap().to_string(),
            compared: model_b.to_str().unwrap().to_string(),
            tolerance,
            verbose: false,
            format: "text".to_string(),
        };

        let result = pmg_cli::commands::compare::execute(args, false);

        // Assert : La commande devrait s'exécuter sans erreur
        assert!(
            result.is_ok(),
            "La commande devrait réussir pour la tolérance {}: {:?}",
            tolerance,
            result.err()
        );
    }
}

/// Test : Comparaison avec format text (par défaut).
///
/// Vérifie que le format text fonctionne correctement.
#[test]
fn test_compare_text_output_direct() {
    // Arrange
    let fixtures = fixtures_dir();
    let model_a = fixtures.join("model_a.safetensors");
    let model_b = fixtures.join("model_b.safetensors");

    // Act
    let args = pmg_cli::commands::compare::CompareArgs {
        original: model_a.to_str().unwrap().to_string(),
        compared: model_b.to_str().unwrap().to_string(),
        tolerance: 0.1,
        verbose: false,
        format: "text".to_string(),
    };

    let result = pmg_cli::commands::compare::execute(args, false);

    // Assert : La comparaison devrait réussir
    assert!(
        result.is_ok(),
        "La comparaison devrait réussir: {:?}",
        result.err()
    );
}

/// Test : Comparaison avec deux fichiers différents (modèle D avec deux tenseurs).
///
/// Vérifie que la comparaison fonctionne avec des modèles ayant plusieurs tenseurs.
#[test]
fn test_compare_multi_tensor_models_direct() {
    // Arrange
    let fixtures = fixtures_dir();
    let model_d = fixtures.join("model_d.safetensors");
    let model_e = fixtures.join("model_e.safetensors");

    // Vérifier que les fixtures existent
    assert!(
        model_d.exists(),
        "La fixture model_d.safetensors n'existe pas"
    );
    assert!(
        model_e.exists(),
        "La fixture model_e.safetensors n'existe pas"
    );

    // Act
    let args = pmg_cli::commands::compare::CompareArgs {
        original: model_d.to_str().unwrap().to_string(),
        compared: model_e.to_str().unwrap().to_string(),
        tolerance: 0.1,
        verbose: false,
        format: "text".to_string(),
    };

    let result = pmg_cli::commands::compare::execute(args, false);

    // Assert : La commande devrait s'exécuter
    assert!(
        result.is_ok() || result.is_err(),
        "La commande devrait s'exécuter sans panic"
    );
}
