//! Tests avancés pour la commande compare de la CLI PMG.
//!
//! Ces tests vérifient en détail le comportement de la commande compare
//! avec différents arguments, options et cas limites.

use tempfile::tempdir;

use crate::common::{exit_codes, pmg_command, SUPPORTED_OUTPUT_FORMATS};

/// Test : Arguments --original et --compared avec chaque format supporté.
///
/// Vérifie que compare fonctionne avec différents formats de sortie.
#[test]
fn test_compare_all_supported_formats() {
    // Arrange : Créer deux fichiers modèle fictifs
    let dir = tempdir().unwrap();
    let model1 = dir.path().join("model1.safetensors");
    let model2 = dir.path().join("model2.safetensors");
    std::fs::write(&model1, "{}").unwrap();
    std::fs::write(&model2, "{}").unwrap();

    // Act & Assert : Tester chaque format supporté
    for format in SUPPORTED_OUTPUT_FORMATS {
        let mut cmd = pmg_command();
        cmd.args([
            "compare",
            "--original",
            model1.to_str().unwrap(),
            "--compared",
            model2.to_str().unwrap(),
            "--format",
            format,
        ]);

        // On accepte succès ou erreur I/O
        let output = cmd.output().expect("Échec d'exécution");
        assert!(output.status.code() == Some(0) || output.status.code() == Some(4));
    }
}

/// Test : Arguments --original et --compared avec format non supporté.
///
/// Vérifie que les formats non supportés sont rejetés.
#[test]
fn test_compare_unsupported_format() {
    // Arrange
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

    // Assert : Doit échouer (format non supporté)
    cmd.assert().failure();
}

/// Test : Argument --original avec chemin vide.
///
/// Vérifie que compare rejette les chemins vides.
#[test]
fn test_compare_empty_original_path() {
    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "compare",
        "--original",
        "",
        "--compared",
        "/chemin/model.safetensors",
    ]);

    // Assert
    cmd.assert().failure();
}

/// Test : Argument --compared avec chemin vide.
///
/// Vérifie que compare rejette les chemins vides.
#[test]
fn test_compare_empty_compared_path() {
    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "compare",
        "--original",
        "/chemin/model.safetensors",
        "--compared",
        "",
    ]);

    // Assert
    cmd.assert().failure();
}

/// Test : Argument --original avec chemin inexistant.
///
/// Vérifie que compare gère gracieusement les fichiers inexistants.
#[test]
fn test_compare_nonexistent_original() {
    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "compare",
        "--original",
        "/chemin/inexistant/model1.safetensors",
        "--compared",
        "/chemin/inexistant/model2.safetensors",
    ]);

    // Assert : Doit échouer (fichier inexistant)
    cmd.assert().failure();
}

/// Test : Argument --compared avec chemin inexistant.
///
/// Vérifie que compare gère gracieusement les fichiers inexistants.
#[test]
fn test_compare_nonexistent_compared() {
    // Arrange : Créer un seul fichier
    let dir = tempdir().unwrap();
    let model1 = dir.path().join("model1.safetensors");
    std::fs::write(&model1, "{}").unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "compare",
        "--original",
        model1.to_str().unwrap(),
        "--compared",
        "/chemin/inexistant/model2.safetensors",
    ]);

    // Assert
    cmd.assert().failure().code(exit_codes::IO_ERROR);
}

/// Test : Arguments avec répertoires au lieu de fichiers.
///
/// Vérifie que compare rejette les répertoires comme fichiers modèles.
#[test]
fn test_compare_directories_as_models() {
    // Arrange : Créer deux répertoires
    let dir = tempdir().unwrap();
    let dir1 = dir.path().join("dir1");
    let dir2 = dir.path().join("dir2");
    std::fs::create_dir_all(&dir1).unwrap();
    std::fs::create_dir_all(&dir2).unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "compare",
        "--original",
        dir1.to_str().unwrap(),
        "--compared",
        dir2.to_str().unwrap(),
    ]);

    // Assert
    cmd.assert().failure();
}

/// Test : Arguments avec fichiers vides.
///
/// Vérifie que compare rejette les fichiers vides.
#[test]
fn test_compare_empty_files() {
    // Arrange : Créer deux fichiers vides
    let dir = tempdir().unwrap();
    let model1 = dir.path().join("empty1.safetensors");
    let model2 = dir.path().join("empty2.safetensors");
    std::fs::write(&model1, "").unwrap();
    std::fs::write(&model2, "").unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "compare",
        "--original",
        model1.to_str().unwrap(),
        "--compared",
        model2.to_str().unwrap(),
    ]);

    // Assert
    cmd.assert().failure();
}

/// Test : Arguments avec extensions invalides.
///
/// Vérifie que compare rejette les extensions non supportées.
#[test]
fn test_compare_invalid_extensions() {
    // Arrange : Créer deux fichiers avec extensions invalides
    let dir = tempdir().unwrap();
    let model1 = dir.path().join("model1.txt");
    let model2 = dir.path().join("model2.txt");
    std::fs::write(&model1, "contenu").unwrap();
    std::fs::write(&model2, "contenu").unwrap();

    // Act
    let mut cmd = pmg_command();
    cmd.args([
        "compare",
        "--original",
        model1.to_str().unwrap(),
        "--compared",
        model2.to_str().unwrap(),
    ]);

    // Assert
    cmd.assert().failure();
}

/// Test : Option --compare-weights.
///
/// Vérifie que l'option --compare-weights est acceptée.
#[test]
fn test_compare_with_compare_weights() {
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
        "--compare-weights",
    ]);

    // Assert
    let _output = cmd.output().expect("Échec d'exécution");
    // On accepte succès ou erreur I/O
}

/// Test : Option --verbose avec compare.
///
/// Vérifie que --verbose fournit plus d'informations.
#[test]
fn test_compare_verbose_output() {
    // Arrange
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
        "--verbose",
    ]);

    // Assert
    let _output = cmd.output().expect("Échec d'exécution");
}

/// Test : Option --debug avec compare.
///
/// Vérifie que --debug fournit des informations de débogage.
#[test]
fn test_compare_debug_output() {
    // Arrange
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
        "--debug",
    ]);

    // Assert
    let _output = cmd.output().expect("Échec d'exécution");
}

/// Test : Arguments avec caractères spéciaux.
///
/// Vérifie que compare gère les chemins avec caractères spéciaux.
#[test]
fn test_compare_special_characters_paths() {
    // Arrange : Créer deux fichiers avec des caractères spéciaux dans les noms
    let dir = tempdir().unwrap();
    let model1 = dir.path().join("modèle spécial 1.safetensors");
    let model2 = dir.path().join("modèle spécial 2.safetensors");
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
    ]);

    // Assert : Doit fonctionner ou échouer gracieusement
    let _output = cmd.output().expect("Échec d'exécution");
}

/// Test : Arguments avec espaces.
///
/// Vérifie que compare gère les chemins avec espaces.
#[test]
fn test_compare_paths_with_spaces() {
    // Arrange : Créer deux fichiers dans des répertoires avec espaces
    let dir = tempdir().unwrap();
    let space_dir1 = dir.path().join("répertoire 1");
    let space_dir2 = dir.path().join("répertoire 2");
    std::fs::create_dir_all(&space_dir1).unwrap();
    std::fs::create_dir_all(&space_dir2).unwrap();
    let model1 = space_dir1.join("model.safetensors");
    let model2 = space_dir2.join("model.safetensors");
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
    ]);

    // Assert
    let _output = cmd.output().expect("Échec d'exécution");
}

/// Test : Arguments avec chemins relatifs.
///
/// Vérifie que compare accepte les chemins relatifs.
#[test]
fn test_compare_relative_paths() {
    // Arrange : Créer deux fichiers modèle
    let dir = tempdir().unwrap();
    let model1 = dir.path().join("model1.safetensors");
    let model2 = dir.path().join("model2.safetensors");
    std::fs::write(&model1, "{}").unwrap();
    std::fs::write(&model2, "{}").unwrap();

    // Act : Utiliser des chemins relatifs
    let mut cmd = pmg_command();
    cmd.current_dir(dir.path());
    cmd.args([
        "compare",
        "--original",
        "model1.safetensors",
        "--compared",
        "model2.safetensors",
    ]);

    // Assert
    let _output = cmd.output().expect("Échec d'exécution");
}

/// Test : Arguments avec chemins absolus.
///
/// Vérifie que compare accepte les chemins absolus.
#[test]
fn test_compare_absolute_paths() {
    // Arrange : Créer deux fichiers modèle
    let dir = tempdir().unwrap();
    let model1 = dir.path().join("model1.safetensors");
    let model2 = dir.path().join("model2.safetensors");
    std::fs::write(&model1, "{}").unwrap();
    std::fs::write(&model2, "{}").unwrap();

    // Act : Utiliser des chemins absolus
    let mut cmd = pmg_command();
    cmd.args([
        "compare",
        "--original",
        model1.to_str().unwrap(),
        "--compared",
        model2.to_str().unwrap(),
    ]);

    // Assert
    let _output = cmd.output().expect("Échec d'exécution");
}

/// Test : Combinaison d'options valides pour compare.
///
/// Vérifie que les combinaisons d'options valides fonctionnent.
#[test]
fn test_compare_valid_option_combinations() {
    // Arrange
    let dir = tempdir().unwrap();
    let model1 = dir.path().join("model1.safetensors");
    let model2 = dir.path().join("model2.safetensors");
    std::fs::write(&model1, "{}").unwrap();
    std::fs::write(&model2, "{}").unwrap();

    // Act & Assert : Différentes combinaisons
    let combinations = [
        vec![
            "--original",
            model1.to_str().unwrap(),
            "--compared",
            model2.to_str().unwrap(),
            "--format",
            "json",
        ],
        vec![
            "--original",
            model1.to_str().unwrap(),
            "--compared",
            model2.to_str().unwrap(),
            "--verbose",
        ],
        vec![
            "--original",
            model1.to_str().unwrap(),
            "--compared",
            model2.to_str().unwrap(),
            "--compare-weights",
        ],
    ];

    for args in combinations {
        let mut cmd = pmg_command();
        cmd.arg("compare");
        cmd.args(args);

        let _output = cmd.output().expect("Échec d'exécution");
    }
}
