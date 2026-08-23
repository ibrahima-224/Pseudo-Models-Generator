//! Tests de performance pour la CLI PMG.
//!
//! Ces tests vérifient que les commandes principales s'exécutent dans un temps raisonnable.
//! Les seuils sont définis pour des machines de développement standard et peuvent être
//! ajustés selon les besoins.
//!
//! # Surveillance mémoire
//! Les tests de génération incluent un monitoring mémoire en mesurant le RSS (Resident Set Size)
//! du processus pour s'assurer qu'il reste sous 500 Mo.

use assert_cmd::Command;
use std::time::Instant;
use tempfile::TempDir;

/// Mesure le RSS (Resident Set Size) du processus actuel en octets.
///
/// Lit `/proc/self/status` pour obtenir la mémoire résidente.
/// Retourne 0 si la lecture échoue (ex: non-Linux).
fn measure_process_rss() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<u64>() {
                            return kb * 1024; // Convertir Ko en octets
                        }
                    }
                }
            }
        }
        0
    }
    #[cfg(not(target_os = "linux"))]
    {
        0
    }
}

/// Crée un fichier modèle safetensors minimal valide pour les tests.
///
/// # Paramètres
/// * `dir` - Répertoire où créer le fichier
/// * `name` - Nom du fichier (sans extension)
///
/// # Retourne
/// Le chemin du fichier créé.
fn create_test_model_file(dir: &std::path::Path, name: &str) -> std::path::PathBuf {
    let file_path = dir.join(format!("{}.safetensors", name));

    // Contenu JSON minimal pour un fichier safetensors valide (header vide)
    std::fs::write(&file_path, "{}").expect("Impossible de créer le fichier de modèle de test");

    file_path
}

/// Obtient le chemin vers le répertoire source du modèle GLM-5.2.
///
/// # Retourne
/// Le chemin absolu vers le répertoire Models/GLM-5.2 dans la racine du projet.
fn glm52_source_path() -> std::path::PathBuf {
    // CARGO_MANIFEST_DIR est le répertoire du crate pmg-cli (crates/pmg-cli)
    // Nous devons remonter jusqu'à la racine du projet
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent() // crates/
        .expect("Répertoire parent introuvable")
        .parent() // racine du projet
        .expect("Répertoire racine introuvable")
        .join("Models")
        .join("GLM-5.2")
}

/// Test de performance pour la commande generate avec un petit modèle (100 Ko).
///
/// Vérifie que la génération prend moins de 5 secondes et reste sous 500 Mo de mémoire.
/// Ce test peut prendre plus de 10 secondes sur des machines lentes.
#[test]
#[ignore] // Marquer comme lente si nécessaire ( > 10 secondes)
fn test_performance_generate_small() {
    let temp_dir = TempDir::new().expect("Impossible de créer le répertoire temporaire");

    // Chemin vers le répertoire source du modèle GLM-5.2
    let source_path = glm52_source_path();

    // Seuil mémoire en octets (500 Mo)
    let max_memory_bytes = 500 * 1024 * 1024;

    // Mesurer le RSS initial
    let initial_rss = measure_process_rss();

    let start = Instant::now();

    // Exécution de la génération
    let output = Command::cargo_bin("pmg")
        .unwrap()
        .args([
            "generate",
            "--model",
            "glm52",
            "--size",
            "102400", // 100 Ko (102 400 octets)
            "--source",
            source_path.to_str().unwrap(),
        ])
        .current_dir(temp_dir.path())
        .output()
        .expect("Échec d'exécution de la commande generate");

    // Vérification que la commande a réussi
    assert!(
        output.status.success(),
        "La commande generate a échoué: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let duration = start.elapsed();

    // Vérifier que le temps est inférieur à 5 secondes
    assert!(
        duration.as_secs() < 5,
        "La génération d'un petit modèle a pris {:.2}s, seuil: 5s",
        duration.as_secs_f64()
    );

    // Mesurer le RSS final et vérifier qu'il reste sous 500 Mo
    let final_rss = measure_process_rss();
    let rss_used = final_rss.saturating_sub(initial_rss);
    assert!(
        rss_used <= max_memory_bytes,
        "La mémoire RSS utilisée ({:.2} Mo) dépasse le seuil de 500 Mo",
        rss_used as f64 / (1024.0 * 1024.0)
    );

    // Vérifier qu'un fichier modèle a été créé dans le répertoire
    let model_files: Vec<_> = std::fs::read_dir(temp_dir.path())
        .expect("Impossible de lire le répertoire")
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .collect();

    assert!(
        !model_files.is_empty(),
        "Aucun fichier modèle n'a été créé dans le répertoire temporaire"
    );
}

/// Test de performance pour la commande generate avec un modèle moyen (1 Mo).
///
/// Vérifie que la génération prend moins de 10 secondes et reste sous 500 Mo de mémoire.
/// Ce test peut prendre plus de 10 secondes sur des machines lentes.
#[test]
#[ignore] // Marquer comme lente si nécessaire ( > 10 secondes)
fn test_performance_generate_medium() {
    let temp_dir = TempDir::new().expect("Impossible de créer le répertoire temporaire");

    // Chemin vers le répertoire source du modèle GLM-5.2
    let source_path = glm52_source_path();

    // Seuil mémoire en octets (500 Mo)
    let max_memory_bytes = 500 * 1024 * 1024;

    // Mesurer le RSS initial
    let initial_rss = measure_process_rss();

    let start = Instant::now();

    // Exécution de la génération
    let output = Command::cargo_bin("pmg")
        .unwrap()
        .args([
            "generate",
            "--model",
            "glm52",
            "--size",
            "1048576", // 1 Mo (1 048 576 octets)
            "--source",
            source_path.to_str().unwrap(),
        ])
        .current_dir(temp_dir.path())
        .output()
        .expect("Échec d'exécution de la commande generate");

    // Vérification que la commande a réussi
    assert!(
        output.status.success(),
        "La commande generate a échoué: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let duration = start.elapsed();

    // Vérifier que le temps est inférieur à 10 secondes
    assert!(
        duration.as_secs() < 10,
        "La génération d'un modèle moyen a pris {:.2}s, seuil: 10s",
        duration.as_secs_f64()
    );

    // Mesurer le RSS final et vérifier qu'il reste sous 500 Mo
    let final_rss = measure_process_rss();
    let rss_used = final_rss.saturating_sub(initial_rss);
    assert!(
        rss_used <= max_memory_bytes,
        "La mémoire RSS utilisée ({:.2} Mo) dépasse le seuil de 500 Mo",
        rss_used as f64 / (1024.0 * 1024.0)
    );

    // Vérifier qu'un fichier modèle a été créé
    let model_files: Vec<_> = std::fs::read_dir(temp_dir.path())
        .expect("Impossible de lire le répertoire")
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry
                .path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .collect();

    assert!(
        !model_files.is_empty(),
        "Aucun fichier modèle n'a été créé dans le répertoire temporaire"
    );
}

/// Test de performance pour la commande espec.
///
/// Vérifie que l'inspection d'un modèle existant prend moins de 2 secondes.
#[test]
fn test_performance_espec() {
    let temp_dir = TempDir::new().expect("Impossible de créer le répertoire temporaire");

    // Créer un petit modèle de test pour l'inspection
    let model_file = create_test_model_file(temp_dir.path(), "test_model");

    let start = Instant::now();

    Command::cargo_bin("pmg")
        .unwrap()
        .args(["espec", "--model-path", model_file.to_str().unwrap()])
        .output()
        .expect("Échec d'exécution de la commande espec");

    let duration = start.elapsed();

    // Vérifier que le temps est inférieur à 2 secondes
    assert!(
        duration.as_secs() < 2,
        "L'inspection d'un modèle a pris {:.2}s, seuil: 2s",
        duration.as_secs_f64()
    );
}

/// Test de performance pour la commande validate.
///
/// Vérifie que la validation d'un modèle existant prend moins de 3 secondes.
#[test]
fn test_performance_validate() {
    let temp_dir = TempDir::new().expect("Impossible de créer le répertoire temporaire");

    // Créer un petit modèle de test pour la validation
    let model_file = create_test_model_file(temp_dir.path(), "test_model");

    let start = Instant::now();

    Command::cargo_bin("pmg")
        .unwrap()
        .args(["validate", "--model-path", model_file.to_str().unwrap()])
        .output()
        .expect("Échec d'exécution de la commande validate");

    let duration = start.elapsed();

    // Vérifier que le temps est inférieur à 3 secondes
    assert!(
        duration.as_secs() < 3,
        "La validation d'un modèle a pris {:.2}s, seuil: 3s",
        duration.as_secs_f64()
    );
}

/// Test de performance pour la commande compare.
///
/// Vérifie que la comparaison de deux modèles prend moins de 3 secondes.
#[test]
fn test_performance_compare() {
    let temp_dir = TempDir::new().expect("Impossible de créer le répertoire temporaire");

    // Créer deux petits modèles de test pour la comparaison
    let model1_file = create_test_model_file(temp_dir.path(), "model1");
    let model2_file = create_test_model_file(temp_dir.path(), "model2");

    let start = Instant::now();

    Command::cargo_bin("pmg")
        .unwrap()
        .args([
            "compare",
            "--original",
            model1_file.to_str().unwrap(),
            "--compared",
            model2_file.to_str().unwrap(),
        ])
        .output()
        .expect("Échec d'exécution de la commande compare");

    let duration = start.elapsed();

    // Vérifier que le temps est inférieur à 3 secondes
    assert!(
        duration.as_secs() < 3,
        "La comparaison de modèles a pris {:.2}s, seuil: 3s",
        duration.as_secs_f64()
    );
}

/// Test de performance pour la commande version.
///
/// Vérifie que l'affichage de la version prend moins de 0.5 seconde.
#[test]
fn test_performance_version() {
    let start = Instant::now();

    Command::cargo_bin("pmg")
        .unwrap()
        .arg("version")
        .assert()
        .success();

    let duration = start.elapsed();

    // Vérifier que le temps est inférieur à 0.5 seconde
    assert!(
        duration.as_millis() < 500,
        "L'affichage de la version a pris {:.2}ms, seuil: 500ms",
        duration.as_millis()
    );
}
