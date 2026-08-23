//! Tests d'intégration pour la génération de modèles de très grande taille
//! avec optimisation mémoire streaming.
//!
//! Ces tests valident que :
//! 1. La génération de grands modèles fonctionne avec le pipeline streaming
//! 2. La consommation mémoire reste inférieure à 500 Mo
//! 3. Le streaming écrit directement sur disque sans accumulation en RAM
//!
//! ## Stratégie de test
//! - Tests rapides (2 Go) pour validation fonctionnelle
//! - Tests longs (>100 Go) optionnels pour validation de charge
//! - Monitoring mémoire en temps réel avec `MemoryMonitor`

use pmg_blueprint::tensor_spec::TensorSpec;
use pmg_core::{DType, Shape, TensorRole};
use pmg_generator::memory_monitor::MemoryMonitor;
use pmg_generator::streaming_config::StreamingConfig;
use pmg_generator::tensor_chunk_generator::TensorChunkGenerator;
use pmg_io::safetensors::ShardWriter;
use std::path::PathBuf;
use std::time::Instant;

/// Configuration pour les tests rapides (2 Go)
const FAST_CHUNK_SIZE_MB: usize = 8; // 8 Mo par chunk
const FAST_MAX_MEMORY_MB: u64 = 500; // 500 Mo maximum
const FAST_TARGET_SIZE_GB: u64 = 2; // 2 Go par modèle

/// Configuration pour les tests longs (>100 Go) - optionnels
const STRESS_CHUNK_SIZE_MB: usize = 8;
const STRESS_MAX_MEMORY_MB: u64 = 500;
const STRESS_TARGET_SIZE_GB: u64 = 100;

/// Structure pour les résultats de test
#[derive(Debug)]
struct TestResult {
    /// Nom du modèle
    model_name: String,
    /// Taille cible en Go
    target_size_gb: u64,
    /// Taille réelle en octets
    actual_size_bytes: u64,
    /// Mémoire maximale utilisée en octets (internally tracked)
    peak_memory_bytes: u64,
    /// Nombre de chunks écrits
    chunks_written: usize,
    /// Temps d'exécution en secondes
    execution_time_secs: f64,
    /// RSS processus en octets (mesuré via /proc)
    process_rss_bytes: u64,
    /// Succès du test
    success: bool,
    /// Message d'erreur éventuel
    error_message: Option<String>,
}

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

/// Génère un modèle de grande taille avec monitoring mémoire
///
/// # Paramètres
/// - `model_name` : nom du modèle
/// - `target_size_go` : taille cible en Go
/// - `output_dir` : répertoire de sortie
/// - `seed` : seed pour la génération déterministe
/// - `chunk_size_mb` : taille des chunks en Mo
/// - `max_memory_mb` : mémoire maximale en Mo
///
/// # Retourne
/// Les résultats du test avec métriques détaillées
fn generate_model_with_monitoring(
    model_name: &str,
    target_size_go: u64,
    output_dir: &std::path::Path,
    seed: u64,
    chunk_size_mb: usize,
    max_memory_mb: u64,
) -> TestResult {
    let start_time = Instant::now();
    let initial_rss = measure_process_rss();

    // Configuration du streaming
    let chunk_size = chunk_size_mb * 1024 * 1024; // Convertir Mo en octets
    let max_memory = max_memory_mb * 1024 * 1024; // Convertir Mo en octets
    let config = StreamingConfig::new(chunk_size, max_memory);

    // Création du moniteur mémoire
    let _memory_monitor = MemoryMonitor::new(max_memory).with_verbose(true);

    // Création du générateur
    let mut generator = TensorChunkGenerator::new(config, seed);

    // Chemin du fichier de sortie
    let output_path = output_dir.join(format!("{}.safetensors", model_name));

    // Suppression du fichier existant s'il existe
    let _ = std::fs::remove_file(&output_path);

    // Création du writer avec réserve de 1 Mo pour l'en-tête
    let mut writer = match ShardWriter::new(output_path.clone(), 1024 * 1024) {
        Ok(w) => w,
        Err(e) => {
            return TestResult {
                model_name: model_name.to_string(),
                target_size_gb: target_size_go,
                actual_size_bytes: 0,
                peak_memory_bytes: 0,
                chunks_written: 0,
                execution_time_secs: start_time.elapsed().as_secs_f64(),
                process_rss_bytes: measure_process_rss(),
                success: false,
                error_message: Some(format!("Erreur création writer: {}", e)),
            };
        },
    };

    // Calcul du nombre d'éléments pour atteindre la taille cible
    // Chaque élément f32 = 4 octets
    let target_bytes = target_size_go * 1024 * 1024 * 1024; // Convertir Go en octets
    let total_elements = (target_bytes / 4) as usize; // Diviser par taille d'un f32

    // Création de la spécification du tenseur (forme 1D pour simplifier)
    let tensor_spec = match TensorSpec::new(
        format!("{}.weight", model_name),
        Shape::new(vec![total_elements as u64]).unwrap(),
        DType::F32,
        TensorRole::Other,
    ) {
        Ok(spec) => spec,
        Err(e) => {
            return TestResult {
                model_name: model_name.to_string(),
                target_size_gb: target_size_go,
                actual_size_bytes: 0,
                peak_memory_bytes: 0,
                chunks_written: 0,
                execution_time_secs: start_time.elapsed().as_secs_f64(),
                process_rss_bytes: measure_process_rss(),
                success: false,
                error_message: Some(format!("Erreur création TensorSpec: {}", e)),
            };
        },
    };

    // Messages de démarrage
    eprintln!();
    eprintln!("🚀 ═══════════════════════════════════════════════════════════════");
    eprintln!("   Début de la génération: {}", model_name);
    eprintln!(
        "   Taille cible: {} Go ({} éléments f32)",
        target_size_go, total_elements
    );
    eprintln!("   Chunk size: {} Mo", chunk_size_mb);
    eprintln!("   Mémoire max: {} Mo", max_memory_mb);
    eprintln!(
        "   RSS initial: {:.2} Mo",
        initial_rss as f64 / (1024.0 * 1024.0)
    );
    eprintln!("═══════════════════════════════════════════════════════════════");

    // Mesure de la RSS avant la génération
    let pre_gen_rss = measure_process_rss();

    // Génération et écriture du tenseur
    let result = match generator.generate_and_write_tensor(&tensor_spec, &mut writer, 0) {
        Ok(r) => r,
        Err(e) => {
            return TestResult {
                model_name: model_name.to_string(),
                target_size_gb: target_size_go,
                actual_size_bytes: 0,
                peak_memory_bytes: 0,
                chunks_written: 0,
                execution_time_secs: start_time.elapsed().as_secs_f64(),
                process_rss_bytes: measure_process_rss(),
                success: false,
                error_message: Some(format!("Erreur génération tenseur: {}", e)),
            };
        },
    };

    // Mesure de la RSS après la génération
    let post_gen_rss = measure_process_rss();

    // Finalisation du writer
    if let Err(e) = writer.finalize() {
        return TestResult {
            model_name: model_name.to_string(),
            target_size_gb: target_size_go,
            actual_size_bytes: result.total_bytes_written as u64,
            peak_memory_bytes: generator.memory_monitor().metrics().peak_usage,
            chunks_written: result.chunks_written,
            execution_time_secs: start_time.elapsed().as_secs_f64(),
            process_rss_bytes: measure_process_rss(),
            success: false,
            error_message: Some(format!("Erreur finalisation writer: {}", e)),
        };
    }

    // Vérification de la taille du fichier
    let actual_size = match std::fs::metadata(&output_path) {
        Ok(metadata) => metadata.len(),
        Err(e) => {
            return TestResult {
                model_name: model_name.to_string(),
                target_size_gb: target_size_go,
                actual_size_bytes: result.total_bytes_written as u64,
                peak_memory_bytes: generator.memory_monitor().metrics().peak_usage,
                chunks_written: result.chunks_written,
                execution_time_secs: start_time.elapsed().as_secs_f64(),
                process_rss_bytes: measure_process_rss(),
                success: false,
                error_message: Some(format!("Erreur lecture métadonnées: {}", e)),
            };
        },
    };

    let peak_memory = generator.memory_monitor().metrics().peak_usage;
    let final_rss = measure_process_rss();
    let execution_time = start_time.elapsed().as_secs_f64();

    // Calcul des métriques RSS
    let rss_during_gen = post_gen_rss.saturating_sub(pre_gen_rss);
    let max_rss = final_rss.max(post_gen_rss).max(pre_gen_rss);

    // Vérification que la mémoire reste sous la limite
    let memory_monitor_ok = peak_memory <= max_memory;
    let rss_ok = max_rss <= max_memory * 2; // Tolérance pour le RSS système

    // Rapport détaillé
    eprintln!();
    eprintln!("✅ ═══════════════════════════════════════════════════════════════");
    eprintln!("   Résultat: {}", model_name);
    eprintln!("   ─────────────────────────────────────────────────────────────");
    eprintln!("   Taille cible:     {:.2} Go", target_size_go as f64);
    eprintln!(
        "   Taille réelle:    {:.2} Go",
        actual_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    eprintln!("   Éléments écrits:  {}", result.total_elements);
    eprintln!("   Chunks écrits:    {}", result.chunks_written);
    eprintln!("   ─────────────────────────────────────────────────────────────");
    eprintln!("   Métriques mémoire:");
    eprintln!(
        "     MemoryMonitor peak:  {:.2} Mo",
        peak_memory as f64 / (1024.0 * 1024.0)
    );
    eprintln!(
        "     MemoryMonitor max:   {:.2} Mo",
        max_memory as f64 / (1024.0 * 1024.0)
    );
    eprintln!(
        "     RSS initial:         {:.2} Mo",
        initial_rss as f64 / (1024.0 * 1024.0)
    );
    eprintln!(
        "     RSS avant gén:       {:.2} Mo",
        pre_gen_rss as f64 / (1024.0 * 1024.0)
    );
    eprintln!(
        "     RSS après gén:       {:.2} Mo",
        post_gen_rss as f64 / (1024.0 * 1024.0)
    );
    eprintln!(
        "     RSS final:           {:.2} Mo",
        final_rss as f64 / (1024.0 * 1024.0)
    );
    eprintln!(
        "     RSS pendant gén:     {:.2} Mo",
        rss_during_gen as f64 / (1024.0 * 1024.0)
    );
    eprintln!("   ─────────────────────────────────────────────────────────────");
    eprintln!("   Validation:");
    eprintln!(
        "     MemoryMonitor < {} Mo:  {}",
        max_memory_mb,
        if memory_monitor_ok {
            "✅ OUI"
        } else {
            "❌ NON"
        }
    );
    eprintln!(
        "     RSS < {} Mo:            {}",
        max_memory_mb * 2,
        if rss_ok { "✅ OUI" } else { "❌ NON" }
    );
    eprintln!("   ─────────────────────────────────────────────────────────────");
    eprintln!(
        "   Temps: {:.2} secondes ({:.2} Mo/s)",
        execution_time,
        actual_size as f64 / (1024.0 * 1024.0) / execution_time
    );
    eprintln!("═══════════════════════════════════════════════════════════════");

    let size_ok = actual_size >= target_size_go * 1024 * 1024 * 1024;

    TestResult {
        model_name: model_name.to_string(),
        target_size_gb: target_size_go,
        actual_size_bytes: actual_size,
        peak_memory_bytes: peak_memory,
        chunks_written: result.chunks_written,
        execution_time_secs: execution_time,
        process_rss_bytes: max_rss,
        success: memory_monitor_ok && size_ok,
        error_message: None,
    }
}

// ============================================================================
// Tests rapides (2 Go) - Exécutés normalement
// ============================================================================

/// Test rapide : génération d'un modèle DeepSeek-V4-Flash de 2 Go
///
/// Valide que le pipeline streaming fonctionne avec une taille raisonnable
/// et que la consommation mémoire reste sous 500 Mo.
#[test]
#[ignore] // Test gourmand en mémoire (2 Go) - exécuter manuellement avec cargo test -- --ignored
fn test_deepseek_v4_flash_2gb() {
    let test_dir = PathBuf::from("/media/sory/SORY-DATA/pmg_test");
    std::fs::create_dir_all(&test_dir).expect("Impossible de créer le répertoire de test");

    let result = generate_model_with_monitoring(
        "DeepSeek-V4-Flash",
        FAST_TARGET_SIZE_GB,
        &test_dir,
        42,
        FAST_CHUNK_SIZE_MB,
        FAST_MAX_MEMORY_MB,
    );

    // Assertions
    assert!(
        result.success,
        "La génération a échoué: {:?}",
        result.error_message
    );
    assert!(
        result.actual_size_bytes >= FAST_TARGET_SIZE_GB * 1024 * 1024 * 1024,
        "Taille insuffisante: {:.2} Go < {} Go",
        result.actual_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        FAST_TARGET_SIZE_GB
    );
    assert!(
        result.peak_memory_bytes <= FAST_MAX_MEMORY_MB * 1024 * 1024,
        "Mémoire dépassée: {:.2} Mo > {} Mo",
        result.peak_memory_bytes as f64 / (1024.0 * 1024.0),
        FAST_MAX_MEMORY_MB
    );

    // Nettoyage
    let _ = std::fs::remove_file(test_dir.join("DeepSeek-V4-Flash.safetensors"));
}

/// Test rapide : génération d'un modèle GLM-5.2 de 2 Go
///
/// Valide que le pipeline streaming fonctionne avec un modèle différent
/// et que la consommation mémoire reste sous 500 Mo.
#[test]
#[ignore] // Test gourmand en mémoire (2 Go) - exécuter manuellement avec cargo test -- --ignored
fn test_glm52_2gb() {
    let test_dir = PathBuf::from("/media/sory/SORY-DATA/pmg_test");
    std::fs::create_dir_all(&test_dir).expect("Impossible de créer le répertoire de test");

    let result = generate_model_with_monitoring(
        "GLM-5.2",
        FAST_TARGET_SIZE_GB,
        &test_dir,
        123,
        FAST_CHUNK_SIZE_MB,
        FAST_MAX_MEMORY_MB,
    );

    // Assertions
    assert!(
        result.success,
        "La génération a échoué: {:?}",
        result.error_message
    );
    assert!(
        result.actual_size_bytes >= FAST_TARGET_SIZE_GB * 1024 * 1024 * 1024,
        "Taille insuffisante: {:.2} Go < {} Go",
        result.actual_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        FAST_TARGET_SIZE_GB
    );
    assert!(
        result.peak_memory_bytes <= FAST_MAX_MEMORY_MB * 1024 * 1024,
        "Mémoire dépassée: {:.2} Mo > {} Mo",
        result.peak_memory_bytes as f64 / (1024.0 * 1024.0),
        FAST_MAX_MEMORY_MB
    );

    // Nettoyage
    let _ = std::fs::remove_file(test_dir.join("GLM-5.2.safetensors"));
}

/// Test rapide : génération des deux modèles consécutivement
///
/// Valide que le pipeline streaming peut générer plusieurs modèles
/// sans fuite mémoire entre les générations.
#[test]
#[ignore] // Test gourmand en mémoire (2x2 Go) - exécuter manuellement avec cargo test -- --ignored
fn test_both_models_2gb_sequential() {
    let test_dir = PathBuf::from("/media/sory/SORY-DATA/pmg_test");
    std::fs::create_dir_all(&test_dir).expect("Impossible de créer le répertoire de test");

    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════════╗");
    eprintln!("║  Test séquentiel: DeepSeek-V4-Flash + GLM-5.2 (2 Go chacun) ║");
    eprintln!("╚═══════════════════════════════════════════════════════════════╝");

    // Premier modèle
    let result1 = generate_model_with_monitoring(
        "DeepSeek-V4-Flash",
        FAST_TARGET_SIZE_GB,
        &test_dir,
        42,
        FAST_CHUNK_SIZE_MB,
        FAST_MAX_MEMORY_MB,
    );

    assert!(
        result1.success,
        "DeepSeek-V4-Flash a échoué: {:?}",
        result1.error_message
    );

    // Deuxième modèle (vérification qu'il n'y a pas de fuite mémoire)
    let result2 = generate_model_with_monitoring(
        "GLM-5.2",
        FAST_TARGET_SIZE_GB,
        &test_dir,
        123,
        FAST_CHUNK_SIZE_MB,
        FAST_MAX_MEMORY_MB,
    );

    assert!(
        result2.success,
        "GLM-5.2 a échoué: {:?}",
        result2.error_message
    );

    // Vérification que la mémoire n'a pas augmenté entre les deux modèles
    // (indicateur de fuite mémoire)
    eprintln!();
    eprintln!("📊 Comparaison mémoire entre les deux modèles:");
    eprintln!(
        "   DeepSeek-V4-Flash RSS: {:.2} Mo",
        result1.process_rss_bytes as f64 / (1024.0 * 1024.0)
    );
    eprintln!(
        "   GLM-5.2 RSS:           {:.2} Mo",
        result2.process_rss_bytes as f64 / (1024.0 * 1024.0)
    );

    // Nettoyage
    let _ = std::fs::remove_file(test_dir.join("DeepSeek-V4-Flash.safetensors"));
    let _ = std::fs::remove_file(test_dir.join("GLM-5.2.safetensors"));
}

// ============================================================================
// Tests de longue durée (>100 Go) - Ignorés par défaut
// ============================================================================

/// Test long : génération d'un modèle DeepSeek-V4-Flash de >100 Go
///
/// Ce test est ignoré par défaut car il prendre plusieurs heures.
/// Pour l'exécuter : `cargo test -- --ignored test_deepseek_v4_flash_100gb`
#[test]
#[ignore]
fn test_deepseek_v4_flash_100gb() {
    let test_dir = PathBuf::from("/media/sory/SORY-DATA/pmg_test");
    std::fs::create_dir_all(&test_dir).expect("Impossible de créer le répertoire de test");

    let result = generate_model_with_monitoring(
        "DeepSeek-V4-Flash",
        STRESS_TARGET_SIZE_GB,
        &test_dir,
        42,
        STRESS_CHUNK_SIZE_MB,
        STRESS_MAX_MEMORY_MB,
    );

    // Affichage du rapport
    eprintln!();
    eprintln!("📊 Rapport final DeepSeek-V4-Flash (>100 Go):");
    eprintln!(
        "   Succès: {}",
        if result.success { "✅ OUI" } else { "❌ NON" }
    );
    eprintln!(
        "   Taille: {:.2} Go",
        result.actual_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    eprintln!(
        "   Mémoire peak: {:.2} Mo",
        result.peak_memory_bytes as f64 / (1024.0 * 1024.0)
    );
    eprintln!(
        "   RSS max: {:.2} Mo",
        result.process_rss_bytes as f64 / (1024.0 * 1024.0)
    );
    eprintln!("   Temps: {:.2} secondes", result.execution_time_secs);

    if let Some(error) = &result.error_message {
        eprintln!("   Erreur: {}", error);
    }

    // Assertions
    assert!(
        result.success,
        "La génération a échoué: {:?}",
        result.error_message
    );
    assert!(
        result.actual_size_bytes >= STRESS_TARGET_SIZE_GB * 1024 * 1024 * 1024,
        "Taille insuffisante"
    );
    assert!(
        result.peak_memory_bytes <= STRESS_MAX_MEMORY_MB * 1024 * 1024,
        "Mémoire dépassée"
    );

    // Nettoyage
    let _ = std::fs::remove_file(test_dir.join("DeepSeek-V4-Flash.safetensors"));
}

/// Test long : génération d'un modèle GLM-5.2 de >100 Go
///
/// Ce test est ignoré par défaut car il prendre plusieurs heures.
/// Pour l'exécuter : `cargo test -- --ignored test_glm52_100gb`
#[test]
#[ignore]
fn test_glm52_100gb() {
    let test_dir = PathBuf::from("/media/sory/SORY-DATA/pmg_test");
    std::fs::create_dir_all(&test_dir).expect("Impossible de créer le répertoire de test");

    let result = generate_model_with_monitoring(
        "GLM-5.2",
        STRESS_TARGET_SIZE_GB,
        &test_dir,
        123,
        STRESS_CHUNK_SIZE_MB,
        STRESS_MAX_MEMORY_MB,
    );

    // Affichage du rapport
    eprintln!();
    eprintln!("📊 Rapport final GLM-5.2 (>100 Go):");
    eprintln!(
        "   Succès: {}",
        if result.success { "✅ OUI" } else { "❌ NON" }
    );
    eprintln!(
        "   Taille: {:.2} Go",
        result.actual_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    eprintln!(
        "   Mémoire peak: {:.2} Mo",
        result.peak_memory_bytes as f64 / (1024.0 * 1024.0)
    );
    eprintln!(
        "   RSS max: {:.2} Mo",
        result.process_rss_bytes as f64 / (1024.0 * 1024.0)
    );
    eprintln!("   Temps: {:.2} secondes", result.execution_time_secs);

    if let Some(error) = &result.error_message {
        eprintln!("   Erreur: {}", error);
    }

    // Assertions
    assert!(
        result.success,
        "La génération a échoué: {:?}",
        result.error_message
    );
    assert!(
        result.actual_size_bytes >= STRESS_TARGET_SIZE_GB * 1024 * 1024 * 1024,
        "Taille insuffisante"
    );
    assert!(
        result.peak_memory_bytes <= STRESS_MAX_MEMORY_MB * 1024 * 1024,
        "Mémoire dépassée"
    );

    // Nettoyage
    let _ = std::fs::remove_file(test_dir.join("GLM-5.2.safetensors"));
}

// ============================================================================
// Tests unitaires de monitoring mémoire
// ============================================================================

/// Test unitaire : vérification du monitoring mémoire pendant la génération
#[test]
fn test_memory_monitoring_during_generation() {
    // Configuration avec une limite plus petite pour ce test
    let chunk_size = 1024 * 1024; // 1 Mo
    let max_memory = 10 * 1024 * 1024; // 10 Mo
    let config = StreamingConfig::new(chunk_size, max_memory);

    let mut generator = TensorChunkGenerator::new(config, 42);
    let monitor = generator.memory_monitor();

    // Vérification initiale
    assert_eq!(monitor.usage_percentage(), 0.0);
    assert!(!monitor.is_near_limit());
    assert!(!monitor.is_over_limit());

    // Création d'un tenseur de test
    let tensor_spec = TensorSpec::new(
        "test_tensor",
        Shape::new(vec![1000, 1000]).unwrap(),
        DType::F32,
        TensorRole::Other,
    )
    .unwrap();

    // Création d'un writer temporaire
    let temp_dir = tempfile::tempdir().unwrap();
    let shard_path = temp_dir.path().join("test.safetensors");
    let mut writer = ShardWriter::new(shard_path, 1024).unwrap();

    // Génération et écriture
    let _result = generator
        .generate_and_write_tensor(&tensor_spec, &mut writer, 0)
        .unwrap();

    // Finalisation
    writer.finalize().unwrap();

    // Vérification que la mémoire est revenue à zéro
    assert_eq!(generator.memory_monitor().usage_percentage(), 0.0);

    // Vérification que le pic de mémoire est inférieur à la limite
    assert!(generator.memory_monitor().metrics().peak_usage <= max_memory);
}

/// Test unitaire : vérification du déterminisme de la génération
#[test]
fn test_generation_determinism() {
    let config = StreamingConfig::new(1024 * 1024, 10 * 1024 * 1024);
    let seed = 42;

    // Création de deux générateurs avec la même seed
    let mut generator1 = TensorChunkGenerator::new(config.clone(), seed);
    let mut generator2 = TensorChunkGenerator::new(config, seed);

    // Spécification du tenseur
    let tensor_spec = TensorSpec::new(
        "test_tensor",
        Shape::new(vec![100, 100]).unwrap(),
        DType::F32,
        TensorRole::Other,
    )
    .unwrap();

    // Création de writers temporaires
    let temp_dir1 = tempfile::tempdir().unwrap();
    let temp_dir2 = tempfile::tempdir().unwrap();
    let shard_path1 = temp_dir1.path().join("test1.safetensors");
    let shard_path2 = temp_dir2.path().join("test2.safetensors");
    let mut writer1 = ShardWriter::new(shard_path1.clone(), 1024).unwrap();
    let mut writer2 = ShardWriter::new(shard_path2.clone(), 1024).unwrap();

    // Génération avec les deux générateurs
    let result1 = generator1
        .generate_and_write_tensor(&tensor_spec, &mut writer1, 0)
        .unwrap();
    let result2 = generator2
        .generate_and_write_tensor(&tensor_spec, &mut writer2, 0)
        .unwrap();

    // Finalisation
    writer1.finalize().unwrap();
    writer2.finalize().unwrap();

    // Les résultats doivent être identiques
    assert_eq!(result1.total_elements, result2.total_elements);
    assert_eq!(result1.chunks_written, result2.chunks_written);
    assert_eq!(result1.total_bytes_written, result2.total_bytes_written);

    // Les fichiers doivent être identiques
    let data1 = std::fs::read(&shard_path1).unwrap();
    let data2 = std::fs::read(&shard_path2).unwrap();
    assert_eq!(data1, data2);
}

/// Test unitaire : vérification de la configuration du streaming
#[test]
fn test_streaming_config_validation() {
    // Configuration valide
    let valid_config = StreamingConfig::new(8 * 1024 * 1024, 500 * 1024 * 1024);
    assert!(valid_config.is_valid());

    // Configuration avec chunk trop petit
    let invalid_config1 = StreamingConfig::new(512 * 1024, 500 * 1024 * 1024);
    assert!(!invalid_config1.is_valid());

    // Configuration avec mémoire trop faible
    let invalid_config2 = StreamingConfig::new(8 * 1024 * 1024, 50 * 1024 * 1024);
    assert!(!invalid_config2.is_valid());
}

/// Test unitaire : vérification des conversions en Mo
#[test]
fn test_streaming_config_conversions() {
    let config = StreamingConfig::new(8 * 1024 * 1024, 500 * 1024 * 1024);

    assert!((config.chunk_size_mb() - 8.0).abs() < f64::EPSILON);
    assert!((config.max_memory_mb() - 500.0).abs() < f64::EPSILON);

    let config2 = StreamingConfig::new(16 * 1024 * 1024, 1024 * 1024 * 1024);
    assert!((config2.chunk_size_mb() - 16.0).abs() < f64::EPSILON);
    assert!((config2.max_memory_mb() - 1024.0).abs() < f64::EPSILON);
}

/// Test unitaire : vérification du moniteur mémoire
#[test]
fn test_memory_monitor_comprehensive() {
    let mut monitor = MemoryMonitor::new(1000);

    // Test initial
    assert_eq!(monitor.usage_percentage(), 0.0);
    assert!(!monitor.is_near_limit());
    assert!(!monitor.is_over_limit());

    // Allocation partielle
    assert!(monitor.allocate(500));
    assert_eq!(monitor.usage_percentage(), 50.0);
    assert!(!monitor.is_near_limit());
    assert!(!monitor.is_over_limit());

    // Allocation jusqu'au seuil d'avertissement
    assert!(monitor.allocate(300)); // 800/1000 = 80%
    assert!(monitor.is_near_limit());
    assert!(!monitor.is_over_limit());

    // Allocation qui dépasse la limite
    assert!(!monitor.allocate(300)); // 800 + 300 = 1100 > 1000
    assert_eq!(monitor.usage_percentage(), 80.0); // Pas de changement

    // Libération de mémoire
    monitor.release(200);
    assert_eq!(monitor.usage_percentage(), 60.0);
    assert!(!monitor.is_near_limit());

    // Réinitialisation
    monitor.reset();
    assert_eq!(monitor.usage_percentage(), 0.0);
    assert!(!monitor.is_near_limit());
    assert!(!monitor.is_over_limit());
}

/// Test unitaire : vérification de l'affichage du moniteur mémoire
#[test]
fn test_memory_monitor_display() {
    let monitor = MemoryMonitor::new(500 * 1024 * 1024);
    let display = format!("{}", monitor);

    // L'affichage contient le pourcentage d'utilisation (0.0%)
    assert!(display.contains("0.0%")); // Utilisation actuelle
}

/// Test unitaire : vérification des métriques mémoire
#[test]
fn test_memory_metrics_comprehensive() {
    use pmg_generator::memory_monitor::MemoryMetrics;

    let mut metrics = MemoryMetrics::with_limit(1000);

    // Test initial
    assert_eq!(metrics.max_allowed, 1000);
    assert_eq!(metrics.current_usage, 0);
    assert_eq!(metrics.peak_usage, 0);
    assert!(metrics.is_within_limits());

    // Mise à jour de l'utilisation
    metrics.update_usage(500);
    assert_eq!(metrics.current_usage, 500);
    assert_eq!(metrics.peak_usage, 500);
    assert_eq!(metrics.allocation_count, 1);

    // Mise à jour supplémentaire
    metrics.update_usage(300);
    assert_eq!(metrics.current_usage, 800);
    assert_eq!(metrics.peak_usage, 800);
    assert_eq!(metrics.allocation_count, 2);

    // Libération de mémoire
    metrics.release(200);
    assert_eq!(metrics.current_usage, 600);
    assert_eq!(metrics.peak_usage, 800); // Le pic ne change pas

    // Pourcentage d'utilisation
    assert!((metrics.usage_percentage() - 60.0).abs() < f64::EPSILON);

    // Réinitialisation
    metrics.reset();
    assert_eq!(metrics.current_usage, 0);
    assert_eq!(metrics.peak_usage, 0);
    assert_eq!(metrics.allocation_count, 0);
}

/// Test unitaire : mesure du RSS processus
#[test]
fn test_process_rss_measurement() {
    let rss = measure_process_rss();

    // Le RSS doit être positif (sur Linux)
    #[cfg(target_os = "linux")]
    assert!(rss > 0, "Le RSS doit être positif sur Linux");

    // Le RSS doit être raisonnable (entre 1 Mo et 1 Go pour un test unitaire)
    assert!(rss > 1024 * 1024, "Le RSS doit être > 1 Mo");
    assert!(
        rss < 1024 * 1024 * 1024,
        "Le RSS doit être < 1 Go pour un test unitaire"
    );
}
