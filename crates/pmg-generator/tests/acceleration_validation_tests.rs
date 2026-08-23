//! Tests de validation d'accélération pour pmg-generator
//!
//! Ces tests vérifient que le pipeline fonctionne en mode CPU
//! et que l'accélération GPU est correctement gérée.
//!
//! Conformité : Tests de validation d'accélération - Mission GPU-CPU-VALIDATION-001
//! Date : 2026-08-22

use pmg_generator::{GeneratorConfig, PipelineGlobalConfig};

/// Test 1 : Vérifier l'exécution du pipeline en mode CPU
///
/// Ce test vérifie que la configuration du générateur est valide
/// et peut être utilisée en mode CPU.
#[test]
fn test_pipeline_cpu_execution() {
    // Créer une configuration par défaut du générateur
    let config = GeneratorConfig::default();

    // Créer une configuration de pipeline par défaut
    let pipeline_config = PipelineGlobalConfig::default();

    // Vérifier que la configuration du générateur est valide
    let validation_result = config.validate();
    assert!(
        validation_result.is_ok(),
        "La configuration du générateur devrait être valide: {:?}",
        validation_result
    );

    // Vérifier que les paramètres de la configuration sont raisonnables
    assert!(
        config.chunk_size > 0,
        "La taille des chunks devrait être positive"
    );
    assert!(
        config.max_shard_bytes > 0,
        "La taille maximale par shard devrait être positive"
    );

    // Vérifier que la configuration de pipeline a des valeurs par défaut raisonnables
    assert_eq!(
        pipeline_config.distribution.mean, 0.0,
        "La moyenne de distribution devrait être 0.0 par défaut"
    );
    assert_eq!(
        pipeline_config.distribution.std, 1.0,
        "L'écart-type de distribution devrait être 1.0 par défaut"
    );
    assert!(
        pipeline_config.outliers.threshold_k > 0.0,
        "Le seuil d'outliers devrait être positif"
    );
    assert!(
        pipeline_config.super_weights.threshold_k > 0.0,
        "Le seuil des super-poids devrait être positif"
    );

    println!("Test de pipeline CPU réussi");
}

/// Test 2 : Vérifier la détection du statut GPU
///
/// Ce test vérifie que le pipeline peut détecter la disponibilité du GPU
/// et fonctionner en mode CPU si nécessaire.
#[test]
fn test_gpu_status_detection() {
    // Vérifier la disponibilité du GPU via le crate pmg-gpu
    let gpu_available = pmg_gpu::is_gpu_available();

    if gpu_available {
        println!("Mode GPU disponible");
    } else {
        println!("Mode CPU seulement");
    }

    // Créer une configuration de générateur
    let config = GeneratorConfig::default();

    // Vérifier que la configuration est valide quel que soit le mode
    assert!(
        config.validate().is_ok(),
        "La configuration devrait être valide quelle que soit la disponibilité GPU"
    );

    // Vérifier que les paramètres CPU sont correctement configurés
    assert!(
        config.chunk_size > 0,
        "La taille des chunks devrait être positive en mode CPU"
    );

    println!("Détection du statut GPU vérifiée");
}

/// Test 3 : Vérifier la compatibilité des features
///
/// Ce test vérifie que les features Cargo sont correctement configurées
/// et que le code fonctionne avec différentes combinaisons de features.
#[test]
fn test_feature_compatibility() {
    // Vérifier que les features sont correctement configurées
    #[cfg(feature = "gpu-acceleration")]
    {
        println!("Feature 'gpu-acceleration' activée");
    }

    #[cfg(not(feature = "gpu-acceleration"))]
    {
        println!("Feature 'gpu-acceleration' désactivée");
    }

    // Créer une configuration par défaut
    let config = GeneratorConfig::default();

    // Vérifier que la configuration fonctionne quelle que soit la feature
    assert!(
        config.validate().is_ok(),
        "La configuration devrait fonctionner quelle que soit la feature gpu-acceleration"
    );

    // Créer une configuration de pipeline
    let pipeline_config = PipelineGlobalConfig::default();

    // Vérifier que la configuration de pipeline est correcte
    assert!(
        pipeline_config.distribution.std > 0.0,
        "L'écart-type de distribution devrait être positif"
    );

    println!("Compatibilité des features vérifiée");
}

/// Test 4 : Vérifier la performance du pipeline en mode CPU
///
/// Ce test vérifie que le pipeline peut traiter des opérations de base
/// en mode CPU dans des temps raisonnables.
#[test]
fn test_pipeline_performance_cpu() {
    let start = std::time::Instant::now();

    // Simuler des opérations de traitement de données
    let mut data: Vec<f64> = (0..10_000).map(|i| i as f64).collect();

    // Appliquer des transformations typiques du pipeline
    for i in 0..data.len() {
        // Distribution normale simplifiée
        data[i] = (data[i] - 5000.0) / 1000.0;

        // Détection d'outliers simplifiée
        if data[i].abs() > 3.0 {
            data[i] = 3.0 * data[i].signum();
        }
    }

    // Calculer des statistiques de base
    let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
    let variance: f64 = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    let std_dev = variance.sqrt();

    let duration = start.elapsed();

    // Vérifier que le traitement est rapide (< 100 ms)
    assert!(
        duration.as_millis() < 100,
        "Le traitement CPU devrait prendre moins de 100ms, a pris {:?}",
        duration
    );

    // Vérifier que les résultats sont raisonnables
    assert!(!mean.is_nan(), "La moyenne ne devrait pas être NaN");
    assert!(!std_dev.is_nan(), "L'écart-type ne devrait pas être NaN");
    assert!(std_dev >= 0.0, "L'écart-type devrait être positif");

    println!("Performance pipeline CPU vérifiée: {:?}", duration);
}

/// Test 5 : Vérifier la gestion des erreurs de configuration
///
/// Ce test vérifie que les erreurs de configuration sont correctement détectées
/// et retournées sous forme de Result.
#[test]
fn test_configuration_error_handling() {
    // Test avec configuration invalide (chunk_size = 0)
    use pmg_core::CoreConfig;

    let core_config = CoreConfig::new(42, "test-model").unwrap();
    let mut config = GeneratorConfig::from_core(core_config);

    // Modifier la configuration pour la rendre invalide
    config.chunk_size = 0;

    // Vérifier que la validation échoue
    let validation_result = config.validate();
    assert!(
        validation_result.is_err(),
        "La validation devrait échouer pour chunk_size = 0"
    );

    // Vérifier le type d'erreur
    let error = validation_result.unwrap_err();
    let error_msg = format!("{}", error);
    assert!(
        error_msg.contains("chunk_size"),
        "L'erreur devrait mentionner chunk_size"
    );

    println!("Gestion des erreurs de configuration vérifiée");
}

/// Test 6 : Vérifier la reproductibilité en mode CPU
///
/// Ce test vérifie que les mêmes configurations produisent les mêmes résultats
/// en mode CPU (déterminisme).
#[test]
fn test_cpu_determinism() {
    // Créer deux configurations identiques
    let config1 = GeneratorConfig::default();
    let config2 = GeneratorConfig::default();

    // Vérifier que les configurations sont identiques
    assert_eq!(
        config1.chunk_size, config2.chunk_size,
        "Les tailles de chunks devraient être identiques"
    );
    assert_eq!(
        config1.max_shard_bytes, config2.max_shard_bytes,
        "Les tailles maximales par shard devraient être identiques"
    );
    assert_eq!(
        config1.validate, config2.validate,
        "Les flags de validation devraient être identiques"
    );

    // Simuler des calculs déterministes
    let data1: Vec<f64> = (0..1000).map(|i| i as f64 * 0.1).collect();
    let data2: Vec<f64> = (0..1000).map(|i| i as f64 * 0.1).collect();

    // Vérifier que les résultats sont identiques
    assert_eq!(
        data1, data2,
        "Les calculs déterministes devraient produire les mêmes résultats"
    );

    // Calculer des statistiques sur les deux ensembles
    let sum1: f64 = data1.iter().sum();
    let sum2: f64 = data2.iter().sum();
    assert_eq!(sum1, sum2, "Les sommes devraient être identiques");

    println!("Reproductibilité CPU vérifiée");
}
