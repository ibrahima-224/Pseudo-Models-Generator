//! Tests de Limite Mémoire pour PMG
//!
//! Ce module contient des tests pour vérifier que la mémoire utilisée
//! ne dépasse jamais la limite de 1 Go, même pour de grandes générations.

use std::sync::Arc;
use tempfile::tempdir;

use pmg_blueprint::tensor_spec::TensorSpec;
use pmg_core::memory::{GlobalMemoryManager, MemoryMonitor};
use pmg_core::{DType, Shape, TensorRole};
use pmg_generator::optimized_generator::OptimizedTensorChunkGenerator;
use pmg_io::safetensors::writer::zero_copy::{TensorWriterConfig, ZeroCopyTensorWriter};
use pmg_io::safetensors::ShardWriter;

/// Test de la limite mémoire avec le GlobalMemoryManager
#[test]
fn test_limite_memoire_1_go() {
    // Créer un gestionnaire avec limite de 1 Go
    let manager = Arc::new(GlobalMemoryManager::with_limit(1_073_741_824));

    // Vérifier que la limite est correcte
    assert_eq!(manager.max_memory(), 1_073_741_824);

    // Simuler des allocations successives
    let chunk_size = 8 * 1024 * 1024; // 8 Mo
    let mut allocated = 0;

    // Allouer jusqu'à la limite
    while manager.can_allocate(chunk_size) {
        manager.allocate(chunk_size as u64).unwrap();
        allocated += chunk_size;
    }

    // Vérifier qu'on ne dépasse pas la limite
    assert!(allocated <= 1_073_741_824);
    assert!(manager.current_usage() <= 1_073_741_824);

    // Vérifier qu'on ne peut plus allouer
    assert!(!manager.can_allocate(chunk_size));

    // Libérer la mémoire
    manager.deallocate(allocated as u64);
    assert_eq!(manager.current_usage(), 0);
}

/// Test de génération avec limite mémoire stricte
#[test]
fn test_generation_avec_limite_memoire() {
    // Créer un gestionnaire avec limite de 1 Go
    let manager = Arc::new(GlobalMemoryManager::with_limit(1_073_741_824));

    // Créer un générateur optimisé
    let config = pmg_generator::streaming_config::StreamingConfig {
        chunk_size: 8 * 1024 * 1024, // 8 Mo
        ..Default::default()
    };

    let mut generator = OptimizedTensorChunkGenerator::new(config, manager.clone(), 12345);

    // Créer un spec de tenseur de taille raisonnable
    let spec = TensorSpec::new(
        "test.limite",
        Shape::new(vec![1024, 1024]).unwrap(), // 1M éléments
        DType::F32,
        TensorRole::Embedding,
    )
    .unwrap();

    // Créer un writer temporaire
    let dir = tempdir().unwrap();
    let path = dir.path().join("test_limite.safetensors");
    let shard_writer = ShardWriter::new(path, 1024).unwrap();
    let writer_config = TensorWriterConfig::default();
    let mut writer =
        ZeroCopyTensorWriter::new(shard_writer, manager.clone(), writer_config).unwrap();

    // Générer le tenseur
    let result = generator
        .generate_and_write_tensor(&spec, &mut writer, 0)
        .unwrap();

    // Vérifier que la génération a réussi
    assert!(result.total_elements > 0);
    assert!(result.chunks_written > 0);

    // Vérifier que la mémoire n'a pas dépassé la limite
    assert!(manager.current_usage() <= 1_073_741_824);
    assert!(manager.peak_usage() <= 1_073_741_824);
}

/// Test de monitoring mémoire en temps réel
#[test]
fn test_monitoring_temps_reel() {
    let manager = Arc::new(GlobalMemoryManager::with_limit(1024 * 1024)); // 1 Mo pour test

    // Simuler des allocations et désallocations
    manager.allocate(512 * 1024).unwrap(); // 512 Ko
    assert_eq!(manager.current_usage(), 512 * 1024);

    // Vérifier l'état
    let status = manager.status();
    assert_eq!(status.current_usage, 512 * 1024);
    assert_eq!(status.max_memory, 1024 * 1024);

    // Vérifier les métriques
    let metrics = manager.detailed_metrics();
    assert_eq!(metrics.operation_peak, 512 * 1024);
    assert_eq!(metrics.allocations_by_type.len(), 0);

    // Libérer
    manager.deallocate(512 * 1024);
    assert_eq!(manager.current_usage(), 0);
}

/// Test de performance avec buffer optimisé
#[test]
fn test_performance_buffer_optimise() {
    // Créer un gestionnaire avec limite de 1 Mo
    let manager = Arc::new(GlobalMemoryManager::with_limit(1024 * 1024));

    // Configuration personnalisée avec des tailles adaptées à la limite de 1 Mo
    let pool_config = pmg_io::pool::PoolConfig {
        max_memory_per_pool: 512 * 1024, // 512 Ko par pool (total 1 Mo pour u8 et f64)
        min_buffer_size: 1024,           // 1 Ko - taille minimale des buffers
        max_buffer_size: 1024 * 1024,    // 1 Mo - taille maximale des buffers
        enable_metrics: true,
    };

    let pool = pmg_io::pool::OptimizedBufferPool::new(manager.clone(), pool_config);

    // Acquérir et retourner des buffers
    let buffer1 = pool.acquire_u8(1024).unwrap();
    let buffer2 = pool.acquire_f64(1024).unwrap();

    // Vérifier les tailles
    assert_eq!(buffer1.len(), 1024);
    assert_eq!(buffer2.len(), 1024);

    // Retourner les buffers
    pool.release_u8(buffer1);
    pool.release_f64(buffer2);

    // Vérifier les statistiques
    let stats = pool.stats();
    assert_eq!(stats.total_allocations, 2);
    assert_eq!(stats.total_reuses, 0);
}

/// Test de détection de fuites mémoire
#[test]
fn test_detection_fuites_memoire() {
    let manager = Arc::new(GlobalMemoryManager::with_limit(1024 * 1024));

    // Simuler des allocations/désallocations
    for _ in 0..10 {
        manager.allocate(1024).unwrap();
        manager.deallocate(1024);
    }

    // Vérifier qu'il n'y a pas de fuite
    assert_eq!(manager.current_usage(), 0);
    assert_eq!(manager.peak_usage(), 1024);

    // Vérifier les métriques
    let metrics = manager.detailed_metrics();
    assert_eq!(metrics.allocations_by_type.len(), 0);
}

/// Test de stress mémoire avec grand tenseur
#[test]
#[ignore] // Test gourmand en mémoire (optimisation mémoire)
fn test_stress_grand_tenseur() {
    // Créer un gestionnaire avec limite de 1 Go
    let manager = Arc::new(GlobalMemoryManager::with_limit(1_073_741_824));

    // Créer un générateur optimisé
    let config = pmg_generator::streaming_config::StreamingConfig {
        chunk_size: 16 * 1024 * 1024, // 16 Mo
        ..Default::default()
    };

    let mut generator = OptimizedTensorChunkGenerator::new(config, manager.clone(), 54321);

    // Créer un spec de tenseur très grand (2 Go de données)
    let spec = TensorSpec::new(
        "test.stress",
        Shape::new(vec![16384, 16384]).unwrap(), // 256M éléments = 1 Go en f32
        DType::F32,
        TensorRole::Embedding,
    )
    .unwrap();

    // Créer un writer temporaire
    let dir = tempdir().unwrap();
    let path = dir.path().join("test_stress.safetensors");
    let shard_writer = ShardWriter::new(path, 1024).unwrap();
    let writer_config = TensorWriterConfig::default();
    let mut writer =
        ZeroCopyTensorWriter::new(shard_writer, manager.clone(), writer_config).unwrap();

    // Générer le tenseur (devrait fonctionner avec streaming)
    let result = generator
        .generate_and_write_tensor(&spec, &mut writer, 0)
        .unwrap();

    // Vérifier que la génération a réussi
    assert!(result.total_elements > 0);
    assert!(result.chunks_written > 0);

    // Vérifier que la mémoire n'a jamais dépassé 1 Go
    assert!(manager.peak_usage() <= 1_073_741_824);

    // Vérifier les métriques du générateur
    let gen_metrics = generator.metrics();
    assert!(gen_metrics.total_bytes_written > 0);
    assert!(gen_metrics.peak_memory_usage <= 1_073_741_824);
}

/// Test de concurrence multi-thread pour la gestion mémoire
///
/// Ce test vérifie que le GlobalMemoryManager reste thread-safe
/// même avec plusieurs threads qui allouent et désallouent simultanément.
#[test]
#[ignore] // Test de concurrence gourmand (optimisation mémoire)
fn test_concurrence_multi_thread() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;

    // Créer un gestionnaire avec limite de 1 Mo pour les tests
    let manager = Arc::new(GlobalMemoryManager::with_limit(1024 * 1024));
    let thread_count = 10;
    let allocations_per_thread = 100;
    let allocation_size = 1024; // 1 Ko par allocation

    // Compteur pour suivre les allocations réussies
    let successful_allocations = Arc::new(AtomicUsize::new(0));
    let mut handles = vec![];

    // Lancer plusieurs threads qui allouent et désallouent simultanément
    for thread_id in 0..thread_count {
        let manager_clone = manager.clone();
        let counter_clone = successful_allocations.clone();

        let handle = thread::spawn(move || {
            for i in 0..allocations_per_thread {
                // Essayer d'allouer
                if manager_clone.can_allocate(allocation_size as u64) {
                    match manager_clone.allocate(allocation_size as u64) {
                        Ok(()) => {
                            counter_clone.fetch_add(1, Ordering::SeqCst);

                            // Simuler un traitement
                            thread::yield_now();

                            // Désallouer
                            manager_clone.deallocate(allocation_size as u64);
                        },
                        Err(_) => {
                            // L'allocation a échoué (limite atteinte), c'est normal
                        },
                    }
                }
            }
        });

        handles.push(handle);
    }

    // Attendre que tous les threads terminent
    for handle in handles {
        handle.join().unwrap();
    }

    // Vérifier que la mémoire est entièrement libérée
    assert_eq!(manager.current_usage(), 0);

    // Vérifier que le pic n'a pas dépassé la limite
    assert!(manager.peak_usage() <= 1024 * 1024);

    // Vérifier qu'il y a eu des allocations réussies
    let total_allocations = successful_allocations.load(Ordering::SeqCst);
    assert!(total_allocations > 0, "Aucune allocation réussie");

    println!(
        "Allocations réussies: {}/{}",
        total_allocations,
        thread_count * allocations_per_thread
    );
}

/// Test de concurrence pour le pool de buffers
///
/// Ce test vérifie que le OptimizedBufferPool reste thread-safe
/// même avec plusieurs threads qui acquièrent et retournent des buffers.
#[test]
#[ignore] // Test de pool gourmand (optimisation mémoire)
fn test_concurrence_pool_buffers() {
    use std::sync::Arc;
    use std::thread;

    // Créer un gestionnaire avec limite de 1 Mo
    let manager = Arc::new(GlobalMemoryManager::with_limit(1024 * 1024));

    // Configuration du pool
    let pool_config = pmg_io::pool::PoolConfig {
        max_memory_per_pool: 512 * 1024, // 512 Ko par pool
        min_buffer_size: 256,            // 256 octets
        max_buffer_size: 1024 * 1024,    // 1 Mo
        enable_metrics: true,
    };

    let pool = Arc::new(pmg_io::pool::OptimizedBufferPool::new(
        manager.clone(),
        pool_config,
    ));
    let thread_count = 8;
    let iterations_per_thread = 50;
    let mut handles = vec![];

    // Lancer plusieurs threads qui acquièrent et retournent des buffers
    for thread_id in 0..thread_count {
        let pool_clone = pool.clone();
        let is_f64 = thread_id % 2 == 0; // Alterner entre u8 et f64

        let handle = thread::spawn(move || {
            for i in 0..iterations_per_thread {
                if is_f64 {
                    // Acquérir un buffer f64
                    match pool_clone.acquire_f64(256) {
                        Ok(buffer) => {
                            // Simuler un traitement
                            assert_eq!(buffer.len(), 256);
                            thread::yield_now();
                            // Le buffer est automatiquement retourné au pool via Drop
                        },
                        Err(_) => {
                            // L'acquisition a échoué, c'est normal
                        },
                    }
                } else {
                    // Acquérir un buffer u8
                    match pool_clone.acquire_u8(1024) {
                        Ok(buffer) => {
                            // Simuler un traitement
                            assert_eq!(buffer.len(), 1024);
                            thread::yield_now();
                            // Le buffer est automatiquement retourné au pool via Drop
                        },
                        Err(_) => {
                            // L'acquisition a échoué, c'est normal
                        },
                    }
                }
            }
        });

        handles.push(handle);
    }

    // Attendre que tous les threads terminent
    for handle in handles {
        handle.join().unwrap();
    }

    // Vérifier que la mémoire est entièrement libérée
    // Note: Le pool peut conserver des buffers en interne, donc on vérifie
    // que la mémoire utilisée ne dépasse pas la limite
    assert!(manager.current_usage() <= 1024 * 1024);

    // Vérifier les statistiques du pool
    let stats = pool.stats();
    assert!(
        stats.total_allocations > 0,
        "Aucune allocation dans le pool"
    );
    println!("Statistiques pool: {:?}", stats);
}
