//! Tests pour le module multi-GPU
//!
//! Ce module contient les tests unitaires et d'intégration
//! pour la gestion multi-GPU et la distribution de travail.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::device::GpuDevice;
use crate::error::GpuResult;
use crate::multi_gpu::work_distributor::{create_multi_gpu_system, WorkDistributor};
use crate::multi_gpu::{DeviceStrategy, MultiGpuPool};

/// Fonction worker pour les tests de multiplication
fn multiply_worker(item: i32, _device: Arc<Mutex<GpuDevice>>) -> GpuResult<i32> {
    Ok(item * 2)
}

/// Fonction worker pour les tests avec compteur
fn counting_worker(
    item: i32,
    _device: Arc<Mutex<GpuDevice>>,
    counter: &AtomicUsize,
) -> GpuResult<i32> {
    counter.fetch_add(1, Ordering::SeqCst);
    Ok(item * 2)
}

/// Fonction worker pour les tests de base
fn identity_worker(item: i32, _device: Arc<Mutex<GpuDevice>>) -> GpuResult<i32> {
    Ok(item)
}

/// Fonction worker pour les tests d'addition
fn add_one_worker(item: i32, _device: Arc<Mutex<GpuDevice>>) -> GpuResult<i32> {
    Ok(item + 1)
}

/// Fonction worker pour les tests de multiplication par 3
fn triple_worker(item: i32, _device: Arc<Mutex<GpuDevice>>) -> GpuResult<i32> {
    Ok(item * 3)
}

/// Fonction worker pour les tests de multiplication par 4
fn quadruple_worker(item: i32, _device: Arc<Mutex<GpuDevice>>) -> GpuResult<i32> {
    Ok(item * 4)
}

/// Test de création du pool multi-GPU
#[test]
fn test_multi_gpu_pool_creation() {
    let pool = MultiGpuPool::new();
    // Le pool peut être vide si pas de GPU disponible (mode fallback simule 1 device)
    assert!(pool.is_ok());
}

/// Test de la stratégie round-robin
#[test]
fn test_round_robin_strategy() {
    let pool = Arc::new(MultiGpuPool::default());
    let distributor = WorkDistributor::new(pool.clone(), 10);

    let work_items: Vec<i32> = (0..100).collect();
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_clone = counter.clone();

    // Créer un worker avec compteur via closure
    let worker = Arc::new(
        move |item: i32, device: Arc<Mutex<GpuDevice>>| -> GpuResult<i32> {
            counting_worker(item, device, &counter_clone)
        },
    );

    let results = distributor.distribute(work_items, worker);

    assert!(results.is_ok());
    let all_results: Vec<i32> = results.unwrap().into_iter().flatten().collect();
    assert_eq!(all_results.len(), 100);
    assert_eq!(counter.load(Ordering::SeqCst), 100);
}

/// Test de la stratégie least-used
#[test]
fn test_least_used_strategy() {
    let pool = Arc::new(MultiGpuPool::with_strategy(DeviceStrategy::LeastUsed).unwrap());
    let distributor = WorkDistributor::new(pool, 5);

    let work_items: Vec<i32> = (0..50).collect();
    let worker = Arc::new(triple_worker);

    let results = distributor.distribute(work_items, worker);

    assert!(results.is_ok());
}

/// Test de distribution automatique
#[test]
fn test_auto_distribution() {
    let pool = Arc::new(MultiGpuPool::default());
    let distributor = WorkDistributor::new(pool, 10);

    let work_items: Vec<i32> = (0..200).collect();
    let worker = Arc::new(add_one_worker);

    let results = distributor.distribute_auto(work_items, worker);

    assert!(results.is_ok());
}

/// Test des statistiques
#[test]
fn test_statistics_tracking() {
    let pool = Arc::new(MultiGpuPool::default());
    let distributor = WorkDistributor::new(pool.clone(), 5);

    let work_items: Vec<i32> = (0..20).collect();
    let worker = Arc::new(identity_worker);

    let _ = distributor.distribute(work_items, worker);

    let stats = pool.all_stats();
    assert!(!stats.is_empty());

    // Vérifier que les compteurs ont été mis à jour
    for (_, device_stats) in stats.iter() {
        assert!(device_stats.task_count > 0);
    }
}

/// Test du helper de création
#[test]
fn test_create_multi_gpu_system() {
    let result = create_multi_gpu_system(DeviceStrategy::RoundRobin, 1024);
    assert!(result.is_ok());

    let (pool, distributor) = result.unwrap();
    assert_eq!(pool.device_count(), distributor.pool().device_count());
}

/// Test de la stratégie most-memory
#[test]
fn test_most_memory_strategy() {
    let pool = Arc::new(MultiGpuPool::with_strategy(DeviceStrategy::MostMemory).unwrap());
    let distributor = WorkDistributor::new(pool, 10);

    let work_items: Vec<i32> = (0..30).collect();
    let worker = Arc::new(quadruple_worker);

    let results = distributor.distribute(work_items, worker);

    assert!(results.is_ok());
}

/// Test de réinitialisation des statistiques
#[test]
fn test_reset_stats() {
    let pool = Arc::new(MultiGpuPool::default());
    let distributor = WorkDistributor::new(pool.clone(), 5);

    let work_items: Vec<i32> = (0..10).collect();
    let worker = Arc::new(identity_worker);

    let _ = distributor.distribute(work_items, worker);

    // Vérifier que les stats ont été mises à jour
    let stats_before = pool.all_stats();
    assert!(stats_before.values().any(|s| s.task_count > 0));

    // Réinitialiser
    pool.reset_stats();

    let stats_after = pool.all_stats();
    assert!(stats_after.values().all(|s| s.task_count == 0));
}

/// Test de changement de stratégie
#[test]
fn test_strategy_change() {
    let pool = Arc::new(MultiGpuPool::default());

    assert_eq!(pool.default_strategy(), DeviceStrategy::RoundRobin);

    pool.set_default_strategy(DeviceStrategy::LeastUsed);
    assert_eq!(pool.default_strategy(), DeviceStrategy::LeastUsed);

    pool.set_default_strategy(DeviceStrategy::MostMemory);
    assert_eq!(pool.default_strategy(), DeviceStrategy::MostMemory);
}

/// Test de get_device avec différentes stratégies
#[test]
fn test_get_device_strategies() {
    let pool = MultiGpuPool::default();

    // RoundRobin
    let device1 = pool.get_device(DeviceStrategy::RoundRobin);
    assert!(device1.is_ok());

    // LeastUsed
    let device2 = pool.get_device(DeviceStrategy::LeastUsed);
    assert!(device2.is_ok());

    // MostMemory
    let device3 = pool.get_device(DeviceStrategy::MostMemory);
    assert!(device3.is_ok());
}

/// Test de thread-safety avec accès concurrent
#[test]
fn test_concurrent_access() {
    let pool = Arc::new(MultiGpuPool::default());
    let distributor = Arc::new(WorkDistributor::new(pool.clone(), 5));

    let mut handles = Vec::new();

    // Lancer plusieurs threads qui distribuent du travail
    for i in 0..4 {
        let distributor = distributor.clone();
        let handle = std::thread::spawn(move || {
            let work_items: Vec<i32> = (0..20).map(|x| x + i * 20).collect();
            let worker = Arc::new(multiply_worker);
            distributor.distribute(work_items, worker)
        });
        handles.push(handle);
    }

    // Vérifier que tous les threads ont réussi
    for handle in handles {
        let result = handle.join().unwrap();
        assert!(result.is_ok());
    }

    // Vérifier que les statistiques sont cohérentes
    // Chaque thread distribue 20 items en chunks de 5 = 4 chunks par thread
    // 4 threads * 4 chunks = 16 tâches au total
    let stats = pool.all_stats();
    let total_tasks: u64 = stats.values().map(|s| s.task_count).sum();
    assert!(total_tasks > 0, "Au moins une tâche doit être enregistrée");
    assert!(
        total_tasks <= 16,
        "Maximum 16 tâches attendues (4 threads × 4 chunks)"
    );
}
