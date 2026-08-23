//! Tests unitaires pour le module d'allocation GPU
//!
//! Ce module contient les tests pour les allocateurs GPU et les pools.

use super::*;
use std::sync::Arc;

use crate::device::GpuDevice;
#[test]
fn test_allocator_creation() {
    let device = Arc::new(GpuDevice::new(0).unwrap());
    let allocator = GpuAllocator::new(device);
    assert!(allocator.is_ok());
}

#[test]
fn test_allocator_stats() {
    let device = Arc::new(GpuDevice::new(0).unwrap());
    let allocator = Arc::new(GpuAllocator::new(device).unwrap());

    let stats = allocator.stats();
    assert_eq!(stats.total_allocations, 0);
    assert_eq!(stats.active_allocations, 0);
}

#[test]
fn test_allocation_and_release() {
    let device = Arc::new(GpuDevice::new(0).unwrap());
    let allocator = Arc::new(GpuAllocator::new(device).unwrap());

    // Allouer un bloc
    let block = Arc::clone(&allocator).allocate(1024, Some("test"));
    assert!(block.is_ok());

    let block = block.unwrap();
    assert_eq!(block.size(), 1024);

    // Vérifier les statistiques
    let stats = allocator.stats();
    assert_eq!(stats.total_allocations, 1);
    assert_eq!(stats.active_allocations, 1);

    // Libérer le bloc
    drop(block);

    // Vérifier les statistiques après libération
    let stats = allocator.stats();
    assert_eq!(stats.total_deallocations, 1);
    assert_eq!(stats.active_allocations, 0);
}

#[test]
fn test_pool_allocator() {
    let device = Arc::new(GpuDevice::new(0).unwrap());
    let allocator = Arc::new(GpuAllocator::new(device).unwrap());
    let pool = GpuPoolAllocator::new(allocator, 1024 * 1024, 10);

    // Allouer un bloc
    let block = pool.allocate(1024);
    assert!(block.is_ok());

    let block = block.unwrap();
    assert_eq!(block.size(), 1024);

    // Retourner au pool
    let returned = pool.release(block);
    assert!(returned);

    // Vérifier les statistiques du pool
    let stats = pool.pool_stats();
    assert_eq!(stats.total_blocks, 1);
}

#[test]
fn test_thread_safety() {
    // Test de thread-safety avec Arc<GpuAllocator>
    let device = Arc::new(GpuDevice::new(0).unwrap());
    let allocator = Arc::new(GpuAllocator::new(device).unwrap());

    let mut handles = vec![];

    for i in 0..10 {
        let alloc_clone = Arc::clone(&allocator);
        handles.push(std::thread::spawn(move || {
            // Allouer depuis différents threads
            let block = alloc_clone.allocate(1024 * (i + 1), Some(&format!("thread_{}", i)));
            assert!(block.is_ok());

            let block = block.unwrap();
            assert_eq!(block.size(), 1024 * (i + 1));

            // Laisser le bloc se libérer automatiquement
            drop(block);
        }));
    }

    // Attendre que tous les threads terminent
    for handle in handles {
        handle.join().unwrap();
    }

    // Vérifier que toutes les allocations ont été libérées
    let stats = allocator.stats();
    assert_eq!(stats.active_allocations, 0);
    assert_eq!(stats.total_allocations, 10);
    assert_eq!(stats.total_deallocations, 10);
}

#[test]
fn test_lifetime_safety() {
    // Test de lifetime - l'allocateur reste vivant tant qu'il y a des blocs
    let device = Arc::new(GpuDevice::new(0).unwrap());
    let allocator = Arc::new(GpuAllocator::new(device).unwrap());

    let block = Arc::clone(&allocator)
        .allocate(2048, Some("lifetime_test"))
        .unwrap();

    // L'allocateur est toujours vivant
    assert!(Arc::strong_count(&allocator) >= 2); // allocator + block

    // Créer une autre référence
    let allocator_ref = Arc::clone(&allocator);

    // Libérer le bloc
    drop(block);

    // L'allocateur est toujours vivant grâce à allocator_ref
    let stats = allocator_ref.stats();
    assert_eq!(stats.active_allocations, 0);
}
