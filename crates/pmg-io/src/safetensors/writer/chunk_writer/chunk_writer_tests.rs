//! Tests unitaires pour le ChunkWriter
//!
//! Ce module contient les tests unitaires pour le ChunkWriter, le BufferPool
//! et les métriques de performance.

use super::*;
use std::io::Read;
use tempfile::tempdir;

use crate::pool::{PoolConfig, UnifiedBufferPool};

/// Test de création d'un UnifiedBufferPool
#[test]
fn test_buffer_pool_new() {
    let config = PoolConfig::new(
        DEFAULT_MAX_POOL_MEMORY,
        MIN_CHUNK_SIZE,
        MAX_CHUNK_SIZE,
        true,
    );
    let pool = UnifiedBufferPool::new(config);
    assert_eq!(pool.memory_usage(), 0);
    assert_eq!(pool.buffer_count(), 0);
}

/// Test d'acquisition et de libération de buffers
#[test]
fn test_buffer_pool_acquire_release() {
    let config = PoolConfig::new(
        DEFAULT_MAX_POOL_MEMORY,
        MIN_CHUNK_SIZE,
        MAX_CHUNK_SIZE,
        true,
    );
    let pool = UnifiedBufferPool::new(config);

    // Vérification initiale
    assert_eq!(pool.memory_usage(), 0);
    assert_eq!(pool.buffer_count(), 0);

    // Acquisition d'un buffer
    let buffer = pool.acquire_u8(MIN_CHUNK_SIZE);
    // Le buffer a len=0 et capacity>=MIN_CHUNK_SIZE
    assert!(buffer.capacity() >= MIN_CHUNK_SIZE);

    // Libération du buffer
    let capacity = buffer.capacity();
    pool.release_u8(buffer);
    // La mémoire est comptabilisée lorsque le buffer est dans le pool
    assert!(pool.memory_usage() > 0);
    assert_eq!(pool.buffer_count(), 1);

    // Réutilisation du buffer avec la même taille
    let buffer2 = pool.acquire_u8(capacity);
    // Après réutilisation, le buffer n'est plus dans le pool
    assert_eq!(pool.buffer_count(), 0);
    // Le buffer a len=0 et capacity>=capacity
    assert!(buffer2.capacity() >= capacity);
}

/// Test de création d'un ChunkWriter
#[test]
fn test_chunk_writer_new() -> std::io::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("test.safetensors");

    let writer = ChunkWriter::new(&path, 0)?;
    assert_eq!(writer.current_offset(), 0);
    assert_eq!(writer.chunk_size(), DEFAULT_CHUNK_SIZE);

    // Vérification que le fichier a été créé
    assert!(path.exists());

    Ok(())
}

/// Test d'écriture d'un petit tenseur
#[test]
fn test_chunk_writer_write_small_tensor() -> std::io::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("test_small.safetensors");

    let mut writer = ChunkWriter::new(&path, 0)?;

    // Données de test (1 Ko)
    let data = vec![42u8; 1024];
    let result = writer.write_tensor("test_tensor", &data, DType::F32, &[1024])?;

    assert_eq!(result.bytes_written, 1024);
    assert_eq!(result.chunks_used, 1); // Un seul chunk pour les petites données

    // Finalisation pour flush les données sur disque
    writer.finalize()?;

    // Vérification que le fichier contient les données
    let mut file = File::open(&path)?;
    let mut file_data = Vec::new();
    file.read_to_end(&mut file_data)?;
    assert_eq!(file_data, data);

    Ok(())
}

/// Test d'écriture d'un grand tenseur
#[test]
fn test_chunk_writer_write_large_tensor() -> std::io::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("test_large.safetensors");

    let mut writer = ChunkWriter::new(&path, 0)?;

    // Données de test (2 Mo)
    let data = vec![123u8; 2 * 1024 * 1024];
    let result = writer.write_tensor("large_tensor", &data, DType::F32, &[2 * 1024 * 1024])?;

    assert_eq!(result.bytes_written, 2 * 1024 * 1024);
    // Devrait utiliser plusieurs chunks
    assert!(result.chunks_used > 1);

    Ok(())
}

/// Test de la taille adaptative des chunks
#[test]
fn test_adaptive_chunk_size() -> std::io::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("test_adaptive.safetensors");

    let writer = ChunkWriter::new(&path, 0)?;

    // Test pour un petit tenseur (< 1 Mo)
    let small_size = 512 * 1024; // 512 Ko
    assert_eq!(writer.adaptive_chunk_size(small_size), small_size);

    // Test pour un tenseur moyen (1-32 Mo)
    let medium_size = 16 * 1024 * 1024; // 16 Mo
    assert_eq!(writer.adaptive_chunk_size(medium_size), MIN_CHUNK_SIZE);

    // Test pour un grand tenseur (> 32 Mo)
    let large_size = 64 * 1024 * 1024; // 64 Mo
    assert_eq!(writer.adaptive_chunk_size(large_size), DEFAULT_CHUNK_SIZE);

    Ok(())
}

/// Test des métriques de performance
#[test]
fn test_writer_metrics() {
    let mut metrics = ChunkWriterMetrics::new();

    // Simulation d'écritures
    metrics.update_write(1024, 1, 10);
    metrics.update_write(2048, 2, 20);
    metrics.increment_buffer_reuse();
    metrics.update_peak_memory(1024);

    assert_eq!(metrics.bytes_written, 3072);
    assert_eq!(metrics.chunks_written, 3);
    assert_eq!(metrics.total_write_time_ms, 30);
    assert_eq!(metrics.buffer_reuses, 1);
    assert_eq!(metrics.peak_memory_usage, 1024);
}

/// Test de finalisation du ChunkWriter
#[test]
fn test_chunk_writer_finalize() -> std::io::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("test_finalize.safetensors");

    let mut writer = ChunkWriter::new(&path, 0)?;

    // Écriture de quelques données
    let data = vec![1u8; 1024];
    writer.write_tensor("tensor1", &data, DType::F32, &[1024])?;

    // Finalisation
    let metrics = writer.finalize()?;

    assert!(metrics.bytes_written > 0);
    assert!(metrics.chunks_written > 0);

    Ok(())
}
