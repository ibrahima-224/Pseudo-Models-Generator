//! Tests unitaires pour le ChunkWriter
//!
//! Ce fichier contient les tests pour le module chunk_writer,
//! vérifiant le bon fonctionnement de l'écriture par chunks
//! et du pool de buffers.

use pmg_io::pool::{PoolConfig, UnifiedBufferPool};
use pmg_io::safetensors::DType;
use pmg_io::safetensors::{
    ChunkWriter, ChunkWriterMetrics, DEFAULT_MAX_POOL_MEMORY, MAX_CHUNK_SIZE, MIN_CHUNK_SIZE,
};
use tempfile::tempdir;

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
    pool.release_u8(buffer);
    assert!(pool.memory_usage() > 0);
    assert_eq!(pool.buffer_count(), 1);

    // Réutilisation du buffer
    let _buffer2 = pool.acquire_u8(MIN_CHUNK_SIZE);
    assert_eq!(pool.buffer_count(), 0);
}

/// Test de création d'un ChunkWriter
#[test]
fn test_chunk_writer_new() -> std::io::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("test.safetensors");

    let writer = ChunkWriter::new(&path, 0)?;
    assert_eq!(writer.current_offset(), 0);
    // La taille du chunk par défaut est de 8 Mo (définie dans chunk_writer.rs)
    assert_eq!(writer.chunk_size(), 8 * 1024 * 1024);

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
    let _metrics = writer.finalize()?;

    // Vérification que le fichier existe
    assert!(path.exists());

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

    let mut writer = ChunkWriter::new(&path, 0)?;

    // Test pour un petit tenseur (< 1 Mo)
    let small_size = 512 * 1024; // 512 Ko
                                 // La méthode adaptive_chunk_size est privée, donc nous testons via write_tensor
                                 // Nous vérifions simplement que l'écriture fonctionne
    let data_small = vec![42u8; small_size];
    let result_small =
        writer.write_tensor("small", &data_small, DType::F32, &[small_size as u64 / 4])?;
    assert!(result_small.bytes_written > 0);

    // Test pour un tenseur moyen (1-32 Mo)
    let medium_size = 16 * 1024 * 1024; // 16 Mo
    let data_medium = vec![42u8; medium_size];
    let result_medium = writer.write_tensor(
        "medium",
        &data_medium,
        DType::F32,
        &[medium_size as u64 / 4],
    )?;
    assert!(result_medium.bytes_written > 0);

    // Test pour un grand tenseur (> 32 Mo)
    let large_size = 64 * 1024 * 1024; // 64 Mo
    let data_large = vec![42u8; large_size];
    let result_large =
        writer.write_tensor("large", &data_large, DType::F32, &[large_size as u64 / 4])?;
    assert!(result_large.bytes_written > 0);

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

/// Test d'écriture d'un chunk unique
#[test]
fn test_chunk_writer_write_tensor_chunk() -> std::io::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("test_chunk.safetensors");

    let mut writer = ChunkWriter::new(&path, 0)?;

    // Données de test
    let data = vec![1u8; 1024];
    writer.write_tensor_chunk("test_tensor", &data, 0, 2048, DType::F32, &[2048])?;

    // Vérification que l'offset a été mis à jour
    assert_eq!(writer.current_offset(), 1024);

    Ok(())
}

/// Test de vidage et synchronisation
#[test]
fn test_chunk_writer_flush_and_sync() -> std::io::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("test_flush.safetensors");

    let mut writer = ChunkWriter::new(&path, 0)?;

    // Écriture de données
    let data = vec![1u8; 1024];
    writer.write_tensor("tensor1", &data, DType::F32, &[1024])?;

    // Vidage et synchronisation
    writer.flush_and_sync()?;

    Ok(())
}

/// Test de création avec taille de chunk invalide
#[test]
fn test_chunk_writer_invalid_chunk_size() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("test_invalid.safetensors");

    // Taille trop petite
    let result = ChunkWriter::with_chunk_size(&path, 0, MIN_CHUNK_SIZE - 1);
    assert!(result.is_err());

    // Taille trop grande
    let result = ChunkWriter::with_chunk_size(&path, 0, MAX_CHUNK_SIZE + 1);
    assert!(result.is_err());
}

/// Test du pool de buffers avec limite mémoire
#[test]
fn test_buffer_pool_memory_limit() {
    // Utiliser un pool avec limite mémoire de 4 Mo
    let config = PoolConfig::new(
        4 * 1024 * 1024, // Limite de 4 Mo
        MIN_CHUNK_SIZE,
        MAX_CHUNK_SIZE,
        true,
    );
    let pool = UnifiedBufferPool::new(config);

    // Acquisition de buffers (MIN_CHUNK_SIZE = 1 Mo)
    let buffer1 = pool.acquire_u8(MIN_CHUNK_SIZE);
    // Le buffer a len=0 et capacity>=MIN_CHUNK_SIZE
    assert!(buffer1.capacity() >= MIN_CHUNK_SIZE);
    pool.release_u8(buffer1);

    // Le pool stocke un buffer
    assert!(pool.memory_usage() > 0);
    assert_eq!(pool.buffer_count(), 1);

    // Réutilisation du buffer (pas de nouveau buffer créé)
    let buffer2 = pool.acquire_u8(MIN_CHUNK_SIZE);
    pool.release_u8(buffer2);

    // Le pool stocke toujours un seul buffer
    assert!(pool.memory_usage() > 0);
    assert_eq!(pool.buffer_count(), 1);
}

/// Test de nettoyage du pool
#[test]
fn test_buffer_pool_clear() {
    let config = PoolConfig::new(
        DEFAULT_MAX_POOL_MEMORY,
        MIN_CHUNK_SIZE,
        MAX_CHUNK_SIZE,
        true,
    );
    let pool = UnifiedBufferPool::new(config);

    // Ajout de buffers
    let buffer = pool.acquire_u8(MIN_CHUNK_SIZE);
    pool.release_u8(buffer);

    assert_eq!(pool.buffer_count(), 1);

    // Nettoyage
    pool.clear();

    assert_eq!(pool.buffer_count(), 0);
    assert_eq!(pool.memory_usage(), 0);
}
