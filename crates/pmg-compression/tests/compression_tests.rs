//! Tests de compression pour pmg-compression

use pmg_compression::{CompressionAlgorithm, CompressionConfig, CompressionStats, Compressor};
use std::time::Instant;

#[test]
fn test_compression_lz4_basic() {
    // Données de test répétitives (facilement compressibles)
    let data = vec![0u8; 1024 * 1024]; // 1 Mo de zéros
    let config = CompressionConfig {
        algorithm: CompressionAlgorithm::Lz4,
        level: 6,
        block_size: 64 * 1024,
        in_memory: true,
        use_dictionary: false,
        max_buffer_size: 8 * 1024 * 1024,
    };

    let mut compressor = Compressor::new(config).unwrap();

    // Compression
    let start = Instant::now();
    let compressed = compressor.compress(&data).unwrap();
    let compression_time = start.elapsed().as_millis() as f64;

    // Décompression
    let start = Instant::now();
    let decompressed = compressor.decompress(&compressed).unwrap();
    let decompression_time = start.elapsed().as_millis() as f64;

    // Vérifications
    assert_eq!(data, decompressed);
    assert!(
        compressed.len() < data.len(),
        "La compression devrait réduire la taille"
    );

    // Statistiques
    let stats = CompressionStats {
        original_size: data.len(),
        compressed_size: compressed.len(),
        compression_time_ms: compression_time,
        decompression_time_ms: decompression_time,
        algorithm: CompressionAlgorithm::Lz4,
    };

    println!("LZ4 Compression:");
    println!("  Taille originale: {} octets", stats.original_size);
    println!("  Taille compressée: {} octets", stats.compressed_size);
    println!("  Ratio: {:.2}", stats.ratio());
    println!("  Réduction: {:.1}%", stats.reduction_percent());
    println!(
        "  Débit compression: {:.2} MB/s",
        stats.compression_throughput_mbps()
    );
    println!(
        "  Débit décompression: {:.2} MB/s",
        stats.decompression_throughput_mbps()
    );
}

#[cfg(feature = "zstd")]
#[test]
fn test_compression_zstd_basic() {
    // Données de test répétitives
    let data = vec![42u8; 1024 * 1024]; // 1 Mo de 42
    let config = CompressionConfig {
        algorithm: CompressionAlgorithm::Zstd,
        level: 6,
        block_size: 64 * 1024,
        in_memory: true,
        use_dictionary: false,
        max_buffer_size: 8 * 1024 * 1024,
    };

    let mut compressor = Compressor::new(config).unwrap();

    // Compression
    let start = Instant::now();
    let compressed = compressor.compress(&data).unwrap();
    let compression_time = start.elapsed().as_millis() as f64;

    // Décompression
    let start = Instant::now();
    let decompressed = compressor.decompress(&compressed).unwrap();
    let decompression_time = start.elapsed().as_millis() as f64;

    // Vérifications
    assert_eq!(data, decompressed);
    assert!(
        compressed.len() < data.len(),
        "La compression devrait réduire la taille"
    );

    // Statistiques
    let stats = CompressionStats {
        original_size: data.len(),
        compressed_size: compressed.len(),
        compression_time_ms: compression_time,
        decompression_time_ms: decompression_time,
        algorithm: CompressionAlgorithm::Zstd,
    };

    println!("Zstd Compression:");
    println!("  Taille originale: {} octets", stats.original_size);
    println!("  Taille compressée: {} octets", stats.compressed_size);
    println!("  Ratio: {:.2}", stats.ratio());
    println!("  Réduction: {:.1}%", stats.reduction_percent());
    println!(
        "  Débit compression: {:.2} MB/s",
        stats.compression_throughput_mbps()
    );
    println!(
        "  Débit décompression: {:.2} MB/s",
        stats.decompression_throughput_mbps()
    );
}

#[test]
fn test_compression_none() {
    let data = vec![1, 2, 3, 4, 5];
    let config = CompressionConfig {
        algorithm: CompressionAlgorithm::None,
        level: 0,
        block_size: 1024,
        in_memory: true,
        use_dictionary: false,
        max_buffer_size: 8 * 1024 * 1024,
    };

    let mut compressor = Compressor::new(config).unwrap();

    let compressed = compressor.compress(&data).unwrap();
    let decompressed = compressor.decompress(&compressed).unwrap();

    assert_eq!(data, compressed);
    assert_eq!(data, decompressed);
}

#[test]
fn test_compression_ratio_calculation() {
    let original = vec![0u8; 1000];
    let compressed = vec![0u8; 500];

    let config = CompressionConfig::default();
    let compressor = Compressor::new(config).unwrap();

    let ratio = compressor.compression_ratio(&original, &compressed);
    assert_eq!(ratio, 0.5);
}

#[test]
fn test_compression_stats() {
    let stats = CompressionStats {
        original_size: 1000,
        compressed_size: 500,
        compression_time_ms: 100.0,
        decompression_time_ms: 50.0,
        algorithm: CompressionAlgorithm::Lz4,
    };

    assert_eq!(stats.ratio(), 0.5);
    assert_eq!(stats.reduction_percent(), 50.0);
    assert!(stats.compression_throughput_mbps() > 0.0);
    assert!(stats.decompression_throughput_mbps() > 0.0);
}

#[test]
fn test_algorithm_from_str() {
    assert_eq!(
        CompressionAlgorithm::parse_from_str("lz4"),
        Some(CompressionAlgorithm::Lz4)
    );
    assert_eq!(
        CompressionAlgorithm::parse_from_str("zstd"),
        Some(CompressionAlgorithm::Zstd)
    );
    assert_eq!(
        CompressionAlgorithm::parse_from_str("none"),
        Some(CompressionAlgorithm::None)
    );
    assert_eq!(
        CompressionAlgorithm::parse_from_str("gzip"),
        Some(CompressionAlgorithm::Gzip)
    );
    assert_eq!(CompressionAlgorithm::parse_from_str("invalid"), None);
}

#[test]
fn test_algorithm_available() {
    let available = CompressionAlgorithm::available();
    assert!(available.contains(&CompressionAlgorithm::None));
    // LZ4 devrait être disponible par défaut
    assert!(available.contains(&CompressionAlgorithm::Lz4));
}

/// Tests pour le streaming avec buffer borné
#[cfg(test)]
mod streaming_tests {
    use super::*;

    #[test]
    fn test_streaming_config_default() {
        let config = CompressionConfig::default();
        assert_eq!(config.max_buffer_size, 8 * 1024 * 1024);
    }

    #[test]
    fn test_streaming_config_custom() {
        let config = CompressionConfig {
            max_buffer_size: 1024 * 1024, // 1 Mo
            ..CompressionConfig::default()
        };
        assert_eq!(config.max_buffer_size, 1024 * 1024);
    }
}
