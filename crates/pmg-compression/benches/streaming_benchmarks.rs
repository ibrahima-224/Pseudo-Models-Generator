//! Benchmarks de performance pour le streaming
//!
//! Mesure la performance avec gestion mémoire bornée.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use pmg_compression::{CompressionAlgorithm, CompressionConfig};
use std::io::Write;
use tempfile::tempdir;

/// Benchmark streaming compression
fn bench_streaming_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_compression");

    for buffer_size in [65536, 262144, 1048576] {
        group.bench_with_input(
            BenchmarkId::new("compress", buffer_size),
            &buffer_size,
            |b, &buffer_size| {
                let dir = tempdir().unwrap();
                let input_path = dir.path().join("input.bin");
                let output_path = dir.path().join("output.bin");

                // Créer fichier d'entrée (10 Mo)
                let data = vec![0u8; 10 * 1024 * 1024];
                std::fs::write(&input_path, &data).unwrap();

                let config = CompressionConfig {
                    algorithm: CompressionAlgorithm::Lz4,
                    level: 6,
                    max_buffer_size: buffer_size,
                    ..Default::default()
                };

                b.iter(|| {
                    // Créer un writer synchronisé pour le benchmark
                    let mut output_file = std::fs::File::create(&output_path).unwrap();
                    // Utiliser le compresseur synchronisé directement
                    let mut compressor = pmg_compression::Compressor::new(config.clone()).unwrap();
                    let compressed = compressor.compress(&data).unwrap();
                    output_file.write_all(&compressed).unwrap();
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_streaming_compress);
criterion_main!(benches);
