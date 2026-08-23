//! Benchmarks de performance pour la compression en mémoire
//!
//! Mesure le débit, ratio et latence pour différents algorithmes et tailles.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use pmg_compression::{CompressionAlgorithm, CompressionConfig, Compressor};

/// Benchmark compression LZ4
fn bench_compress_lz4(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_lz4");

    for size in [1024, 65536, 1048576, 10485760] {
        group.bench_with_input(BenchmarkId::new("compress", size), &size, |b, &size| {
            let data = vec![0u8; size];
            let config = CompressionConfig {
                algorithm: CompressionAlgorithm::Lz4,
                level: 6,
                ..Default::default()
            };
            let mut compressor = Compressor::new(config).unwrap();
            b.iter(|| compressor.compress(black_box(&data)));
        });
    }
    group.finish();
}

/// Benchmark compression Zstd
fn bench_compress_zstd(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_zstd");

    for size in [1024, 65536, 1048576, 10485760] {
        group.bench_with_input(BenchmarkId::new("compress", size), &size, |b, &size| {
            let data = vec![0u8; size];
            let config = CompressionConfig {
                algorithm: CompressionAlgorithm::Zstd,
                level: 6,
                ..Default::default()
            };
            let mut compressor = Compressor::new(config).unwrap();
            b.iter(|| compressor.compress(black_box(&data)));
        });
    }
    group.finish();
}

/// Benchmark décompression
fn bench_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompression");

    for size in [1024, 65536, 1048576] {
        // Préparer données compressées
        let data = vec![0u8; size];
        let config = CompressionConfig {
            algorithm: CompressionAlgorithm::Lz4,
            level: 6,
            ..Default::default()
        };
        let mut compressor = Compressor::new(config).unwrap();
        let compressed = compressor.compress(&data).unwrap();

        group.bench_with_input(
            BenchmarkId::new("decompress_lz4", size),
            &compressed,
            |b, compressed| {
                b.iter(|| compressor.decompress(black_box(compressed)));
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_compress_lz4,
    bench_compress_zstd,
    bench_decompress
);
criterion_main!(benches);
