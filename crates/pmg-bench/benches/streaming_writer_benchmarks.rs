//! Benchmarks de performance pour l'écriture streaming
//! dans le crate `pmg-io`.
//!
//! Ce fichier mesure les performances d'écriture de blocs de données
//! dans des fichiers temporaires, opération critique pour la génération
//! de fichiers SafeTensors de grande taille.

use criterion::{criterion_group, criterion_main, Criterion};
use std::io::Write;
use tempfile::NamedTempFile;

/// Benchmark d'écriture streaming.
fn bench_streaming_writer(c: &mut Criterion) {
    // Mesure de la performance d'écriture de blocs de 1 Mo.
    c.bench_function("write_1mb_chunks", |b| {
        b.iter(|| {
            let mut file = NamedTempFile::new().unwrap();
            let chunk = vec![0u8; 1024 * 1024]; // 1 Mo
            for _ in 0..10 {
                file.write_all(&chunk).unwrap();
            }
            file.flush().unwrap();
        })
    });
}

criterion_group!(benches, bench_streaming_writer);
criterion_main!(benches);
