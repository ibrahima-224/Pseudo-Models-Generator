//! Benchmarks de performance pour le packing/dépacking de données
//! dans le crate `pmg-io`.
//!
//! Ce fichier mesure les performances de conversion entre types f32
//! et représentation en octets (little-endian), opérations critiques
//! pour l'écriture et la lecture de fichiers SafeTensors.

use criterion::{criterion_group, criterion_main, Criterion};

/// Benchmark de packing/dépacking de données.
fn bench_packing(c: &mut Criterion) {
    // Données de test : 10 000 valeurs f32.
    let values: Vec<f32> = (0..10000).map(|i| i as f32 * 0.001).collect();

    // Mesure de la performance de conversion f32 → octets.
    c.bench_function("f32_to_bytes", |b| {
        b.iter(|| {
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>()
        })
    });

    // Mesure de la performance de conversion octets → f32.
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    c.bench_function("bytes_to_f32", |b| {
        b.iter(|| {
            bytes
                .chunks(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<f32>>()
        })
    });
}

criterion_group!(benches, bench_packing);
criterion_main!(benches);
