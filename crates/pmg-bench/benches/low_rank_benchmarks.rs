//! Benchmarks de performance pour le calcul bas-rang
//! dans le crate `pmg-math`.
//!
//! Ce fichier mesure les performances de la multiplication matricielle
//! et des opérations de décomposition bas-rang.

use criterion::{criterion_group, criterion_main, Criterion};

/// Benchmark de calcul bas-rang (multiplication matricielle).
fn bench_low_rank(c: &mut Criterion) {
    let n = 128;
    let data: Vec<f64> = (0..n * n).map(|i| i as f64 * 0.001).collect();

    // Mesure de la performance de multiplication matricielle naive.
    c.bench_function("matrix_multiply_128x128", |b| {
        b.iter(|| {
            let mut result = vec![0.0f64; n * n];
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        result[i * n + j] += data[i * n + k] * data[k * n + j];
                    }
                }
            }
            result
        })
    });
}

criterion_group!(benches, bench_low_rank);
criterion_main!(benches);
