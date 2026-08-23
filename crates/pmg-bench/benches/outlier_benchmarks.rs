//! Benchmarks de performance pour la détection d'outliers
//! dans le crate `pmg-validate`.
//!
//! Ce fichier mesure les performances du calcul de la médiane des écarts absolus
//! (MAD), une mesure robuste de la dispersion utilisée pour la détection d'outliers.

use criterion::{criterion_group, criterion_main, Criterion};

/// Benchmark de détection d'outliers via la MAD.
fn bench_outlier_detection(c: &mut Criterion) {
    // Données de test : séquence de 10 000 valeurs avec quelques outliers.
    let data: Vec<f64> = (0..10000)
        .map(|i| {
            if i % 100 == 0 {
                1000.0
            } else {
                i as f64 * 0.001
            }
        })
        .collect();

    // Mesure de la performance du calcul de la MAD.
    c.bench_function("outlier_mad", |b| {
        b.iter(|| {
            // Calcul de la MAD (Median Absolute Deviation).
            let mut sorted = data.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = sorted[sorted.len() / 2];
            let deviations: Vec<f64> = data.iter().map(|x| (x - median).abs()).collect();
            let mut dev_sorted = deviations;
            dev_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            dev_sorted[dev_sorted.len() / 2]
        })
    });
}

criterion_group!(benches, bench_outlier_detection);
criterion_main!(benches);
