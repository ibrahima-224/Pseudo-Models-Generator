//! Benchmarks de performance pour les opérations statistiques de base
//! dans le crate `pmg-math`.
//!
//! Ce fichier mesure les performances du calcul de moyenne, variance,
//! écart-type, et autres statistiques descriptives.

use criterion::{criterion_group, criterion_main, Criterion};
use pmg_math::statistics::{mean, min_max, skewness, std_sample, variance_sample};

/// Benchmark des opérations statistiques de base.
fn bench_statistics(c: &mut Criterion) {
    // Données de test : séquence de 10 000 valeurs.
    let data: Vec<f64> = (0..10000).map(|i| i as f64 * 0.001).collect();

    // Mesure de la performance du calcul de la moyenne.
    c.bench_function("statistics_mean", |b| b.iter(|| mean(&data).unwrap()));

    // Mesure de la performance du calcul de la variance d'échantillon.
    c.bench_function("statistics_variance_sample", |b| {
        b.iter(|| variance_sample(&data).unwrap())
    });

    // Mesure de la performance du calcul de l'écart-type d'échantillon.
    c.bench_function("statistics_std_sample", |b| {
        b.iter(|| std_sample(&data).unwrap())
    });

    // Mesure de la performance du calcul du min et max.
    c.bench_function("statistics_min_max", |b| b.iter(|| min_max(&data).unwrap()));

    // Mesure de la performance du calcul de l'asymétrie (skewness).
    c.bench_function("statistics_skewness", |b| {
        b.iter(|| skewness(&data).unwrap())
    });
}

criterion_group!(benches, bench_statistics);
criterion_main!(benches);
