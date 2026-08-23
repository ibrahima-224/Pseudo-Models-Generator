//! Benchmarks de performance pour le générateur de nombres aléatoires (RNG)
//! et les distributions associées dans le crate `pmg-math`.
//!
//! Ce fichier mesure les performances de base du RNG déterministe
//! (génération d'entiers, de flottants, et d'échantillons de loi normale).

use criterion::{criterion_group, criterion_main, Criterion};
use pmg_math::distribution::Distribution;
use pmg_math::distributions::normal::Normal;
use pmg_math::rng::DeterministicRng;

/// Benchmark de génération d'entiers u64 via le RNG.
fn bench_rng_generation(c: &mut Criterion) {
    // Création du RNG avec une seed fixe pour la reproductibilité.
    let mut rng = DeterministicRng::from_seed([42u8; 32]);

    // Mesure de la performance de génération d'entiers non signés 64 bits.
    c.bench_function("rng_next_u64", |b| b.iter(|| rng.next_u64()));

    // Mesure de la performance de génération de flottants f64.
    c.bench_function("rng_next_f64", |b| b.iter(|| rng.next_f64()));

    // Mesure de la performance d'échantillonnage d'une loi normale.
    let mut normal = Normal::new(0.0, 1.0).expect("paramètres valides");
    c.bench_function("rng_normal_sample", |b| b.iter(|| normal.sample(&mut rng)));
}

criterion_group!(benches, bench_rng_generation);
criterion_main!(benches);
