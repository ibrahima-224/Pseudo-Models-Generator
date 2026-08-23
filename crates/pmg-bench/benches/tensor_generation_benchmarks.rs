//! Benchmarks de performance pour la génération de tenseurs
//! dans le crate `pmg-core`.
//!
//! Ce fichier mesure les performances de création et remplissage de tenseurs
//! en mémoire, opérations fondamentales pour la génération de poids.

use criterion::{criterion_group, criterion_main, Criterion};
use pmg_core::shape::Shape;
use rand::Rng;

/// Benchmark de génération de tenseurs.
fn bench_tensor_generation(c: &mut Criterion) {
    // Shape de test : tenseur 1024x1024 (environ 1 million d'éléments).
    let shape = Shape::new(vec![1024, 1024]).expect("dimensions valides");
    let num_elements = shape.num_elements_usize().expect("taille raisonnable");

    // Mesure de la performance de création d'un vecteur de zéros.
    c.bench_function("tensor_creation_1k", |b| {
        b.iter(|| vec![0.0f32; num_elements])
    });

    // Mesure de la performance de remplissage avec des valeurs aléatoires.
    c.bench_function("tensor_fill_normal", |b| {
        b.iter(|| {
            let mut rng = rand::thread_rng();
            let mut data = vec![0.0f32; num_elements];
            for v in data.iter_mut() {
                *v = rng.gen();
            }
            data
        })
    });
}

criterion_group!(benches, bench_tensor_generation);
criterion_main!(benches);
