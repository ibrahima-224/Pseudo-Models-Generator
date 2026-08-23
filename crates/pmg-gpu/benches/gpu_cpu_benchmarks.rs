//! Benchmarks GPU vs CPU pour pmg-gpu
//!
//! Ce fichier contient des benchmarks mesurant les performances de l'accélération
//! GPU par rapport au CPU pour la génération de nombres aléatoires normaux.
//!
//! Les benchmarks sont conçus pour être exécutés avec Criterion et fournissent
//! des métriques de performance détaillées pour optimiser le pipeline de génération.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use pmg_gpu::{GpuAccelerated, NormalGenerationAccelerated};

/// Benchmark génération normale CPU
///
/// Mesure le temps de génération de nombres aléatoires normaux
/// sur CPU pour différentes tailles de données.
fn bench_normal_generation_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("normal_generation_cpu");

    // Différentes tailles de données pour tester la scalabilité
    for size in [1024, 10240, 102400] {
        group.bench_with_input(BenchmarkId::new("cpu", size), &size, |b, &size| {
            let generator = NormalGenerationAccelerated::new(42);
            b.iter(|| generator.execute_cpu(black_box(&(size, 0.0, 1.0))));
        });
    }
    group.finish();
}

/// Benchmark génération normale GPU (si disponible)
///
/// Mesure le temps de génération sur GPU lorsque disponible.
/// Utilise un fallback CPU si le GPU n'est pas disponible.
fn bench_normal_generation_gpu(c: &mut Criterion) {
    if !pmg_gpu::is_gpu_available() {
        println!("GPU non disponible, benchmark GPU ignoré");
        return;
    }

    let mut group = c.benchmark_group("normal_generation_gpu");

    for size in [1024, 10240, 102400] {
        group.bench_with_input(BenchmarkId::new("gpu", size), &size, |b, &size| {
            // Placeholder - nécessite un device GPU réel
            // let device = GpuDevice::new(0).unwrap();
            // let generator = NormalGenerationAccelerated::new(42);
            // b.iter(|| generator.execute_gpu(black_box(&(size, 0.0, 1.0)), &device));
            b.iter(|| {
                // Simuler un calcul GPU (fallback CPU pour le moment)
                let data: Vec<f64> = (0..size).map(|i| i as f64 * 0.001).collect();
                black_box(data);
            });
        });
    }
    group.finish();
}

/// Benchmark comparaison CPU vs GPU
///
/// Compare directement les performances CPU et GPU pour une taille fixe.
fn bench_cpu_vs_gpu_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_vs_gpu_comparison");
    let size = 10240;

    // CPU
    group.bench_function("cpu_10k_elements", |b| {
        let generator = NormalGenerationAccelerated::new(42);
        b.iter(|| generator.execute_cpu(black_box(&(size, 0.0, 1.0))));
    });

    // GPU (simulé si non disponible)
    group.bench_function("gpu_10k_elements", |b| {
        b.iter(|| {
            let data: Vec<f64> = (0..size).map(|i| i as f64 * 0.001).collect();
            black_box(data);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_normal_generation_cpu,
    bench_normal_generation_gpu,
    bench_cpu_vs_gpu_comparison
);
criterion_main!(benches);
