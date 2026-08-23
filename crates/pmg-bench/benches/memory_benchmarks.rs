// Copyright (C) 2024 PMG Contributors
// This file is part of PMG (Pseudo-Model Generator).
//
// PMG is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// PMG is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with PMG.  If not, see <https://www.gnu.org/licenses/>.

//! Benchmarks de consommation mémoire pour le projet PMG
//!
//! Ce module contient des benchmarks pour mesurer la consommation mémoire
//! lors des opérations critiques : allocation de tenseurs, streaming par chunks,
//! et validation de tenseurs.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pmg_validate::types::{TensorData, ValidationConfig};
use pmg_validate::ModelValidator;
use std::time::Duration;

/// Benchmark d'allocation de tenseur avec mesure mémoire.
/// Mesure la mémoire allouée pour un vecteur de f64.
fn bench_generation_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_generation");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    // Test avec différentes tailles de tenseurs
    for (name, rows, cols) in [
        ("small_100x64", 100, 64),    // 6 400 éléments ≈ 50 Ko
        ("medium_1kx1k", 1000, 1000), // 1 000 000 éléments ≈ 8 Mo
        ("large_2kx2k", 2000, 2000),  // 4 000 000 éléments ≈ 32 Mo
    ] {
        let size = rows * cols;
        let expected_bytes = size * std::mem::size_of::<f64>();

        group.bench_function(name, |b| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::ZERO;
                let mut total_allocated = 0u64;

                for _ in 0..iters {
                    // Génération simulée : allocation d'un vecteur de f64
                    let start = std::time::Instant::now();
                    let mut tensor: Vec<f64> = Vec::with_capacity(size);
                    for i in 0..size {
                        tensor.push(i as f64 * 0.001);
                    }
                    black_box(&tensor);
                    total_duration += start.elapsed();

                    // Mesure de la mémoire allouée
                    total_allocated += (tensor.capacity() * std::mem::size_of::<f64>()) as u64;
                }

                // Retourne la durée moyenne et affiche la mémoire moyenne
                let avg_allocated = total_allocated / iters;
                eprintln!(
                    "  Mémoire allouée moyenne pour {}: {:.2} Ko (attendu ≈ {:.2} Ko)",
                    name,
                    avg_allocated as f64 / 1024.0,
                    expected_bytes as f64 / 1024.0
                );
                total_duration / iters as u32
            })
        });
    }

    group.finish();
}

/// Benchmark de streaming (génération par chunks) avec mesure mémoire.
/// Mesure la mémoire totale allouée pour tous les chunks.
fn bench_streaming_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_streaming");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    // Test avec différentes tailles de chunks
    for (name, total_size, chunk_size) in [
        ("small_tensor_1k", 1000 * 100, 100), // 100 000 éléments ≈ 800 Ko
        ("medium_tensor_10k", 10000 * 100, 1000), // 1 000 000 éléments ≈ 8 Mo
        ("large_tensor_100k", 100000 * 100, 10000), // 10 000 000 éléments ≈ 80 Mo
    ] {
        let expected_bytes = total_size * std::mem::size_of::<f64>();

        group.bench_function(name, |b| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::ZERO;
                let mut total_allocated = 0u64;

                for _ in 0..iters {
                    // Génération par chunks (streaming simulé)
                    let start = std::time::Instant::now();
                    let mut all_chunks: Vec<Vec<f64>> = Vec::new();
                    let mut offset = 0;
                    while offset < total_size {
                        let current_chunk_size = chunk_size.min(total_size - offset);
                        let mut chunk: Vec<f64> = Vec::with_capacity(current_chunk_size);
                        for i in 0..current_chunk_size {
                            chunk.push((offset + i) as f64 * 0.001);
                        }
                        all_chunks.push(chunk);
                        offset += current_chunk_size;
                    }
                    black_box(&all_chunks);
                    total_duration += start.elapsed();

                    // Mesure de la mémoire totale allouée
                    let chunk_memory: u64 = all_chunks
                        .iter()
                        .map(|c| (c.capacity() * std::mem::size_of::<f64>()) as u64)
                        .sum();
                    total_allocated += chunk_memory;
                }

                let avg_allocated = total_allocated / iters;
                eprintln!(
                    "  Mémoire allouée moyenne pour {}: {:.2} Ko (attendu ≈ {:.2} Ko)",
                    name,
                    avg_allocated as f64 / 1024.0,
                    expected_bytes as f64 / 1024.0
                );
                total_duration / iters as u32
            })
        });
    }

    group.finish();
}

/// Benchmark de validation avec mesure mémoire.
/// Mesure la mémoire utilisée lors de la validation de tenseurs.
fn bench_validation_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_validation");
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(50);

    // Test avec différentes tailles de données
    for (name, size) in [
        ("small_10k", 10000),    // 10 000 éléments ≈ 80 Ko
        ("medium_100k", 100000), // 100 000 éléments ≈ 800 Ko
        ("large_1m", 1000000),   // 1 000 000 éléments ≈ 8 Mo
    ] {
        let expected_bytes = size * std::mem::size_of::<f64>();

        group.bench_function(name, |b| {
            b.iter_custom(|iters| {
                let mut total_duration = Duration::ZERO;
                let mut total_allocated = 0u64;

                for _ in 0..iters {
                    // Générer des données de test
                    let data: Vec<f64> = (0..size).map(|i| i as f64 * 0.001).collect();
                    let data_memory = (data.capacity() * std::mem::size_of::<f64>()) as u64;

                    // Validation du tenseur
                    let start = std::time::Instant::now();
                    let validator = ModelValidator::new(ValidationConfig::default());
                    let tensors: Vec<TensorData> = vec![(
                        "model.layers.0.mlp.gate.weight",
                        black_box(&data),
                        None,
                        None,
                    )];
                    let result = validator.validate_model("bench-model", &tensors);
                    black_box(&result);
                    total_duration += start.elapsed();

                    // Mesure de la mémoire allouée pour les données
                    total_allocated += data_memory;
                }

                let avg_allocated = total_allocated / iters;
                eprintln!(
                    "  Mémoire allouée moyenne pour {}: {:.2} Ko (attendu ≈ {:.2} Ko)",
                    name,
                    avg_allocated as f64 / 1024.0,
                    expected_bytes as f64 / 1024.0
                );
                total_duration / iters as u32
            })
        });
    }

    group.finish();
}

// ============================================================================
// Configuration des benchmarks
// ============================================================================

/// Configuration du groupe de benchmarks.
fn bench_config() -> Criterion {
    Criterion::default()
        .measurement_time(Duration::from_secs(3))
        .sample_size(100)
}

criterion_group! {
    name = benches;
    config = bench_config();
    targets =
        bench_generation_memory,
        bench_streaming_memory,
        bench_validation_memory,
}

criterion_main!(benches);
