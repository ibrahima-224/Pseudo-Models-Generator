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

//! Benchmarks de génération pour le projet PMG
//!
//! Ce module contient des benchmarks pour évaluer les performances
//! des opérations de génération critiques : modèle complet, tenseurs individuels
//! et pipeline de génération.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pmg_bench::{create_seed_plan, create_tensor_spec};
use pmg_generator::model_generator::ModelGeneratorComplete;
use pmg_generator::pipeline::GenerationPipeline;
use pmg_generator::tensor_generator::TensorGenerator;
use std::time::Duration;

/// Crée un pipeline de génération pour les benchmarks.
fn create_generation_pipeline() -> GenerationPipeline {
    GenerationPipeline::default()
}

/// Benchmark de génération d'un tenseur individuel (petit : 100×64).
/// Mémoire utilisée : 100 × 64 × 8 octets (f64) = 51 200 octets ≈ 50 Ko.
fn bench_tensor_generation_simple(c: &mut Criterion) {
    let spec = create_tensor_spec("model.layers.0.mlp.gate.weight", 100, 64);
    let plan = create_seed_plan(42, "bench-model", "0.1.0");

    c.bench_function("tensor_generation_simple", |b| {
        b.iter(|| {
            let generator =
                TensorGenerator::new(black_box(spec.clone()), black_box(plan.clone()), None);
            let values = generator.generate().unwrap();
            black_box(values);
        })
    });
}

/// Benchmark de génération d'un tenseur individuel (moyen : 1024×512).
/// Mémoire utilisée : 1024 × 512 × 8 octets (f64) = 4 194 304 octets ≈ 4 Mo.
fn bench_tensor_generation_medium(c: &mut Criterion) {
    let spec = create_tensor_spec("model.layers.0.mlp.gate.weight", 1024, 512);
    let plan = create_seed_plan(42, "bench-model", "0.1.0");

    c.bench_function("tensor_generation_medium", |b| {
        b.iter(|| {
            let generator =
                TensorGenerator::new(black_box(spec.clone()), black_box(plan.clone()), None);
            let values = generator.generate().unwrap();
            black_box(values);
        })
    });
}

/// Benchmark de génération d'un tenseur individuel (grand : 2048×1024).
/// Mémoire utilisée : 2048 × 1024 × 8 octets (f64) = 16 777 216 octets ≈ 16 Mo.
fn bench_tensor_generation_large(c: &mut Criterion) {
    let spec = create_tensor_spec("model.layers.0.mlp.gate.weight", 2048, 1024);
    let plan = create_seed_plan(42, "bench-model", "0.1.0");

    c.bench_function("tensor_generation_large", |b| {
        b.iter(|| {
            let generator =
                TensorGenerator::new(black_box(spec.clone()), black_box(plan.clone()), None);
            let values = generator.generate().unwrap();
            black_box(values);
        })
    });
}

/// Benchmark de génération complète du modèle.
/// Utilise un blueprint simplifié pour mesurer les performances.
fn bench_model_generation(c: &mut Criterion) {
    let blueprint = pmg_bench::create_model_blueprint_for_bench();
    let pipeline = create_generation_pipeline();

    c.bench_function("model_generation_complete", |b| {
        b.iter(|| {
            let generator = ModelGeneratorComplete::new(
                black_box(blueprint.clone()),
                black_box(42),
                black_box("0.1.0"),
                black_box(pipeline.clone()),
                black_box(1024),
            );
            let results = generator.generate_all().unwrap();
            black_box(results);
        })
    });
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
        bench_tensor_generation_simple,
        bench_tensor_generation_medium,
        bench_tensor_generation_large,
        bench_model_generation,
}

criterion_main!(benches);
