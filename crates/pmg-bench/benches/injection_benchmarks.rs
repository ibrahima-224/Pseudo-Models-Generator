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

//! Benchmarks d'injection pour le projet PMG
//!
//! Ce module contient des benchmarks pour évaluer les performances
//! des opérations d'injection critiques : pipeline complet, outliers,
//! corrélation et bas-rang.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pmg_blueprint::tensor_spec::TensorSpec;
use pmg_core::dtype::DType;
use pmg_core::shape::Shape;
use pmg_core::TensorRole;
use pmg_injector::injection_policy::InjectionPolicy;
use pmg_injector::tensor_injector::TensorInjector;
use pmg_math::rng::SeedPlan;
use std::time::Duration;

/// Crée un SeedPlan pour les benchmarks.
fn create_seed_plan<'a>(tensor_name: &'a str, layer_id: Option<u32>) -> SeedPlan<'a> {
    SeedPlan {
        seed_global: 42,
        model_id: "bench-model",
        tensor_name,
        layer_id,
        generation_version: "0.1.0",
    }
}

/// Crée un TensorSpec pour les benchmarks.
fn create_tensor_spec(name: &str, rows: usize, cols: usize) -> TensorSpec {
    TensorSpec::new(
        name,
        Shape::new(vec![rows as u64, cols as u64]).unwrap(),
        DType::F32,
        TensorRole::MlpGate,
    )
    .unwrap()
}

/// Crée une InjectionPolicy avec des valeurs typiques pour les benchmarks.
fn create_injection_policy() -> InjectionPolicy {
    InjectionPolicy::new(
        0.05, // outlier_frequency
        5.0,  // outlier_scale
        0.3,  // correlation_strength
        0.7,  // low_rank_probability
        8,    // low_rank_rank
        0.1,  // low_rank_alpha
        0.2,  // heavy_tail_probability
        3.0,  // heavy_tail_df
        0.1,  // sparse_structure_probability
        0.5,  // sparse_density
    )
    .unwrap()
}

/// Benchmark du pipeline d'injection complet sur un petit tenseur (100×64).
/// Mémoire utilisée : 100 × 64 × 8 octets (f64) = 51 200 octets ≈ 50 Ko.
fn bench_injection_pipeline_simple(c: &mut Criterion) {
    let spec = create_tensor_spec("model.layers.0.mlp.gate.weight", 100, 64);
    let policy = create_injection_policy();
    let plan = create_seed_plan("model.layers.0.mlp.gate.weight", Some(0));

    c.bench_function("injection_pipeline_simple", |b| {
        b.iter(|| {
            let injector = TensorInjector::from_seed_plan(
                black_box(&spec),
                black_box(policy.clone()),
                black_box(&plan),
            );
            let tensor = injector.inject().unwrap();
            black_box(tensor);
        })
    });
}

/// Benchmark du pipeline d'injection complet sur un tenseur moyen (1024×512).
/// Mémoire utilisée : 1024 × 512 × 8 octets (f64) = 4 194 304 octets ≈ 4 Mo.
fn bench_injection_pipeline_large(c: &mut Criterion) {
    let spec = create_tensor_spec("model.layers.0.mlp.gate.weight", 1024, 512);
    let policy = create_injection_policy();
    let plan = create_seed_plan("model.layers.0.mlp.gate.weight", Some(0));

    c.bench_function("injection_pipeline_large", |b| {
        b.iter(|| {
            let injector = TensorInjector::from_seed_plan(
                black_box(&spec),
                black_box(policy.clone()),
                black_box(&plan),
            );
            let tensor = injector.inject().unwrap();
            black_box(tensor);
        })
    });
}

/// Benchmark d'injection avec outliers (fréquence élevée).
fn bench_outlier_injection(c: &mut Criterion) {
    let spec = create_tensor_spec("model.layers.0.mlp.gate.weight", 256, 128);
    let mut policy = create_injection_policy();
    policy.outlier_frequency = 0.2; // 20% d'outliers
    policy.outlier_scale = 10.0; // Amplification forte
    let plan = create_seed_plan("model.layers.0.mlp.gate.weight", Some(0));

    c.bench_function("outlier_injection_bench", |b| {
        b.iter(|| {
            let injector = TensorInjector::from_seed_plan(
                black_box(&spec),
                black_box(policy.clone()),
                black_box(&plan),
            );
            let tensor = injector.inject().unwrap();
            black_box(tensor);
        })
    });
}

/// Benchmark d'injection avec corrélation colonnes (ρ élevé).
fn bench_correlated_injection(c: &mut Criterion) {
    let spec = create_tensor_spec("model.layers.0.mlp.gate.weight", 256, 128);
    let mut policy = create_injection_policy();
    policy.correlation_strength = 0.8; // Forte corrélation
    let plan = create_seed_plan("model.layers.0.mlp.gate.weight", Some(0));

    c.bench_function("correlated_injection_bench", |b| {
        b.iter(|| {
            let injector = TensorInjector::from_seed_plan(
                black_box(&spec),
                black_box(policy.clone()),
                black_box(&plan),
            );
            let tensor = injector.inject().unwrap();
            black_box(tensor);
        })
    });
}

/// Benchmark d'injection bas-rang par blocs (rang élevé).
fn bench_low_rank_injection(c: &mut Criterion) {
    let spec = create_tensor_spec("model.layers.0.mlp.gate.weight", 256, 128);
    let mut policy = create_injection_policy();
    policy.low_rank_probability = 1.0; // Toujours appliquer
    policy.low_rank_rank = 32; // Rang élevé
    policy.low_rank_alpha = 0.5; // Amplitude forte
    let plan = create_seed_plan("model.layers.0.mlp.gate.weight", Some(0));

    c.bench_function("low_rank_injection_bench", |b| {
        b.iter(|| {
            let injector = TensorInjector::from_seed_plan(
                black_box(&spec),
                black_box(policy.clone()),
                black_box(&plan),
            );
            let tensor = injector.inject().unwrap();
            black_box(tensor);
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
        bench_injection_pipeline_simple,
        bench_injection_pipeline_large,
        bench_outlier_injection,
        bench_correlated_injection,
        bench_low_rank_injection,
}

criterion_main!(benches);
