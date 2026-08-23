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

//! Benchmarks de comparaison pour le projet PMG
//!
//! Ce module contient des benchmarks pour évaluer les performances
//! des opérations de comparaison critiques : comparaison de métadonnées
//! de tenseurs et comparaison de configurations.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pmg_compare::config_compare::{compare_configs, ConfigValue, ModelConfig};
use pmg_compare::tensor_compare::{compare_tensors, TensorInfo};
use std::time::Duration;

/// Crée une liste de TensorInfo pour les benchmarks.
fn create_tensor_infos(count: usize, prefix: &str) -> Vec<TensorInfo> {
    (0..count)
        .map(|i| TensorInfo::new(format!("{}.tensor_{}", prefix, i)))
        .collect()
}

/// Crée une ModelConfig pour les benchmarks.
fn create_model_config(name: &str) -> ModelConfig {
    ModelConfig {
        name: name.to_string(),
        parameters: vec![
            ("vocab_size".to_string(), ConfigValue::Integer(32000)),
            ("hidden_size".to_string(), ConfigValue::Integer(4096)),
            ("num_layers".to_string(), ConfigValue::Integer(32)),
            ("num_heads".to_string(), ConfigValue::Integer(32)),
            ("num_experts".to_string(), ConfigValue::Integer(8)),
            ("intermediate_size".to_string(), ConfigValue::Integer(11008)),
        ],
    }
}

/// Benchmark de comparaison de tenseurs identiques (petit : 10 tenseurs).
fn bench_tensor_comparison_identical_simple(c: &mut Criterion) {
    let tensors1 = create_tensor_infos(10, "model.layers.0");
    let tensors2 = tensors1.clone();

    c.bench_function("tensor_comparison_identical_simple", |b| {
        b.iter(|| {
            let result = compare_tensors(black_box(&tensors1), black_box(&tensors2));
            black_box(result);
        })
    });
}

/// Benchmark de comparaison de tenseurs identiques (moyen : 100 tenseurs).
fn bench_tensor_comparison_identical_medium(c: &mut Criterion) {
    let tensors1 = create_tensor_infos(100, "model.layers.0");
    let tensors2 = tensors1.clone();

    c.bench_function("tensor_comparison_identical_medium", |b| {
        b.iter(|| {
            let result = compare_tensors(black_box(&tensors1), black_box(&tensors2));
            black_box(result);
        })
    });
}

/// Benchmark de comparaison de tenseurs différents (moyen : 100 tenseurs, 10 manquants).
fn bench_tensor_comparison_different(c: &mut Criterion) {
    let tensors1 = create_tensor_infos(100, "model.layers.0");
    let mut tensors2 = create_tensor_infos(90, "model.layers.0"); // 10 tenseurs manquants
                                                                  // Ajouter quelques tenseurs supplémentaires dans tensors2
    for i in 100..110 {
        tensors2.push(TensorInfo::new(format!(
            "model.layers.0.extra_tensor_{}",
            i
        )));
    }

    c.bench_function("tensor_comparison_different", |b| {
        b.iter(|| {
            let result = compare_tensors(black_box(&tensors1), black_box(&tensors2));
            black_box(result);
        })
    });
}

/// Benchmark de comparaison de configurations identiques.
fn bench_config_comparison_identical(c: &mut Criterion) {
    let config1 = create_model_config("model-a");
    let config2 = create_model_config("model-a");

    c.bench_function("config_comparison_identical", |b| {
        b.iter(|| {
            let result = compare_configs(black_box(&config1), black_box(&config2));
            black_box(result);
        })
    });
}

/// Benchmark de comparaison de configurations différentes.
fn bench_config_comparison_different(c: &mut Criterion) {
    let config1 = create_model_config("model-a");
    let mut config2 = create_model_config("model-b");
    // Modifier quelques paramètres
    config2.parameters[0] = ("vocab_size".to_string(), ConfigValue::Integer(50000));
    config2.parameters[1] = ("hidden_size".to_string(), ConfigValue::Integer(8192));

    c.bench_function("config_comparison_different", |b| {
        b.iter(|| {
            let result = compare_configs(black_box(&config1), black_box(&config2));
            black_box(result);
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
        bench_tensor_comparison_identical_simple,
        bench_tensor_comparison_identical_medium,
        bench_tensor_comparison_different,
        bench_config_comparison_identical,
        bench_config_comparison_different,
}

criterion_main!(benches);
