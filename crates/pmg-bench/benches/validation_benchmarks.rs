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

//! Benchmarks de validation pour le projet PMG
//!
//! Ce module contient des benchmarks pour évaluer les performances
//! des opérations de validation critiques : validation de tenseurs,
//! validation statistique et validation de distribution.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pmg_bench::{create_test_data, create_validation_config};
use pmg_validate::ModelValidator;
use std::time::Duration;

/// Benchmark de validation d'un tenseur (petit : 10 000 éléments).
/// Mémoire utilisée : 10 000 × 8 octets (f64) = 80 000 octets ≈ 80 Ko.
fn bench_validation_simple(c: &mut Criterion) {
    let validator = ModelValidator::new(create_validation_config());
    let data = create_test_data(10_000);

    c.bench_function("validation_simple", |b| {
        b.iter(|| {
            let result = validator.validate_tensor(
                black_box("model.layers.0.mlp.gate.weight"),
                black_box(&data),
                None,
                None,
            );
            black_box(result);
        })
    });
}

/// Benchmark de validation d'un tenseur (moyen : 100 000 éléments).
/// Mémoire utilisée : 100 000 × 8 octets (f64) = 800 000 octets ≈ 800 Ko.
fn bench_validation_medium(c: &mut Criterion) {
    let validator = ModelValidator::new(create_validation_config());
    let data = create_test_data(100_000);

    c.bench_function("validation_medium", |b| {
        b.iter(|| {
            let result = validator.validate_tensor(
                black_box("model.layers.0.mlp.gate.weight"),
                black_box(&data),
                None,
                None,
            );
            black_box(result);
        })
    });
}

/// Benchmark de validation d'un tenseur (grand : 1 000 000 éléments).
/// Mémoire utilisée : 1 000 000 × 8 octets (f64) = 8 000 000 octets ≈ 8 Mo.
fn bench_validation_large(c: &mut Criterion) {
    let validator = ModelValidator::new(create_validation_config());
    let data = create_test_data(1_000_000);

    c.bench_function("validation_large", |b| {
        b.iter(|| {
            let result = validator.validate_tensor(
                black_box("model.layers.0.mlp.gate.weight"),
                black_box(&data),
                None,
                None,
            );
            black_box(result);
        })
    });
}

/// Benchmark de validation statistique avec paramètres attendus.
fn bench_validation_with_params(c: &mut Criterion) {
    let validator = ModelValidator::new(create_validation_config());
    let data = create_test_data(100_000);

    c.bench_function("validation_with_params", |b| {
        b.iter(|| {
            let result = validator.validate_tensor(
                black_box("model.layers.0.mlp.gate.weight"),
                black_box(&data),
                Some(50.0),  // mean attendue
                Some(28.87), // std attendue
            );
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
        bench_validation_simple,
        bench_validation_medium,
        bench_validation_large,
        bench_validation_with_params,
}

criterion_main!(benches);
