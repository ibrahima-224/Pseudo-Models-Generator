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

//! Benchmarks d'E/S pour le projet PMG
//!
//! Ce module contient des benchmarks pour évaluer les performances
//! des opérations d'entrée-sortie critiques : écriture config,
//! écriture métadonnées, écriture streaming et parsing headers.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pmg_core::generator_config::GeneratorConfig;
use pmg_io::config_writer::write_config;
use pmg_io::metadata_writer::write_metadata;
use std::time::Duration;

/// Crée une GeneratorConfig pour les benchmarks.
fn create_generator_config() -> GeneratorConfig {
    GeneratorConfig::new(42, "bench-model").unwrap()
}

/// Benchmark d'écriture de configuration JSON.
/// Taille typique : ~300 octets JSON.
fn bench_write_config(c: &mut Criterion) {
    let config = create_generator_config();

    c.bench_function("write_config_bench", |b| {
        b.iter(|| {
            let json = write_config(black_box(&config)).unwrap();
            black_box(json);
        })
    });
}

/// Benchmark d'écriture de métadonnées.
/// Taille typique : ~200 octets JSON.
fn bench_write_metadata(c: &mut Criterion) {
    let config = create_generator_config();

    c.bench_function("write_metadata_bench", |b| {
        b.iter(|| {
            let metadata = write_metadata(black_box(&config)).unwrap();
            black_box(metadata);
        })
    });
}

/// Benchmark d'écriture streaming de tenseurs (simulation).
/// Données : 1 Mo de tenseurs en mémoire.
fn bench_streaming_write(c: &mut Criterion) {
    // Simulation d'écriture streaming avec des données en mémoire
    let data_size = 1024 * 1024; // 1 Mo
    let data: Vec<u8> = vec![0u8; data_size];

    c.bench_function("streaming_write_bench", |b| {
        b.iter(|| {
            // Simule l'écriture streaming en mémoire
            let mut buffer = Vec::with_capacity(data_size);
            buffer.extend_from_slice(black_box(&data));
            black_box(buffer);
        })
    });
}

/// Benchmark de parsing de headers Safetensors (simulation).
/// Header typique : 2 tenseurs, ~300 octets JSON.
fn bench_header_parse(c: &mut Criterion) {
    // Simulation de parsing de header JSON
    let header_json = r#"{
        "model.embed_tokens.weight": {
            "dtype": "F32",
            "shape": [1000, 256],
            "data_offsets": [0, 1024000]
        },
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F32",
            "shape": [256, 256],
            "data_offsets": [1024000, 1280000]
        }
    }"#;

    c.bench_function("header_parse_bench", |b| {
        b.iter(|| {
            // Simule le parsing de header JSON
            let _parsed: serde_json::Value = serde_json::from_str(black_box(header_json)).unwrap();
            black_box(());
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
        bench_write_config,
        bench_write_metadata,
        bench_streaming_write,
        bench_header_parse,
}

criterion_main!(benches);
