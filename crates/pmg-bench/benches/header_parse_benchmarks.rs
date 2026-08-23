//! Benchmarks de performance pour le parsing de header SafeTensors
//! dans le crate `pmg-io`.
//!
//! Ce fichier mesure les performances de désérialisation JSON
//! des en-têtes de fichiers SafeTensors, opération critique
//! pour la lecture et la validation des métadonnées.

use criterion::{criterion_group, criterion_main, Criterion};

/// Benchmark de parsing de header SafeTensors.
fn bench_header_parse(c: &mut Criterion) {
    // Header JSON simulé avec deux tenseurs de grande taille.
    let header = serde_json::json!({
        "tensor_0": {"dtype": "F32", "shape": [1024, 1024], "data_offsets": [0, 4194304]},
        "tensor_1": {"dtype": "F32", "shape": [1024, 1024], "data_offsets": [4194304, 8388608]},
    })
    .to_string();

    // Mesure de la performance de désérialisation JSON.
    c.bench_function("parse_header_json", |b| {
        b.iter(|| {
            let _: serde_json::Value = serde_json::from_str(&header).unwrap();
        })
    });
}

criterion_group!(benches, bench_header_parse);
criterion_main!(benches);
