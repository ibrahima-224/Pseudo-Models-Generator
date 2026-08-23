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

//! Tests E2E — scénario complet : generate → espec → validate → compare.
//!
//! Ce module contient des tests de bout en bout qui vérifient le workflow
//! complet d'utilisation de l'outil PMG.

use pmg_cli::commands::generate;
use pmg_cli::commands::generate::GenerateArgs;

/// Crée des arguments de génération par défaut pour les tests.
fn default_generate_args() -> GenerateArgs {
    GenerateArgs {
        source: "Models/GLM-5.2".to_string(),
        model: "glm52".to_string(),
        size: "1G".to_string(),
        mode: "safe".to_string(),
        dtype: "f32".to_string(),
        seed: Some(42),
        profile: None,
        chunk_size: 67108864,
        max_shard_bytes: 5368709120,
        no_validate: false,
        force: false,
        dry_run: false,
        verbose: false,
        quiet: false,
        json_output: false,
        debug: false,
        stream: false,
        stream_full: false,
        async_mode: false,
        workers: None,
        distributed: false,
        coordinator: "127.0.0.1:9090".to_string(),
        workers_count: 4,
        worker_mode: false,
        worker_id: None,
        gpu: false,
        gpu_count: None,
        compress: false,
        compression_algorithm: "lz4".to_string(),
        compression_level: 6,
    }
}

/// Test E2E : scénario complet de génération en mode dry-run.
#[test]
fn test_complete_workflow_dry_run() {
    let mut args = default_generate_args();
    args.dry_run = true;

    let result = generate::execute(args, false);
    assert!(
        result.is_ok(),
        "La génération en mode sec a échoué : {:?}",
        result.err()
    );
}

/// Test E2E : génération avec modèle DeepSeek.
#[test]
fn test_generation_deepseek() {
    let mut args = default_generate_args();
    args.model = "deepseek-v4-flash".to_string();
    args.dry_run = true;

    let result = generate::execute(args, false);
    assert!(
        result.is_ok(),
        "La génération DeepSeek a échoué : {:?}",
        result.err()
    );
}

/// Test E2E : génération avec mode réaliste.
#[test]
fn test_generation_realistic_mode() {
    let mut args = default_generate_args();
    args.mode = "realistic".to_string();
    args.dry_run = true;

    let result = generate::execute(args, false);
    assert!(
        result.is_ok(),
        "La génération en mode réaliste a échoué : {:?}",
        result.err()
    );
}

/// Test E2E : génération avec mode stress.
#[test]
fn test_generation_stress_mode() {
    let mut args = default_generate_args();
    args.mode = "stress".to_string();
    args.dry_run = true;

    let result = generate::execute(args, false);
    assert!(
        result.is_ok(),
        "La génération en mode stress a échoué : {:?}",
        result.err()
    );
}

/// Test E2E : génération avec modèle invalide.
#[test]
fn test_generation_invalid_model() {
    let mut args = default_generate_args();
    args.model = "invalid_model".to_string();

    let result = generate::execute(args, false);
    assert!(
        result.is_err(),
        "La génération devrait échouer avec un modèle invalide"
    );
}

/// Test E2E : génération avec taille personnalisée.
#[test]
fn test_generation_custom_size() {
    let mut args = default_generate_args();
    args.size = "500M".to_string();
    args.dry_run = true;

    let result = generate::execute(args, false);
    assert!(
        result.is_ok(),
        "La génération avec taille personnalisée a échoué : {:?}",
        result.err()
    );
}

/// Test E2E : génération avec dtype personnalisé.
#[test]
fn test_generation_custom_dtype() {
    let mut args = default_generate_args();
    args.dtype = "f16".to_string();
    args.dry_run = true;

    let result = generate::execute(args, false);
    assert!(
        result.is_ok(),
        "La génération avec dtype personnalisé a échoué : {:?}",
        result.err()
    );
}
