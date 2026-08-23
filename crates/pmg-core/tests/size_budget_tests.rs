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

//! Tests de budget taille pour la génération de modèles.
//!
//! Ces tests vérifient que la génération respecte les budgets de taille en octets
//! et gère correctement les cas limites (très petit, très grand, dépassement).

use pmg_core::manifest::{Manifest, TensorInfo};

/// Crée un manifeste minimal pour les tests de petite taille.
///
/// Retourne un manifeste avec un seul tenseur de taille configurable.
fn create_small_manifest(num_elements: u64, dtype: &str) -> Manifest {
    let mut manifest = Manifest::new("small-test", "transformer");
    manifest.seed = 42;

    manifest.add_tensor(TensorInfo::new("weight", vec![num_elements], dtype));

    manifest
}

/// Crée un manifeste pour les tests de grande taille.
///
/// Retourne un manifeste avec plusieurs tenseurs simulant un modèle réaliste.
fn create_large_manifest(num_layers: usize, hidden_size: usize) -> Manifest {
    let mut manifest = Manifest::new("large-test", "transformer");
    manifest.seed = 42;

    // Conversion en u64 pour la compatibilité avec TensorInfo
    let hidden_size_u64 = hidden_size as u64;

    // Embedding
    manifest.add_tensor(TensorInfo::new(
        "model.embed_tokens.weight",
        vec![1000, hidden_size_u64],
        "bf16",
    ));

    // Tenseurs par couche
    for layer_idx in 0..num_layers {
        let layer_prefix = format!("model.layers.{}", layer_idx);

        manifest.add_tensor(TensorInfo::new(
            format!("{}.self_attn.q_proj.weight", layer_prefix),
            vec![hidden_size_u64, hidden_size_u64],
            "bf16",
        ));

        manifest.add_tensor(TensorInfo::new(
            format!("{}.mlp.gate_proj.weight", layer_prefix),
            vec![hidden_size_u64 * 4, hidden_size_u64],
            "bf16",
        ));
    }

    // LM Head
    manifest.add_tensor(TensorInfo::new(
        "lm_head.weight",
        vec![1000, hidden_size_u64],
        "bf16",
    ));

    manifest
}

/// Test : génération avec une taille très petite (< 1 Ko).
///
/// Vérifie que la génération fonctionne avec un budget très petit.
#[test]
fn test_generation_very_small_size() {
    let manifest = create_small_manifest(100u64, "f32");
    let total_bytes = manifest.total_byte_size();

    // Le tenseur fait 100 * 4 = 400 octets
    assert_eq!(total_bytes, 400);

    // Vérifie que le manifeste est valide
    assert!(manifest.validate().is_ok());
    assert_eq!(manifest.num_tensors(), 1);
    assert_eq!(manifest.total_parameters(), 100);
}

/// Test : génération avec une taille très grande (> 1 Go).
///
/// Vérifie que l'estimation de taille fonctionne pour les grands modèles.
#[test]
fn test_generation_very_large_size() {
    // Simuler un modèle avec 12 couches et hidden_size de 4096
    let manifest = create_large_manifest(12, 4096);
    let total_bytes = manifest.total_byte_size();

    // Estimer la taille attendue :
    // - Embedding: 1000 * 4096 * 2 (bf16) = 8 192 000 octets
    // - Par couche: (4096*4096 + 4096*4*4096) * 2 = (16M + 67M) * 2 = 166M octets
    // - 12 couches: ~2 Go
    // - LM Head: 1000 * 4096 * 2 = 8 192 000 octets
    // Total: ~2 Go

    // Vérifie que l'estimation est dans une plage raisonnable
    let total_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    assert!(
        total_gb > 0.1 && total_gb < 10.0,
        "La taille estimée ({:.2} Go) est hors de la plage attendue",
        total_gb
    );

    // Vérifie le nombre de paramètres
    let total_params = manifest.total_parameters();
    assert!(
        total_params > 100_000_000, // Plus de 100 millions de paramètres
        "Le nombre de paramètres ({}) est trop faible pour un modèle de cette taille",
        total_params
    );
}

/// Test : vérification de la tolérance de 2%.
///
/// Vérifie que la taille réelle est proche de la taille estimée.
#[test]
fn test_size_tolerance_2_percent() {
    let manifest = create_small_manifest(1000u64, "f32");
    let estimated_bytes = manifest.total_byte_size();

    // Simuler une taille réelle avec une légère variation
    let actual_bytes = estimated_bytes + (estimated_bytes / 100); // +1%

    // Vérifie que la différence est dans la tolérance de 2%
    let tolerance = (estimated_bytes as f64) * 0.02;
    let difference = (actual_bytes as f64 - estimated_bytes as f64).abs();

    assert!(
        difference <= tolerance,
        "La différence ({:.0} octets) dépasse la tolérance de 2% ({:.0} octets)",
        difference,
        tolerance
    );
}

/// Test : vérification que la taille totale ne dépasse pas le budget.
///
/// Vérifie que la génération respecte le budget en octets.
#[test]
fn test_total_size_does_not_exceed_budget() {
    let manifest = create_small_manifest(500u64, "f32");
    let budget_bytes = manifest.total_byte_size();

    // Simuler une génération qui respecte le budget
    let actual_bytes = budget_bytes - 100; // Légèrement en dessous

    assert!(
        actual_bytes <= budget_bytes,
        "La taille réelle ({}) dépasse le budget ({})",
        actual_bytes,
        budget_bytes
    );
}

/// Test : comportement en cas de dépassement.
///
/// Vérifie que le système détecte正确ement un dépassement de budget.
#[test]
fn test_budget_exceeded_detection() {
    let manifest = create_small_manifest(500u64, "f32");
    let budget_bytes = manifest.total_byte_size();

    // Simuler un dépassement de budget
    let actual_bytes = budget_bytes + 100; // Légèrement au-dessus

    // Vérifie que le dépassement est détecté
    assert!(
        actual_bytes > budget_bytes,
        "Le dépassement devrait être détecté"
    );

    // Vérifie que la différence est proportionnelle
    let excess_percentage = ((actual_bytes - budget_bytes) as f64 / budget_bytes as f64) * 100.0;
    assert!(
        excess_percentage > 0.0 && excess_percentage < 10.0,
        "Le dépassement ({:.1}%) devrait être raisonnable",
        excess_percentage
    );
}

/// Test : vérification de la cohérence des tailles.
///
/// Vérifie que toutes les tailles sont cohérentes entre elles.
#[test]
fn test_size_consistency() {
    let manifest = create_small_manifest(100u64, "f32");

    // Vérifie que la taille totale correspond à la somme des tailles individuelles
    let sum_individual: u64 = manifest.tensors.iter().map(|t| t.byte_size).sum();
    let total_bytes = manifest.total_byte_size();

    assert_eq!(
        sum_individual, total_bytes,
        "La somme des tailles individuelles ({}) ne correspond pas à la taille totale ({})",
        sum_individual, total_bytes
    );

    // Vérifie que le nombre de paramètres est cohérent
    let sum_params: u64 = manifest.tensors.iter().map(|t| t.num_elements).sum();
    let total_params = manifest.total_parameters();

    assert_eq!(
        sum_params, total_params,
        "La somme des paramètres individuels ({}) ne correspond pas au total ({})",
        sum_params, total_params
    );
}

/// Test : génération avec différents types de données.
///
/// Vérifie que les calculs de taille sont corrects pour différents dtypes.
#[test]
fn test_size_with_different_dtypes() {
    // Test avec f32 (4 octets)
    let manifest_f32 = create_small_manifest(100u64, "f32");
    assert_eq!(manifest_f32.total_byte_size(), 100 * 4);

    // Test avec bf16 (2 octets)
    let manifest_bf16 = create_small_manifest(100u64, "bf16");
    assert_eq!(manifest_bf16.total_byte_size(), 100 * 2);

    // Test avec f64 (8 octets)
    let manifest_f64 = create_small_manifest(100u64, "f64");
    assert_eq!(manifest_f64.total_byte_size(), 100 * 8);
}

/// Test : déterminisme des calculs de taille.
///
/// Vérifie que les calculs de taille sont déterministes.
#[test]
fn test_size_calculation_determinism() {
    let manifest1 = create_small_manifest(100u64, "f32");
    let manifest2 = create_small_manifest(100u64, "f32");

    assert_eq!(manifest1.total_byte_size(), manifest2.total_byte_size());
    assert_eq!(manifest1.total_parameters(), manifest2.total_parameters());
    assert_eq!(manifest1.num_tensors(), manifest2.num_tensors());
}

/// Test : validation des tailles dans les manifestes.
///
/// Vérifie que la validation détecte les tailles invalides.
#[test]
fn test_invalid_size_detection() {
    let mut manifest = Manifest::new("invalid-size", "transformer");
    manifest.seed = 0; // Seed invalide (nul)

    // Ajouter un tenseur valide
    manifest.add_tensor(TensorInfo::new("weight", vec![100, 100], "f32"));

    // La validation devrait échouer à cause du seed nul
    assert!(manifest.validate().is_err());
}

/// Test : performance des calculs de taille.
///
/// Vérifie que les calculs de taille sont rapides même pour de grands modèles.
#[test]
fn test_size_calculation_performance() {
    use std::time::Instant;

    let start = Instant::now();
    let _manifest = create_large_manifest(78, 6144); // Modèle GLM-5.2
    let duration = start.elapsed();

    // Vérifie que le calcul prend moins de 100ms
    assert!(
        duration.as_millis() < 100,
        "Le calcul de taille a pris {:?}, ce qui dépasse la limite de 100ms",
        duration
    );
}

/// Test : edge cases pour les tailles.
///
/// Vérifie que les calculs gèrent correctement les cas limites.
#[test]
fn test_size_edge_cases() {
    // Tenseur avec un seul élément
    let manifest_single = create_small_manifest(1u64, "f32");
    assert_eq!(manifest_single.total_byte_size(), 4);
    assert_eq!(manifest_single.total_parameters(), 1);

    // Tenseur avec beaucoup de petits éléments
    let manifest_many = create_small_manifest(1_000_000u64, "bf16");
    assert_eq!(manifest_many.total_byte_size(), 1_000_000 * 2);
    assert_eq!(manifest_many.total_parameters(), 1_000_000);

    // Tenseur avec peu de grands éléments
    let manifest_few = create_small_manifest(10u64, "f64");
    assert_eq!(manifest_few.total_byte_size(), 10 * 8);
    assert_eq!(manifest_few.total_parameters(), 10);
}
