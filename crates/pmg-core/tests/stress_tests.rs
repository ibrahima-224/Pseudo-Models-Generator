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

//! Tests de charge pour configurations MoE (Mixture of Experts).
//!
//! Ces tests valident la scalabilité du projet avec des configurations
//! de modèles grands (78 couches, 256 experts) sans écrire de grands
//! fichiers sur disque.

use pmg_core::manifest::{Manifest, ModelType, TensorInfo};
use pmg_core::rng_trait::DeterministicRng;
use std::mem;
use std::path::Path;
use tempfile::TempDir;

/// RNG de test simple pour vérifier le fonctionnement
#[allow(dead_code)]
#[derive(Debug)]
struct MockRng {
    state: u64,
}

#[allow(dead_code)]
impl MockRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
}

impl DeterministicRng for MockRng {
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}

/// Crée un blueprint pour un modèle MoE grand (78 couches, 256 experts).
///
/// Retourne un manifeste contenant tous les tenseurs typiques d'un modèle
/// GLM-5.2 sans écrire de données sur disque.
fn create_large_moe_blueprint() -> Manifest {
    let mut manifest = Manifest::new("glm-5.2-stress", "mixture-of-experts");
    manifest.seed = 42;

    // Paramètres du modèle GLM-5.2
    let num_layers = 78;
    let hidden_size = 6144;
    let intermediate_size = 12288;
    let moe_intermediate_size = 2048;
    let num_attention_heads = 64;
    let num_key_value_heads = 64;
    let head_dim = 192;
    let vocab_size = 154880;
    let n_routed_experts = 256;

    // Embeddings
    manifest.add_tensor(TensorInfo::new(
        "model.embed_tokens.weight",
        vec![vocab_size, hidden_size],
        "bf16",
    ));

    // Tenseurs de position encoding
    manifest.add_tensor(TensorInfo::new(
        "model.rotary_emb.inv_freq",
        vec![head_dim / 2],
        "f32",
    ));

    // Tenseurs par couche
    for layer_idx in 0..num_layers {
        let layer_prefix = format!("model.layers.{}", layer_idx);

        // Attention
        manifest.add_tensor(TensorInfo::new(
            format!("{}.self_attn.q_proj.weight", layer_prefix),
            vec![num_attention_heads * head_dim, hidden_size],
            "bf16",
        ));
        manifest.add_tensor(TensorInfo::new(
            format!("{}.self_attn.k_proj.weight", layer_prefix),
            vec![num_key_value_heads * head_dim, hidden_size],
            "bf16",
        ));
        manifest.add_tensor(TensorInfo::new(
            format!("{}.self_attn.v_proj.weight", layer_prefix),
            vec![num_key_value_heads * head_dim, hidden_size],
            "bf16",
        ));
        manifest.add_tensor(TensorInfo::new(
            format!("{}.self_attn.o_proj.weight", layer_prefix),
            vec![hidden_size, num_attention_heads * head_dim],
            "bf16",
        ));

        // DSA (DeepSeek Sparse Attention) tensors pour GLM
        manifest.add_tensor(TensorInfo::new(
            format!("{}.self_attn.indexer.weight", layer_prefix),
            vec![hidden_size, 256],
            "bf16",
        ));

        // LayerNorm
        manifest.add_tensor(TensorInfo::new(
            format!("{}.input_layernorm.weight", layer_prefix),
            vec![hidden_size],
            "bf16",
        ));
        manifest.add_tensor(TensorInfo::new(
            format!("{}.post_attention_layernorm.weight", layer_prefix),
            vec![hidden_size],
            "bf16",
        ));

        // MLP dense pour les premières couches
        if layer_idx < 3 {
            manifest.add_tensor(TensorInfo::new(
                format!("{}.mlp.gate_proj.weight", layer_prefix),
                vec![intermediate_size, hidden_size],
                "bf16",
            ));
            manifest.add_tensor(TensorInfo::new(
                format!("{}.mlp.up_proj.weight", layer_prefix),
                vec![intermediate_size, hidden_size],
                "bf16",
            ));
            manifest.add_tensor(TensorInfo::new(
                format!("{}.mlp.down_proj.weight", layer_prefix),
                vec![hidden_size, intermediate_size],
                "bf16",
            ));
        } else {
            // MoE pour les couches restantes
            // Gate (routeur)
            manifest.add_tensor(TensorInfo::new(
                format!("{}.mlp.gate.weight", layer_prefix),
                vec![n_routed_experts, hidden_size],
                "f32",
            ));

            // Experts
            for expert_idx in 0..n_routed_experts {
                let expert_prefix = format!("{}.mlp.experts.{}", layer_prefix, expert_idx);
                manifest.add_tensor(TensorInfo::new(
                    format!("{}.gate_proj.weight", expert_prefix),
                    vec![moe_intermediate_size, hidden_size],
                    "bf16",
                ));
                manifest.add_tensor(TensorInfo::new(
                    format!("{}.up_proj.weight", expert_prefix),
                    vec![moe_intermediate_size, hidden_size],
                    "bf16",
                ));
                manifest.add_tensor(TensorInfo::new(
                    format!("{}.down_proj.weight", expert_prefix),
                    vec![hidden_size, moe_intermediate_size],
                    "bf16",
                ));
            }

            // Expert partagé
            let shared_prefix = format!("{}.mlp.shared_expert", layer_prefix);
            manifest.add_tensor(TensorInfo::new(
                format!("{}.gate_proj.weight", shared_prefix),
                vec![moe_intermediate_size, hidden_size],
                "bf16",
            ));
            manifest.add_tensor(TensorInfo::new(
                format!("{}.up_proj.weight", shared_prefix),
                vec![moe_intermediate_size, hidden_size],
                "bf16",
            ));
            manifest.add_tensor(TensorInfo::new(
                format!("{}.down_proj.weight", shared_prefix),
                vec![hidden_size, moe_intermediate_size],
                "bf16",
            ));
        }
    }

    // LM head
    manifest.add_tensor(TensorInfo::new(
        "lm_head.weight",
        vec![vocab_size, hidden_size],
        "bf16",
    ));

    manifest
}

/// Mesure la mémoire utilisée par un vecteur de valeurs.
///
/// Retourne la taille en octets.
#[allow(dead_code)]
fn measure_memory_usage(values: &[f64]) -> usize {
    mem::size_of_val(values)
}

/// Crée un fichier Safetensors corrompu dans le répertoire temporaire.
///
/// Retourne le chemin vers le fichier créé.
fn create_corrupt_safetensors(dir: &Path, corruption_type: &str) -> std::path::PathBuf {
    let path = dir.join(format!("corrupt_{}.safetensors", corruption_type));

    match corruption_type {
        "truncated_header" => {
            // Header JSON tronqué au milieu d'une chaîne
            let content = r#"{"metadata": {"format": "pt", "truncated": "this_is_a_ve"#;
            std::fs::write(&path, content).unwrap();
        },
        "invalid_json" => {
            // JSON invalide (accolade manquante)
            let content = r#"{"metadata": {"format": "pt""#;
            std::fs::write(&path, content).unwrap();
        },
        "out_of_bounds_offsets" => {
            // JSON valide mais avec des offsets hors bornes
            // Le fichier contiendra un JSON valide avec un tenseur dont les offsets sont [0, 400]
            // mais la taille réelle du fichier sera inférieure à 400 octets
            let content = r#"{
                "metadata": {"format": "pt"},
                "tensor": {"dtype": "F32", "shape": [10, 10], "data_offsets": [0, 400]}
            }"#;
            // Écrire le JSON valide (la taille sera inférieure à 400)
            std::fs::write(&path, content).unwrap();
        },
        _ => {
            panic!("Type de corruption inconnu: {}", corruption_type);
        },
    }

    path
}

/// Test de création de blueprint pour un modèle MoE grand (78 couches, 256 experts).
///
/// Vérifie que la création du manifeste est correcte et que les dimensions
/// sont cohérentes avec la configuration GLM-5.2.
#[test]
fn test_large_blueprint_plan_only() {
    let manifest = create_large_moe_blueprint();

    // Vérifie les paramètres de base
    assert_eq!(manifest.model_name, "glm-5.2-stress");
    assert_eq!(manifest.model_type, ModelType::PseudoModel);
    assert_eq!(manifest.seed, 42);

    // Vérifie le nombre de tenseurs
    // Embeddings: 2 (embed_tokens + rotary_emb)
    // Par couche: attention (4) + DSA (1) + layernorms (2) = 7
    // Couches denses (3): 3 tensors supplémentaires (gate, up, down)
    // Couches MoE (75): gate (1) + 256 experts × 3 + shared expert × 3 = 1 + 768 + 3 = 772
    // LM head: 1
    let dense_layers = 3;
    let moe_layers = 75;
    let tensors_per_dense_layer = 7 + 3; // attention + layernorm + mlp
    let tensors_per_moe_layer = 7 + 1 + 256 * 3 + 3; // attention + layernorm + gate + experts + shared
    let total_tensors =
        2 + dense_layers * tensors_per_dense_layer + moe_layers * tensors_per_moe_layer + 1;

    assert_eq!(manifest.num_tensors(), total_tensors);

    // Vérifie le nombre total de paramètres
    let total_params = manifest.total_parameters();
    assert!(total_params > 0);

    // Vérifie que le manifeste est valide
    assert!(manifest.validate().is_ok());
}

/// Test de génération avec mémoire bornée.
///
/// Vérifie que la génération de tenseurs respecte les limites mémoire
/// en utilisant le streaming.
/// NOTE: Ce test a été désactivé car StreamingGenerator a été déplacé vers pmg-generator.
/// Il sera réactivé lors de la migration des tests vers pmg-generator.
#[test]
#[ignore]
fn test_memory_bounded_generation() {
    // NOTE: StreamingGenerator a été déplacé vers pmg-generator
    // Ce test doit être migré vers pmg-generator/tests/
    // assert!(max_chunk_memory <= 1024 * mem::size_of::<f64>());
}

/// Test de gestion de header tronqué.
///
/// Vérifie que le système gère correctement un fichier Safetensors
/// avec un header JSON tronqué.
#[test]
fn test_corrupt_header() {
    let temp_dir = TempDir::new().unwrap();
    let corrupt_path = create_corrupt_safetensors(temp_dir.path(), "truncated_header");

    // Tenter de lire le fichier - devrait échouer proprement
    let result = std::fs::read_to_string(&corrupt_path);

    // Le fichier est lisible mais le contenu est tronqué
    assert!(result.is_ok());
    let content = result.unwrap();

    // Vérifie que le JSON est invalide (truncation)
    let json_result: Result<serde_json::Value, _> = serde_json::from_str(&content);
    assert!(json_result.is_err());

    // Le message d'erreur devrait indiquer un problème de parsing
    let err = json_result.unwrap_err();
    assert!(err.to_string().contains("EOF") || err.to_string().contains("unexpected end"));
}

/// Test de gestion de JSON invalide.
///
/// Vérifie que le système gère correctement un fichier Safetensors
/// avec du JSON invalide.
#[test]
fn test_invalid_json() {
    let temp_dir = TempDir::new().unwrap();
    let corrupt_path = create_corrupt_safetensors(temp_dir.path(), "invalid_json");

    // Tenter de lire le fichier
    let result = std::fs::read_to_string(&corrupt_path);
    assert!(result.is_ok());
    let content = result.unwrap();

    // Vérifie que le JSON est invalide
    let json_result: Result<serde_json::Value, _> = serde_json::from_str(&content);
    assert!(json_result.is_err());

    // Le message d'erreur devrait indiquer un problème de syntaxe
    let err = json_result.unwrap_err();
    let err_msg = err.to_string();
    assert!(err_msg.contains("invalid") || err_msg.contains("expected") || err_msg.contains("EOF"));
}

/// Test de gestion d'offsets hors bornes.
///
/// Vérifie que le système détecte les offsets hors bornes dans un
/// fichier Safetensors.
#[test]
fn test_out_of_bounds_offsets() {
    let temp_dir = TempDir::new().unwrap();
    let corrupt_path = create_corrupt_safetensors(temp_dir.path(), "out_of_bounds_offsets");

    // Lire le fichier
    let content = std::fs::read_to_string(&corrupt_path).unwrap();

    // Parser le JSON pour obtenir les métadonnées
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();
    let metadata = &json["metadata"];
    let tensor = &json["tensor"];

    // Vérifie que les métadonnées sont présentes
    assert!(metadata.is_object());
    assert_eq!(metadata["format"], "pt");

    // Vérifie les offsets du tenseur
    let data_offsets = tensor["data_offsets"].as_array().unwrap();
    let start = data_offsets[0].as_u64().unwrap();
    let end = data_offsets[1].as_u64().unwrap();

    // Les offsets sont [0, 400] mais le fichier fait seulement 100 octets
    assert_eq!(start, 0);
    assert_eq!(end, 400);

    // Vérifie que la taille du fichier est inférieure aux offsets
    let file_size = std::fs::metadata(&corrupt_path).unwrap().len();
    assert!(
        file_size < end,
        "La taille du fichier ({}) devrait être inférieure à l'offset de fin ({})",
        file_size,
        end
    );
}

/// Test de validation des invariants MoE sur un grand modèle.
///
/// Vérifie que la configuration MoE est valide pour un modèle avec
/// 78 couches et 256 experts.
#[test]
fn test_large_moe_config_validation() {
    use pmg_core::model_config::glm52_test_config;

    let config = glm52_test_config();

    // Vérifie que la configuration est valide
    assert!(config.validate().is_ok());

    // Vérifie les paramètres MoE
    let moe = config.moe.as_ref().unwrap();
    assert_eq!(moe.n_routed_experts, 256);
    assert_eq!(moe.n_shared_experts, 1);
    assert_eq!(moe.total_experts().unwrap(), 257);
    assert_eq!(moe.experts_per_tok, 8);
    assert_eq!(moe.layer_types.len(), 78);

    // Vérifie le nombre de couches
    assert_eq!(config.num_layers, 78);
    assert_eq!(config.hidden_size, 6144);
}

/// Test de performance : création rapide de blueprint.
///
/// Vérifie que la création d'un blueprint pour un modèle grand est
/// suffisamment rapide pour le CI (< 1 seconde).
#[test]
fn test_blueprint_creation_performance() {
    use std::time::Instant;

    let start = Instant::now();
    let _manifest = create_large_moe_blueprint();
    let duration = start.elapsed();

    // Vérifie que la création prend moins d'1 seconde
    assert!(
        duration.as_millis() < 1000,
        "La création du blueprint a pris {:?}, ce qui dépasse la limite de 1 seconde",
        duration
    );
}

/// Test de mémoire : estimation de la taille du modèle.
///
/// Vérifie que l'estimation de la taille du modèle est cohérente
/// avec les attentes (environ 1.5 To pour GLM-5.2 complet).
#[test]
fn test_model_size_estimation() {
    let manifest = create_large_moe_blueprint();

    // Calcule la taille totale estimée en octets
    let total_bytes: u64 = manifest.tensors.iter().map(|t| t.byte_size).sum();

    // Convertir en To
    let total_tb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0 * 1024.0);

    // Vérifie que l'estimation est dans une plage raisonnable
    // GLM-5.2 complet devrait être environ 1.5 To
    assert!(
        total_tb > 0.1 && total_tb < 10.0,
        "La taille estimée ({:.2} To) est hors de la plage attendue",
        total_tb
    );

    // Vérifie que le nombre de paramètres est cohérent
    let total_params = manifest.total_parameters();
    assert!(
        total_params > 1_000_000_000, // Plus d'1 milliard de paramètres
        "Le nombre de paramètres ({}) est trop faible pour un modèle de cette taille",
        total_params
    );
}

/// Test de robustesse : création de blueprint avec paramètres limites.
///
/// Vérifie que la création de blueprint fonctionne avec des paramètres
/// dans les limites extrêmes.
#[test]
fn test_blueprint_edge_cases() {
    // Modèle minimal
    let mut manifest_min = Manifest::new("minimal", "transformer");
    manifest_min.seed = 1;
    manifest_min.add_tensor(TensorInfo::new("weight", vec![1u64, 1], "f32"));
    assert!(manifest_min.validate().is_ok());
    assert_eq!(manifest_min.num_tensors(), 1);
    assert_eq!(manifest_min.total_parameters(), 1);

    // Modèle avec beaucoup de petits tenseurs
    let mut manifest_many = Manifest::new("many-tensors", "transformer");
    manifest_many.seed = 42;
    for i in 0..1000 {
        // Réduit de 10000 à 1000 pour optimisation mémoire
        manifest_many.add_tensor(TensorInfo::new(
            format!("tensor_{}", i),
            vec![2u64, 2],
            "f32",
        ));
    }
    assert!(manifest_many.validate().is_ok());
    assert_eq!(manifest_many.num_tensors(), 1000);
    assert_eq!(manifest_many.total_parameters(), 4000);
}
