//! Tests de stress pour grandes configurations
//!
//! Ce module contient des tests pour valider le comportement du système
//! avec des configurations de grande taille : nombre élevé de couches,
//! tenseurs, experts MoE, et grandes dimensions.

use pmg_blueprint::blueprint::ModelBlueprint;
use pmg_blueprint::layer::{LayerConfig, LayerKind};
use pmg_blueprint::moe::MoeBlock;
use pmg_blueprint::planner::plan_blueprint;
use pmg_core::dtype::DType;
use pmg_core::shape::Shape;
use pmg_blueprint::tensor_spec::TensorSpec;

/// Crée un blueprint avec un grand nombre de couches.
///
/// # Paramètres
/// - `n_layers` : nombre de couches à créer
///
/// # Retourne
/// Un `ModelBlueprint` avec `n_layers` couches standard.
fn blueprint_with_many_layers(n_layers: usize) -> ModelBlueprint {
    let mut bp = ModelBlueprint::new("stress-test-model".to_string());

    // Embedding
    bp.add_tensor(
        TensorSpec::new(
            "model.embed_tokens.weight",
            Shape::new(vec![32000, 4096]).unwrap(),
            DType::Bf16,
        )
        .unwrap(),
    );

    // Couches
    for i in 0..n_layers {
        let mut layer = LayerConfig::new(i, LayerKind::Transformer);
        layer.hidden_size = 4096;
        layer.intermediate_size = 11008;
        bp.add_layer(layer);

        // Tenseurs de la couche
        bp.add_tensor(
            TensorSpec::new(
                &format!("model.layers.{}.self_attn.q_proj.weight", i),
                Shape::new(vec![4096, 4096]).unwrap(),
                DType::Bf16,
            )
            .unwrap(),
        );
        bp.add_tensor(
            TensorSpec::new(
                &format!("model.layers.{}.mlp.gate_proj.weight", i),
                Shape::new(vec![11008, 4096]).unwrap(),
                DType::Bf16,
            )
            .unwrap(),
        );
    }

    // Norme finale
    bp.add_tensor(
        TensorSpec::new(
            "model.norm.weight",
            Shape::new(vec![4096]).unwrap(),
            DType::Bf16,
        )
        .unwrap(),
    );

    // LM Head
    bp.add_tensor(
        TensorSpec::new(
            "lm_head.weight",
            Shape::new(vec![32000, 4096]).unwrap(),
            DType::Bf16,
        )
        .unwrap(),
    );

    bp
}

/// Crée un blueprint avec un grand nombre de tenseurs par couche.
fn blueprint_with_many_tensors_per_layer(n_tensors: usize) -> ModelBlueprint {
    let mut bp = ModelBlueprint::new("stress-test-model".to_string());

    // Embedding
    bp.add_tensor(
        TensorSpec::new(
            "model.embed_tokens.weight",
            Shape::new(vec![32000, 4096]).unwrap(),
            DType::Bf16,
        )
        .unwrap(),
    );

    // Une couche avec beaucoup de tenseurs
    let mut layer = LayerConfig::new(0, LayerKind::Transformer);
    layer.hidden_size = 4096;
    bp.add_layer(layer);

    for i in 0..n_tensors {
        bp.add_tensor(
            TensorSpec::new(
                &format!("model.layers.0.tensor_{}.weight", i),
                Shape::new(vec![256, 256]).unwrap(),
                DType::F32,
            )
            .unwrap(),
        );
    }

    // Norme finale
    bp.add_tensor(
        TensorSpec::new(
            "model.norm.weight",
            Shape::new(vec![4096]).unwrap(),
            DType::Bf16,
        )
        .unwrap(),
    );

    bp
}

/// Crée un blueprint avec une configuration MoE à beaucoup d'experts.
fn blueprint_with_many_experts(n_experts: usize) -> ModelBlueprint {
    let mut bp = ModelBlueprint::new("stress-test-moe".to_string());

    // Embedding
    bp.add_tensor(
        TensorSpec::new(
            "model.embed_tokens.weight",
            Shape::new(vec![32000, 4096]).unwrap(),
            DType::Bf16,
        )
        .unwrap(),
    );

    // Couche MoE
    let mut layer = LayerConfig::new(0, LayerKind::MoE);
    layer.hidden_size = 4096;
    layer.intermediate_size = 1024;
    bp.add_layer(layer);

    // Router
    bp.add_tensor(
        TensorSpec::new(
            "model.layers.0.mlp.gate.weight",
            Shape::new(vec![n_experts, 4096]).unwrap(),
            DType::Bf16,
        )
        .unwrap(),
    );

    // Experts
    for i in 0..n_experts {
        bp.add_tensor(
            TensorSpec::new(
                &format!("model.layers.0.mlp.experts.{}.gate_proj.weight", i),
                Shape::new(vec![1024, 4096]).unwrap(),
                DType::Bf16,
            )
            .unwrap(),
        );
        bp.add_tensor(
            TensorSpec::new(
                &format!("model.layers.0.mlp.experts.{}.up_proj.weight", i),
                Shape::new(vec![1024, 4096]).unwrap(),
                DType::Bf16,
            )
            .unwrap(),
        );
        bp.add_tensor(
            TensorSpec::new(
                &format!("model.layers.0.mlp.experts.{}.down_proj.weight", i),
                Shape::new(vec![4096, 1024]).unwrap(),
                DType::Bf16,
            )
            .unwrap(),
        );
    }

    // Norme finale
    bp.add_tensor(
        TensorSpec::new(
            "model.norm.weight",
            Shape::new(vec![4096]).unwrap(),
            DType::Bf16,
        )
        .unwrap(),
    );

    bp
}

/// Crée un blueprint avec de grandes dimensions de tenseurs.
fn blueprint_with_large_dimensions() -> ModelBlueprint {
    let mut bp = ModelBlueprint::new("stress-test-large-dims".to_string());

    // Grand embedding
    bp.add_tensor(
        TensorSpec::new(
            "model.embed_tokens.weight",
            Shape::new(vec![100000, 8192]).unwrap(),
            DType::Bf16,
        )
        .unwrap(),
    );

    // Couche avec grandes dimensions
    let mut layer = LayerConfig::new(0, LayerKind::Transformer);
    layer.hidden_size = 8192;
    layer.intermediate_size = 28672;
    bp.add_layer(layer);

    bp.add_tensor(
        TensorSpec::new(
            "model.layers.0.self_attn.q_proj.weight",
            Shape::new(vec![8192, 8192]).unwrap(),
            DType::Bf16,
        )
        .unwrap(),
    );

    bp.add_tensor(
        TensorSpec::new(
            "model.layers.0.mlp.gate_proj.weight",
            Shape::new(vec![28672, 8192]).unwrap(),
            DType::Bf16,
        )
        .unwrap(),
    );

    // Norme finale
    bp.add_tensor(
        TensorSpec::new(
            "model.norm.weight",
            Shape::new(vec![8192]).unwrap(),
            DType::Bf16,
        )
        .unwrap(),
    );

    // LM Head
    bp.add_tensor(
        TensorSpec::new(
            "lm_head.weight",
            Shape::new(vec![100000, 8192]).unwrap(),
            DType::Bf16,
        )
        .unwrap(),
    );

    bp
}

// ============================================================================
// Tests de stress
// ============================================================================

/// Test avec 100 couches.
#[test]
fn stress_100_layers() {
    let bp = blueprint_with_many_layers(100);
    let plan = plan_blueprint(&bp).unwrap();

    // 1 embed + 100 couches × 2 tenseurs + 1 norme + 1 lm_head = 203 tenseurs
    assert_eq!(plan.tensors.len(), 203);
    assert!(bp.validate().is_valid());
}

/// Test avec 500 tenseurs dans une seule couche.
#[test]
fn stress_500_tensors_per_layer() {
    let bp = blueprint_with_many_tensors_per_layer(500);
    let plan = plan_blueprint(&bp).unwrap();

    // 1 embed + 500 tenseurs + 1 norme = 502 tenseurs
    assert_eq!(plan.tensors.len(), 502);
    assert!(bp.validate().is_valid());
}

/// Test avec 128 experts MoE.
#[test]
fn stress_128_experts_moe() {
    let bp = blueprint_with_many_experts(128);
    let plan = plan_blueprint(&bp).unwrap();

    // 1 embed + (1 router + 128 experts × 3 tenseurs) + 1 norme = 387 tenseurs
    assert_eq!(plan.tensors.len(), 387);
    assert!(bp.validate().is_valid());
}

/// Test avec de grandes dimensions (100k vocab, 8192 hidden).
#[test]
fn stress_large_dimensions() {
    let bp = blueprint_with_large_dimensions();
    let plan = plan_blueprint(&bp).unwrap();

    // Vérifie que le plan est valide
    assert!(plan.tensors.len() >= 4);
    assert!(bp.validate().is_valid());

    // Vérifie les dimensions
    let embed = &plan.tensors[0];
    assert_eq!(embed.shape.dims(), &[100000, 8192]);
}

/// Test avec un modèle complet de type GLM-5.2 (78 couches).
#[test]
fn stress_glm52_like() {
    let mut bp = ModelBlueprint::new("glm52-stress".to_string());

    // Embedding
    bp.add_tensor(
        TensorSpec::new(
            "model.embed_tokens.weight",
            Shape::new(vec![154880, 6144]).unwrap(),
            DType::Bf16,
        )
        .unwrap(),
    );

    // 78 couches
    for i in 0..78 {
        let mut layer = LayerConfig::new(i, LayerKind::Transformer);
        layer.hidden_size = 6144;
        layer.intermediate_size = 16384;
        bp.add_layer(layer);

        // 4 tenseurs par couche (q, k, v, o projections)
        for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            bp.add_tensor(
                TensorSpec::new(
                    &format!("model.layers.0.self_attn.{}.weight", proj),
                    Shape::new(vec![6144, 6144]).unwrap(),
                    DType::Bf16,
                )
                .unwrap(),
            );
        }
    }

    // Norme finale
    bp.add_tensor(
        TensorSpec::new(
            "model.norm.weight",
            Shape::new(vec![6144]).unwrap(),
            DType::Bf16,
        )
        .unwrap(),
    );

    // LM Head
    bp.add_tensor(
        TensorSpec::new(
            "lm_head.weight",
            Shape::new(vec![154880, 6144]).unwrap(),
            DType::Bf16,
        )
        .unwrap(),
    );

    let plan = plan_blueprint(&bp).unwrap();

    // Vérifie le nombre total de tenseurs
    // 1 embed + 78 couches × 4 tenseurs + 1 norme + 1 lm_head = 315 tenseurs
    assert_eq!(plan.tensors.len(), 315);
    assert!(bp.validate().is_valid());
}

/// Test de déterminisme avec de grandes configurations.
#[test]
fn stress_determinism_large_config() {
    let bp = blueprint_with_many_layers(50);

    let plan1 = plan_blueprint(&bp).unwrap();
    let plan2 = plan_blueprint(&bp).unwrap();

    // Vérifie la déterminisme
    assert_eq!(plan1.tensors.len(), plan2.tensors.len());

    for (t1, t2) in plan1.tensors.iter().zip(plan2.tensors.iter()) {
        assert_eq!(t1.name, t2.name);
        assert_eq!(t1.shape, t2.shape);
        assert_eq!(t1.dtype, t2.dtype);
    }
}

/// Test de performance : temps de planification avec beaucoup de tenseurs.
#[test]
fn stress_planning_performance() {
    let bp = blueprint_with_many_tensors_per_layer(1000);

    let start = std::time::Instant::now();
    let plan = plan_blueprint(&bp).unwrap();
    let duration = start.elapsed();

    // La planification doit prendre moins de 1 seconde
    assert!(
        duration.as_secs() < 1,
        "La planification a pris {:?} (max 1s)",
        duration
    );

    // 1 embed + 1000 tenseurs + 1 norme = 1002 tenseurs
    assert_eq!(plan.tensors.len(), 1002);
}

/// Test de validation avec de grandes configurations.
#[test]
fn stress_validation_large_config() {
    let bp = blueprint_with_many_layers(100);

    // La validation ne doit pas panic
    let result = bp.validate();
    assert!(result.is_valid(), "La validation a échoué : {:?}", result);

    // Vérifie le nombre de paramètres
    let param_count = bp.parameter_count().unwrap();
    assert!(param_count > 0, "Le nombre de paramètres doit être positif");
}

/// Test de mémoire : vérifie que les grandes configurations ne provoquent pas d'OOM.
#[test]
fn stress_memory_large_config() {
    // Crée un blueprint avec 200 couches
    let bp = blueprint_with_many_layers(200);

    // La création du plan ne doit pas provoquer d'OOM
    let plan = plan_blueprint(&bp).unwrap();

    // Vérifie que tous les tenseurs ont des dimensions valides
    for tensor in &plan.tensors {
        assert!(tensor.shape.num_elements().unwrap() > 0);
    }
}

/// Test de边界值 : nombre maximum de couches raisonnable.
#[test]
fn stress_max_layers_boundary() {
    // Test avec 1000 couches (limite raisonnable)
    let bp = blueprint_with_many_layers(1000);
    let plan = plan_blueprint(&bp).unwrap();

    // 1 embed + 1000 couches × 2 tenseurs + 1 norme + 1 lm_head = 2003 tenseurs
    assert_eq!(plan.tensors.len(), 2003);
    assert!(bp.validate().is_valid());
}

/// Test de边界值 : nombre maximum d'experts MoE.
#[test]
fn stress_max_experts_boundary() {
    // Test avec 256 experts (limite raisonnable)
    let bp = blueprint_with_many_experts(256);
    let plan = plan_blueprint(&bp).unwrap();

    // 1 embed + (1 router + 256 experts × 3 tenseurs) + 1 norme = 771 tenseurs
    assert_eq!(plan.tensors.len(), 771);
    assert!(bp.validate().is_valid());
}
