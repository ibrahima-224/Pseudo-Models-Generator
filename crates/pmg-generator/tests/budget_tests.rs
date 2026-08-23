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

//! Tests d'intégration pour le planner de budget D8.
//!
//! Ces tests vérifient le calcul du budget, la gestion des modes,
//! le refus PMG-204 et la tolérance de 2%.

use pmg_blueprint::architecture::ArchitectureKind;
use pmg_blueprint::naming::NamingRules;
use pmg_blueprint::tensor_spec::TensorSpec;
use pmg_blueprint::ModelBlueprint;
use pmg_core::model_config::glm52_test_config;
use pmg_core::{DType, Shape, TensorRole};

use pmg_generator::budget::{BudgetError, BudgetPlanner, GenerationMode};
use pmg_generator::seed_plan::GeneratorSeedPlan;
use pmg_generator::ModelGenerator;

/// Blueprint minimal pour les tests de budget.
fn budget_blueprint() -> ModelBlueprint {
    let config = glm52_test_config();
    let mut bp = ModelBlueprint::new(
        "glm-5.2",
        ArchitectureKind::MoETransformer,
        config,
        NamingRules::glm52(),
    );

    // Embedding avec 10 000 éléments (1000x10) pour avoir required_budget > metadata_est
    bp.embeddings.push(
        TensorSpec::new(
            "model.embed_tokens.weight",
            Shape::new(vec![1000, 10]).unwrap(),
            DType::F32,
            TensorRole::Embedding,
        )
        .unwrap(),
    );

    bp
}

/// Test : création du planner avec marge personnalisée.
#[test]
fn test_budget_planner_creation() {
    let planner = BudgetPlanner::new(0.05); // 5%
    assert_eq!(planner.margin(), 0.05);
}

/// Test : création du planner avec valeur par défaut.
#[test]
fn test_budget_planner_default() {
    let planner = BudgetPlanner::default();
    assert_eq!(planner.margin(), 0.02); // 2%
}

/// Test : calcul du budget avec des valeurs simples.
#[test]
fn test_calculate_budget_simple() {
    let planner = BudgetPlanner::new(0.0); // Pas de marge
    let budget = planner.calculate_budget(1_000_000, 100_000, 50_000);
    // 1_000_000 - 100_000 - 50_000 - 0 = 850_000
    assert_eq!(budget, 850_000);
}

/// Test : calcul du budget avec marge de 2%.
#[test]
fn test_calculate_budget_with_margin() {
    let planner = BudgetPlanner::new(0.02);
    let budget = planner.calculate_budget(1_000_000, 100_000, 50_000);
    // 1_000_000 - 100_000 - 50_000 - 20_000 (2% de 1M) = 830_000
    assert_eq!(budget, 830_000);
}

/// Test : estimation des en-têtes pour un plan vide.
#[test]
fn test_estimate_headers_empty_plan() {
    let planner = BudgetPlanner::default();
    let plan = pmg_blueprint::Plan {
        tensors: vec![],
        parameter_count: 0,
    };
    let headers = planner.estimate_headers(&plan);
    // 16 (global) + 0 * 88 = 16
    assert_eq!(headers, 16);
}

/// Test : estimation des en-tètres pour un plan avec 1 tenseur.
#[test]
fn test_estimate_headers_one_tensor() {
    let planner = BudgetPlanner::default();
    let spec = TensorSpec::new(
        "model.embed_tokens.weight",
        Shape::new(vec![100, 10]).unwrap(),
        DType::F32,
        TensorRole::Embedding,
    )
    .unwrap();

    let plan = pmg_blueprint::Plan {
        tensors: vec![spec],
        parameter_count: 1000,
    };
    let headers = planner.estimate_headers(&plan);
    // 16 (global) + 1 * 88 = 104
    assert_eq!(headers, 104);
}

/// Test : estimation des métadonnées.
#[test]
fn test_estimate_metadata() {
    let planner = BudgetPlanner::default();
    let config = glm52_test_config();
    let metadata = planner.estimate_metadata(&config);
    // 24 Ko
    assert_eq!(metadata, 24 * 1024);
}

/// Test : validation du budget dans la tolérance.
#[test]
fn test_validate_budget_within_tolerance() {
    let planner = BudgetPlanner::default();
    assert!(planner.validate_budget(100, 100, 0.0).is_ok());
    assert!(planner.validate_budget(98, 100, 0.02).is_ok());
    assert!(planner.validate_budget(102, 100, 0.02).is_ok());
}

/// Test : validation du budget hors tolérance.
#[test]
fn test_validate_budget_exceeds_tolerance() {
    let planner = BudgetPlanner::default();
    let result = planner.validate_budget(95, 100, 0.02);
    assert!(result.is_err());

    match result {
        Err(BudgetError::ToleranceExceeded {
            actual,
            target,
            tolerance,
        }) => {
            assert_eq!(actual, 95);
            assert_eq!(target, 100);
            assert_eq!(tolerance, 0.02);
        },
        _ => panic!("Erreur attendue : ToleranceExceeded"),
    }
}

/// Test : refus PMG-204 en mode full-structural.
#[test]
fn test_pmg204_refusal_full_structural() {
    let planner = BudgetPlanner::default();
    let result = planner.check_budget_for_mode(&GenerationMode::FullStructural, 800, 1000);
    assert!(result.is_err());

    match result {
        Err(BudgetError::InsufficientBudget { actual, target }) => {
            assert_eq!(actual, 800);
            assert_eq!(target, 1000);
        },
        _ => panic!("Erreur attendue : InsufficientBudget"),
    }
}

/// Test : pas de refus en mode size-constrained.
#[test]
fn test_no_refusal_size_constrained() {
    let planner = BudgetPlanner::default();
    let result = planner.check_budget_for_mode(&GenerationMode::SizeConstrained, 800, 1000);
    assert!(result.is_ok());
}

/// Test : pas de refus en mode dtype-constrained.
#[test]
fn test_no_refusal_dtype_constrained() {
    let planner = BudgetPlanner::default();
    let result = planner.check_budget_for_mode(&GenerationMode::DtypeConstrained, 800, 1000);
    assert!(result.is_ok());
}

/// Test : intégration avec ModelGenerator.
#[test]
fn test_model_generator_budget_integration() {
    let mut gen = ModelGenerator::with_seed(42, "glm-5.2");
    let blueprint = budget_blueprint();
    gen.set_blueprint(blueprint);
    gen.plan().unwrap();

    // Définir un planner avec marge de 1%
    let planner = BudgetPlanner::new(0.01);
    gen.set_budget_planner(planner);

    // Définir le mode
    gen.set_generation_mode(GenerationMode::FullStructural);

    // Calculer le budget (budget total : 10 Mo)
    let tensor_budget = gen.calculate_tensor_budget(10 * 1024 * 1024).unwrap();

    // Vérifier que le budget est positif
    assert!(tensor_budget > 0);
}

/// Test : intégration avec mode size-constrained.
#[test]
fn test_model_generator_size_constrained() {
    let mut gen = ModelGenerator::with_seed(42, "glm-5.2");
    let blueprint = budget_blueprint();
    gen.set_blueprint(blueprint);
    gen.plan().unwrap();

    // Définir un mode size-constrained
    gen.set_generation_mode(GenerationMode::SizeConstrained);

    // Calculer le budget requis en mode full-structural
    let plan = gen.plan_ref().unwrap();
    let required_budget = plan.parameter_count * 4; // 4 octets par paramètre (F32)
                                                    // required_budget = 10 000 * 4 = 40 000 octets

    // Créer un budget total tel que le budget tensoriel calculé soit inférieur au requis
    // mais positif. Nous utilisons un budget total de 40 000 octets.
    let total_budget = 40_000;

    // En mode size-constrained, même si le budget est insuffisant, cela ne doit pas échouer
    let tensor_budget = gen.calculate_tensor_budget(total_budget).unwrap();
    assert!(tensor_budget > 0);
    // Le budget tensoriel doit être inférieur au requis (car les métadonnées sont estimées à 24 Ko)
    assert!(tensor_budget < required_budget);
}

/// Test : vérification que le planner est accessible.
#[test]
fn test_budget_planner_access() {
    let mut gen = ModelGenerator::with_seed(42, "glm-5.2");
    let planner = BudgetPlanner::new(0.05);
    gen.set_budget_planner(planner);

    assert_eq!(gen.budget_planner().margin(), 0.05);
}

/// Test : vérification que le mode est accessible.
#[test]
fn test_generation_mode_access() {
    let mut gen = ModelGenerator::with_seed(42, "glm-5.2");
    gen.set_generation_mode(GenerationMode::DtypeConstrained);

    assert_eq!(gen.generation_mode(), &GenerationMode::DtypeConstrained);
}

/// Test : rétrocompatibilité avec budget = None.
#[test]
fn test_budget_none_backward_compatibility() {
    // Créer un tenseur simple pour tester la génération sans budget
    let spec = TensorSpec::new(
        "model.test.weight",
        Shape::new(vec![100, 10]).unwrap(),
        DType::F32,
        TensorRole::Embedding,
    )
    .unwrap();

    // Créer un plan de seed
    let seed_plan = GeneratorSeedPlan::new(42, "test-model", "1.0.0");

    // Générer avec budget = None (pas de limite)
    let generator = pmg_generator::tensor_generator::TensorGenerator::new(
        spec.clone(),
        seed_plan.clone(),
        None,
    );

    let values = generator.generate().unwrap();
    // Vérifier que nous avons le nombre complet d'éléments (100 * 10 = 1000)
    assert_eq!(
        values.len(),
        1000,
        "avec budget = None, doit générer tous les éléments"
    );

    // Générer avec un budget limité
    let budget_limit = 50 * 4; // 50 éléments * 4 octets = 200 octets
    let generator_with_budget =
        pmg_generator::tensor_generator::TensorGenerator::new(spec, seed_plan, Some(budget_limit));

    let values_limited = generator_with_budget.generate().unwrap();
    // Vérifier que la génération est tronquée
    assert!(
        values_limited.len() <= 50,
        "avec budget limité, doit tronquer les éléments"
    );
    assert!(
        !values_limited.is_empty(),
        "doit générer au moins un élément avec budget"
    );
}
