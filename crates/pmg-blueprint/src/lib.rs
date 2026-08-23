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

//! Crate `pmg-blueprint` — description abstraite des pseudo-modèles.
//!
//! Un *blueprint* décrit la structure complète d'un pseudo-modèle
//! (architecture, couches, spécifications de tenseurs, règles) **sans aucun
//! poids** : c'est la « recette » consommée par le générateur.
//!
//! ## Responsabilité
//!
//! - [`ModelBlueprint`] : description globale (architecture + config + tenseurs) ;
//! - [`LayerSpec`] : spécification d'une couche (attention, MLP, normes, MoE) ;
//! - [`TensorSpec`] : spécification de génération d'un tenseur ;
//! - [`NamingRules`] : conventions de nommage alignées sur les index réels ;
//! - [`plan_blueprint`] : planification déterministe Blueprint → liste de specs.
//!
//! ## Dépendances
//!
//! `pmg-core`. Interdit : RNG, I/O, données numériques.
//!
//! ## Modules
//!
//! | Module | Contenu |
//! |---|---|
//! | [`blueprint`] | [`ModelBlueprint`] |
//! | [`layer`] | [`LayerSpec`], [`LayerKind`], [`LayerPolicy`] |
//! | [`tensor_spec`] | [`TensorSpec`], distribution/structure/outliers |
//! | [`architecture`] | [`ArchitectureKind`], [`ArchitectureMeta`] |
//! | [`moe`] | [`MoeBlockSpec`], [`ExpertSpec`] |
//! | [`naming`] | [`NamingRules`], [`ExpertProj`] |
//! | [`planner`] | [`plan_blueprint`], [`planner::build_moe_specs`] |
//! | [`validation`] | [`BlueprintValidation`] |
//! | [`error`] | [`BlueprintError`] |
//!
//! # Exemple
//!
//! ```
//! use pmg_blueprint::{ArchitectureKind, ModelBlueprint, NamingRules, plan_blueprint};
//! use pmg_core::model_config::glm52_test_config;
//! use pmg_core::{DType, Shape, TensorRole};
//! use pmg_blueprint::tensor_spec::TensorSpec;
//!
//! let mut bp = ModelBlueprint::new(
//!     "glm-5.2",
//!     ArchitectureKind::MoETransformer,
//!     glm52_test_config(),
//!     NamingRules::glm52(),
//! );
//! bp.embeddings.push(
//!     TensorSpec::new(
//!         "model.embed_tokens.weight",
//!         Shape::new(vec![154880, 6144]).unwrap(),
//!         DType::Bf16,
//!         TensorRole::Embedding,
//!     )
//!     .unwrap(),
//! );
//! let plan = plan_blueprint(&bp).expect("plan valide");
//! assert_eq!(plan.tensors.len(), 1);
//! ```

pub mod architecture;
pub mod blueprint;
pub mod error;
pub mod layer;
pub mod moe;
pub mod naming;
pub mod planner;
pub mod tensor_spec;
pub mod validation;

// Ré-exports pratiques.
pub use architecture::{ArchitectureKind, ArchitectureMeta};
pub use blueprint::ModelBlueprint;
pub use error::{BlueprintError, BlueprintResult};
pub use layer::{LayerKind, LayerPolicy, LayerSpec};
pub use moe::{ExpertSpec, MoeBlockSpec};
pub use naming::{ExpertProj, NamingRules};
pub use planner::{plan_blueprint, Plan};
pub use tensor_spec::{
    DistributionFamily, DistributionSpec, OutlierSpec, StructureSpec, TensorSpec,
};
pub use validation::BlueprintValidation;
