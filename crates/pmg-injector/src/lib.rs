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

//! Crate `pmg-injector` — moteur d'injection structurelle.
//!
//! Introduit volontairement les structures statistiques et anomalies contrôlées
//! nécessaires aux pseudo-modèles : **super-poids**, **corrélation**, **bas-rang**,
//! **structures sparse** et **patterns de couche**.
//!
//! # Responsabilité
//!
//! | Module | Rôle |
//! |---|---|
//! | [`injection_policy`] | Décrit **quoi** injecter (fréquences, amplitudes, probabilités), sans injecter |
//! | [`outlier_mask`] | Détermine les positions affectées (masque booléen déterministe) |
//! | [`super_weight`] | Transforme des valeurs ordinaires en valeurs extrêmes contrôlées |
//! | [`correlated`] | Introduit une dépendance contrôlée `X = ρZ + √(1−ρ²)ε` entre colonnes |
//! | [`low_rank`] | Applique `W' = W + α·UVᵀ` avec `r ≪ min(m, n)` |
//! | [`sparse_structure`] | Crée des structures localisées (blocs, bande, lignes/colonnes) |
//! | [`layer_pattern`] | Fait varier les injections selon la profondeur `θ_l = f(l)` |
//! | [`tensor_injector`] | Orchestre l'ordre canonique **Distribution → Structure → Corrélation → Bas-rang → Super-poids** |
//! | [`injection_validator`] | Mesure l'effet réel (`p̂`, mean, std, quantiles…) et compare au profil |
//!
//! # Reproductibilité
//!
//! Toute génération passe par un RNG déterministe dérivé de seed
//! ([`pmg_math::rng`]) : mêmes entrées ⇒ mêmes sorties, bit à bit.
//! Aucune source aléatoire globale (`thread_rng`) — conformité
//! `docs/documents/CAHIER DE PLAN DEVELOPPEMENT SPRINT_0_6.md` §14.
//!
//! # Dépendances
//!
//! `pmg-core` (types fondamentaux), `pmg-blueprint` (`TensorSpec`,
//! `DistributionFamily`…), `pmg-math` (RNG, distributions, statistiques,
//! bas-rang, covariance). Interdit : I/O, CLI.
//!
//! # Exemple
//!
//! ```
//! use pmg_blueprint::tensor_spec::TensorSpec;
//! use pmg_core::{DType, Shape, TensorRole};
//! use pmg_injector::injection_policy::InjectionPolicy;
//! use pmg_injector::tensor_injector::TensorInjector;
//! use pmg_math::rng::SeedPlan;
//!
//! let spec = TensorSpec::new(
//!     "model.layers.0.mlp.gate.weight",
//!     Shape::new(vec![64, 32]).unwrap(),
//!     DType::F32,
//!     TensorRole::Other,
//! )
//! .unwrap();
//! let plan = SeedPlan {
//!     seed_global: 42,
//!     model_id: "glm-5.2",
//!     tensor_name: &spec.name,
//!     layer_id: Some(0),
//!     generation_version: "1.0.0",
//! };
//! let injector = TensorInjector::from_seed_plan(&spec, InjectionPolicy::default(), &plan);
//! let tensor = injector.inject().expect("injection valide");
//! assert_eq!(tensor.len(), 64 * 32);
//! ```

pub mod correlated;
pub mod distribution_mapping;
pub mod error;
pub mod injection_policy;
pub mod injection_validator;
pub mod layer_pattern;
pub mod low_rank;
pub mod outlier_mask;
pub mod sparse_structure;
pub mod super_weight;
pub mod tensor_injector;
pub use distribution_mapping::{distribution_from_family, DEFAULT_STUDENT_T_DF};
// Ré-exports pratiques pour les consommateurs (une seule ligne d'import).
pub use error::{InjectorError, InjectorResult};
pub use injection_policy::{InjectionPolicy, LayerDepthProfile};
pub use injection_validator::{InjectionReport, InjectionValidation, ValidationTolerances};
pub use layer_pattern::DepthProfileKind;
pub use outlier_mask::OutlierMask;
pub use sparse_structure::{BlockPattern, SparseStructureSpec};
pub use super_weight::SuperWeightStrategy;
pub use tensor_injector::{InjectionStage, TensorInjector};
