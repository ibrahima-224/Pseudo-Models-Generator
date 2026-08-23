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

//! Crate `pmg-math` — moteur mathématique et statistique déterministe.
//!
//! Primitives statistiques pures, reproductibles et sans entrée-sortie,
//! utilisées par l'injection, la génération et la validation
//! (`docs/architecture/04-moteurs-math-injection-generation.md`).
//!
//! ## Responsabilité
//!
//! - RNG déterministe (ChaCha12 via `rand_chacha`) et dérivation hiérarchique
//!   des seeds (`derive_seed`, SHA-256, concaténation canonique à taille
//!   préfixée) — module [`rng`] ;
//! - statistiques descriptives sur slices `f64` (mean, variance, quantiles,
//!   skewness, kurtosis) — module [`statistics`] ;
//! - distributions (normale, Student-t, Laplace, log-normale, Weibull,
//!   Pareto, mélanges) avec contrat commun [`Distribution`] — modules
//!   [`distributions`] et [`distribution`] ;
//! - covariance PSD (Cholesky strict, échec explicite non-PSD) et génération
//!   corrélée `X = LZ + μ` — module [`covariance`] ;
//! - structures bas-rang `W = α·UVᵀ` (génération complète et par blocs) et
//!   estimation de rang effectif — module [`low_rank`].
//!
//! ## Dépendances
//!
//! `pmg-core` (types fondamentaux), `rand`/`rand_chacha` (flux ChaCha12),
//! `sha2` (dérivation de seeds), `serde` (configs sérialisables),
//! `thiserror` (erreurs typées). Interdit : I/O, CLI, dépendances ML lourdes.
//!
//! ## Reproductibilité
//!
//! Même entrée (seed, plan, version du générateur) ⇒ même sortie, testée en
//! stricte égalité binaire sur une même plateforme ; inter-plateformes =
//! « meilleure effort » documenté (doc 4 §1.3.4). Zéro `thread_rng`.
//!
//! # Exemple
//!
//! ```
//! use pmg_math::distribution::{Distribution, from_config};
//! use pmg_core::distribution_config::DistributionConfig;
//! use pmg_math::rng::{derive_seed, DeterministicRng, SeedPlan};
//!
//! // 1) Dérive la seed canonique du tenseur, puis crée son flux.
//! let plan = SeedPlan {
//!     seed_global: 42,
//!     model_id: "glm-5.2",
//!     tensor_name: "model.layers.0.mlp.gate.weight",
//!     layer_id: Some(0),
//!     generation_version: "1.0.0",
//! };
//! let seed = derive_seed(&plan);
//! let mut rng = DeterministicRng::from_seed(seed);
//!
//! // 2) Construit une distribution depuis sa config et échantillonne.
//! let cfg = DistributionConfig::normal(0.0, 1.0);
//! let mut dist = pmg_math::distribution::from_config(&cfg).unwrap();
//! let x: f64 = dist.sample(&mut rng);
//! assert!(x.is_finite());
//! ```

pub mod correlation_analysis;
pub mod covariance;
pub mod distribution;
pub mod distributions;
pub mod error;
pub mod generator;
pub mod low_rank;
pub mod low_rank_analysis;
pub mod outlier_analysis;
pub mod outliers;
pub mod rng;
pub mod special;
pub mod statistics;
pub mod structure;

pub use distribution::{from_config, Distribution};
pub use error::{MathError, MathResult};
pub use generator::{generate_normal, generate_uniform};
pub use pmg_core::distribution_config::{DistributionConfig, DistributionKind};
pub use rng::{derive_seed, derive_sub_seed, seed_to_u64, DeterministicRng, SeedPlan};
