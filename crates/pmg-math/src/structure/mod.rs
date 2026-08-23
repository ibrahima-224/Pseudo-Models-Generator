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

//! Structures et corrélations contrôlées pour la génération de tenseurs.
//!
//! Conformité : `docs/architecture/04-moteurs-math-injection-generation.md`
//! §5. Ce module fournit des modèles de structure contrôlés pour les tenseurs,
//! allant du modèle de base indépendant aux structures par blocs avec
//! corrélations locales.
//!
//! ## Sous-modules
//!
//! - [`base_structure`] : modèle de base indépendant W = E (chaque élément issu du générateur statistique) ;
//! - [`factors`] : génération de facteurs U et V pour la décomposition bas-rang L = UVᵀ ;
//! - [`correlation`] : corrélation contrôlée globale via matrice de covariance ;
//! - [`local_correlation`] : corrélations locales par blocs sans matrice globale ;
//! - [`block_structure`] : structure par blocs W = diag(W₁, W₂, W₃) avec interactions contrôlées.
//!
//! ## Utilisation
//!
//! Ces structures sont utilisées par l'injecteur (`pmg-injector`) pour contrôler
//! les propriétés statistiques et structurelles des tenseurs générés.

pub mod base_structure;
pub mod block_structure;
pub mod correlation;
pub mod factors;
pub mod local_correlation;

// Ré-exports pratiques
pub use base_structure::BaseStructure;
pub use block_structure::BlockStructure;
pub use correlation::{Correlation, CorrelationConfig};
pub use factors::FactorGenerator;
pub use local_correlation::LocalCorrelation;
