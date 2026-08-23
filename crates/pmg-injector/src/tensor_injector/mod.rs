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

//! Sous-module contenant l'orchestrateur d'injection de tenseurs.
//!
//! Ce module fournit :
//! - [`TensorInjector`] : orchestrateur principal
//! - [`InjectionStage`] : étapes du pipeline canonique
//! - Fonctions utilitaires pour la manipulation des shapes et des structures sparses

// Autoriser le warning module_inception pour ce module spécifique
#![allow(clippy::module_inception)]

// Sous-modules
mod helpers;
mod injection_stage;
mod tensor_injector;

// Ré-exports publics pour maintenir la compatibilité avec l'ancienne API.
pub use injection_stage::InjectionStage;
pub use tensor_injector::TensorInjector;

// Ré-exports des fonctions utilitaires
pub use helpers::{matrix_dims, sparse_spec_from_policy};

// Tests unitaires
#[cfg(test)]
mod tests;
