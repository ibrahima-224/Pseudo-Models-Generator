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

//! Sous-module contenant le validateur principal pour les pseudo-modèles.
//!
//! Ce module fournit le validateur principal qui combine les différents
//! contrôles de validation pour les pseudo-modèles générés.

// Sous-modules
mod model_validator;

// Ré-exports publics pour maintenir la compatibilité avec l'ancienne API.
pub use model_validator::ModelValidator;

// Tests unitaires
#[cfg(test)]
mod tests;
