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

//! Sous-module contenant les politiques de génération pour les profils de modèles.
//!
//! Ce module définit les structures de configuration qui pilotent le
//! comportement spécifique à chaque modèle lors de la génération de
//! pseudo-modèles. Ces politiques sont utilisées par le crate `pmg-generator`
//! pour adapter les paramètres de génération aux caractéristiques observées
//! de chaque modèle cible.

// Sous-modules
mod config;
mod strategies;

// Ré-exports publics pour maintenir la compatibilité avec l'ancienne API.
pub use config::{
    CorrelationPolicy, DtypePolicy, GenerationPolicy, LayerPolicyGlm, LowRankPolicy, ModelPolicies,
    OutlierPolicy, SerializationPolicy, TensorRule,
};
pub use strategies::{
    CompressionStrategy, CorrelationStrategy, LowRankStrategy, OutlierStrategy, SeedStrategy,
};

// Tests unitaires
#[cfg(test)]
mod tests;
