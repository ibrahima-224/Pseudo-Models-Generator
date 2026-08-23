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

//! Sous-module contenant les informations de provenance et traçabilité.
//!
//! Ce module fournit :
//! - [`ProvenanceInfo`] : informations globales de génération
//! - [`InputMetadata`] : métadonnées des sources d'entrée
//! - [`GeneratedArtifact`] : artifacts produits
//! - [`GenerationEnvironment`] : environnement d'exécution
//! - [`GranularProvenance`] : provenance granulaire pour tenseurs et champs

// Sous-modules
mod granular;
mod provenance_info;
mod types;

// Ré-exports publics pour maintenir la compatibilité avec l'ancienne API.
pub use granular::{FieldProvenance, GranularProvenance, TensorProvenance};
pub use provenance_info::ProvenanceInfo;
pub use types::{GeneratedArtifact, GenerationEnvironment, InputMetadata, SourceMetadata};

// Tests unitaires
#[cfg(test)]
mod tests;
