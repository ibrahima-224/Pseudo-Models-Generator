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

//! Crate `pmg-core` — types fondamentaux et invariants de PMG.
//!
//! Cette crate est le **socle** du workspace : elle définit le langage commun
//! utilisé par toutes les autres crates (`DType`, `Shape`, `TensorMetadata`,
//! `ModelConfig`, `TensorRole`, erreurs typées…) sans jamais dépendre d'elles.
//!
//! ## Responsabilité
//!
//! - Types de données fondamentaux et leurs invariants (dimensions strictement
//!   positives, produit vérifié, divisibilité des têtes…) ;
//! - erreurs typées [`CoreError`] (via `thiserror`) ;
//! - catégories transverses [`Origin`] / [`Confidence`] (provenance des valeurs).
//!
//! ## Dépendances
//!
//! Légères uniquement : `serde`, `serde_json`, `thiserror` (pinées dans
//! `[workspace.dependencies]`). Interdit : I/O, RNG, algorithmes métier.
//!
//! ## Modules
//!
//! | Module | Contenu |
//! |---|---|
//! | [`dtype`] | [`DType`] : 19 variantes, taille/bits, noms Safetensors |
//! | [`shape`] | [`Shape`] : dimensions, produit vérifié |
//! | [`tensor_metadata`] | [`TensorMetadata`] : métadonnées sans données |
//! | [`tensor_role`] | [`TensorRole`] : rôle fonctionnel + mapping de noms |
//! | [`model_config`] | [`ModelConfig`], [`AttentionKind`] : config normalisée |
//! | [`moe`] | [`MoeConfig`] : experts, top-k, couches denses |
//! | [`statistical_profile`] | [`StatisticalProfile`] : profils statistiques externes |
//! | [`error`] | [`CoreError`] : erreurs typées françaises |
//! | [`validation`] | prédicats de validation partagés |
//! | [`origin`] | [`Origin`] / [`Confidence`] : provenance des valeurs |
//! | [`storage_vs_quant`] | [`StorageDType`] / [`QuantizationScheme`] |
//!
//! # Exemple
//!
//! ```
//! use pmg_core::{CoreResult, DType, Shape, TensorMetadata};
//!
//! fn build_metadata() -> CoreResult<TensorMetadata> {
//!     let shape = Shape::new(vec![6144, 6144])?; // dims > 0 exigé
//!     TensorMetadata::new("model.embed_tokens.weight", shape, DType::Bf16)
//! }
//!
//! let meta = build_metadata().expect("métadonnées valides");
//! assert_eq!(meta.byte_size().unwrap(), Some(6144 * 6144 * 2));
//! ```

pub mod core_config;
pub mod distribution_config;
pub mod dtype;
pub mod error;
pub mod generation_plan;
pub mod generator_config;
pub mod manifest;
pub mod memory;
pub mod metadata_hash;
pub mod model_config;
pub mod moe;
pub mod origin;
pub mod outlier_metadata;
pub mod provenance;
pub mod rng_trait;
pub mod shape;
pub mod shard_plan;
pub mod statistical_profile;
pub mod storage_vs_quant;
pub mod structure_config;
pub mod tensor_metadata;
pub mod tensor_role;
pub mod validation;

/// Version du projet PMG, centralisée ici pour éviter toute divergence entre
/// les crates (voir `docs/architecture/02-workspace-et-crates.md` §5).
pub const PMG_VERSION: &str = "1.0.0";

// Ré-exports pratiques pour les consommateurs (une seule ligne d'import).
pub use core_config::{AmplitudeStrategy, CoreConfig, GenerationMode, OutlierConfig};
pub use distribution_config::{DistributionConfig, DistributionKind};
pub use dtype::DType;
pub use error::{CoreError, CoreResult};
pub use generation_plan::GenerationPlan;
// NOTE: `GeneratorConfig` est un alias pour `CoreConfig` dans le module `generator_config`
// pour assurer la transition sans cassure API. Il sera supprimé dans une version future.
pub use generator_config::GeneratorConfig;
pub use manifest::{Architecture, Manifest, ModelType, TensorInfo};
pub use metadata_hash::{HashAlgorithm, MetadataHash};
pub use model_config::{AttentionKind, ModelConfig};
pub use moe::MoeConfig;
pub use origin::{Confidence, Origin};
pub use outlier_metadata::{OutlierMetadata, OutlierStrategyKind};
pub use provenance::{Provenance, ProvenanceError, ProvenanceOrigin};
pub use shape::Shape;
pub use shard_plan::{ShardPlan, TensorShard};
pub use statistical_profile::{
    CorrelationConfig, LowRankConfig, ProfileDistributionConfig as StatisticalDistributionConfig,
    StatisticalProfile, SuperWeightConfig, WeightDistribution,
};
pub use storage_vs_quant::{QuantizationScheme, StorageDType};
pub use structure_config::{StructureConfig, StructureStrength};
pub use tensor_metadata::TensorMetadata;
pub use tensor_role::TensorRole;

#[cfg(test)]
mod tests {
    use super::PMG_VERSION;

    #[test]
    fn version_matches_cargo_manifest() {
        // La constante centralisée doit rester alignée sur le manifeste Cargo.
        assert_eq!(PMG_VERSION, env!("CARGO_PKG_VERSION"));
    }
}
