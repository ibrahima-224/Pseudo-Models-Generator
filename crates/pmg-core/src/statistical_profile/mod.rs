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

//! Module des profils statistiques externes pour la génération de pseudo-modèles.
//!
//! Ce module définit les structures de données pour les profils statistiques
//! qui configurent la génération des poids. Les profils sont chargés depuis
//! des fichiers JSON externes et permettent de personnaliser les paramètres
//! statistiques par modèle.
//!
//! ## Structure d'un profil statistique
//!
//! Un profil statistique contient :
//! - **distributions** : types de distributions pour les poids et outliers
//! - **outlier_config** : configuration des outliers
//! - **correlation_config** : configuration des corrélations entre paramètres
//! - **low_rank_config** : configuration de la structure à faible rang
//! - **super_weight_config** : configuration des super-poids (magnitude élevée)
//!
//! ## Utilisation
//!
//! ```rust,ignore
//! // Chargement depuis un fichier (via pmg-io)
//! use pmg_io::statistical_profile::load_from_file;
//! use std::path::Path;
//!
//! let profile = load_from_file(Path::new("statistical_profiles/glm52.json"))
//!     .expect("profil valide");
//!
//! // Ou création avec valeurs par défaut (via pmg-core)
//! use pmg_core::statistical_profile::StatisticalProfile;
//! let default_profile = StatisticalProfile::glm52_default();
//! ```

// Sous-modules contenant les configurations et le profil principal.
mod configs;
mod profile;

// Ré-exports publics pour maintenir la compatibilité avec l'ancienne API.
pub use configs::{
    CorrelationConfig, LowRankConfig, OutlierProfileConfig, ProfileDistributionConfig,
    SuperWeightConfig, WeightDistribution,
};
pub use profile::StatisticalProfile;
