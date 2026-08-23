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

//! Politique d'injection : décrire **quoi** injecter, sans encore injecter.
//!
//! [`InjectionPolicy`] est la description sérialisable des anomalies et
//! structures statistiques à introduire dans un tenseur : fréquence et
//! amplitude des outliers (super-poids), force de corrélation, probabilité et
//! rang des composantes bas-rang, queues lourdes et structures parcimonieuses.
//!
//! Conformité : `docs/documents/CAHIER DE PLAN DEVELOPPEMENT SPRINT_0_6.md`
//! étape 4.1. La politique est purement déclarative : aucun RNG, aucune valeur
//! générée ici. Sa validation est faite à la construction ([`InjectionPolicy::new`])
//! et peut être rejouée via [`InjectionPolicy::validate`] après désérialisation.
//!
//! # Exemple
//!
//! ```
//! use pmg_injector::injection_policy::InjectionPolicy;
//!
//! // Politique par défaut : anomalies rares mais réalistes
//! let policy = InjectionPolicy::default();
//! assert!(policy.validate().is_ok());
//!
//! // Politique personnalisée avec validation
//! let policy = InjectionPolicy::new(
//!     0.02,  // outlier_frequency
//!     5.0,   // outlier_scale
//!     0.4,   // correlation_strength
//!     0.15,  // low_rank_probability
//!     10,    // low_rank_rank
//!     0.3,   // low_rank_alpha
//!     0.1,   // heavy_tail_probability
//!     4.0,   // heavy_tail_df
//!     0.05,  // sparse_structure_probability
//!     0.3,   // sparse_density
//! ).unwrap();
//!
//! // Politique neutre : aucune injection
//! let policy = InjectionPolicy::none();
//! assert_eq!(policy.outlier_frequency, 0.0);
//! ```

use serde::{Deserialize, Serialize};

use crate::error::{InjectorError, InjectorResult};

/// Profil de profondeur d'une couche : multiplicateur de probabilité et
/// d'amplitude des injections en fonction de la profondeur normalisée.
///
/// Voir [`crate::layer_pattern`] pour la modulation `θ_l = f(l)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LayerDepthProfile {
    /// Multiplicateur appliqué à la fréquence d'outliers en première couche
    /// (profondeur normalisée `0.0`).
    pub probability_start: f64,
    /// Multiplicateur appliqué à la fréquence d'outliers en dernière couche
    /// (profondeur normalisée `1.0`).
    pub probability_end: f64,
    /// Multiplicateur appliqué à l'amplitude des outliers en première couche.
    pub amplitude_start: f64,
    /// Multiplicateur appliqué à l'amplitude des outliers en dernière couche.
    pub amplitude_end: f64,
}

impl LayerDepthProfile {
    /// Profil neutre : aucun effet de profondeur (multiplicateurs unitaires).
    pub fn neutral() -> Self {
        Self {
            probability_start: 1.0,
            probability_end: 1.0,
            amplitude_start: 1.0,
            amplitude_end: 1.0,
        }
    }

    /// Valide que les multiplicateurs sont finis et non négatifs.
    pub fn validate(&self) -> InjectorResult<()> {
        for (name, v) in [
            ("probability_start", self.probability_start),
            ("probability_end", self.probability_end),
            ("amplitude_start", self.amplitude_start),
            ("amplitude_end", self.amplitude_end),
        ] {
            if !v.is_finite() || v < 0.0 {
                return Err(InjectorError::InvalidPolicy(format!(
                    "{name} doit être fini et ≥ 0, reçu {v}"
                )));
            }
        }
        Ok(())
    }
}

impl Default for LayerDepthProfile {
    fn default() -> Self {
        Self::neutral()
    }
}

/// Politique d'injection structurelle d'un tenseur.
///
/// # Invariants (vérifiés par [`InjectionPolicy::validate`])
/// - toutes les probabilités sont dans `[0, 1]` et finies ;
/// - `outlier_scale ≥ 1` (un super-poids doit amplifier, jamais atténuer) ;
/// - `correlation_strength ∈ [0, 1)` (ρ = 1 est exclu : variance nulle de la
///   composante indépendante, construction instable) ;
/// - `low_rank_rank ≥ 1` ;
/// - amplitudes finies et non négatives ;
/// - le profil de profondeur est valide.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct InjectionPolicy {
    /// Probabilité qu'un élément soit un outlier (super-poids) — `[0, 1]`.
    pub outlier_frequency: f64,
    /// Amplitude multiplicative des outliers — `≥ 1` (`w' = s·w`).
    pub outlier_scale: f64,
    /// Force de corrélation entre colonnes — `[0, 1)` (`X = ρZ + √(1−ρ²)ε`).
    pub correlation_strength: f64,
    /// Probabilité d'appliquer une composante bas-rang — `[0, 1]`.
    pub low_rank_probability: f64,
    /// Rang cible de la composante bas-rang — `≥ 1`.
    pub low_rank_rank: usize,
    /// Amplitude α de la composante bas-rang (`W' = W + α·UVᵀ`).
    pub low_rank_alpha: f64,
    /// Probabilité d'un mélange à queues lourdes pour les outliers — `[0, 1]`.
    pub heavy_tail_probability: f64,
    /// Degrés de liberté de la Student-t utilisée pour les outliers statistiques.
    pub heavy_tail_df: f64,
    /// Probabilité d'introduire une structure sparse localisée — `[0, 1]`.
    pub sparse_structure_probability: f64,
    /// Densité cible de la structure sparse — `(0, 1]` (fraction non nulle).
    pub sparse_density: f64,
    /// Profil de profondeur (modulation par couche).
    pub depth_profile: LayerDepthProfile,
}

impl InjectionPolicy {
    /// Construit une politique et la valide immédiatement.
    ///
    /// # Erreurs
    /// [`InjectorError::InvalidPolicy`] dès qu'un paramètre viole ses bornes.
    ///
    /// # Complexité
    /// O(1).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        outlier_frequency: f64,
        outlier_scale: f64,
        correlation_strength: f64,
        low_rank_probability: f64,
        low_rank_rank: usize,
        low_rank_alpha: f64,
        heavy_tail_probability: f64,
        heavy_tail_df: f64,
        sparse_structure_probability: f64,
        sparse_density: f64,
    ) -> InjectorResult<Self> {
        let policy = Self {
            outlier_frequency,
            outlier_scale,
            correlation_strength,
            low_rank_probability,
            low_rank_rank,
            low_rank_alpha,
            heavy_tail_probability,
            heavy_tail_df,
            sparse_structure_probability,
            sparse_density,
            depth_profile: LayerDepthProfile::neutral(),
        };
        policy.validate()?;
        Ok(policy)
    }

    /// Politique strictement neutre : aucune injection (toutes les
    /// probabilités nulles, aucune amplitude).
    pub fn none() -> Self {
        Self {
            outlier_frequency: 0.0,
            outlier_scale: 1.0,
            correlation_strength: 0.0,
            low_rank_probability: 0.0,
            low_rank_rank: 1,
            low_rank_alpha: 0.0,
            heavy_tail_probability: 0.0,
            heavy_tail_df: 5.0,
            sparse_structure_probability: 0.0,
            sparse_density: 0.5,
            depth_profile: LayerDepthProfile::neutral(),
        }
    }

    /// Valide tous les invariants de la politique.
    ///
    /// Cette méthode est rejouée après désérialisation : une politique
    /// chargée depuis du JSON doit être vérifiée avant toute injection.
    ///
    /// # Erreurs
    /// [`InjectorError::InvalidPolicy`] avec le nom du champ fautif.
    pub fn validate(&self) -> InjectorResult<()> {
        for (name, v) in [
            ("outlier_frequency", self.outlier_frequency),
            ("low_rank_probability", self.low_rank_probability),
            ("heavy_tail_probability", self.heavy_tail_probability),
            (
                "sparse_structure_probability",
                self.sparse_structure_probability,
            ),
        ] {
            if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                return Err(InjectorError::InvalidPolicy(format!(
                    "{name} doit être dans [0, 1], reçu {v}"
                )));
            }
        }
        if !self.outlier_scale.is_finite() || self.outlier_scale < 1.0 {
            return Err(InjectorError::InvalidPolicy(format!(
                "outlier_scale doit être fini et ≥ 1 (amplifier, jamais atténuer), reçu {}",
                self.outlier_scale
            )));
        }
        if !self.correlation_strength.is_finite()
            || !(0.0..1.0).contains(&self.correlation_strength)
        {
            return Err(InjectorError::InvalidPolicy(format!(
                "correlation_strength doit être dans [0, 1), reçu {}",
                self.correlation_strength
            )));
        }
        if self.low_rank_rank < 1 {
            return Err(InjectorError::InvalidPolicy(format!(
                "low_rank_rank doit être ≥ 1, reçu {}",
                self.low_rank_rank
            )));
        }
        if !self.low_rank_alpha.is_finite() || self.low_rank_alpha < 0.0 {
            return Err(InjectorError::InvalidPolicy(format!(
                "low_rank_alpha doit être fini et ≥ 0, reçu {}",
                self.low_rank_alpha
            )));
        }
        if !self.heavy_tail_df.is_finite() || self.heavy_tail_df <= 0.0 {
            return Err(InjectorError::InvalidPolicy(format!(
                "heavy_tail_df doit être fini et > 0 (Student-t), reçu {}",
                self.heavy_tail_df
            )));
        }
        if !self.sparse_density.is_finite()
            || !(0.0..=1.0).contains(&self.sparse_density)
            || self.sparse_density == 0.0
        {
            return Err(InjectorError::InvalidPolicy(format!(
                "sparse_density doit être dans (0, 1], reçu {}",
                self.sparse_density
            )));
        }
        self.depth_profile.validate()
    }
}

impl Default for InjectionPolicy {
    /// Politique par défaut : anomalies rares mais réalistes
    /// (outlier 1 %, corrélation modérée 0.3, bas-rang 20 % de rang 8,
    /// queues lourdes 5 %, sparse 10 %).
    fn default() -> Self {
        Self {
            outlier_frequency: 0.01,
            outlier_scale: 5.0,
            correlation_strength: 0.3,
            low_rank_probability: 0.2,
            low_rank_rank: 8,
            low_rank_alpha: 0.5,
            heavy_tail_probability: 0.05,
            heavy_tail_df: 4.0,
            sparse_structure_probability: 0.1,
            sparse_density: 0.5,
            depth_profile: LayerDepthProfile::neutral(),
        }
    }
}

/// Construit une [`InjectionPolicy`] adaptée à la profondeur d'une couche :
/// les multiplicateurs du profil sont appliqués aux paramètres de base.
///
/// # Entrées
/// - `base` : politique de référence ;
/// - `depth` : profondeur normalisée de la couche dans `[0, 1]`
///   (`0.0` = première couche, `1.0` = dernière) ;
/// - `total_layers` : nombre total de couches (`1` = modèle monocouche).
///
/// # Sorties
/// Politique modulée : `fréquence_l = fréquence · p(depth)` et
/// `amplitude_l = amplitude · a(depth)` (interpolation linéaire du profil).
///
/// # Erreurs
/// [`InjectorError::InvalidTensor`] si `depth` est hors `[0, 1]` ou
/// `total_layers == 0`.
///
/// # Complexité
/// O(1).
///
/// # Exemple
///
/// ```
/// use pmg_injector::injection_policy::{InjectionPolicy, LayerDepthProfile, policy_for_depth};
///
/// let mut base = InjectionPolicy::default();
/// base.depth_profile = LayerDepthProfile {
///     probability_start: 1.0,
///     probability_end: 2.0,
///     amplitude_start: 1.0,
///     amplitude_end: 3.0,
/// };
///
/// let first_layer = policy_for_depth(&base, 0.0, 4).unwrap();
/// let last_layer = policy_for_depth(&base, 1.0, 4).unwrap();
///
/// assert_eq!(first_layer.outlier_frequency, base.outlier_frequency);
/// assert_eq!(last_layer.outlier_frequency, 2.0 * base.outlier_frequency);
/// assert_eq!(last_layer.outlier_scale, 3.0 * base.outlier_scale);
/// ```
pub fn policy_for_depth(
    base: &InjectionPolicy,
    depth: f64,
    total_layers: usize,
) -> InjectorResult<InjectionPolicy> {
    if !(0.0..=1.0).contains(&depth) {
        return Err(InjectorError::InvalidTensor(format!(
            "profondeur normalisée hors [0, 1] : {depth}"
        )));
    }
    if total_layers == 0 {
        return Err(InjectorError::InvalidTensor("nombre de couches nul".into()));
    }
    let p = base.depth_profile;
    let prob_factor = lerp(p.probability_start, p.probability_end, depth);
    let amp_factor = lerp(p.amplitude_start, p.amplitude_end, depth);
    let mut modulated = base.clone();
    // Les fréquences restent bornées à 1 par construction des multiplicateurs
    // non négatifs : on reclamp par sécurité (aucune probabilité > 1).
    modulated.outlier_frequency = (base.outlier_frequency * prob_factor).min(1.0);
    modulated.heavy_tail_probability = (base.heavy_tail_probability * prob_factor).min(1.0);
    modulated.sparse_structure_probability =
        (base.sparse_structure_probability * prob_factor).min(1.0);
    modulated.low_rank_probability = (base.low_rank_probability * prob_factor).min(1.0);
    modulated.outlier_scale = base.outlier_scale * amp_factor;
    modulated.low_rank_alpha = base.low_rank_alpha * amp_factor;
    modulated.validate()?;
    Ok(modulated)
}

/// Interpolation linéaire `a + (b − a)·t`.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::{policy_for_depth, InjectionPolicy, LayerDepthProfile};
    use crate::error::InjectorError;

    #[test]
    fn default_policy_is_valid() {
        let p = InjectionPolicy::default();
        p.validate().unwrap();
    }

    #[test]
    fn none_policy_disables_everything() {
        let p = InjectionPolicy::none();
        p.validate().unwrap();
        assert_eq!(p.outlier_frequency, 0.0);
        assert_eq!(p.correlation_strength, 0.0);
        assert_eq!(p.low_rank_probability, 0.0);
    }

    #[test]
    fn probabilities_out_of_bounds_rejected() {
        // Probabilité > 1 rejetée.
        assert!(matches!(
            InjectionPolicy::new(1.5, 5.0, 0.3, 0.2, 8, 0.5, 0.05, 4.0, 0.1, 0.5),
            Err(InjectorError::InvalidPolicy(_))
        ));
        // Probabilité négative rejetée.
        assert!(matches!(
            InjectionPolicy::new(-0.1, 5.0, 0.3, 0.2, 8, 0.5, 0.05, 4.0, 0.1, 0.5),
            Err(InjectorError::InvalidPolicy(_))
        ));
        // NaN rejeté.
        assert!(matches!(
            InjectionPolicy::new(f64::NAN, 5.0, 0.3, 0.2, 8, 0.5, 0.05, 4.0, 0.1, 0.5),
            Err(InjectorError::InvalidPolicy(_))
        ));
    }

    #[test]
    fn scale_below_one_rejected() {
        // Un super-poids doit amplifier : scale < 1 rejeté.
        assert!(matches!(
            InjectionPolicy::new(0.01, 0.5, 0.3, 0.2, 8, 0.5, 0.05, 4.0, 0.1, 0.5),
            Err(InjectorError::InvalidPolicy(_))
        ));
    }

    #[test]
    fn correlation_strength_one_rejected() {
        // ρ = 1 exclu (variance nulle de la composante indépendante).
        assert!(matches!(
            InjectionPolicy::new(0.01, 5.0, 1.0, 0.2, 8, 0.5, 0.05, 4.0, 0.1, 0.5),
            Err(InjectorError::InvalidPolicy(_))
        ));
        // ρ = 0 accepté.
        assert!(InjectionPolicy::new(0.01, 5.0, 0.0, 0.2, 8, 0.5, 0.05, 4.0, 0.1, 0.5).is_ok());
    }

    #[test]
    fn rank_zero_rejected() {
        assert!(matches!(
            InjectionPolicy::new(0.01, 5.0, 0.3, 0.2, 0, 0.5, 0.05, 4.0, 0.1, 0.5),
            Err(InjectorError::InvalidPolicy(_))
        ));
    }

    #[test]
    fn sparse_density_zero_rejected() {
        assert!(matches!(
            InjectionPolicy::new(0.01, 5.0, 0.3, 0.2, 8, 0.5, 0.05, 4.0, 0.1, 0.0),
            Err(InjectorError::InvalidPolicy(_))
        ));
    }

    #[test]
    fn serde_roundtrip_preserves_policy() {
        let p = InjectionPolicy::default();
        let json = serde_json::to_string(&p).unwrap();
        let back: InjectionPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn policy_for_depth_interpolates() {
        let base = InjectionPolicy {
            depth_profile: LayerDepthProfile {
                probability_start: 1.0,
                probability_end: 2.0,
                amplitude_start: 1.0,
                amplitude_end: 3.0,
            },
            ..Default::default()
        };
        let first = policy_for_depth(&base, 0.0, 4).unwrap();
        let last = policy_for_depth(&base, 1.0, 4).unwrap();
        assert_eq!(first.outlier_frequency, base.outlier_frequency);
        assert_eq!(last.outlier_frequency, 2.0 * base.outlier_frequency);
        assert_eq!(last.outlier_scale, 3.0 * base.outlier_scale);
        // Milieu : interpolation linéaire.
        let mid = policy_for_depth(&base, 0.5, 4).unwrap();
        assert!((mid.outlier_frequency - 1.5 * base.outlier_frequency).abs() < 1e-12);
        assert!((mid.outlier_scale - 2.0 * base.outlier_scale).abs() < 1e-12);
    }

    #[test]
    fn policy_for_depth_rejects_bad_input() {
        let base = InjectionPolicy::default();
        assert!(policy_for_depth(&base, -0.1, 4).is_err());
        assert!(policy_for_depth(&base, 1.1, 4).is_err());
        assert!(policy_for_depth(&base, 0.5, 0).is_err());
    }

    #[test]
    fn depth_profile_validation() {
        let mut profile = LayerDepthProfile::neutral();
        assert!(profile.validate().is_ok());
        profile.amplitude_end = -1.0;
        assert!(profile.validate().is_err());
    }
}
