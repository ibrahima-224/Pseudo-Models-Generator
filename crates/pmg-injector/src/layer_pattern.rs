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

//! Patterns de couche : faire varier les injections selon la profondeur.
//!
//! La spécification (étape 4.7) propose le profil linéaire
//! `p_l = p_0 + Δp·l/(L−1)`. Ce module généralise en séparant **probabilité**
//! et **amplitude** : les multiplicateurs de profil
//! ([`crate::injection_policy::LayerDepthProfile`]) sont interpolés
//! linéairement entre la première couche (`depth = 0.0`) et la dernière
//! (`depth = 1.0`), puis appliqués à une politique de base par
//! [`crate::injection_policy::policy_for_depth`].
//!
//! Trois profils prédéfinis couvrent les cas d'usage des modèles réels :
//! - [`DepthProfileKind::Uniform`] : aucune modulation (toutes couches égales) ;
//! - [`DepthProfileKind::Increasing`] : anomalies concentrées vers la fin
//!   (couches profondes, proches de la tête de sortie) ;
//! - [`DepthProfileKind::Decreasing`] : anomalies concentrées vers le début
//!   (couches d'entrée, près des embeddings).

use serde::{Deserialize, Serialize};

use crate::error::{InjectorError, InjectorResult};
use crate::injection_policy::{policy_for_depth, InjectionPolicy, LayerDepthProfile};

/// Profil de profondeur prédéfini.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum DepthProfileKind {
    /// Aucune modulation : même politique à toutes les profondeurs.
    Uniform,
    /// Anomalies croissantes avec la profondeur (fin de réseau accentuée).
    Increasing,
    /// Anomalies décroissantes avec la profondeur (début de réseau accentuée).
    Decreasing,
}

impl DepthProfileKind {
    /// Construit le [`LayerDepthProfile`] correspondant au profil prédéfini.
    ///
    /// # Entrées
    /// - `intensity` : écart total entre première et dernière couche dans
    ///   `[0, ∞)` — `0` = uniforme, `1` = modulation ×2 entre les extrêmes.
    ///
    /// # Erreurs
    /// [`InjectorError::InvalidPolicy`] si `intensity` est négatif ou non fini.
    pub fn profile(self, intensity: f64) -> InjectorResult<LayerDepthProfile> {
        if !intensity.is_finite() || intensity < 0.0 {
            return Err(InjectorError::InvalidPolicy(format!(
                "intensity doit être fini et ≥ 0, reçu {intensity}"
            )));
        }
        let p = match self {
            DepthProfileKind::Uniform => LayerDepthProfile {
                probability_start: 1.0,
                probability_end: 1.0,
                amplitude_start: 1.0,
                amplitude_end: 1.0,
            },
            DepthProfileKind::Increasing => LayerDepthProfile {
                probability_start: 1.0 / (1.0 + intensity),
                probability_end: 1.0 + intensity,
                amplitude_start: 1.0 / (1.0 + intensity),
                amplitude_end: 1.0 + intensity,
            },
            DepthProfileKind::Decreasing => LayerDepthProfile {
                probability_start: 1.0 + intensity,
                probability_end: 1.0 / (1.0 + intensity),
                amplitude_start: 1.0 + intensity,
                amplitude_end: 1.0 / (1.0 + intensity),
            },
        };
        p.validate()?;
        Ok(p)
    }
}

/// Normalise l'index d'une couche en profondeur dans `[0, 1]`.
///
/// # Entrées
/// - `layer_index` : index 0-based de la couche ;
/// - `total_layers` : nombre total de couches (`> 0`).
///
/// # Sorties
/// `l/(L−1)` si `L > 1`, `0.5` pour un modèle monocouche (position médiane :
/// une couche unique n'est ni début ni fin, elle reçoit la valeur moyenne du
/// profil).
///
/// # Erreurs
/// [`InjectorError::InvalidTensor`] si `total_layers == 0` ou si
/// `layer_index ≥ total_layers`.
///
/// # Complexité
/// O(1).
pub fn normalized_depth(layer_index: u64, total_layers: u64) -> InjectorResult<f64> {
    if total_layers == 0 {
        return Err(InjectorError::InvalidTensor("nombre de couches nul".into()));
    }
    if layer_index >= total_layers {
        return Err(InjectorError::InvalidTensor(format!(
            "layer_index {layer_index} ≥ total_layers {total_layers}"
        )));
    }
    if total_layers == 1 {
        // Une couche unique n'est ni début ni fin : position médiane (valeur
        // moyenne du profil), cohérente avec l'interpolation linéaire.
        return Ok(0.5);
    }
    Ok(layer_index as f64 / (total_layers - 1) as f64)
}

/// Applique le profil de profondeur `kind` à la politique `base` pour la
/// couche d'index `layer_index` parmi `total_layers`.
///
/// # Entrées
/// - `base` : politique de référence (avant modulation) ;
/// - `kind` : profil prédéfini ;
/// - `intensity` : écart total de modulation ;
/// - `layer_index`, `total_layers` : position de la couche.
///
/// # Sorties
/// Politique modulée pour cette couche (voir
/// [`crate::injection_policy::policy_for_depth`]).
///
/// # Erreurs
/// [`InjectorError::InvalidPolicy`] / [`InjectorError::InvalidTensor`] selon
/// le paramètre fautif.
///
/// # Complexité
/// O(1).
pub fn policy_for_layer(
    base: &InjectionPolicy,
    kind: DepthProfileKind,
    intensity: f64,
    layer_index: u64,
    total_layers: u64,
) -> InjectorResult<InjectionPolicy> {
    let profile = kind.profile(intensity)?;
    let mut modulated = base.clone();
    modulated.depth_profile = profile;
    let depth = normalized_depth(layer_index, total_layers)?;
    policy_for_depth(&modulated, depth, total_layers as usize)
}

/// Construit la liste des politiques modulées pour toutes les couches d'un
/// modèle (index 0 à `total_layers − 1`), dans l'ordre.
///
/// # Erreurs
/// [`InjectorError::InvalidTensor`] si `total_layers == 0` ;
/// [`InjectorError::InvalidPolicy`] si `intensity` est invalide.
///
/// # Complexité
/// O(total_layers).
pub fn policies_for_all_layers(
    base: &InjectionPolicy,
    kind: DepthProfileKind,
    intensity: f64,
    total_layers: u64,
) -> InjectorResult<Vec<InjectionPolicy>> {
    if total_layers == 0 {
        return Err(InjectorError::InvalidTensor("nombre de couches nul".into()));
    }
    (0..total_layers)
        .map(|l| policy_for_layer(base, kind, intensity, l, total_layers))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{normalized_depth, policies_for_all_layers, policy_for_layer, DepthProfileKind};
    use crate::error::InjectorError;
    use crate::injection_policy::InjectionPolicy;

    #[test]
    fn normalized_depth_is_linear() {
        assert_eq!(normalized_depth(0, 4).unwrap(), 0.0);
        assert_eq!(normalized_depth(3, 4).unwrap(), 1.0);
        assert!((normalized_depth(1, 4).unwrap() - 1.0 / 3.0).abs() < 1e-12);
        // Modèle monocouche : position médiane (ni début ni fin).
        assert_eq!(normalized_depth(0, 1).unwrap(), 0.5);
    }

    #[test]
    fn normalized_depth_rejects_bad_input() {
        assert!(matches!(
            normalized_depth(0, 0),
            Err(InjectorError::InvalidTensor(_))
        ));
        assert!(matches!(
            normalized_depth(4, 4),
            Err(InjectorError::InvalidTensor(_))
        ));
    }

    #[test]
    fn uniform_profile_leaves_policy_unchanged() {
        let base = InjectionPolicy::default();
        for l in 0..4u64 {
            let p = policy_for_layer(&base, DepthProfileKind::Uniform, 0.5, l, 4).unwrap();
            assert_eq!(p.outlier_frequency, base.outlier_frequency);
            assert_eq!(p.outlier_scale, base.outlier_scale);
        }
    }

    #[test]
    fn increasing_profile_concentrates_on_last_layer() {
        let base = InjectionPolicy::default();
        let first = policy_for_layer(&base, DepthProfileKind::Increasing, 1.0, 0, 4).unwrap();
        let last = policy_for_layer(&base, DepthProfileKind::Increasing, 1.0, 3, 4).unwrap();
        // Première couche : fréquence × 1/2 ; dernière : × 2.
        assert!((first.outlier_frequency - 0.5 * base.outlier_frequency).abs() < 1e-12);
        assert!((last.outlier_frequency - 2.0 * base.outlier_frequency).abs() < 1e-12);
        assert!(first.outlier_frequency < last.outlier_frequency);
        assert!(first.outlier_scale < last.outlier_scale);
    }

    #[test]
    fn decreasing_profile_inverts() {
        let base = InjectionPolicy::default();
        let first = policy_for_layer(&base, DepthProfileKind::Decreasing, 1.0, 0, 4).unwrap();
        let last = policy_for_layer(&base, DepthProfileKind::Decreasing, 1.0, 3, 4).unwrap();
        assert!(first.outlier_frequency > last.outlier_frequency);
    }

    #[test]
    fn policies_for_all_layers_ordered_and_consistent() {
        let base = InjectionPolicy::default();
        let policies =
            policies_for_all_layers(&base, DepthProfileKind::Increasing, 1.0, 5).unwrap();
        assert_eq!(policies.len(), 5);
        // La fréquence d'outliers croît strictement avec la profondeur.
        for w in policies.windows(2) {
            assert!(w[0].outlier_frequency < w[1].outlier_frequency);
        }
        // Chaque politique reste valide.
        for p in &policies {
            p.validate().unwrap();
        }
    }

    #[test]
    fn mono_layer_gets_mid_depth_policy() {
        // Une couche unique reçoit la valeur moyenne du profil (depth = 0.5).
        let base = InjectionPolicy::default();
        let p = policy_for_layer(&base, DepthProfileKind::Increasing, 3.0, 0, 1).unwrap();
        // Profil Increasing intensity=3 : start = 1/(1+3) = 0.25, end = 1+3 = 4,
        // moyenne (0.25 + 4)/2 = 2.125.
        let expected_freq = base.outlier_frequency * (0.25 + 4.0) / 2.0;
        assert!((p.outlier_frequency - expected_freq).abs() < 1e-12);
        assert!(p.validate().is_ok());
    }

    #[test]
    fn invalid_intensity_rejected() {
        let base = InjectionPolicy::default();
        assert!(matches!(
            DepthProfileKind::Uniform.profile(-1.0),
            Err(InjectorError::InvalidPolicy(_))
        ));
        assert!(policy_for_layer(&base, DepthProfileKind::Uniform, -1.0, 0, 4).is_err());
        assert!(policies_for_all_layers(&base, DepthProfileKind::Uniform, -1.0, 4).is_err());
    }

    #[test]
    fn intensity_zero_is_uniform() {
        let base = InjectionPolicy::default();
        let p = policy_for_layer(&base, DepthProfileKind::Increasing, 0.0, 3, 4).unwrap();
        assert_eq!(p.outlier_frequency, base.outlier_frequency);
        assert_eq!(p.outlier_scale, base.outlier_scale);
    }
}
