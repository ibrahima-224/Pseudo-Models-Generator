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

//! Calcul de l'amplitude des super-poids selon différentes stratégies.
//!
//! Ce module fournit les primitives pour déterminer l'amplitude (magnitude)
//! des anomalies à injecter, en fonction de propriétés statistiques du tenseur
//! ou de paramètres fixes.
//!
//! # Stratégies implémentées
//!
//! - [`AmplitudeStrategy::Fixed`] : amplitude constante spécifiée par l'utilisateur ;
//! - [`AmplitudeStrategy::RelativeToStd`] : amplitude proportionnelle à l'écart-type ;
//! - [`AmplitudeStrategy::QuantileBased`] : amplitude basée sur un quantile de la distribution ;
//! - [`AmplitudeStrategy::HeavyTail`] : amplitude générée par une distribution à queue lourde.
//!
//! # Conformité
//!
//! Spécification Sprint 9, étape 5.2 : « Amplitude des super-poids ».

use crate::error::{MathError, MathResult};
use crate::rng::DeterministicRng;
use crate::statistics;

/// Stratégie de calcul de l'amplitude d'un outlier.
#[derive(Debug, Clone, PartialEq)]
pub enum AmplitudeStrategy {
    /// Amplitude constante (fixe).
    Fixed(f64),
    /// Amplitude = k × écart-type d'échantillon, où k est un facteur multiplicatif.
    RelativeToStd { k: f64 },
    /// Amplitude = quantile p de la distribution (p ∈ [0,1]).
    QuantileBased { p: f64 },
    /// Amplitude tirée d'une distribution à queue lourde (Student-t de df degrés de liberté).
    HeavyTail { df: f64 },
}

impl AmplitudeStrategy {
    /// Valide les paramètres de la stratégie.
    ///
    /// # Erreurs
    /// [`MathError::InvalidParameter`] si les paramètres sont hors bornes.
    pub fn validate(&self) -> MathResult<()> {
        match self {
            Self::Fixed(a) => {
                if !a.is_finite() || *a < 0.0 {
                    return Err(MathError::InvalidParameter(format!(
                        "amplitude fixe doit être finie et ≥ 0, reçu {a}"
                    )));
                }
            },
            Self::RelativeToStd { k } => {
                if !k.is_finite() || *k <= 0.0 {
                    return Err(MathError::InvalidParameter(format!(
                        "facteur k de relative_to_std doit être fini et > 0, reçu {k}"
                    )));
                }
            },
            Self::QuantileBased { p } => {
                if !p.is_finite() || *p < 0.0 || *p > 1.0 {
                    return Err(MathError::InvalidParameter(format!(
                        "quantile p doit être dans [0, 1], reçu {p}"
                    )));
                }
            },
            Self::HeavyTail { df } => {
                if !df.is_finite() || *df <= 0.0 {
                    return Err(MathError::InvalidParameter(format!(
                        "degrés de liberté df de Student-t doit être fini et > 0, reçu {df}"
                    )));
                }
            },
        }
        Ok(())
    }
}

/// Calcule l'amplitude d'un outlier selon la stratégie et les données du tenseur.
///
/// # Arguments
/// - `strategy` : stratégie de calcul ;
/// - `data` : slice des valeurs du tenseur (peut être vide pour certaines stratégies) ;
/// - `rng` : flux déterministe pour les stratégies stochastiques.
///
/// # Retour
/// L'amplitude calculée (toujours ≥ 0).
///
/// # Erreurs
/// - [`MathError::InvalidParameter`] si les paramètres de la stratégie sont invalides ;
/// - [`MathError::EmptyData`] si les données sont requises mais vides.
pub fn compute_amplitude(
    strategy: &AmplitudeStrategy,
    data: &[f64],
    rng: &mut DeterministicRng,
) -> MathResult<f64> {
    strategy.validate()?;

    match strategy {
        AmplitudeStrategy::Fixed(a) => Ok(*a),

        AmplitudeStrategy::RelativeToStd { k } => {
            if data.is_empty() {
                return Err(MathError::EmptyData(
                    "données requises pour relative_to_std".into(),
                ));
            }
            let std = statistics::std_sample(data)?;
            Ok(k * std)
        },

        AmplitudeStrategy::QuantileBased { p } => {
            if data.is_empty() {
                return Err(MathError::EmptyData(
                    "données requises pour quantile_based".into(),
                ));
            }
            // Utilise le quantile comme amplitude (valeur absolue).
            let q = quantile(data, *p)?;
            Ok(q.abs())
        },

        AmplitudeStrategy::HeavyTail { df } => {
            // Génère un tirage d'une Student-t standard (centrée, scale=1)
            // et prend la valeur absolue comme amplitude.
            let t = sample_student_t(*df, rng);
            Ok(t.abs())
        },
    }
}

/// Calcule le quantile d'un échantillon par interpolation linéaire.
///
/// # Algorithme
/// 1. Trie les données ;
/// 2. Calcule l'index positionnel : `i = p × (n - 1)` ;
/// 3. Interpolation linéaire entre les valeurs entourant l'index.
///
/// # Erreurs
/// [`MathError::EmptyData`] si les données sont vides.
fn quantile(data: &[f64], p: f64) -> MathResult<f64> {
    if data.is_empty() {
        return Err(MathError::EmptyData(
            "données vides pour calcul de quantile".into(),
        ));
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let index = p * (n - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    if lower == upper {
        Ok(sorted[lower])
    } else {
        let fraction = index - lower as f64;
        Ok(sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction)
    }
}

/// Échantillonne une Student-t standard de `df` degrés de liberté.
///
/// Utilise la relation : T = Z / sqrt(V/df) où Z ~ N(0,1), V ~ Chi2(df).
///
/// NOTE : Pour `df` non entier, cette implémentation utilise une approximation
/// par interpolation linéaire entre les distributions Chi2(k) et Chi2(k+1),
/// où `k = floor(df)`. Une implémentation complète nécessiterait une loi gamma.
fn sample_student_t(df: f64, rng: &mut DeterministicRng) -> f64 {
    // Génère un nombre normal standard via la méthode de Box-Muller simplifiée.
    let z = sample_normal(rng);
    // Génère une variable chi-deux via la somme de carrés de normales.
    let mut chi2 = 0.0;
    let k = df.floor() as usize;
    for _ in 0..k {
        let n = sample_normal(rng);
        chi2 += n * n;
    }
    // Approximation pour df non entier : interpolation linéaire entre chi2(k) et chi2(k+1)
    let v = if df == k as f64 {
        chi2
    } else {
        let mut chi2_next = chi2;
        let n = sample_normal(rng);
        chi2_next += n * n;
        let frac = df - k as f64;
        chi2 * (1.0 - frac) + chi2_next * frac
    };
    // Protection contre la division par zéro
    let denominator = (v / df).sqrt();
    if denominator.abs() < f64::EPSILON {
        0.0
    } else {
        z / denominator
    }
}

/// Échantillonne une normale standard N(0,1) via Box-Muller.
fn sample_normal(rng: &mut DeterministicRng) -> f64 {
    // Protège contre ln(0) qui donnerait -∞.
    let u1 = rng.next_f64().max(1e-300);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_amplitude() {
        let strategy = AmplitudeStrategy::Fixed(5.0);
        let mut rng = DeterministicRng::from_seed([42u8; 32]);
        let amp = compute_amplitude(&strategy, &[], &mut rng).unwrap();
        assert_eq!(amp, 5.0);
    }

    #[test]
    fn test_relative_to_std() {
        let strategy = AmplitudeStrategy::RelativeToStd { k: 2.0 };
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut rng = DeterministicRng::from_seed([42u8; 32]);
        let amp = compute_amplitude(&strategy, &data, &mut rng).unwrap();
        let std = statistics::std_sample(&data).unwrap();
        assert!((amp - 2.0 * std).abs() < 1e-10);
    }

    #[test]
    fn test_empty_data_error() {
        let strategy = AmplitudeStrategy::RelativeToStd { k: 1.0 };
        let mut rng = DeterministicRng::from_seed([42u8; 32]);
        assert!(compute_amplitude(&strategy, &[], &mut rng).is_err());
    }

    #[test]
    fn test_quantile_based() {
        let strategy = AmplitudeStrategy::QuantileBased { p: 0.5 };
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut rng = DeterministicRng::from_seed([42u8; 32]);
        let amp = compute_amplitude(&strategy, &data, &mut rng).unwrap();
        // Le quantile médian de [1,2,3,4,5] est 3.0
        assert!((amp - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_heavy_tail() {
        let strategy = AmplitudeStrategy::HeavyTail { df: 3.0 };
        let mut rng = DeterministicRng::from_seed([42u8; 32]);
        let amp = compute_amplitude(&strategy, &[], &mut rng).unwrap();
        // L'amplitude doit être finie et positive.
        assert!(amp.is_finite());
        assert!(amp >= 0.0);
    }

    #[test]
    fn test_heavy_tail_non_integer_df() {
        // Test avec df non entier pour vérifier l'approximation par interpolation
        let strategy = AmplitudeStrategy::HeavyTail { df: 2.5 };
        let mut rng = DeterministicRng::from_seed([42u8; 32]);
        let amp = compute_amplitude(&strategy, &[], &mut rng).unwrap();
        // L'amplitude doit être finie et positive même avec df non entier
        assert!(amp.is_finite(), "amplitude avec df=2.5 doit être finie");
        assert!(amp >= 0.0, "amplitude avec df=2.5 doit être ≥ 0");
    }

    #[test]
    fn test_heavy_tail_small_df() {
        // Test avec df très petit mais positif (proche de 0)
        let strategy = AmplitudeStrategy::HeavyTail { df: 0.1 };
        let mut rng = DeterministicRng::from_seed([42u8; 32]);
        let amp = compute_amplitude(&strategy, &[], &mut rng).unwrap();
        // Doit gérer df < 1 sans panic ni division par zéro
        assert!(amp.is_finite(), "amplitude avec df=0.1 doit être finie");
        assert!(amp >= 0.0, "amplitude avec df=0.1 doit être ≥ 0");
    }

    #[test]
    fn test_heavy_tail_very_small_df() {
        // Test avec df extrêmement petit pour vérifier la protection contre division par zéro
        let strategy = AmplitudeStrategy::HeavyTail { df: f64::EPSILON };
        let mut rng = DeterministicRng::from_seed([42u8; 32]);
        let amp = compute_amplitude(&strategy, &[], &mut rng).unwrap();
        // Doit retourner une valeur finie (0.0 si denominator < EPSILON)
        assert!(amp.is_finite(), "amplitude avec df=EPSILON doit être finie");
        assert!(amp >= 0.0, "amplitude avec df=EPSILON doit être ≥ 0");
    }
}
