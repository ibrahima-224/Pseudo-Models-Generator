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

//! Contrat commun des distributions : trait [`Distribution`] et fabrique
//! [`from_config`].
//!
//! Conformité : `docs/architecture/04-moteurs-math-injection-generation.md`
//! §2. Chaque distribution implémente `sample`, `pdf`, `cdf` (quand la cdf
//! est définissable numériquement), `mean`, `variance` et `name`.

use pmg_core::distribution_config::{DistributionConfig, DistributionKind};

use crate::distributions::{
    Laplace, LogNormal, Mixture, Normal, Pareto, StudentT, Uniform, Weibull,
};
use crate::error::{MathError, MathResult};
use crate::rng::DeterministicRng;

/// Trait commun des distributions de probabilité univariées.
///
/// # Contrat
/// - `sample` : tire une valeur selon la loi (déterministe étant donné le RNG) ;
/// - `pdf` : densité de probabilité en `x` ;
/// - `cdf` : fonction de répartition en `x`, `None` si non définissable
///   numériquement ;
/// - `mean` / `variance` : moments, `None` si non définis (ex. Pareto α ≤ 1) ;
/// - `name` : identifiant stable (ex. `"normal"`).
pub trait Distribution {
    /// Tire un échantillon selon la loi.
    fn sample(&mut self, rng: &mut DeterministicRng) -> f64;

    /// Densité de probabilité `f(x)`.
    fn pdf(&self, x: f64) -> f64;

    /// Fonction de répartition `P(X ≤ x)`.
    fn cdf(&self, x: f64) -> Option<f64>;

    /// Espérance, ou `None` si non définie.
    fn mean(&self) -> Option<f64>;

    /// Variance, ou `None` si non définie.
    fn variance(&self) -> Option<f64>;

    /// Nom stable de la famille.
    fn name(&self) -> &'static str;
}

/// Fabrique une distribution concrète depuis sa configuration.
///
/// # Entrées
/// - `config` : description sérialisable.
///
/// # Sorties
/// `Box<dyn Distribution>` prête à l'emploi.
///
/// # Erreurs
/// [`MathError::InvalidParameter`] si un paramètre viole les bornes
/// documentées de la famille (σ ≤ 0, df ≤ 0, poids du mélange invalides…).
///
/// # Complexité
/// O(C) pour un mélange de C composantes, O(1) sinon.
pub fn from_config(config: &DistributionConfig) -> MathResult<Box<dyn Distribution>> {
    match config.kind {
        DistributionKind::Normal => {
            let std = config
                .p2
                .ok_or_else(|| invalid("normale : paramètre σ manquant"))?;
            Ok(Box::new(Normal::new(config.p1, std)?))
        },
        DistributionKind::StudentT => Ok(Box::new(StudentT::new(config.p1)?)),
        DistributionKind::Laplace => {
            let scale = config
                .p2
                .ok_or_else(|| invalid("laplace : paramètre b manquant"))?;
            Ok(Box::new(Laplace::new(config.p1, scale)?))
        },
        DistributionKind::LogNormal => {
            let sigma = config
                .p2
                .ok_or_else(|| invalid("log-normale : paramètre σ_log manquant"))?;
            Ok(Box::new(LogNormal::new(config.p1, sigma)?))
        },
        DistributionKind::Weibull => {
            let shape = config
                .p2
                .ok_or_else(|| invalid("weibull : paramètre k manquant"))?;
            Ok(Box::new(Weibull::new(config.p1, shape)?))
        },
        DistributionKind::Pareto => {
            let shape = config
                .p2
                .ok_or_else(|| invalid("pareto : paramètre α manquant"))?;
            Ok(Box::new(Pareto::new(config.p1, shape)?))
        },
        DistributionKind::Mixture => {
            let components = config
                .mixture_components
                .iter()
                .map(|(w, c)| Ok((*w, from_config(c)?)))
                .collect::<MathResult<Vec<_>>>()?;

            // Validation des poids : doivent sommer à 1.0 (tolérance 1e-6)
            let total_weight: f64 = components.iter().map(|(w, _)| w).sum();
            if (total_weight - 1.0).abs() > 1e-6 {
                return Err(invalid(&format!(
                    "les poids du mélange doivent sommer à 1.0, reçu {total_weight}"
                )));
            }

            Ok(Box::new(Mixture::new(components)?))
        },
        DistributionKind::Uniform => {
            let max = config
                .p2
                .ok_or_else(|| invalid("uniforme : paramètre max manquant"))?;
            Ok(Box::new(Uniform::new(config.p1, max)?))
        },
    }
}

fn invalid(msg: &str) -> MathError {
    MathError::InvalidParameter(msg.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::DeterministicRng;

    #[test]
    fn serde_round_trip_snake_case() {
        let cfg = DistributionConfig::normal(0.0, 1.0);
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains("\"kind\":\"normal\""), "json={json}");
        let back: DistributionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn from_config_builds_expected_kinds() {
        let cases: &[(DistributionConfig, &str)] = &[
            (DistributionConfig::normal(1.0, 2.0), "normal"),
            (DistributionConfig::student_t(5.0), "student_t"),
            (DistributionConfig::laplace(0.0, 1.0), "laplace"),
            (DistributionConfig::log_normal(0.0, 1.0), "log_normal"),
            (DistributionConfig::weibull(1.0, 2.0), "weibull"),
            (DistributionConfig::pareto(1.0, 2.0), "pareto"),
            (DistributionConfig::uniform(0.0, 1.0), "uniform"),
        ];
        for (cfg, expected_name) in cases {
            let d = from_config(cfg).unwrap();
            assert_eq!(d.name(), *expected_name);
        }
    }

    #[test]
    fn from_config_mixture_recursive() {
        let cfg = DistributionConfig::mixture(vec![
            (0.7, DistributionConfig::normal(0.0, 1.0)),
            (0.3, DistributionConfig::laplace(0.0, 1.0)),
        ]);
        let mut d = from_config(&cfg).unwrap();
        assert_eq!(d.name(), "mixture");
        let mut rng = DeterministicRng::from_seed([7u8; 32]);
        for _ in 0..100 {
            let _ = d.sample(&mut rng);
        }
    }

    #[test]
    fn from_config_rejects_invalid_parameters() {
        assert!(from_config(&DistributionConfig::normal(0.0, -1.0)).is_err());
        assert!(from_config(&DistributionConfig::student_t(0.0)).is_err());
        assert!(from_config(&DistributionConfig::pareto(1.0, 0.0)).is_err());
        assert!(from_config(&DistributionConfig::weibull(1.0, 0.0)).is_err());
        // Mélange à poids non normalisés.
        let bad_mix = DistributionConfig::mixture(vec![
            (0.5, DistributionConfig::normal(0.0, 1.0)),
            (0.4, DistributionConfig::normal(0.0, 1.0)),
        ]);
        assert!(from_config(&bad_mix).is_err());

        // Mélange avec poids qui ne somment pas à 1.0
        let bad_mix2 = DistributionConfig::mixture(vec![
            (0.3, DistributionConfig::normal(0.0, 1.0)),
            (0.3, DistributionConfig::normal(0.0, 1.0)),
        ]);
        assert!(from_config(&bad_mix2).is_err());
        // Uniforme avec min >= max
        assert!(from_config(&DistributionConfig::uniform(1.0, 0.0)).is_err());
        assert!(from_config(&DistributionConfig::uniform(1.0, 1.0)).is_err());
    }

    #[test]
    fn mixture_weights_sum_to_one_validation() {
        // Mélange avec poids valides
        let good_mix = DistributionConfig::mixture(vec![
            (0.7, DistributionConfig::normal(0.0, 1.0)),
            (0.3, DistributionConfig::laplace(0.0, 0.5)),
        ]);
        assert!(from_config(&good_mix).is_ok());

        // Mélange avec poids ne sommant pas à 1.0
        let bad_mix = DistributionConfig::mixture(vec![
            (0.5, DistributionConfig::normal(0.0, 1.0)),
            (0.4, DistributionConfig::normal(0.0, 1.0)),
        ]);
        assert!(from_config(&bad_mix).is_err());

        // Mélange avec poids trop grands
        let bad_mix2 = DistributionConfig::mixture(vec![
            (0.8, DistributionConfig::normal(0.0, 1.0)),
            (0.3, DistributionConfig::normal(0.0, 1.0)),
        ]);
        assert!(from_config(&bad_mix2).is_err());
    }

    #[test]
    fn from_config_uniform_generates_correct_range() {
        let cfg = DistributionConfig::uniform(0.0, 1.0);
        let mut d = from_config(&cfg).unwrap();
        assert_eq!(d.name(), "uniform");
        let mut rng = DeterministicRng::from_seed([42u8; 32]);
        for _ in 0..1000 {
            let x = d.sample(&mut rng);
            assert!((0.0..=1.0).contains(&x), "échantillon hors [0,1]: {x}");
        }
    }
}
