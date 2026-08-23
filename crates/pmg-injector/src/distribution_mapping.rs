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

//! Mapping des familles de distribution蓝图 vers les implémentations `pmg-math`.
//!
//! Ce module fournit la fonction [`distribution_from_family`] qui convertit une
//! [`DistributionFamily`] (blueprint) en une distribution concrète utilisable
//! par le pipeline d'injection. Chaque famille est mappée vers une implémentation
//! de `pmg-math` avec des propriétés statistiques documentées.
//!
//! # Familles supportées
//!
//! | Famille | Distribution pmg-math | Propriétés |
//! |---------|----------------------|------------|
//! | `Normal` | `N(mean, stddev)` | Exacte |
//! | `StudentT` | `t(5)` re-centrée/re-échelonnée | Exacte |
//! | `Laplace` | `L(mean, stddev/√2)` | Exacte |
//! | `LogNormal` | Paramètres log depuis mean/stddev | Exacte si mean > 0 |
//! | `Uniform` | `U(mean − √3·σ, mean + √3·σ)` | Exacte |
//! | `Mixture` | Mélange bimodal 50/50 de deux normales | Moyenne et variance exactes |
//! | `Weibull` | `W(1,2)` re-échelonnée | Approximation (queues lourdes) |
//! | `Pareto` | `P(1,3)` re-échelonnée | Approximation (queues lourdes) |
//!
//! # Exemple
//!
//! ```
//! use pmg_blueprint::tensor_spec::DistributionFamily;
//! use pmg_injector::distribution_mapping::distribution_from_family;
//!
//! // Distribution normale standard
//! let mut dist = distribution_from_family(DistributionFamily::Normal, 0.0, 1.0).unwrap();
//! // ... utiliser dist.sample(rng)
//!
//! // Distribution Student-t avec df=5
//! let mut dist = distribution_from_family(DistributionFamily::StudentT, 0.0, 1.0).unwrap();
//!
//! // Mélange bimodal contrôlé
//! let mut dist = distribution_from_family(DistributionFamily::Mixture, 0.0, 1.0).unwrap();
//! ```

use pmg_blueprint::tensor_spec::DistributionFamily;
use pmg_core::distribution_config::DistributionConfig;
use pmg_math::distribution::{from_config, Distribution};
use pmg_math::rng::DeterministicRng;

use crate::error::{InjectorError, InjectorResult};

/// Degrés de liberté par défaut de la Student-t utilisée pour les outliers
/// statistiques lorsque la politique n'en spécifie pas explicitement.
pub const DEFAULT_STUDENT_T_DF: f64 = 5.0;

/// Distribution d'un `DistributionFamily` blueprint, mappée vers `pmg-math`.
///
/// # Paramètres
/// - `family` : famille de distribution du blueprint ;
/// - `mean` : moyenne cible (doit être > 0 pour LogNormal) ;
/// - `stddev` : écart-type cible (doit être > 0 et fini).
///
/// # Mappings documentés (moyenne/écart-type reproduits exactement quand la
/// famille le permet)
/// - `Normal` : `N(mean, stddev)` — exact ;
/// - `StudentT` : `t(5)` re-centrée/re-échelonnée — exact ;
/// - `Laplace` : `L(mean, stddev/√2)` — exact ;
/// - `LogNormal` : paramètres log calculés depuis mean/stddev — exact si
///   `mean > 0`, sinon [`InjectorError::InvalidPolicy`] ;
/// - `Uniform` : `U(mean − √3·σ, mean + √3·σ)` — exact ;
/// - `Mixture` : mélange bimodal 50/50 de deux normales (moyenne et variance
///   exactes) ;
/// - `Weibull` / `Pareto` : paramètres de forme fixes documentés
///   (`W(1,2)` / `P(1,3)`), re-échelonnés — **approximation** (moyenne non
///   reproduite), limité aux queues lourdes.
///
/// # Erreurs
/// [`InjectorError::InvalidPolicy`] si la famille n'est pas mappable ou si
/// `stddev ≤ 0`.
///
/// # Exemple
///
/// ```
/// use pmg_blueprint::tensor_spec::DistributionFamily;
/// use pmg_injector::distribution_mapping::distribution_from_family;
///
/// // Distribution normale standard
/// let mut dist = distribution_from_family(DistributionFamily::Normal, 0.0, 1.0).unwrap();
///
/// // Distribution Student-t avec df=5
/// let mut dist = distribution_from_family(DistributionFamily::StudentT, 0.0, 1.0).unwrap();
///
/// // Mélange bimodal contrôlé
/// let mut dist = distribution_from_family(DistributionFamily::Mixture, 0.0, 1.0).unwrap();
///
/// // Erreur si stddev invalide
/// let result = distribution_from_family(DistributionFamily::Normal, 0.0, -1.0);
/// assert!(result.is_err());
/// ```
pub fn distribution_from_family(
    family: DistributionFamily,
    mean: f64,
    stddev: f64,
) -> InjectorResult<Box<dyn Distribution>> {
    if !stddev.is_finite() || stddev <= 0.0 {
        return Err(InjectorError::InvalidPolicy(format!(
            "stddev de distribution doit être fini et > 0, reçu {stddev}"
        )));
    }
    let dist: Box<dyn Distribution> = match family {
        DistributionFamily::Normal => from_config(&DistributionConfig::normal(mean, stddev))?,
        DistributionFamily::StudentT => {
            let df = DEFAULT_STUDENT_T_DF;
            let scale = stddev / (df / (df - 2.0)).sqrt();
            Box::new(Scaled::new(
                from_config(&DistributionConfig::student_t(df))?,
                mean,
                scale,
            ))
        },
        DistributionFamily::Laplace => {
            from_config(&DistributionConfig::laplace(mean, stddev / 2.0f64.sqrt()))?
        },
        DistributionFamily::LogNormal => {
            if mean <= 0.0 {
                return Err(InjectorError::InvalidPolicy(format!(
                    "log-normale exige une moyenne > 0, reçu {mean}"
                )));
            }
            let sigma2 = (1.0 + (stddev / mean).powi(2)).ln();
            let mu = mean.ln() - sigma2 / 2.0;
            from_config(&DistributionConfig::log_normal(mu, sigma2.sqrt()))?
        },
        DistributionFamily::Uniform => Box::new(Uniform {
            lo: mean - 3.0f64.sqrt() * stddev,
            hi: mean + 3.0f64.sqrt() * stddev,
        }),
        DistributionFamily::Mixture => {
            // Deux normales décalées de ±0.6σ avec écart 0.8σ :
            // variance totale = 0.36σ² + 0.64σ² = σ² (bimodal contrôlé).
            let cfg = DistributionConfig::mixture(vec![
                (
                    0.5,
                    DistributionConfig::normal(mean - 0.6 * stddev, 0.8 * stddev),
                ),
                (
                    0.5,
                    DistributionConfig::normal(mean + 0.6 * stddev, 0.8 * stddev),
                ),
            ]);
            from_config(&cfg)?
        },
        DistributionFamily::Weibull => {
            // Approximation : forme fixe k = 2, échelle ∝ σ. Les queues lourdes
            // sont le but ; la moyenne n'est pas reproduite (limite documentée).
            Box::new(Scaled::new(
                from_config(&DistributionConfig::weibull(1.0, 2.0))?,
                0.0,
                stddev,
            ))
        },
        DistributionFamily::Pareto => {
            // Approximation : forme fixe α = 3, échelle ∝ σ (limite documentée).
            Box::new(Scaled::new(
                from_config(&DistributionConfig::pareto(1.0, 3.0))?,
                0.0,
                stddev,
            ))
        },
        _ => {
            return Err(InjectorError::InvalidPolicy(format!(
                "famille de distribution non supportée par l'injection : {family:?}"
            )))
        },
    };
    Ok(dist)
}

/// Distribution `Y = shift + scale·X` autour d'une distribution `pmg-math`.
struct Scaled {
    inner: Box<dyn Distribution>,
    shift: f64,
    scale: f64,
}

impl Scaled {
    fn new(inner: Box<dyn Distribution>, shift: f64, scale: f64) -> Self {
        Self {
            inner,
            shift,
            scale,
        }
    }
}

impl Distribution for Scaled {
    fn sample(&mut self, rng: &mut DeterministicRng) -> f64 {
        self.shift + self.scale * self.inner.sample(rng)
    }

    fn pdf(&self, x: f64) -> f64 {
        self.inner.pdf((x - self.shift) / self.scale) / self.scale
    }

    fn cdf(&self, x: f64) -> Option<f64> {
        self.inner.cdf((x - self.shift) / self.scale)
    }

    fn mean(&self) -> Option<f64> {
        self.inner.mean().map(|m| self.shift + self.scale * m)
    }

    fn variance(&self) -> Option<f64> {
        self.inner.variance().map(|v| self.scale * self.scale * v)
    }

    fn name(&self) -> &'static str {
        "scaled"
    }
}

/// Distribution uniforme locale sur `[lo, hi]` (famille `Uniform` du
/// blueprint, absente de `pmg-math`).
struct Uniform {
    lo: f64,
    hi: f64,
}

impl Distribution for Uniform {
    fn sample(&mut self, rng: &mut DeterministicRng) -> f64 {
        self.lo + (self.hi - self.lo) * rng.next_f64()
    }

    fn pdf(&self, x: f64) -> f64 {
        if x >= self.lo && x <= self.hi {
            1.0 / (self.hi - self.lo)
        } else {
            0.0
        }
    }

    fn cdf(&self, x: f64) -> Option<f64> {
        if x < self.lo {
            Some(0.0)
        } else if x > self.hi {
            Some(1.0)
        } else {
            Some((x - self.lo) / (self.hi - self.lo))
        }
    }

    fn mean(&self) -> Option<f64> {
        Some((self.lo + self.hi) / 2.0)
    }

    fn variance(&self) -> Option<f64> {
        let w = self.hi - self.lo;
        Some(w * w / 12.0)
    }

    fn name(&self) -> &'static str {
        "uniform"
    }
}
