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

//! Distribution de Pareto (Type I) `P(x_m, α)` — support `[x_m, +∞)`.
//!
//! Échantillonnage par **inverse CDF** : `X = x_m·U^(−1/α)`, `U ~ U(0,1)`
//! (spécification doc 4 §2.2).
//!
//! # Avertissement (spécification doc 4 §2.2)
//! Pareto est réservée aux profils qui la justifient (jamais par défaut
//! partout). Les moments non définis retournent `None` (jamais de valeur
//! arbitraire).

use crate::distribution::Distribution;
use crate::error::{MathError, MathResult};
use crate::rng::DeterministicRng;

/// Loi de Pareto `P(scale, shape)` avec `scale > 0`, `shape > 0`.
///
/// # Moments
/// - Espérance : `scale·α/(α−1)` (si `α > 1`) ;
/// - Variance : `scale²·α/((α−1)²·(α−2))` (si `α > 2`).
#[derive(Debug, Clone, PartialEq)]
pub struct Pareto {
    scale: f64,
    shape: f64,
}

impl Pareto {
    /// Construit `P(scale, shape)`.
    ///
    /// # Erreurs
    /// [`MathError::InvalidParameter`] si `scale ≤ 0` ou `shape ≤ 0` (non finis).
    ///
    /// # Complexité
    /// O(1).
    pub fn new(scale: f64, shape: f64) -> MathResult<Self> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(MathError::InvalidParameter(format!(
                "scale de Pareto doit être fini et > 0, reçu {scale}"
            )));
        }
        if !shape.is_finite() || shape <= 0.0 {
            return Err(MathError::InvalidParameter(format!(
                "shape de Pareto doit être fini et > 0, reçu {shape}"
            )));
        }
        Ok(Self { scale, shape })
    }
}

impl Distribution for Pareto {
    /// Inverse CDF : `x_m·U^(−1/α)`.
    ///
    /// # Complexité
    /// O(1).
    fn sample(&mut self, rng: &mut DeterministicRng) -> f64 {
        let u = rng.next_f64().max(f64::MIN_POSITIVE);
        self.scale * u.powf(-1.0 / self.shape)
    }

    /// Densité `f(x) = α·x_m^α / x^(α+1)` pour `x ≥ x_m`, nulle sinon.
    ///
    /// # Complexité
    /// O(1).
    fn pdf(&self, x: f64) -> f64 {
        if x < self.scale {
            return 0.0;
        }
        self.shape * self.scale.powf(self.shape) / x.powf(self.shape + 1.0)
    }

    /// Fonction de répartition `F(x) = 1 − (x_m/x)^α` pour `x ≥ x_m`,
    /// nulle sinon.
    ///
    /// # Complexité
    /// O(1).
    fn cdf(&self, x: f64) -> Option<f64> {
        if x < self.scale {
            return Some(0.0);
        }
        Some(1.0 - (self.scale / x).powf(self.shape))
    }

    fn mean(&self) -> Option<f64> {
        (self.shape > 1.0).then_some(self.scale * self.shape / (self.shape - 1.0))
    }

    fn variance(&self) -> Option<f64> {
        if self.shape > 2.0 {
            let denom = (self.shape - 1.0) * (self.shape - 1.0) * (self.shape - 2.0);
            Some(self.scale * self.scale * self.shape / denom)
        } else {
            None
        }
    }

    fn name(&self) -> &'static str {
        "pareto"
    }
}

#[cfg(test)]
mod tests {
    use super::Pareto;
    use crate::distribution::Distribution;
    use crate::rng::DeterministicRng;

    const N: usize = 100_000;

    fn rng() -> DeterministicRng {
        DeterministicRng::from_seed([17u8; 32])
    }

    #[test]
    fn invalid_parameters_rejected() {
        assert!(Pareto::new(0.0, 1.0).is_err());
        assert!(Pareto::new(1.0, 0.0).is_err());
        assert!(Pareto::new(-1.0, 1.0).is_err());
        assert!(Pareto::new(1.0, f64::NAN).is_err());
        assert!(Pareto::new(1.0, 2.0).is_ok());
    }

    #[test]
    fn pdf_known_values() {
        // P(1, 3) : f(1) = 3 ; f(2) = 3/16 = 0.1875.
        let p = Pareto::new(1.0, 3.0).unwrap();
        assert!((p.pdf(1.0) - 3.0).abs() < 1e-9);
        assert!((p.pdf(2.0) - 3.0 / 16.0).abs() < 1e-9);
        // Support : pdf nulle sous le scale.
        assert_eq!(p.pdf(0.5), 0.0);
        assert_eq!(p.pdf(1.0 - 1e-12), 0.0);
    }

    #[test]
    fn cdf_known_values() {
        let p = Pareto::new(1.0, 3.0).unwrap();
        // F(1) = 0 ; F(2) = 1 − (1/2)³ = 0.875.
        assert_eq!(p.cdf(0.5).unwrap(), 0.0);
        assert_eq!(p.cdf(1.0).unwrap(), 0.0);
        assert!((p.cdf(2.0).unwrap() - 0.875).abs() < 1e-9);
    }

    #[test]
    fn empirical_moments_within_tolerance() {
        // P(1, 4) : μ = 4/3 ≈ 1.3333 ; var = 4/(9·2) = 2/9 ≈ 0.2222.
        let p = Pareto::new(1.0, 4.0).unwrap();
        let mu = p.mean().unwrap();
        let sigma = p.variance().unwrap().sqrt();
        let mut dist = p.clone();
        let mut rng = rng();
        let samples: Vec<f64> = (0..N).map(|_| dist.sample(&mut rng)).collect();
        crate::distributions::test_util::assert_mean_tolerance(&samples, mu, sigma).unwrap();
        crate::distributions::test_util::assert_std_tolerance(&samples, sigma).unwrap();
    }

    #[test]
    fn moments_none_for_non_integrable_tails() {
        // α ≤ 1 : espérance infinie ; α ≤ 2 : variance infinie.
        let p1 = Pareto::new(1.0, 1.0).unwrap();
        assert_eq!(p1.mean(), None);
        assert_eq!(p1.variance(), None);
        let p2 = Pareto::new(1.0, 1.5).unwrap();
        assert_eq!(p2.mean(), Some(3.0));
        assert_eq!(p2.variance(), None);
    }

    #[test]
    fn support_is_at_least_scale() {
        let mut dist = Pareto::new(2.0, 3.0).unwrap();
        let mut rng = rng();
        for _ in 0..N {
            let x = dist.sample(&mut rng);
            assert!(x >= 2.0, "échantillon sous le support : {x}");
        }
    }

    #[test]
    fn reproducibility_strict() {
        let mut a = Pareto::new(1.0, 3.0).unwrap();
        let mut b = Pareto::new(1.0, 3.0).unwrap();
        let mut rng_a = rng();
        let mut rng_b = rng();
        for _ in 0..10_000 {
            assert_eq!(a.sample(&mut rng_a), b.sample(&mut rng_b));
        }
    }
}
