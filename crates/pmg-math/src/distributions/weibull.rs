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

//! Distribution de Weibull `W(λ, k)` (échelle λ > 0, forme k > 0).
//!
//! Échantillonnage par **inverse CDF** : `X = λ·(−ln U)^(1/k)`,
//! `U ~ U(0,1)` (spécification doc 4 §2.2). Support `[0, +∞)`.

use crate::distribution::Distribution;
use crate::error::{MathError, MathResult};
use crate::rng::DeterministicRng;

/// Loi de Weibull `W(scale, shape)` avec `scale > 0`, `shape > 0`.
///
/// # Moments
/// - Espérance : `λ·Γ(1 + 1/k)` ;
/// - Variance : `λ²·(Γ(1 + 2/k) − Γ(1 + 1/k)²)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Weibull {
    scale: f64,
    shape: f64,
}

impl Weibull {
    /// Construit `W(scale, shape)`.
    ///
    /// # Erreurs
    /// [`MathError::InvalidParameter`] si `scale ≤ 0` ou `shape ≤ 0` (non finis).
    ///
    /// # Complexité
    /// O(1).
    pub fn new(scale: f64, shape: f64) -> MathResult<Self> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(MathError::InvalidParameter(format!(
                "scale de Weibull doit être fini et > 0, reçu {scale}"
            )));
        }
        if !shape.is_finite() || shape <= 0.0 {
            return Err(MathError::InvalidParameter(format!(
                "shape de Weibull doit être fini et > 0, reçu {shape}"
            )));
        }
        Ok(Self { scale, shape })
    }
}

impl Distribution for Weibull {
    /// Inverse CDF : `λ·(−ln U)^(1/k)`.
    ///
    /// # Complexité
    /// O(1).
    fn sample(&mut self, rng: &mut DeterministicRng) -> f64 {
        let u = rng.next_f64().max(f64::MIN_POSITIVE);
        self.scale * (-u.ln()).powf(1.0 / self.shape)
    }

    /// Densité `f(x) = (k/λ)·(x/λ)^(k−1)·exp(−(x/λ)^k)` pour `x ≥ 0`,
    /// nulle pour `x < 0`.
    ///
    /// # Complexité
    /// O(1).
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        if x == 0.0 {
            // Limite : k/λ si k = 1, sinon 0 (k > 1) ou +∞ (k < 1).
            return if (self.shape - 1.0).abs() < 1e-12 {
                self.shape / self.scale
            } else {
                0.0
            };
        }
        let r = x / self.scale;
        let k = self.shape;
        (k / self.scale) * r.powf(k - 1.0) * (-r.powf(k)).exp()
    }

    /// Fonction de répartition `F(x) = 1 − exp(−(x/λ)^k)` pour `x ≥ 0`,
    /// nulle pour `x < 0`.
    ///
    /// # Complexité
    /// O(1).
    fn cdf(&self, x: f64) -> Option<f64> {
        if x < 0.0 {
            return Some(0.0);
        }
        Some(1.0 - (-(x / self.scale).powf(self.shape)).exp())
    }

    fn mean(&self) -> Option<f64> {
        Some(self.scale * crate::special::gamma(1.0 + 1.0 / self.shape))
    }

    fn variance(&self) -> Option<f64> {
        let g1 = crate::special::gamma(1.0 + 1.0 / self.shape);
        let g2 = crate::special::gamma(1.0 + 2.0 / self.shape);
        Some(self.scale * self.scale * (g2 - g1 * g1))
    }

    fn name(&self) -> &'static str {
        "weibull"
    }
}

#[cfg(test)]
mod tests {
    use super::Weibull;
    use crate::distribution::Distribution;
    use crate::rng::DeterministicRng;

    const N: usize = 100_000;

    fn rng() -> DeterministicRng {
        DeterministicRng::from_seed([13u8; 32])
    }

    #[test]
    fn invalid_parameters_rejected() {
        assert!(Weibull::new(0.0, 1.0).is_err());
        assert!(Weibull::new(1.0, 0.0).is_err());
        assert!(Weibull::new(-1.0, 1.0).is_err());
        assert!(Weibull::new(1.0, f64::NAN).is_err());
        assert!(Weibull::new(1.0, 2.0).is_ok());
    }

    #[test]
    fn pdf_known_values() {
        // W(1, 2) (Rayleigh) : f(1) = 2·e⁻¹ ≈ 0.7357589.
        let w = Weibull::new(1.0, 2.0).unwrap();
        assert!((w.pdf(1.0) - 2.0 * (-1.0f64).exp()).abs() < 1e-9);
        // W(1, 1) = exponentielle(1) : f(x) = e⁻ˣ, f(0) = 1.
        let exp_w = Weibull::new(1.0, 1.0).unwrap();
        assert!((exp_w.pdf(0.0) - 1.0).abs() < 1e-12);
        assert!((exp_w.pdf(2.0) - (-2.0f64).exp()).abs() < 1e-9);
        assert_eq!(w.pdf(-1.0), 0.0);
    }

    #[test]
    fn cdf_known_values() {
        let w = Weibull::new(1.0, 2.0).unwrap();
        // F(1) = 1 − e⁻¹ ≈ 0.6321 ; F(0) = 0.
        assert!((w.cdf(1.0).unwrap() - (1.0 - (-1.0f64).exp())).abs() < 1e-9);
        assert_eq!(w.cdf(0.0).unwrap(), 0.0);
        assert_eq!(w.cdf(-1.0).unwrap(), 0.0);
        // F(λ) = 1 − 1/e indépendant de k.
        assert!((w.cdf(1.0).unwrap() - (1.0 - (-1.0f64).exp())).abs() < 1e-9);
    }

    #[test]
    fn empirical_moments_within_tolerance() {
        // W(2, 3) : μ = 2·Γ(4/3) ≈ 1.7627 ; σ = 2·√(Γ(5/3) − Γ(4/3)²) ≈ 0.630.
        let w = Weibull::new(2.0, 3.0).unwrap();
        let mu = w.mean().unwrap();
        let sigma = w.variance().unwrap().sqrt();
        let mut dist = w.clone();
        let mut rng = rng();
        let samples: Vec<f64> = (0..N).map(|_| dist.sample(&mut rng)).collect();
        crate::distributions::test_util::assert_mean_tolerance(&samples, mu, sigma).unwrap();
        crate::distributions::test_util::assert_std_tolerance(&samples, sigma).unwrap();
    }

    #[test]
    fn support_is_non_negative() {
        let mut dist = Weibull::new(1.0, 0.5).unwrap();
        let mut rng = rng();
        for _ in 0..N {
            assert!(dist.sample(&mut rng) >= 0.0);
        }
    }

    #[test]
    fn reproducibility_strict() {
        let mut a = Weibull::new(1.0, 2.0).unwrap();
        let mut b = Weibull::new(1.0, 2.0).unwrap();
        let mut rng_a = rng();
        let mut rng_b = rng();
        for _ in 0..10_000 {
            assert_eq!(a.sample(&mut rng_a), b.sample(&mut rng_b));
        }
    }
}
