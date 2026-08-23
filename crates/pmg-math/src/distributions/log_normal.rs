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

//! Distribution log-normale `LogN(μ_log, σ_log)`.
//!
//! `X = exp(Y)` avec `Y ~ N(μ_log, σ_log²)`. Échantillonnage via la normale
//! interne (Box-Muller). Le support est `(0, +∞)`.

use crate::distribution::Distribution;
use crate::error::{MathError, MathResult};
use crate::rng::DeterministicRng;

use super::normal::Normal;

/// Loi log-normale de paramètres `(mu_log, sigma_log)` avec `sigma_log > 0`.
///
/// # Moments
/// - Espérance : `exp(μ + σ²/2)` ;
/// - Variance : `(exp(σ²) − 1)·exp(2μ + σ²)`.
#[derive(Debug, Clone, PartialEq)]
pub struct LogNormal {
    mu_log: f64,
    sigma_log: f64,
}

impl LogNormal {
    /// Construit `LogN(mu_log, sigma_log)`.
    ///
    /// # Erreurs
    /// [`MathError::InvalidParameter`] si `sigma_log ≤ 0` ou non fini.
    ///
    /// # Complexité
    /// O(1).
    pub fn new(mu_log: f64, sigma_log: f64) -> MathResult<Self> {
        if !sigma_log.is_finite() || sigma_log <= 0.0 {
            return Err(MathError::InvalidParameter(format!(
                "sigma_log de la log-normale doit être fini et > 0, reçu {sigma_log}"
            )));
        }
        Ok(Self { mu_log, sigma_log })
    }

    /// Loi normale sous-jacente `N(μ_log, σ_log²)`.
    pub fn underlying_normal(&self) -> Normal {
        Normal::new(self.mu_log, self.sigma_log).expect("paramètres valides")
    }
}

impl Distribution for LogNormal {
    /// `exp(Y)` avec `Y ~ N(μ_log, σ_log²)`.
    ///
    /// # Complexité
    /// O(1) — un couple Box-Muller.
    fn sample(&mut self, rng: &mut DeterministicRng) -> f64 {
        self.underlying_normal().sample(rng).exp()
    }

    /// Densité `f(x) = 1/(x σ √(2π))·exp(−(ln x − μ)²/(2σ²))` pour `x > 0`,
    /// nulle pour `x ≤ 0`.
    ///
    /// # Complexité
    /// O(1).
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let lx = x.ln();
        let z = (lx - self.mu_log) / self.sigma_log;
        (-0.5 * z * z).exp() / (x * self.sigma_log * (2.0 * std::f64::consts::PI).sqrt())
    }

    /// Fonction de répartition : `F(x) = Φ((ln x − μ)/σ)` pour `x > 0`,
    /// nulle pour `x ≤ 0`.
    ///
    /// # Complexité
    /// O(1).
    fn cdf(&self, x: f64) -> Option<f64> {
        if x <= 0.0 {
            return Some(0.0);
        }
        Some(super::normal::normal_cdf(
            (x.ln() - self.mu_log) / self.sigma_log,
        ))
    }

    fn mean(&self) -> Option<f64> {
        Some((self.mu_log + self.sigma_log * self.sigma_log / 2.0).exp())
    }

    fn variance(&self) -> Option<f64> {
        let s2 = self.sigma_log * self.sigma_log;
        Some(((s2).exp() - 1.0) * (2.0 * self.mu_log + s2).exp())
    }

    fn name(&self) -> &'static str {
        "log_normal"
    }
}

#[cfg(test)]
mod tests {
    use super::LogNormal;
    use crate::distribution::Distribution;
    use crate::rng::DeterministicRng;

    const N: usize = 100_000;

    fn rng() -> DeterministicRng {
        DeterministicRng::from_seed([11u8; 32])
    }

    #[test]
    fn invalid_parameters_rejected() {
        assert!(LogNormal::new(0.0, 0.0).is_err());
        assert!(LogNormal::new(0.0, -1.0).is_err());
        assert!(LogNormal::new(0.0, f64::NAN).is_err());
        assert!(LogNormal::new(1.0, 0.5).is_ok());
    }

    #[test]
    fn pdf_known_values() {
        // LogN(0,1) : f(1) = 1/(1·1·√(2π)) ≈ 0.3989423 (car ln 1 = 0).
        let l = LogNormal::new(0.0, 1.0).unwrap();
        assert!((l.pdf(1.0) - 1.0 / (2.0 * std::f64::consts::PI).sqrt()).abs() < 1e-9);
        // pdf nulle pour x ≤ 0.
        assert_eq!(l.pdf(0.0), 0.0);
        assert_eq!(l.pdf(-1.0), 0.0);
        // Lien avec la normale : pdf_ln(x) = pdf_normale(ln x) / x.
        let n = super::super::normal::Normal::new(0.0, 1.0).unwrap();
        assert!((l.pdf(2.5) - n.pdf(2.5f64.ln()) / 2.5).abs() < 1e-9);
    }

    #[test]
    fn cdf_known_values() {
        // LogN(0,1) : F(1) = Φ(0) = 0.5 ; F(e) ≈ Φ(1) ≈ 0.8413.
        let l = LogNormal::new(0.0, 1.0).unwrap();
        assert!((l.cdf(1.0).unwrap() - 0.5).abs() < 1e-8);
        assert!((l.cdf(std::f64::consts::E).unwrap() - 0.841_344_746).abs() < 1e-4);
        assert_eq!(l.cdf(-1.0).unwrap(), 0.0);
    }

    #[test]
    fn link_with_normal_distribution() {
        // Moyenne empirique de log(X) ≈ μ_log.
        let mut dist = LogNormal::new(0.5, 1.0).unwrap();
        let mut rng = rng();
        let samples: Vec<f64> = (0..N).map(|_| dist.sample(&mut rng)).collect();
        let logs: Vec<f64> = samples.iter().map(|x| x.ln()).collect();
        crate::distributions::test_util::assert_mean_tolerance(&logs, 0.5, 1.0).unwrap();
    }

    #[test]
    fn moments_match_formula() {
        let l = LogNormal::new(0.0, 1.0).unwrap();
        let exp_half = (0.5f64).exp();
        assert!((l.mean().unwrap() - exp_half).abs() < 1e-12);
        let var = ((1.0f64).exp() - 1.0) * (1.0f64).exp();
        assert!((l.variance().unwrap() - var).abs() < 1e-12);
    }

    #[test]
    fn support_is_positive() {
        let mut dist = LogNormal::new(0.0, 1.0).unwrap();
        let mut rng = rng();
        for _ in 0..N {
            let x = dist.sample(&mut rng);
            assert!(x > 0.0, "échantillon non positif : {x}");
        }
    }

    #[test]
    fn reproducibility_strict() {
        let mut a = LogNormal::new(0.0, 1.0).unwrap();
        let mut b = LogNormal::new(0.0, 1.0).unwrap();
        let mut rng_a = rng();
        let mut rng_b = rng();
        for _ in 0..10_000 {
            assert_eq!(a.sample(&mut rng_a), b.sample(&mut rng_b));
        }
    }
}
