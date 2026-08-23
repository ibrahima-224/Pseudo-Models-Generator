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

//! Distribution de Laplace (double exponentielle) `L(μ, b)`.
//!
//! Échantillonnage par **transformation inverse** : `X = μ − b·sign(U)·ln(1−2|U|)`
//! avec `U ~ U(−0.5, 0.5)` (spécification doc 4 §2.2).

use crate::distribution::Distribution;
use crate::error::{MathError, MathResult};
use crate::rng::DeterministicRng;

/// Loi de Laplace `L(location, scale)` avec `scale > 0`.
///
/// # Moments
/// - Espérance : `location` ;
/// - Variance : `2·scale²`.
#[derive(Debug, Clone, PartialEq)]
pub struct Laplace {
    location: f64,
    scale: f64,
}

impl Laplace {
    /// Construit `L(location, scale)`.
    ///
    /// # Erreurs
    /// [`MathError::InvalidParameter`] si `scale ≤ 0` ou non fini.
    ///
    /// # Complexité
    /// O(1).
    pub fn new(location: f64, scale: f64) -> MathResult<Self> {
        if !scale.is_finite() || scale <= 0.0 {
            return Err(MathError::InvalidParameter(format!(
                "scale de Laplace doit être fini et > 0, reçu {scale}"
            )));
        }
        Ok(Self { location, scale })
    }
}

impl Distribution for Laplace {
    /// Transformation inverse : `μ − b·sign(U)·ln(1−2|U|)`, `U ~ U(−½, ½)`.
    ///
    /// # Complexité
    /// O(1).
    fn sample(&mut self, rng: &mut DeterministicRng) -> f64 {
        // u ∈ [0,1) → v ∈ (−0.5, 0.5).
        let u = rng.next_f64();
        let v = u - 0.5;
        let mag = -self.scale * (1.0 - 2.0 * v.abs()).ln();
        self.location + if v >= 0.0 { mag } else { -mag }
    }

    /// Densité `f(x) = 1/(2b)·exp(−|x−μ|/b)`.
    ///
    /// # Complexité
    /// O(1).
    fn pdf(&self, x: f64) -> f64 {
        (-(x - self.location).abs() / self.scale).exp() / (2.0 * self.scale)
    }

    /// Fonction de répartition :
    /// `F(x) = ½ exp((x−μ)/b)` si `x < μ`, sinon `1 − ½ exp(−(x−μ)/b)`.
    ///
    /// # Complexité
    /// O(1).
    fn cdf(&self, x: f64) -> Option<f64> {
        let z = x - self.location;
        if z < 0.0 {
            Some(0.5 * (z / self.scale).exp())
        } else {
            Some(1.0 - 0.5 * (-z / self.scale).exp())
        }
    }

    fn mean(&self) -> Option<f64> {
        Some(self.location)
    }

    fn variance(&self) -> Option<f64> {
        Some(2.0 * self.scale * self.scale)
    }

    fn name(&self) -> &'static str {
        "laplace"
    }
}

#[cfg(test)]
mod tests {
    use super::Laplace;
    use crate::distribution::Distribution;
    use crate::rng::DeterministicRng;

    const N: usize = 100_000;

    fn rng() -> DeterministicRng {
        DeterministicRng::from_seed([5u8; 32])
    }

    #[test]
    fn invalid_parameters_rejected() {
        assert!(Laplace::new(0.0, 0.0).is_err());
        assert!(Laplace::new(0.0, -2.0).is_err());
        assert!(Laplace::new(0.0, f64::NAN).is_err());
        assert!(Laplace::new(1.0, 2.0).is_ok());
    }

    #[test]
    fn pdf_known_values() {
        // L(0,1) : f(0) = 0.5 ; f(1) = 0.5·e⁻¹ ≈ 0.1839397.
        let l = Laplace::new(0.0, 1.0).unwrap();
        assert!((l.pdf(0.0) - 0.5).abs() < 1e-12);
        assert!((l.pdf(1.0) - 0.5 * (-1.0f64).exp()).abs() < 1e-12);
        assert_eq!(l.pdf(1.0), l.pdf(-1.0)); // symétrie
    }

    #[test]
    fn cdf_known_values() {
        let l = Laplace::new(0.0, 1.0).unwrap();
        assert!((l.cdf(0.0).unwrap() - 0.5).abs() < 1e-12);
        assert!((l.cdf(1.0).unwrap() - (1.0 - 0.5 * (-1.0f64).exp())).abs() < 1e-12);
        assert!((l.cdf(-1.0).unwrap() - 0.5 * (-1.0f64).exp()).abs() < 1e-12);
        // Monotonie et bornes.
        assert!(l.cdf(-10.0).unwrap() < 1e-3);
        assert!(l.cdf(10.0).unwrap() > 1.0 - 1e-3);
    }

    #[test]
    fn empirical_mean_within_tolerance() {
        // L(1, 2) : μ = 1, σ = √(2·4) = √8 ≈ 2.828.
        let mut dist = Laplace::new(1.0, 2.0).unwrap();
        let mut rng = rng();
        let samples: Vec<f64> = (0..N).map(|_| dist.sample(&mut rng)).collect();
        let expected_std = (2.0f64 * 4.0).sqrt();
        crate::distributions::test_util::assert_mean_tolerance(&samples, 1.0, expected_std)
            .unwrap();
        crate::distributions::test_util::assert_std_tolerance(&samples, expected_std).unwrap();
    }

    #[test]
    fn reproducibility_strict() {
        let mut a = Laplace::new(0.0, 1.0).unwrap();
        let mut b = Laplace::new(0.0, 1.0).unwrap();
        let mut rng_a = rng();
        let mut rng_b = rng();
        for _ in 0..10_000 {
            assert_eq!(a.sample(&mut rng_a), b.sample(&mut rng_b));
        }
    }
}
