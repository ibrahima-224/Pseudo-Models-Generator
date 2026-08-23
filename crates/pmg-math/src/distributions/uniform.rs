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

//! Distribution uniforme continue `U(min, max)`.
//!
//! Échantillonnage par transformation inverse : `x = min + u·(max - min)`
//! où `u` est un uniforme `[0,1]`.

use crate::distribution::Distribution;
use crate::error::{MathError, MathResult};
use crate::rng::DeterministicRng;

/// Loi uniforme continue `U(min, max)` avec `min < max`.
///
/// # Moments
/// - Espérance : `(min + max) / 2` ;
/// - Variance : `(max - min)² / 12`.
///
/// # Cas limites
///
/// - **`min = max`** : Rejet avec `MathError::InvalidParameter`.
/// - **`min > max`** : Rejet avec `MathError::InvalidParameter`.
/// - **Valeurs non finies** (NaN, ±∞) : Rejet avec `MathError::InvalidParameter`.
/// - **Valeurs extrêmes** (`min = f64::MIN`, `max = f64::MAX`) : Accepté, mais attention aux précisions.
/// - **Différence minuscule** (`max - min < f64::EPSILON`) : Accepté, mais échantillons très proches.
#[derive(Debug, Clone, PartialEq)]
pub struct Uniform {
    /// Borne inférieure (incluse).
    min: f64,
    /// Borne supérieure (incluse).
    max: f64,
}

impl Uniform {
    /// Crée une distribution uniforme `U(min, max)`.
    ///
    /// # Erreurs
    /// [`MathError::InvalidParameter`] si `min >= max` ou si les valeurs
    /// ne sont pas finies.
    ///
    /// # Complexité
    /// O(1).
    pub fn new(min: f64, max: f64) -> MathResult<Self> {
        if !min.is_finite() || !max.is_finite() {
            return Err(MathError::InvalidParameter(format!(
                "pour Uniform, min ({min}) et max ({max}) doivent être finis"
            )));
        }
        if min >= max {
            return Err(MathError::InvalidParameter(format!(
                "pour Uniform, min ({min}) doit être strictement inférieur à max ({max})"
            )));
        }
        Ok(Self { min, max })
    }

    /// Borne inférieure de la distribution.
    pub fn min_value(&self) -> f64 {
        self.min
    }

    /// Borne supérieure de la distribution.
    pub fn max_value(&self) -> f64 {
        self.max
    }
}

impl Distribution for Uniform {
    /// Tire un échantillon selon `U(min, max)`.
    ///
    /// # Complexité
    /// O(1).
    fn sample(&mut self, rng: &mut DeterministicRng) -> f64 {
        let u = rng.next_f64();
        self.min + u * (self.max - self.min)
    }

    /// Densité `f(x) = 1/(max - min)` si `x ∈ [min, max]`, `0` sinon.
    ///
    /// # Complexité
    /// O(1).
    fn pdf(&self, x: f64) -> f64 {
        if x < self.min || x > self.max {
            0.0
        } else {
            1.0 / (self.max - self.min)
        }
    }

    /// Fonction de répartition `F(x)`.
    ///
    /// # Complexité
    /// O(1).
    fn cdf(&self, x: f64) -> Option<f64> {
        if x < self.min {
            Some(0.0)
        } else if x > self.max {
            Some(1.0)
        } else {
            Some((x - self.min) / (self.max - self.min))
        }
    }

    fn mean(&self) -> Option<f64> {
        Some((self.min + self.max) / 2.0)
    }

    fn variance(&self) -> Option<f64> {
        let range = self.max - self.min;
        Some(range * range / 12.0)
    }

    fn name(&self) -> &'static str {
        "uniform"
    }
}

#[cfg(test)]
mod tests {
    use super::{Distribution, Uniform};
    use crate::distributions::test_util::{assert_mean_tolerance, assert_std_tolerance};
    use crate::rng::DeterministicRng;

    const N: usize = 100_000;

    fn rng() -> DeterministicRng {
        DeterministicRng::from_seed([5u8; 32])
    }

    #[test]
    fn invalid_parameters_rejected() {
        // min >= max
        assert!(Uniform::new(1.0, 1.0).is_err());
        assert!(Uniform::new(2.0, 1.0).is_err());
        // valeurs non finies
        assert!(Uniform::new(f64::NAN, 1.0).is_err());
        assert!(Uniform::new(0.0, f64::INFINITY).is_err());
        // cas valides
        assert!(Uniform::new(0.0, 1.0).is_ok());
        assert!(Uniform::new(-10.0, 10.0).is_ok());
    }

    #[test]
    fn pdf_known_values() {
        let u = Uniform::new(0.0, 1.0).unwrap();
        // f(0.5) = 1.0
        assert!((u.pdf(0.5) - 1.0).abs() < 1e-10);
        // f(0.0) = 1.0 (inclus)
        assert!((u.pdf(0.0) - 1.0).abs() < 1e-10);
        // f(1.0) = 1.0 (inclus)
        assert!((u.pdf(1.0) - 1.0).abs() < 1e-10);
        // f(-0.1) = 0.0
        assert!((u.pdf(-0.1) - 0.0).abs() < 1e-10);
        // f(1.1) = 0.0
        assert!((u.pdf(1.1) - 0.0).abs() < 1e-10);

        // U(2, 5) : f(3) = 1/3 ≈ 0.333333
        let u2 = Uniform::new(2.0, 5.0).unwrap();
        assert!((u2.pdf(3.0) - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn cdf_known_values() {
        let u = Uniform::new(0.0, 1.0).unwrap();
        // F(0.0) = 0.0
        assert!((u.cdf(0.0).unwrap() - 0.0).abs() < 1e-10);
        // F(0.5) = 0.5
        assert!((u.cdf(0.5).unwrap() - 0.5).abs() < 1e-10);
        // F(1.0) = 1.0
        assert!((u.cdf(1.0).unwrap() - 1.0).abs() < 1e-10);
        // F(-0.1) = 0.0
        assert!((u.cdf(-0.1).unwrap() - 0.0).abs() < 1e-10);
        // F(1.1) = 1.0
        assert!((u.cdf(1.1).unwrap() - 1.0).abs() < 1e-10);

        // U(2, 5) : F(3) = (3-2)/(5-2) = 1/3
        let u2 = Uniform::new(2.0, 5.0).unwrap();
        assert!((u2.cdf(3.0).unwrap() - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn moments_known_values() {
        // U(0, 1) : mean = 0.5, var = 1/12
        let u = Uniform::new(0.0, 1.0).unwrap();
        assert!((u.mean().unwrap() - 0.5).abs() < 1e-10);
        assert!((u.variance().unwrap() - 1.0 / 12.0).abs() < 1e-10);

        // U(2, 5) : mean = 3.5, var = 9/12 = 0.75
        let u2 = Uniform::new(2.0, 5.0).unwrap();
        assert!((u2.mean().unwrap() - 3.5).abs() < 1e-10);
        assert!((u2.variance().unwrap() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn sample_range() {
        let mut u = Uniform::new(0.0, 1.0).unwrap();
        let mut r = rng();
        for _ in 0..N {
            let x = u.sample(&mut r);
            assert!((0.0..=1.0).contains(&x), "échantillon hors [0,1]: {x}");
        }

        let mut u2 = Uniform::new(-10.0, 10.0).unwrap();
        for _ in 0..N {
            let x = u2.sample(&mut r);
            assert!(
                (-10.0..=10.0).contains(&x),
                "échantillon hors [-10,10]: {x}"
            );
        }
    }

    #[test]
    fn statistical_properties() {
        let mut u = Uniform::new(0.0, 1.0).unwrap();
        let mut r = rng();
        let samples: Vec<f64> = (0..N).map(|_| u.sample(&mut r)).collect();

        // Vérifier la moyenne (μ = 0.5, σ = √(1/12) ≈ 0.288675)
        assert_mean_tolerance(&samples, 0.5, 1.0 / 12.0_f64.sqrt()).unwrap();
        // Vérifier l'écart-type
        assert_std_tolerance(&samples, 1.0 / 12.0_f64.sqrt()).unwrap();
    }

    #[test]
    fn serialization_round_trip() {
        let u = Uniform::new(2.0, 8.0).unwrap();
        // Vérifier que les accesseurs fonctionnent
        assert_eq!(u.min_value(), 2.0);
        assert_eq!(u.max_value(), 8.0);
        assert_eq!(u.name(), "uniform");
    }
}
