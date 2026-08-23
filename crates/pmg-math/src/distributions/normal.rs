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

//! Distribution normale (gaussienne) `N(μ, σ)`.
//!
//! Échantillonnage par **Box-Muller** (impl. interne, sans dépendance `rand`
//! de distributions) : deux uniformes indépendantes `u1, u2` produisent deux
//! normales standard indépendantes via
//! `z = √(−2 ln u1) · cos(2π u2)`.

use crate::distribution::Distribution;
use crate::error::{MathError, MathResult};
use crate::rng::DeterministicRng;

/// Loi normale `N(mean, std)` avec `std > 0`.
///
/// # Moments
/// - Espérance : `mean` ;
/// - Variance : `std²`.
#[derive(Debug, Clone, PartialEq)]
pub struct Normal {
    mean: f64,
    std: f64,
}

impl Normal {
    /// Construit une normale `N(mean, std)`.
    ///
    /// # Erreurs
    /// [`MathError::InvalidParameter`] si `std ≤ 0` (variance nulle ou
    /// négative interdite — le cas dégénéré σ = 0 n'est pas une loi).
    ///
    /// # Complexité
    /// O(1).
    pub fn new(mean: f64, std: f64) -> MathResult<Self> {
        if !std.is_finite() || std <= 0.0 {
            return Err(MathError::InvalidParameter(format!(
                "std de la normale doit être fini et > 0, reçu {std}"
            )));
        }
        Ok(Self { mean, std })
    }

    /// Moyenne de la loi.
    pub fn mean_value(&self) -> f64 {
        self.mean
    }

    /// Écart-type de la loi.
    pub fn std_value(&self) -> f64 {
        self.std
    }

    /// Remplit un buffer de `n` échantillons indépendants (un seul appel
    /// consomme un nombre pair de valeurs du RNG).
    ///
    /// # Complexité
    /// O(n) — un couple Box-Muller par paire d'échantillons.
    pub fn sample_into(&self, rng: &mut DeterministicRng, buf: &mut [f64]) {
        let mut i = 0;
        while i < buf.len() {
            // Box-Muller : un couple par itération.
            let u1 = rng.next_f64().max(f64::MIN_POSITIVE);
            let u2 = rng.next_f64();
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            buf[i] = self.mean + self.std * r * theta.cos();
            if i + 1 < buf.len() {
                buf[i + 1] = self.mean + self.std * r * theta.sin();
            }
            i += 2;
        }
    }
}

impl Distribution for Normal {
    fn sample(&mut self, rng: &mut DeterministicRng) -> f64 {
        // Même algorithme que sample_into pour un seul échantillon.
        let u1 = rng.next_f64().max(f64::MIN_POSITIVE);
        let u2 = rng.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.mean + self.std * r * theta.cos()
    }

    /// Densité `f(x) = 1/(σ√(2π)) · exp(−½((x−μ)/σ)²)`.
    ///
    /// # Complexité
    /// O(1).
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mean) / self.std;
        (-0.5 * z * z).exp() / (self.std * (2.0 * std::f64::consts::PI).sqrt())
    }

    /// Fonction de répartition `Φ((x−μ)/σ)` (via [`normal_cdf`]).
    fn cdf(&self, x: f64) -> Option<f64> {
        Some(normal_cdf((x - self.mean) / self.std))
    }

    fn mean(&self) -> Option<f64> {
        Some(self.mean)
    }

    fn variance(&self) -> Option<f64> {
        Some(self.std * self.std)
    }

    fn name(&self) -> &'static str {
        "normal"
    }
}

/// Fonction de répartition de la normale standard, `Φ(z) = P(Z ≤ z)`.
///
/// Approximation d'Abramowitz & Stegun 26.2.17 (erreur |ε(z)| < 7.5e-8).
///
/// # Complexité
/// O(1).
pub fn normal_cdf(z: f64) -> f64 {
    const B1: f64 = 0.319_381_530;
    const B2: f64 = -0.356_563_782;
    const B3: f64 = 1.781_477_937;
    const B4: f64 = -1.821_255_978;
    const B5: f64 = 1.330_274_429;
    const P: f64 = 0.231_641_9;

    if z.is_nan() {
        return f64::NAN;
    }
    if z.is_infinite() {
        return if z > 0.0 { 1.0 } else { 0.0 };
    }
    let t = 1.0 / (1.0 + P * z.abs());
    let poly = t * (B1 + t * (B2 + t * (B3 + t * (B4 + t * B5))));
    // A&S 26.2.17 : Φ(z) = 1 − φ(z)·poly, φ(z) = e^(−z²/2)/√(2π).
    let phi = (-z * z / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf = 1.0 - phi * poly;
    if z >= 0.0 {
        cdf
    } else {
        1.0 - cdf
    }
}

#[cfg(test)]
mod tests {
    use super::{normal_cdf, Normal};
    use crate::distribution::Distribution;
    use crate::rng::DeterministicRng;

    const N: usize = 100_000;

    fn rng() -> DeterministicRng {
        DeterministicRng::from_seed([3u8; 32])
    }

    #[test]
    fn invalid_parameters_rejected() {
        assert!(Normal::new(0.0, 0.0).is_err());
        assert!(Normal::new(0.0, -1.0).is_err());
        assert!(Normal::new(0.0, f64::NAN).is_err());
        assert!(Normal::new(0.0, f64::INFINITY).is_err());
        assert!(Normal::new(0.0, 1.0).is_ok());
    }

    #[test]
    fn pdf_known_values() {
        // N(0,1) : f(0) = 1/√(2π) ≈ 0.3989423 ; f(1) = 0.2419707.
        let std_normal = Normal::new(0.0, 1.0).unwrap();
        assert!((std_normal.pdf(0.0) - 0.398_942_280_4).abs() < 1e-7);
        assert!((std_normal.pdf(1.0) - 0.241_970_724_5).abs() < 1e-7);
        // N(2, 0.5) : f(2) = 1/(0.5√(2π)) ≈ 0.7978846.
        let shifted = Normal::new(2.0, 0.5).unwrap();
        assert!((shifted.pdf(2.0) - 0.797_884_560_8).abs() < 1e-7);
    }

    #[test]
    fn cdf_known_values() {
        // Φ(0) = 0.5 ; Φ(1.96) ≈ 0.975 ; Φ(−1) ≈ 0.158655.
        // Tolérance 1e-6 : l'approximation A&S 26.2.17 a une erreur < 7.5e-8
        // mais les arrondis en f64 sur les extrêmes restent ~1e-7.
        let std_normal = Normal::new(0.0, 1.0).unwrap();
        assert!((std_normal.cdf(0.0).unwrap() - 0.5).abs() < 1e-6);
        assert!((std_normal.cdf(1.96).unwrap() - 0.975).abs() < 1e-4);
        assert!((std_normal.cdf(-1.0).unwrap() - 0.158_655).abs() < 1e-4);
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn empirical_mean_within_tolerance() {
        let dist = Normal::new(3.0, 2.0).unwrap();
        let mut rng = rng();
        let mut samples = vec![0.0f64; N];
        dist.sample_into(&mut rng, &mut samples);
        crate::distributions::test_util::assert_mean_tolerance(&samples, 3.0, 2.0).unwrap();
        crate::distributions::test_util::assert_std_tolerance(&samples, 2.0).unwrap();
    }

    #[test]
    fn reproducibility_strict() {
        let mut a = Normal::new(0.0, 1.0).unwrap();
        let mut b = Normal::new(0.0, 1.0).unwrap();
        let mut rng_a = rng();
        let mut rng_b = rng();
        for _ in 0..10_000 {
            assert_eq!(a.sample(&mut rng_a), b.sample(&mut rng_b));
        }
    }

    #[test]
    fn moments_match_formula() {
        let n = Normal::new(4.0, 3.0).unwrap();
        assert_eq!(n.mean(), Some(4.0));
        assert_eq!(n.variance(), Some(9.0));
        assert_eq!(n.name(), "normal");
    }
}
