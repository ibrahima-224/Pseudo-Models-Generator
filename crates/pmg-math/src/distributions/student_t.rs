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

//! Distribution t de Student `t(ν)` (standard, location 0, scale 1).
//!
//! Échantillonnage par la méthode de Marsaglia : `t = Z / √(V/ν)` avec
//! `Z ~ N(0,1)` et `V ~ χ²(ν)` (spécification doc 4 §2.2).

use crate::distribution::Distribution;
use crate::error::{MathError, MathResult};
use crate::rng::DeterministicRng;

use super::normal::Normal;

/// Loi de Student standard à `df > 0` degrés de liberté.
///
/// # Moments
/// - Espérance : 0 (si `df > 1`) ;
/// - Variance : `df/(df−2)` (si `df > 2`).
///
/// # Limites
/// `df → ∞` converge vers la normale standard (testé).
#[derive(Debug, Clone, PartialEq)]
pub struct StudentT {
    df: f64,
}

impl StudentT {
    /// Construit `t(df)`.
    ///
    /// # Erreurs
    /// [`MathError::InvalidParameter`] si `df ≤ 0` ou non fini.
    ///
    /// # Complexité
    /// O(1).
    pub fn new(df: f64) -> MathResult<Self> {
        if !df.is_finite() || df <= 0.0 {
            return Err(MathError::InvalidParameter(format!(
                "df de Student doit être fini et > 0, reçu {df}"
            )));
        }
        Ok(Self { df })
    }

    /// Degrés de liberté.
    pub fn df_value(&self) -> f64 {
        self.df
    }
}

impl Distribution for StudentT {
    /// Tirage par Marsaglia : `Z / √(V/ν)`.
    ///
    /// `V ~ χ²(ν)` est tirée par transformation inverse du quantile chi²
    /// ([`crate::special::chi2_quantile`], exact et déterministe), puis
    /// `t = Z / √(V/ν)` avec `Z ~ N(0,1)`.
    ///
    /// # Complexité
    /// O(60 · coût de la gamma incomplète) — bornée.
    fn sample(&mut self, rng: &mut DeterministicRng) -> f64 {
        let mut normal = Normal::new(0.0, 1.0).expect("paramètres valides");
        let z = normal.sample(rng);
        let p = rng.next_f64().clamp(f64::MIN_POSITIVE, 1.0 - 1e-15);
        let v = crate::special::chi2_quantile(p, self.df);
        z / (v / self.df).sqrt()
    }

    /// Densité `f(x) = Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) · (1 + x²/ν)^(−(ν+1)/2)`.
    ///
    /// # Complexité
    /// O(1) — deux log-gammas.
    fn pdf(&self, x: f64) -> f64 {
        let nu = self.df;
        let lc = crate::special::ln_gamma((nu + 1.0) / 2.0)
            - crate::special::ln_gamma(nu / 2.0)
            - 0.5 * (nu * std::f64::consts::PI).ln();
        (lc - ((nu + 1.0) / 2.0) * (1.0 + x * x / nu).ln()).exp()
    }

    /// Fonction de répartition via [`crate::special::student_t_cdf`]
    /// (implémentation numérique documentée, A&S §26.7.1).
    fn cdf(&self, x: f64) -> Option<f64> {
        Some(crate::special::student_t_cdf(x, self.df))
    }

    fn mean(&self) -> Option<f64> {
        (self.df > 1.0).then_some(0.0)
    }

    fn variance(&self) -> Option<f64> {
        (self.df > 2.0).then_some(self.df / (self.df - 2.0))
    }

    fn name(&self) -> &'static str {
        "student_t"
    }
}

#[cfg(test)]
mod tests {
    use super::StudentT;
    use crate::distribution::Distribution;
    use crate::rng::DeterministicRng;

    const N: usize = 100_000;

    fn rng() -> DeterministicRng {
        DeterministicRng::from_seed([9u8; 32])
    }

    #[test]
    fn invalid_parameters_rejected() {
        assert!(StudentT::new(0.0).is_err());
        assert!(StudentT::new(-3.0).is_err());
        assert!(StudentT::new(f64::NAN).is_err());
        assert!(StudentT::new(1.0).is_ok());
    }

    #[test]
    fn high_df_approaches_normal() {
        // df élevé : pdf et cdf proches de la normale standard.
        let t = StudentT::new(500.0).unwrap();
        let n = super::super::normal::Normal::new(0.0, 1.0).unwrap();
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            assert!(
                (t.pdf(x) - n.pdf(x)).abs() < 5e-3,
                "pdf t vs normale à x={x}"
            );
            assert!(
                (t.cdf(x).unwrap() - n.cdf(x).unwrap()).abs() < 5e-3,
                "cdf t vs normale à x={x}"
            );
        }
    }

    #[test]
    fn symmetry_properties() {
        let t = StudentT::new(4.0).unwrap();
        // Densité paire : f(x) = f(−x) ; cdf : F(x) + F(−x) = 1.
        for x in [0.5, 1.0, 2.0, 3.0] {
            assert!((t.pdf(x) - t.pdf(-x)).abs() < 1e-12, "x={x}");
            assert!((t.cdf(x).unwrap() + t.cdf(-x).unwrap() - 1.0).abs() < 1e-9);
        }
        assert!((t.cdf(0.0).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn empirical_moments_within_tolerance() {
        // df = 10 : variance = 10/8 = 1.25, écart-type ≈ 1.118.
        let df = 10.0;
        let mut dist = StudentT::new(df).unwrap();
        let mut rng = rng();
        let samples: Vec<f64> = (0..N).map(|_| dist.sample(&mut rng)).collect();
        let expected_std = (df / (df - 2.0)).sqrt();
        crate::distributions::test_util::assert_mean_tolerance(&samples, 0.0, expected_std)
            .unwrap();
        crate::distributions::test_util::assert_std_tolerance(&samples, expected_std).unwrap();
    }

    #[test]
    fn moments_none_for_low_df() {
        let t1 = StudentT::new(1.0).unwrap();
        assert_eq!(t1.mean(), None);
        assert_eq!(t1.variance(), None);
        let t2 = StudentT::new(2.0).unwrap();
        assert_eq!(t2.mean(), Some(0.0));
        assert_eq!(t2.variance(), None);
        assert_eq!(StudentT::new(3.0).unwrap().variance(), Some(3.0));
    }

    #[test]
    fn reproducibility_strict() {
        let mut a = StudentT::new(4.0).unwrap();
        let mut b = StudentT::new(4.0).unwrap();
        let mut rng_a = rng();
        let mut rng_b = rng();
        for _ in 0..2000 {
            assert_eq!(a.sample(&mut rng_a), b.sample(&mut rng_b));
        }
    }
}
