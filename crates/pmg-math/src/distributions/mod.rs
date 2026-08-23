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

//! Familles de distributions de probabilité.
//!
//! Chaque famille implémente le trait [`crate::distribution::Distribution`] (tirage `sample`,
//! densité `pdf`, répartition `cdf`, moments) avec :
//! - validation des paramètres à la construction (erreur typée, jamais de
//!   valeur arbitraire) ;
//! - méthodes documentées (formules, hypothèses, complexité, limites) ;
//! - tests de valeurs connues, de propriétés statistiques (tolérances) et de
//!   reproductibilité.

pub mod laplace;
pub mod log_normal;
pub mod mixture;
pub mod normal;
pub mod pareto;
pub mod student_t;
pub mod uniform;
pub mod weibull;

pub use laplace::Laplace;
pub use log_normal::LogNormal;
pub use mixture::Mixture;
pub use normal::Normal;
pub use pareto::Pareto;
pub use student_t::StudentT;
pub use uniform::Uniform;
pub use weibull::Weibull;

#[cfg(test)]
pub(crate) mod test_util {
    use crate::error::MathError;
    use crate::statistics;

    /// Vérifie que la moyenne empirique d'un échantillon est dans la tolérance
    /// `5·σ/√N` (conformité `docs/architecture/09-tests-benchmarks-ci.md`
    /// §1.7) autour de la moyenne théorique.
    ///
    /// # Retour
    /// `Ok(())` si `|μ̂ − μ| < 5σ/√N`, sinon l'écart observé en erreur
    /// [`MathError::Internal`] (message diagnostic).
    pub fn assert_mean_tolerance(
        samples: &[f64],
        expected_mean: f64,
        expected_std: f64,
    ) -> Result<(), MathError> {
        let n = samples.len() as f64;
        let mu_hat = statistics::mean(samples)?;
        let tolerance = 5.0 * expected_std / n.sqrt();
        let err = (mu_hat - expected_mean).abs();
        if err <= tolerance {
            Ok(())
        } else {
            Err(MathError::Internal(format!(
                "moyenne hors tolérance : μ̂={mu_hat:.6} attendu {expected_mean:.6}, \
                 tolérance {tolerance:.6}, écart {err:.6}"
            )))
        }
    }

    /// Vérifie que l'écart-type empirique (population) est dans la tolérance
    /// relative `10%` autour de l'écart-type théorique (grand échantillon).
    pub fn assert_std_tolerance(samples: &[f64], expected_std: f64) -> Result<(), MathError> {
        let s_hat = statistics::std_population(samples)?;
        let rel = (s_hat - expected_std).abs() / expected_std;
        if rel <= 0.10 {
            Ok(())
        } else {
            Err(MathError::Internal(format!(
                "écart-type hors tolérance : σ̂={s_hat:.6} attendu {expected_std:.6}, rel={rel:.6}"
            )))
        }
    }
}
