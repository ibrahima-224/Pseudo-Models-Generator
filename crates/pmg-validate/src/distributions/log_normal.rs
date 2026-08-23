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

//! Estimation des paramètres de la distribution log-normale et CDF.
//!
//! La distribution log-normale est caractérisée par les paramètres :
//! - `mu` : moyenne du logarithme des données ;
//! - `sigma` : écart-type du logarithme des données.
//!
//! On utilise ici la méthode des moments pour estimer ces paramètres.

/// Estime les paramètres de la distribution log-normale.
///
/// # Entrées
/// - `data` : données d'entrée (doivent être > 0).
///
/// # Sorties
/// Tuple `(mu, sigma)` où :
/// - `mu` : moyenne du logarithme des données ;
/// - `sigma` : écart-type du logarithme des données.
///
/// # Algorithme
/// Utilise la méthode des moments :
/// - `mu = mean(ln(x_i))` ;
/// - `sigma = std(ln(x_i))`.
///
/// # Erreurs
/// Retourne `(0.0, 1.0)` si les données sont vides ou si le calcul échoue.
pub fn estimate_lognormal_params(data: &[f64]) -> (f64, f64) {
    if data.is_empty() {
        return (0.0, 1.0);
    }

    // Filtrer les valeurs non positives
    let positive_data: Vec<f64> = data.iter().copied().filter(|&x| x > 0.0).collect();
    if positive_data.is_empty() {
        return (0.0, 1.0);
    }

    let n = positive_data.len() as f64;

    // Calcul des logarithmes
    let ln_data: Vec<f64> = positive_data.iter().map(|x| x.ln()).collect();

    // Calcul de la moyenne des logarithmes
    let mu = ln_data.iter().sum::<f64>() / n;

    // Calcul de la variance des logarithmes
    let variance = ln_data.iter().map(|ln_x| (ln_x - mu).powi(2)).sum::<f64>() / n;

    let sigma = variance.sqrt();

    // Bornes raisonnables pour les paramètres
    let sigma = sigma.clamp(0.01, 10.0);

    (mu, sigma)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_lognormal_params_empty() {
        let (mu, sigma) = estimate_lognormal_params(&[]);
        assert_eq!(mu, 0.0);
        assert_eq!(sigma, 1.0);
    }

    #[test]
    fn estimate_lognormal_params_positive_data() {
        // Données simulées d'une log-normale avec mu=0, sigma=1
        let data = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
        let (_mu, sigma) = estimate_lognormal_params(&data);
        // On s'attend à des paramètres raisonnables
        assert!(sigma > 0.0, "sigma devrait être positif, reçu {sigma}");
    }

    #[test]
    fn estimate_lognormal_params_with_zeros() {
        // Données avec des zéros (doivent être filtrés)
        let data = vec![0.0, 1.0, 2.0, 3.0, 0.0];
        let (_mu, sigma) = estimate_lognormal_params(&data);
        assert!(sigma > 0.0);
    }

    #[test]
    fn estimate_lognormal_params_constant_data() {
        // Données constantes
        let data = vec![5.0, 5.0, 5.0, 5.0];
        let (mu, sigma) = estimate_lognormal_params(&data);
        // Pour des données constantes, on s'attend à un sigma faible
        assert!(
            sigma < 0.1,
            "sigma devrait être faible pour des données constantes, reçu {sigma}"
        );
        assert!((mu - 5.0f64.ln()).abs() < 1e-10);
    }
}
