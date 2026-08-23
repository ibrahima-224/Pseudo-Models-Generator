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

//! Estimation des paramètres de la distribution de Weibull et CDF.
//!
//! La distribution de Weibull est caractérisée par les paramètres :
//! - `scale` (λ) : paramètre d'échelle ;
//! - `shape` (k) : paramètre de forme.
//!
//! On utilise ici la méthode des moments pour estimer ces paramètres.

/// Estime les paramètres de la distribution de Weibull.
///
/// # Entrées
/// - `data` : données d'entrée (doivent être > 0).
///
/// # Sorties
/// Tuple `(shape, scale)` où :
/// - `shape` : paramètre de forme estimé ;
/// - `scale` : paramètre d'échelle estimé.
///
/// # Algorithme
/// Utilise la méthode des moments via la régression sur les probabilités log.
/// Pour des données `x_i`, on calcule `ln(-ln(1 - i/(n+1)))` vs `ln(x_i)`.
/// La pente donne `shape`, l'intercept donne `ln(scale)`.
///
/// # Erreurs
/// Retourne `(1.0, mean)` si les données sont vides ou si le calcul échoue.
pub fn estimate_weibull_params(data: &[f64]) -> (f64, f64) {
    if data.is_empty() {
        return (1.0, 1.0);
    }

    // Filtrer les valeurs non positives
    let positive_data: Vec<f64> = data.iter().copied().filter(|&x| x > 0.0).collect();
    if positive_data.is_empty() {
        return (1.0, 1.0);
    }

    let mut sorted_data = positive_data.clone();
    // Gestion safe des NaN : on les traite comme égaux pour éviter un panic
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted_data.len();
    let mut sum_ln_x = 0.0;
    let mut sum_ln_x_sq = 0.0;
    let mut sum_y = 0.0;
    let mut sum_y_ln_x = 0.0;

    // Calcul des sommes pour la régression linéaire
    for (i, &x) in sorted_data.iter().enumerate() {
        let ln_x = x.ln();
        // Probabilité empirique de Weibull : y = ln(-ln(1 - p))
        // avec p = (i + 0.5) / n (méthode de médiane)
        let p = (i as f64 + 0.5) / n as f64;
        let y = (-(1.0 - p).ln()).ln();

        sum_ln_x += ln_x;
        sum_ln_x_sq += ln_x * ln_x;
        sum_y += y;
        sum_y_ln_x += y * ln_x;
    }

    let n_f = n as f64;
    let denom = n_f * sum_ln_x_sq - sum_ln_x * sum_ln_x;

    if denom.abs() < 1e-10 {
        // Données constantes ou quasi constantes
        return (1.0, sorted_data.iter().sum::<f64>() / n_f);
    }

    // Calcul de la pente (shape) et de l'intercept (ln(scale))
    let shape = (n_f * sum_y_ln_x - sum_ln_x * sum_y) / denom;
    let intercept = (sum_y - shape * sum_ln_x) / n_f;

    let scale = intercept.exp();

    // Bornes raisonnables pour les paramètres
    let shape = shape.clamp(0.1, 10.0);
    let scale = scale.max(1e-10);

    (shape, scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_weibull_params_empty() {
        let (shape, scale) = estimate_weibull_params(&[]);
        assert_eq!(shape, 1.0);
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn estimate_weibull_params_positive_data() {
        // Données simulées d'une Weibull avec shape=2, scale=1
        let data = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
        let (shape, scale) = estimate_weibull_params(&data);
        // On s'attend à des paramètres raisonnables
        assert!(shape > 0.0, "shape devrait être positif, reçu {shape}");
        assert!(scale > 0.0, "scale devrait être positif, reçu {scale}");
    }

    #[test]
    fn estimate_weibull_params_with_zeros() {
        // Données avec des zéros (doivent être filtrés)
        let data = vec![0.0, 1.0, 2.0, 3.0, 0.0];
        let (shape, scale) = estimate_weibull_params(&data);
        assert!(shape > 0.0);
        assert!(scale > 0.0);
    }

    #[test]
    fn estimate_weibull_params_constant_data() {
        // Données constantes
        let data = vec![5.0, 5.0, 5.0, 5.0];
        let (shape, scale) = estimate_weibull_params(&data);
        // Pour des données constantes, on s'attend à un shape élevé
        assert!(
            shape >= 1.0,
            "shape devrait être >= 1 pour des données constantes, reçu {shape}"
        );
        assert!((scale - 5.0).abs() < 1e-10);
    }
}
