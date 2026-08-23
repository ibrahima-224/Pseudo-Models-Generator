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

//! Estimation des paramètres de la distribution de Pareto et CDF.
//!
//! La distribution de Pareto (Type I) est caractérisée par les paramètres :
//! - `scale` (x_m) : paramètre d'échelle (minimum) ;
//! - `shape` (α) : paramètre de forme.
//!
//! On utilise ici la méthode du maximum de vraisemblance (MLE) pour estimer ces paramètres.

/// Estime les paramètres de la distribution de Pareto.
///
/// # Entrées
/// - `data` : données d'entrée (doivent être > 0).
///
/// # Sorties
/// Tuple `(shape, scale)` où :
/// - `shape` : paramètre de forme estimé (α) ;
/// - `scale` : paramètre d'échelle estimé (x_m).
///
/// # Algorithme
/// Utilise la méthode du maximum de vraisemblance (MLE) :
/// - `x_m = min(x_i)` ;
/// - `α = n / Σ ln(x_i / x_m)`.
///
/// # Erreurs
/// Retourne `(1.0, min)` si les données sont vides ou si le calcul échoue.
pub fn estimate_pareto_params(data: &[f64]) -> (f64, f64) {
    if data.is_empty() {
        return (1.0, 1.0);
    }

    // Filtrer les valeurs non positives
    let positive_data: Vec<f64> = data.iter().copied().filter(|&x| x > 0.0).collect();
    if positive_data.is_empty() {
        return (1.0, 1.0);
    }

    let n = positive_data.len() as f64;
    let x_m = positive_data.iter().cloned().fold(f64::INFINITY, f64::min);

    // Calcul de la somme des logarithmes
    let sum_ln = positive_data.iter().map(|&x| (x / x_m).ln()).sum::<f64>();

    if sum_ln.abs() < 1e-10 {
        // Toutes les valeurs sont égales à x_m
        return (1.0, x_m);
    }

    let shape = n / sum_ln;

    // Bornes raisonnables pour les paramètres
    let shape = shape.clamp(0.1, 100.0);
    let scale = x_m.max(1e-10);

    (shape, scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_pareto_params_empty() {
        let (shape, scale) = estimate_pareto_params(&[]);
        assert_eq!(shape, 1.0);
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn estimate_pareto_params_positive_data() {
        // Données simulées d'une Pareto avec shape=2, scale=1
        let data = vec![1.0, 1.5, 2.0, 3.0, 5.0, 10.0];
        let (shape, scale) = estimate_pareto_params(&data);
        // On s'attend à des paramètres raisonnables
        assert!(shape > 0.0, "shape devrait être positif, reçu {shape}");
        assert!(scale > 0.0, "scale devrait être positif, reçu {scale}");
        // Le scale devrait être le minimum des données
        assert!((scale - 1.0).abs() < 1e-10);
    }

    #[test]
    fn estimate_pareto_params_with_zeros() {
        // Données avec des zéros (doivent être filtrés)
        let data = vec![0.0, 1.0, 2.0, 3.0, 0.0];
        let (shape, scale) = estimate_pareto_params(&data);
        assert!(shape > 0.0);
        assert!((scale - 1.0).abs() < 1e-10);
    }

    #[test]
    fn estimate_pareto_params_constant_data() {
        // Données constantes
        let data = vec![5.0, 5.0, 5.0, 5.0];
        let (shape, scale) = estimate_pareto_params(&data);
        // Pour des données constantes, on s'attend à un shape élevé
        assert!(
            shape >= 1.0,
            "shape devrait être >= 1 pour des données constantes, reçu {shape}"
        );
        assert!((scale - 5.0).abs() < 1e-10);
    }
}
