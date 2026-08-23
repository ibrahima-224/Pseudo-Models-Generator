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

//! Estimation des paramètres de la distribution de Student-t et CDF.
//!
//! La distribution de Student-t est caractérisée par les degrés de liberté (df).
//! On utilise ici la méthode des moments pour estimer df à partir du kurtosis.

/// Estime les paramètres de la distribution de Student-t.
///
/// # Entrées
/// - `data` : données d'entrée (non vides).
///
/// # Sorties
/// Tuple `(df, location)` où :
/// - `df` : degrés de liberté estimés ;
/// - `location` : paramètre de localisation (moyenne des données).
///
/// # Algorithme
/// Utilise la méthode des moments : `kurtosis = 6/(df-4) + 3` pour df > 4.
/// Si le kurtosis observé est ≤ 3, on retourne df = 30 (grand df, proche de la normale).
/// Si le kurtosis est > 3, on calcule df = 6/(kurtosis - 3) + 4.
///
/// # Erreurs
/// Retourne `(30.0, mean)` si les données sont vides ou si le calcul échoue.
pub fn estimate_student_t_params(data: &[f64]) -> (f64, f64) {
    if data.is_empty() {
        return (30.0, 0.0);
    }

    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;

    // Calcul de la variance
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    if variance <= 0.0 {
        return (30.0, mean);
    }

    // Calcul du kurtosis (excess kurtosis = kurtosis - 3)
    let kurtosis = data
        .iter()
        .map(|x| ((x - mean) / variance.sqrt()).powi(4))
        .sum::<f64>()
        / n;

    let excess_kurtosis = kurtosis - 3.0;

    // Estimation des degrés de liberté
    let df = if excess_kurtosis <= 0.0 {
        // Kurtosis ≤ 3 : distribution à queues légères, on retourne un df élevé
        30.0
    } else {
        // Kurtosis > 3 : queues lourdes, on calcule df
        // Formule : excess_kurtosis = 6/(df - 4) pour df > 4
        let df_estimated = 6.0 / excess_kurtosis + 4.0;
        // Bornes raisonnables pour df
        df_estimated.clamp(2.1, 100.0)
    };

    (df, mean)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_student_t_params_empty() {
        let (df, mean) = estimate_student_t_params(&[]);
        assert_eq!(df, 30.0);
        assert_eq!(mean, 0.0);
    }

    #[test]
    fn estimate_student_t_params_normal_data() {
        // Données normales (kurtosis ≈ 3)
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let (df, mean) = estimate_student_t_params(&data);
        // Pour des données normales, on s'attend à un df élevé
        assert!(
            df >= 10.0,
            "df devrait être élevé pour des données normales, reçu {df}"
        );
        assert!((mean - 0.0).abs() < 1e-10);
    }

    #[test]
    fn estimate_student_t_params_heavy_tails() {
        // Données avec queues lourdes (kurtosis > 3)
        // On simule avec des valeurs extrêmes
        let data = vec![-10.0, -1.0, 0.0, 1.0, 10.0];
        let (df, mean) = estimate_student_t_params(&data);
        // Avec des valeurs extrêmes, on s'attend à un df plus faible
        // Pour des données avec des valeurs extrêmes, le kurtosis est élevé
        // et le df estimé devrait être plus faible
        assert!(
            df <= 30.0,
            "df devrait être <= 30 pour des queues lourdes, reçu {df}"
        );
        assert!(df >= 2.1, "df devrait être >= 2.1, reçu {df}");
        assert!((mean - 0.0).abs() < 1e-10);
    }

    #[test]
    fn estimate_student_t_params_constant_data() {
        // Données constantes : variance = 0
        let data = vec![5.0, 5.0, 5.0, 5.0];
        let (df, mean) = estimate_student_t_params(&data);
        assert_eq!(df, 30.0);
        assert_eq!(mean, 5.0);
    }
}
