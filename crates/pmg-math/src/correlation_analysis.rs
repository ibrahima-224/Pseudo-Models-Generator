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

//! Analyse de corrélation — corrélation moyenne, maximale, par bloc.
//!
//! Ce module fournit des fonctions pour analyser les corrélations dans
//! un échantillon de données, conformément au Sprint 11 du plan de
//! développement.
//!
//! # Responsabilités
//!
//! - Calcul de la corrélation de Pearson entre deux vecteurs ;
//! - Calcul de la matrice de corrélation complète ;
//! - Statistiques sur les corrélations (moyenne, maximale) ;
//! - Analyse de corrélation par blocs.
//!
//! # Conventions
//!
//! - Toutes les fonctions sont déterministes ;
//! - Les paramètres sont validés et retournent des erreurs typées (`MathError`);
//! - La documentation est en français.

use crate::error::{MathError, MathResult};
use crate::statistics;

/// Résultat de l'analyse de corrélation pour un échantillon.
#[derive(Debug, Clone)]
pub struct CorrelationAnalysis {
    /// Nombre de variables analysées.
    pub variable_count: usize,
    /// Matrice de corrélation (format plat : variable_count × variable_count).
    pub correlation_matrix: Vec<f64>,
    /// Corrélation moyenne (hors diagonale).
    pub mean_correlation: f64,
    /// Corrélation maximale en valeur absolue (hors diagonale).
    pub max_correlation: f64,
}

/// Calcule la corrélation de Pearson entre deux vecteurs.
///
/// # Entrées
/// - `x` : premier vecteur ;
/// - `y` : second vecteur (même longueur que `x`).
///
/// # Sorties
/// Coefficient de corrélation dans [-1, 1].
///
/// # Erreurs
/// - [`MathError::EmptyData`] si les vecteurs sont vides ;
/// - [`MathError::InvalidParameter`] si les vecteurs n'ont pas la même longueur ;
/// - [`MathError::InvalidParameter`] si la variance d'un vecteur est nulle.
///
/// # Exemple
///
/// ```
/// use pmg_math::correlation_analysis::pearson_correlation;
///
/// let x = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = [2.0, 4.0, 5.0, 4.0, 5.0];
/// let r = pearson_correlation(&x, &y).unwrap();
/// assert!((r - 0.7745966692414834).abs() < 1e-10);
/// ```
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> MathResult<f64> {
    if x.is_empty() || y.is_empty() {
        return Err(MathError::EmptyData(
            "pearson_correlation exige des vecteurs non vides".to_string(),
        ));
    }
    if x.len() != y.len() {
        return Err(MathError::InvalidParameter(format!(
            "les vecteurs doivent avoir la même longueur : {} ≠ {}",
            x.len(),
            y.len()
        )));
    }

    let _n = x.len() as f64;
    let mean_x = statistics::mean(x)?;
    let mean_y = statistics::mean(y)?;

    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        sum_xy += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }

    let denom = (sum_x2 * sum_y2).sqrt();
    if denom == 0.0 {
        return Err(MathError::InvalidParameter(
            "la corrélation n'est pas définie (variance nulle)".to_string(),
        ));
    }

    Ok(sum_xy / denom)
}

/// Calcule la matrice de corrélation complète pour un ensemble de variables.
///
/// # Entrées
/// - `data` : matrice de données (format plat : n_variables × n_observations).
///   Chaque variable est une suite consécutive de `n_observations` valeurs.
/// - `n_variables` : nombre de variables ;
/// - `n_observations` : nombre d'observations par variable.
///
/// # Sorties
/// Matrice de corrélation (format plat : n_variables × n_variables).
///
/// # Erreurs
/// - [`MathError::EmptyData`] si les données sont vides ;
/// - [`MathError::InvalidParameter`] si les dimensions sont incohérentes.
pub fn correlation_matrix(
    data: &[f64],
    n_variables: usize,
    n_observations: usize,
) -> MathResult<Vec<f64>> {
    if data.is_empty() {
        return Err(MathError::EmptyData(
            "correlation_matrix exige des données non vides".to_string(),
        ));
    }
    if data.len() != n_variables * n_observations {
        return Err(MathError::InvalidParameter(format!(
            "la taille des données ({}) ne correspond pas aux dimensions ({n_variables} × {n_observations})",
            data.len()
        )));
    }

    let mut matrix = vec![0.0; n_variables * n_variables];

    for i in 0..n_variables {
        for j in 0..n_variables {
            if i == j {
                matrix[i * n_variables + j] = 1.0;
            } else {
                let start_i = i * n_observations;
                let start_j = j * n_observations;
                let var_i = &data[start_i..start_i + n_observations];
                let var_j = &data[start_j..start_j + n_observations];
                matrix[i * n_variables + j] = pearson_correlation(var_i, var_j)?;
            }
        }
    }

    Ok(matrix)
}

/// Calcule les statistiques d'une matrice de corrélation.
///
/// # Entrées
/// - `matrix` : matrice de corrélation (format plat) ;
/// - `n_variables` : taille de la matrice (n × n).
///
/// # Sorties
/// Un [`CorrelationAnalysis`] contenant les statistiques calculées.
///
/// # Erreurs
/// - [`MathError::InvalidParameter`] si la matrice est vide ou de mauvaise taille.
pub fn analyze_correlation_matrix(
    matrix: &[f64],
    n_variables: usize,
) -> MathResult<CorrelationAnalysis> {
    if matrix.is_empty() || matrix.len() != n_variables * n_variables {
        return Err(MathError::InvalidParameter(format!(
            "la matrice doit être de taille n × n : {} ≠ {n_variables}²",
            matrix.len()
        )));
    }

    let mut sum_correlation = 0.0;
    let mut max_correlation = 0.0;
    let mut count_off_diagonal = 0;

    for i in 0..n_variables {
        for j in 0..n_variables {
            if i != j {
                let corr = matrix[i * n_variables + j];
                sum_correlation += corr.abs();
                if corr.abs() > max_correlation {
                    max_correlation = corr.abs();
                }
                count_off_diagonal += 1;
            }
        }
    }

    let mean_correlation = if count_off_diagonal > 0 {
        sum_correlation / count_off_diagonal as f64
    } else {
        0.0
    };

    Ok(CorrelationAnalysis {
        variable_count: n_variables,
        correlation_matrix: matrix.to_vec(),
        mean_correlation,
        max_correlation,
    })
}

/// Calcule la corrélation par blocs pour des données structurées.
///
/// # Entrées
/// - `data` : données complètes ;
/// - `block_size` : taille de chaque bloc.
///
/// # Sorties
/// Vecteur des corrélations moyennes par bloc.
///
/// # Erreurs
/// - [`MathError::EmptyData`] si les données sont vides ;
/// - [`MathError::InvalidParameter`] si la taille du bloc est invalide.
pub fn correlation_by_blocks(data: &[f64], block_size: usize) -> MathResult<Vec<f64>> {
    if data.is_empty() {
        return Err(MathError::EmptyData(
            "correlation_by_blocks exige des données non vides".to_string(),
        ));
    }
    if block_size == 0 || block_size > data.len() {
        return Err(MathError::InvalidParameter(format!(
            "la taille du bloc doit être dans [1, {}] : {block_size}",
            data.len()
        )));
    }

    let n_blocks = data.len() / block_size;
    let mut block_correlations = Vec::with_capacity(n_blocks);

    for i in 0..n_blocks {
        let start = i * block_size;
        let end = start + block_size;
        let block = &data[start..end];

        // Pour un bloc, on calcule la corrélation avec lui-même décalé d'un élément
        if block_size > 1 {
            let x = &block[..block_size - 1];
            let y = &block[1..];
            let corr = pearson_correlation(x, y)?;
            block_correlations.push(corr);
        } else {
            block_correlations.push(0.0);
        }
    }

    Ok(block_correlations)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pearson_correlation_empty_data() {
        let empty: &[f64] = &[];
        let data = [1.0, 2.0, 3.0];
        assert!(matches!(
            pearson_correlation(empty, &data),
            Err(MathError::EmptyData(_))
        ));
        assert!(matches!(
            pearson_correlation(&data, empty),
            Err(MathError::EmptyData(_))
        ));
    }

    #[test]
    fn pearson_correlation_different_lengths() {
        let x = [1.0, 2.0, 3.0];
        let y = [1.0, 2.0];
        assert!(matches!(
            pearson_correlation(&x, &y),
            Err(MathError::InvalidParameter(_))
        ));
    }

    #[test]
    fn pearson_correlation_known_values() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 5.0, 4.0, 5.0];
        let r = pearson_correlation(&x, &y).unwrap();
        assert!((r - 0.7745966692414834).abs() < 1e-10);
    }

    #[test]
    fn correlation_matrix_known_values() {
        // Deux variables parfaitement corrélées
        let data = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]; // var1, var2
        let matrix = correlation_matrix(&data, 2, 3).unwrap();
        assert_eq!(matrix.len(), 4);
        assert_eq!(matrix[0], 1.0); // corr(var1, var1)
        assert_eq!(matrix[3], 1.0); // corr(var2, var2)
        assert!((matrix[1] - 1.0).abs() < 1e-10); // corr(var1, var2)
        assert!((matrix[2] - 1.0).abs() < 1e-10); // corr(var2, var1)
    }

    #[test]
    fn analyze_correlation_matrix_known_values() {
        let matrix = [1.0, 0.5, 0.5, 1.0];
        let analysis = analyze_correlation_matrix(&matrix, 2).unwrap();
        assert_eq!(analysis.variable_count, 2);
        assert!((analysis.mean_correlation - 0.5).abs() < 1e-10);
        assert!((analysis.max_correlation - 0.5).abs() < 1e-10);
    }

    #[test]
    fn correlation_by_blocks_known_values() {
        // Données avec des blocs de taille 4 pour avoir assez d'éléments
        let data = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let correlations = correlation_by_blocks(&data, 4).unwrap();
        assert_eq!(correlations.len(), 3);
        // Chaque bloc [1,2,3,4] a une corrélation parfaite avec lui-même décalé
        for corr in correlations {
            assert!((corr - 1.0).abs() < 1e-10);
        }
    }
}
