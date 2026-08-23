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

//! Analyse bas-rang — concentration d'énergie matricielle E_r.
//!
//! Ce module fournit des fonctions pour analyser la structure bas-rang
//! d'une matrice, conformément au Sprint 11 du plan de développement.
//!
//! # Responsabilités
//!
//! - Calcul de la décomposition en valeurs singulières (SVD) ;
//! - Calcul de la concentration d'énergie E_r pour différents rangs r ;
//! - Estimation du rang effectif d'une matrice ;
//! - Analyse de la décroissance des valeurs singulières.
//!
//! # Conventions
//!
//! - Toutes les fonctions sont déterministes ;
//! - Les paramètres sont validés et retournent des erreurs typées (`MathError`);
//! - La documentation est en français.

use crate::error::{MathError, MathResult};

/// Résultat de l'analyse bas-rang d'une matrice.
#[derive(Debug, Clone)]
pub struct LowRankAnalysis {
    /// Nombre de lignes de la matrice.
    pub rows: usize,
    /// Nombre de colonnes de la matrice.
    pub cols: usize,
    /// Nombre de valeurs singulières calculées.
    pub singular_value_count: usize,
    /// Valeurs singulières (triées par ordre décroissant).
    pub singular_values: Vec<f64>,
    /// Concentration d'énergie E_r pour différents rangs r.
    pub energy_concentration: Vec<f64>,
    /// Rang effectif (premier r tel que E_r > seuil, par défaut 0.9).
    pub effective_rank: usize,
    /// Seuil d'énergie utilisé pour le rang effectif.
    pub energy_threshold: f64,
}

/// Calcule les valeurs singulières d'une matrice (format plat).
///
/// # Entrées
/// - `matrix` : matrice de données (format plat : rows × cols) ;
/// - `rows` : nombre de lignes ;
/// - `cols` : nombre de colonnes.
///
/// # Sorties
/// Vecteur des valeurs singulières triées par ordre décroissant.
///
/// # Erreurs
/// - [`MathError::EmptyData`] si la matrice est vide ;
/// - [`MathError::InvalidParameter`] si les dimensions sont incohérentes.
///
/// # Note
/// Cette implémentation utilise une méthode simplifiée (produit matriciel)
/// pour des raisons de dépendances. Pour une implémentation précise,
/// une bibliothèque linéaire algébrique dédiée serait recommandée.
pub fn compute_singular_values(matrix: &[f64], rows: usize, cols: usize) -> MathResult<Vec<f64>> {
    if matrix.is_empty() {
        return Err(MathError::EmptyData(
            "compute_singular_values exige une matrice non vide".to_string(),
        ));
    }
    if matrix.len() != rows * cols {
        return Err(MathError::InvalidParameter(format!(
            "la taille de la matrice ({}) ne correspond pas aux dimensions ({rows} × {cols})",
            matrix.len()
        )));
    }

    // Pour une implémentation simple, on utilise les valeurs propres de A^T A
    // A^T A est de taille cols × cols
    let mut ata = vec![0.0; cols * cols];

    for i in 0..cols {
        for j in 0..cols {
            let mut sum = 0.0;
            for k in 0..rows {
                sum += matrix[k * cols + i] * matrix[k * cols + j];
            }
            ata[i * cols + j] = sum;
        }
    }

    // Calcul des valeurs propres de A^T A (méthode de Jacobi simplifiée)
    // Pour des matrices petites, on utilise une diagonalisation par rotations
    let eigenvalues = compute_eigenvalues_symmetric(&ata, cols)?;

    // Les valeurs singulières sont les racines carrées des valeurs propres
    let mut singular_values: Vec<f64> = eigenvalues
        .iter()
        .map(|&x| if x > 0.0 { x.sqrt() } else { 0.0 })
        .collect();

    // Tri par ordre décroissant
    singular_values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    Ok(singular_values)
}

/// Calcule les valeurs propres d'une matrice symétrique (méthode de Jacobi).
///
/// # Entrées
/// - `matrix` : matrice symétrique (format plat : n × n) ;
/// - `n` : taille de la matrice.
///
/// # Sorties
/// Vecteur des valeurs propres.
///
/// # Erreurs
/// - [`MathError::InvalidParameter`] si la matrice n'est pas symétrique.
fn compute_eigenvalues_symmetric(matrix: &[f64], n: usize) -> MathResult<Vec<f64>> {
    if matrix.len() != n * n {
        return Err(MathError::InvalidParameter(format!(
            "la matrice doit être de taille n × n : {} ≠ {n}²",
            matrix.len()
        )));
    }

    // Vérification de symétrie
    for i in 0..n {
        for j in 0..n {
            let diff = (matrix[i * n + j] - matrix[j * n + i]).abs();
            if diff > 1e-10 {
                return Err(MathError::InvalidParameter(
                    "la matrice n'est pas symétrique".to_string(),
                ));
            }
        }
    }

    // Méthode de Jacobi pour les matrices symétriques
    let mut a = matrix.to_vec();
    let mut v = vec![0.0; n * n];

    // Initialisation de v comme matrice identité
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    // Itérations de Jacobi
    for _ in 0..100 {
        // Recherche du plus grand élément hors diagonale
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;

        for i in 0..n {
            for j in i + 1..n {
                let val = a[i * n + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-10 {
            break;
        }

        // Calcul de l'angle de rotation
        let theta = if (a[p * n + p] - a[q * n + q]).abs() < 1e-10 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * ((2.0 * a[p * n + q]) / (a[p * n + p] - a[q * n + q])).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Application de la rotation
        let mut a_new = a.clone();
        let mut v_new = v.clone();

        for i in 0..n {
            a_new[i * n + p] = c * a[i * n + p] + s * a[i * n + q];
            a_new[i * n + q] = -s * a[i * n + p] + c * a[i * n + q];
            v_new[i * n + p] = c * v[i * n + p] + s * v[i * n + q];
            v_new[i * n + q] = -s * v[i * n + p] + c * v[i * n + q];
        }

        for j in 0..n {
            a_new[p * n + j] = c * a[p * n + j] + s * a[q * n + j];
            a_new[q * n + j] = -s * a[p * n + j] + c * a[q * n + j];
        }

        a = a_new;
        v = v_new;
    }

    // Les valeurs propres sont sur la diagonale
    let mut eigenvalues = Vec::with_capacity(n);
    for i in 0..n {
        eigenvalues.push(a[i * n + i]);
    }

    Ok(eigenvalues)
}

/// Calcule la concentration d'énergie E_r pour différents rangs r.
///
/// # Entrées
/// - `singular_values` : valeurs singulières triées par ordre décroissant ;
/// - `max_rank` : rang maximum à considerer (0 = toutes les valeurs).
///
/// # Sorties
/// Vecteur des concentrations d'énergie E_r pour r = 1, 2, ..., max_rank.
///
/// # Définition
/// E_r = (Σ_{i=1}^r σ_i²) / (Σ_{i=1}^n σ_i²)
///
/// où σ_i sont les valeurs singulières et n le nombre total de valeurs singulières.
pub fn compute_energy_concentration(
    singular_values: &[f64],
    max_rank: usize,
) -> MathResult<Vec<f64>> {
    if singular_values.is_empty() {
        return Err(MathError::EmptyData(
            "compute_energy_concentration exige des valeurs singulières non vides".to_string(),
        ));
    }

    let total_energy: f64 = singular_values.iter().map(|&x| x * x).sum();
    if total_energy == 0.0 {
        return Err(MathError::InvalidParameter(
            "l'énergie totale est nulle".to_string(),
        ));
    }

    let max_rank = if max_rank == 0 {
        singular_values.len()
    } else {
        max_rank.min(singular_values.len())
    };

    let mut energy_concentration = Vec::with_capacity(max_rank);
    let mut cumulative_energy = 0.0;

    for value in singular_values.iter().take(max_rank) {
        cumulative_energy += value * value;
        energy_concentration.push(cumulative_energy / total_energy);
    }

    Ok(energy_concentration)
}

/// Estime le rang effectif d'une matrice selon un seuil d'énergie.
///
/// # Entrées
/// - `singular_values` : valeurs singulières triées par ordre décroissant ;
/// - `threshold` : seuil d'énergie (par défaut 0.9 pour 90%).
///
/// # Sorties
/// Premier rang r tel que E_r > threshold.
///
/// # Erreurs
/// - [`MathError::EmptyData`] si les valeurs singulières sont vides ;
/// - [`MathError::InvalidParameter`] si le seuil n'est pas dans (0, 1].
pub fn estimate_effective_rank(singular_values: &[f64], threshold: f64) -> MathResult<usize> {
    if singular_values.is_empty() {
        return Err(MathError::EmptyData(
            "estimate_effective_rank exige des valeurs singulières non vides".to_string(),
        ));
    }
    if threshold <= 0.0 || threshold > 1.0 {
        return Err(MathError::InvalidParameter(format!(
            "le seuil doit être dans (0, 1] : {threshold}"
        )));
    }

    let total_energy: f64 = singular_values.iter().map(|&x| x * x).sum();
    if total_energy == 0.0 {
        return Err(MathError::InvalidParameter(
            "l'énergie totale est nulle".to_string(),
        ));
    }

    let mut cumulative_energy = 0.0;
    for (r, &sigma) in singular_values.iter().enumerate() {
        cumulative_energy += sigma * sigma;
        if cumulative_energy / total_energy > threshold {
            return Ok(r + 1);
        }
    }

    Ok(singular_values.len())
}

/// Analyse complète d'une matrice pour sa structure bas-rang.
///
/// # Entrées
/// - `matrix` : matrice de données (format plat : rows × cols) ;
/// - `rows` : nombre de lignes ;
/// - `cols` : nombre de colonnes ;
/// - `energy_threshold` : seuil d'énergie pour le rang effectif (par défaut 0.9).
///
/// # Sorties
/// Un [`LowRankAnalysis`] contenant les résultats de l'analyse.
pub fn analyze_low_rank(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    energy_threshold: f64,
) -> MathResult<LowRankAnalysis> {
    let singular_values = compute_singular_values(matrix, rows, cols)?;
    let energy_concentration = compute_energy_concentration(&singular_values, 0)?;
    let effective_rank = estimate_effective_rank(&singular_values, energy_threshold)?;

    Ok(LowRankAnalysis {
        rows,
        cols,
        singular_value_count: singular_values.len(),
        singular_values,
        energy_concentration,
        effective_rank,
        energy_threshold,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_singular_values_empty_matrix() {
        let empty: &[f64] = &[];
        assert!(matches!(
            compute_singular_values(empty, 0, 0),
            Err(MathError::EmptyData(_))
        ));
    }

    #[test]
    fn compute_singular_values_invalid_dimensions() {
        let matrix = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert!(matches!(
            compute_singular_values(&matrix, 2, 4), // 6 ≠ 2×4
            Err(MathError::InvalidParameter(_))
        ));
    }

    #[test]
    fn compute_singular_values_identity_matrix() {
        let matrix = [1.0, 0.0, 0.0, 1.0]; // 2×2 identité
        let sv = compute_singular_values(&matrix, 2, 2).unwrap();
        assert_eq!(sv.len(), 2);
        assert!((sv[0] - 1.0).abs() < 1e-10);
        assert!((sv[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn compute_energy_concentration_empty() {
        let empty: &[f64] = &[];
        assert!(matches!(
            compute_energy_concentration(empty, 0),
            Err(MathError::EmptyData(_))
        ));
    }

    #[test]
    fn compute_energy_concentration_known_values() {
        let sv = [3.0, 2.0, 1.0]; // Valeurs singulières
        let energy = compute_energy_concentration(&sv, 0).unwrap();
        assert_eq!(energy.len(), 3);
        // E1 = 9/(9+4+1) = 9/14 ≈ 0.6429
        assert!((energy[0] - 9.0 / 14.0).abs() < 1e-10);
        // E2 = (9+4)/14 = 13/14 ≈ 0.9286
        assert!((energy[1] - 13.0 / 14.0).abs() < 1e-10);
        // E3 = 1.0
        assert!((energy[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn estimate_effective_rank_empty() {
        let empty: &[f64] = &[];
        assert!(matches!(
            estimate_effective_rank(empty, 0.9),
            Err(MathError::EmptyData(_))
        ));
    }

    #[test]
    fn estimate_effective_rank_invalid_threshold() {
        let sv = [3.0, 2.0, 1.0];
        assert!(matches!(
            estimate_effective_rank(&sv, 0.0),
            Err(MathError::InvalidParameter(_))
        ));
        assert!(matches!(
            estimate_effective_rank(&sv, 1.5),
            Err(MathError::InvalidParameter(_))
        ));
    }

    #[test]
    fn estimate_effective_rank_known_values() {
        let sv = [3.0, 2.0, 1.0]; // E1 ≈ 0.64, E2 ≈ 0.93, E3 = 1.0
        assert_eq!(estimate_effective_rank(&sv, 0.9).unwrap(), 2);
        assert_eq!(estimate_effective_rank(&sv, 0.95).unwrap(), 3);
    }
}
