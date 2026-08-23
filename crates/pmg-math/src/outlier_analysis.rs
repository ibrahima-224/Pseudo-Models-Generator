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

//! Analyse des outliers — détection par seuil et par quantile.
//!
//! Ce module fournit des fonctions pour détecter les valeurs aberrantes
//! dans un échantillon de données, conformément au Sprint 11 du plan de
//! développement.
//!
//! # Responsabilités
//!
//! - Détection d'outliers par seuil : |x| > kσ (k écart-types) ;
//! - Détection d'outliers par quantile : au-delà d'un quantile donné ;
//! - Statistiques sur les outliers détectés.
//!
//! # Conventions
//!
//! - Toutes les fonctions sont déterministes ;
//! - Les paramètres sont validés et retournent des erreurs typées (`MathError`);
//! - La documentation est en français.

use crate::error::{MathError, MathResult};
use crate::statistics;

/// Représente un outlier détecté dans les données.
#[derive(Debug, Clone, PartialEq)]
pub struct Outlier {
    /// Indice de la valeur dans la slice d'origine.
    pub index: usize,
    /// Valeur de l'outlier.
    pub value: f64,
    /// Type de détection (seuil ou quantile).
    pub detection_type: OutlierDetectionType,
}

/// Type de détection utilisé pour identifier un outlier.
#[derive(Debug, Clone, PartialEq)]
pub enum OutlierDetectionType {
    /// Détection par seuil : |x| > kσ.
    Threshold { k: f64 },
    /// Détection par quantile : au-delà du quantile q.
    Quantile { q: f64 },
}

/// Résultat de l'analyse d'outliers.
#[derive(Debug, Clone)]
pub struct OutlierAnalysis {
    /// Nombre total d'éléments analysés.
    pub total_elements: usize,
    /// Nombre d'outliers détectés.
    pub outlier_count: usize,
    /// Proportion d'outliers dans l'échantillon.
    pub outlier_ratio: f64,
    /// Liste des outliers détectés.
    pub outliers: Vec<Outlier>,
}

/// Détecte les outliers par seuil : |x| > kσ.
///
/// # Entrées
/// - `data` : slice non vide de données ;
/// - `k` : nombre d'écarts-types pour le seuil (k > 0).
///
/// # Sorties
/// Un [`OutlierAnalysis`] contenant les outliers détectés.
///
/// # Erreurs
/// - [`MathError::EmptyData`] si la slice est vide ;
/// - [`MathError::InvalidParameter`] si k ≤ 0.
///
/// # Exemple
///
/// ```
/// use pmg_math::outlier_analysis::{detect_by_threshold, OutlierDetectionType};
///
/// let data = [0.0, 1.0, 2.0, 3.0, 100.0];
/// let result = detect_by_threshold(&data, 2.0).unwrap();
/// assert_eq!(result.outlier_count, 1);
/// assert_eq!(result.outliers[0].value, 100.0);
/// ```
pub fn detect_by_threshold(data: &[f64], k: f64) -> MathResult<OutlierAnalysis> {
    if data.is_empty() {
        return Err(MathError::EmptyData(
            "detect_by_threshold exige une slice non vide".to_string(),
        ));
    }
    if k <= 0.0 {
        return Err(MathError::InvalidParameter(format!(
            "k doit être strictement positif : {k}"
        )));
    }

    let sigma = statistics::std_population(data)?;
    let threshold = k * sigma;
    let mut outliers = Vec::new();

    for (i, &value) in data.iter().enumerate() {
        if value.abs() > threshold {
            outliers.push(Outlier {
                index: i,
                value,
                detection_type: OutlierDetectionType::Threshold { k },
            });
        }
    }

    let outlier_count = outliers.len();
    let outlier_ratio = outlier_count as f64 / data.len() as f64;

    Ok(OutlierAnalysis {
        total_elements: data.len(),
        outlier_count,
        outlier_ratio,
        outliers,
    })
}

/// Détecte les outliers par quantile : au-delà du quantile q.
///
/// # Entrées
/// - `data` : slice non vide de données ;
/// - `q` : quantile supérieur (0.5 < q < 1.0).
///
/// # Sorties
/// Un [`OutlierAnalysis`] contenant les outliers détectés.
///
/// # Erreurs
/// - [`MathError::EmptyData`] si la slice est vide ;
/// - [`MathError::InvalidParameter`] si q n'est pas dans (0.5, 1.0).
///
/// # Exemple
///
/// ```
/// use pmg_math::outlier_analysis::{detect_by_quantile, OutlierDetectionType};
///
/// let data = [0.0, 1.0, 2.0, 3.0, 100.0];
/// let result = detect_by_quantile(&data, 0.9).unwrap();
/// assert_eq!(result.outlier_count, 1);
/// assert_eq!(result.outliers[0].value, 100.0);
/// ```
pub fn detect_by_quantile(data: &[f64], q: f64) -> MathResult<OutlierAnalysis> {
    if data.is_empty() {
        return Err(MathError::EmptyData(
            "detect_by_quantile exige une slice non vide".to_string(),
        ));
    }
    if q <= 0.5 || q >= 1.0 {
        return Err(MathError::InvalidParameter(format!(
            "q doit être dans (0.5, 1.0) : {q}"
        )));
    }

    let quantile_values = statistics::quantiles(data, &[q])?;
    let threshold = quantile_values[0];
    let mut outliers = Vec::new();

    for (i, &value) in data.iter().enumerate() {
        if value.abs() > threshold {
            outliers.push(Outlier {
                index: i,
                value,
                detection_type: OutlierDetectionType::Quantile { q },
            });
        }
    }

    let outlier_count = outliers.len();
    let outlier_ratio = outlier_count as f64 / data.len() as f64;

    Ok(OutlierAnalysis {
        total_elements: data.len(),
        outlier_count,
        outlier_ratio,
        outliers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_by_threshold_empty_data() {
        let empty: &[f64] = &[];
        assert!(matches!(
            detect_by_threshold(empty, 2.0),
            Err(MathError::EmptyData(_))
        ));
    }

    #[test]
    fn detect_by_threshold_invalid_k() {
        let data = [1.0, 2.0, 3.0];
        assert!(matches!(
            detect_by_threshold(&data, 0.0),
            Err(MathError::InvalidParameter(_))
        ));
        assert!(matches!(
            detect_by_threshold(&data, -1.0),
            Err(MathError::InvalidParameter(_))
        ));
    }

    #[test]
    fn detect_by_threshold_known_values() {
        let data = [0.0, 1.0, 2.0, 3.0, 100.0];
        let result = detect_by_threshold(&data, 2.0).unwrap();
        assert_eq!(result.total_elements, 5);
        assert_eq!(result.outlier_count, 1);
        assert_eq!(result.outliers[0].value, 100.0);
        assert_eq!(
            result.outliers[0].detection_type,
            OutlierDetectionType::Threshold { k: 2.0 }
        );
    }

    #[test]
    fn detect_by_quantile_empty_data() {
        let empty: &[f64] = &[];
        assert!(matches!(
            detect_by_quantile(empty, 0.9),
            Err(MathError::EmptyData(_))
        ));
    }

    #[test]
    fn detect_by_quantile_invalid_q() {
        let data = [1.0, 2.0, 3.0];
        assert!(matches!(
            detect_by_quantile(&data, 0.5),
            Err(MathError::InvalidParameter(_))
        ));
        assert!(matches!(
            detect_by_quantile(&data, 1.0),
            Err(MathError::InvalidParameter(_))
        ));
    }

    #[test]
    fn detect_by_quantile_known_values() {
        let data = [0.0, 1.0, 2.0, 3.0, 100.0];
        let result = detect_by_quantile(&data, 0.9).unwrap();
        assert_eq!(result.total_elements, 5);
        assert_eq!(result.outlier_count, 1);
        assert_eq!(result.outliers[0].value, 100.0);
        assert_eq!(
            result.outliers[0].detection_type,
            OutlierDetectionType::Quantile { q: 0.9 }
        );
    }
}
