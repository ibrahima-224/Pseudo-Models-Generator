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

//! Helpers statistiques pour la validation des tenseurs.
//!
//! Ce module contient des fonctions utilitaires pour le calcul
//! des propriétés statistiques de base (moyenne, variance, écart-type).

/// Constante pour éviter la division par zéro dans l'erreur relative.
pub(crate) const EPSILON: f64 = 1e-10;

/// Calcule les statistiques de base en une seule passe.
///
/// # Description
/// Cette fonction calcule la moyenne, la variance et l'écart-type d'un
/// slice de données f64 en une seule passe pour optimiser les performances.
///
/// # Arguments
///
/// * `data` - Slice de données f64.
///
/// # Retour
///
/// Un tuple `(moyenne, variance, écart-type)`.
/// Si le slice est vide, retourne `(0.0, 0.0, 0.0)`.
pub fn compute_basic_stats(data: &[f64]) -> (f64, f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let n = data.len() as f64;
    let sum: f64 = data.iter().sum();
    let mean = sum / n;

    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;

    let std = variance.sqrt();

    (mean, variance, std)
}

/// Calcule l'erreur relative entre une valeur observée et une valeur cible.
///
/// # Formule
/// `E = |θ_obs - θ_target| / max(|θ_target|, ε)`
///
/// # Arguments
///
/// * `observed` - Valeur observée.
/// * `target` - Valeur cible.
///
/// # Retour
///
/// L'erreur relative (valeur positive).
pub fn relative_error(observed: f64, target: f64) -> f64 {
    let denominator = target.abs().max(EPSILON);
    (observed - target).abs() / denominator
}

/// Calcule la moyenne d'un slice de f64.
///
/// # Arguments
///
/// * `data` - Slice de données.
///
/// # Retour
///
/// La moyenne, ou `None` si le slice est vide.
pub fn calculate_mean(data: &[f64]) -> Option<f64> {
    if data.is_empty() {
        None
    } else {
        let sum: f64 = data.iter().sum();
        Some(sum / data.len() as f64)
    }
}

/// Calcule l'écart-type d'un slice de f64.
///
/// # Arguments
///
/// * `data` - Slice de données.
///
/// # Retour
///
/// L'écart-type, ou `None` si le slice est vide ou contient une seule valeur.
pub fn calculate_std(data: &[f64]) -> Option<f64> {
    if data.len() < 2 {
        None
    } else {
        let mean = calculate_mean(data)?;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
        Some(variance.sqrt())
    }
}
