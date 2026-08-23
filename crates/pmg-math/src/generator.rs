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

//! Générateur de base pour des valeurs pseudo-aléatoires.
//!
//! Ce module fournit des fonctions simples pour générer des valeurs
//! pseudo-aléatoires selon différentes distributions, en utilisant le
//! RNG déterministe existant (`DeterministicRng`).
//!
//! # Objectifs
//!
//! - **Simplicité** : fonctions pures, sans état global ;
//! - **Rapidité** : algorithms efficaces, pas d'allocation inutile ;
//! - **Déterminisme** : même seed ⇒ mêmes valeurs (reproductibilité stricte).
//!
//! Conformité : `docs/architecture/04-moteurs-math-injection-generation.md`
//! §1 (reproductibilité) et §2 (distributions).

use crate::error::{MathError, MathResult};
use crate::rng::DeterministicRng;

/// Génère `n` valeurs according à la loi normale `N(μ, σ²)`.
///
/// # Formule
///
/// Pour chaque `i` :
/// ```text
/// x_i = μ + σ * z_i
/// ```
/// où `z_i ~ N(0,1)` est tiré via la méthode de Box-Muller (voir ci-dessous).
///
/// # Paramètres
///
/// - `mu` : espérance (μ) ;
/// - `sigma` : écart-type (σ > 0) ;
/// - `n` : nombre de valeurs à générer ;
/// - `rng` : générateur déterministe (état interne modifié).
///
/// # Retourne
///
/// Un vecteur de `n` valeurs `f64`.
///
/// # Erreurs
///
/// Retourne `Err` si `sigma <= 0` (paramètre invalide).
///
/// # Complexité
///
/// O(n) en temps et espace.
pub fn generate_normal(
    mu: f64,
    sigma: f64,
    n: usize,
    rng: &mut DeterministicRng,
) -> MathResult<Vec<f64>> {
    if sigma <= 0.0 {
        return Err(MathError::InvalidParameter("sigma doit être > 0".into()));
    }

    let mut values = Vec::with_capacity(n);
    for _ in 0..n {
        let z = standard_normal_sample(rng);
        values.push(mu + sigma * z);
    }
    Ok(values)
}

/// Génère `n` valeurs according à la loi uniforme `U(a, b)`.
///
/// # Formule
///
/// Pour chaque `i` :
/// ```text
/// x_i = a + (b - a) * u_i
/// ```
/// où `u_i ~ U(0,1)`.
///
/// # Paramètres
///
/// - `a` : borne inférieure ;
/// - `b` : borne supérieure (`b > a`) ;
/// - `n` : nombre de valeurs à générer ;
/// - `rng` : générateur déterministe.
///
/// # Retourne
///
/// Un vecteur de `n` valeurs `f64`.
///
/// # Erreurs
///
/// Retourne `Err` si `b <= a`.
pub fn generate_uniform(
    a: f64,
    b: f64,
    n: usize,
    rng: &mut DeterministicRng,
) -> MathResult<Vec<f64>> {
    if b <= a {
        return Err(MathError::InvalidParameter("b doit être > a".into()));
    }

    let mut values = Vec::with_capacity(n);
    for _ in 0..n {
        let u = rng.next_f64();
        values.push(a + (b - a) * u);
    }
    Ok(values)
}

/// Échantillonne une valeur according à la loi normale standard `N(0,1)`.
///
/// Utilise la méthode de Box-Muller pour convertir des uniformes en normales.
/// Retourne un seul échantillon (pas de rejet).
///
/// # Complexité
///
/// O(1) (deux appels au RNG, opérations trigonométriques constantes).
fn standard_normal_sample(rng: &mut DeterministicRng) -> f64 {
    // Méthode de Box-Muller (pas de rejet, simple et efficace).
    let u1 = rng.next_f64();
    let u2 = rng.next_f64();

    // Éviter log(0) qui donnerait -∞.
    let r = (-2.0 * (1.0 - u1).max(1e-300).ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;

    r * theta.cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_normal_basic() {
        let mut rng = DeterministicRng::from_seed([42u8; 32]);
        let values = generate_normal(0.0, 1.0, 10, &mut rng).unwrap();
        assert_eq!(values.len(), 10);
        // Toutes les valeurs doivent être finies.
        for v in &values {
            assert!(v.is_finite(), "valeur non finie : {v}");
        }
    }

    #[test]
    fn generate_normal_deterministic() {
        let mut rng1 = DeterministicRng::from_seed([42u8; 32]);
        let mut rng2 = DeterministicRng::from_seed([42u8; 32]);
        let v1 = generate_normal(0.0, 1.0, 100, &mut rng1).unwrap();
        let v2 = generate_normal(0.0, 1.0, 100, &mut rng2).unwrap();
        assert_eq!(v1, v2);
    }

    #[test]
    fn generate_normal_different_seeds() {
        let mut rng1 = DeterministicRng::from_seed([42u8; 32]);
        let mut rng2 = DeterministicRng::from_seed([43u8; 32]);
        let v1 = generate_normal(0.0, 1.0, 100, &mut rng1).unwrap();
        let v2 = generate_normal(0.0, 1.0, 100, &mut rng2).unwrap();
        assert_ne!(v1, v2);
    }

    #[test]
    fn generate_normal_zero_sigma_rejected() {
        let mut rng = DeterministicRng::from_seed([42u8; 32]);
        let err_zero = generate_normal(0.0, 0.0, 10, &mut rng).unwrap_err();
        let err_neg = generate_normal(0.0, -1.0, 10, &mut rng).unwrap_err();

        // Vérifie que les erreurs sont bien de type MathError::InvalidParameter
        assert!(
            matches!(err_zero, MathError::InvalidParameter(_)),
            "erreur pour sigma=0 doit être InvalidParameter"
        );
        assert!(
            matches!(err_neg, MathError::InvalidParameter(_)),
            "erreur pour sigma=-1 doit être InvalidParameter"
        );
    }

    #[test]
    fn generate_normal_zero_elements() {
        let mut rng = DeterministicRng::from_seed([42u8; 32]);
        let values = generate_normal(0.0, 1.0, 0, &mut rng).unwrap();
        assert!(values.is_empty());
    }

    #[test]
    fn generate_normal_mean_and_stddev_approximation() {
        // Vérifie que les statistiques empiriques sont dans les bornes attendues.
        let mut rng = DeterministicRng::from_seed([42u8; 32]);
        let n = 100_000;
        let values = generate_normal(5.0, 2.0, n, &mut rng).unwrap();

        let mean = values.iter().sum::<f64>() / n as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let stddev = variance.sqrt();

        // Bornes larges pour éviter les faux négatifs.
        assert!(
            (mean - 5.0).abs() < 0.1,
            "moyenne empirique {mean} trop éloignée de 5.0"
        );
        assert!(
            (stddev - 2.0).abs() < 0.1,
            "écart-type empirique {stddev} trop éloigné de 2.0"
        );
    }

    #[test]
    fn generate_uniform_basic() {
        let mut rng = DeterministicRng::from_seed([42u8; 32]);
        let values = generate_uniform(0.0, 1.0, 10, &mut rng).unwrap();
        assert_eq!(values.len(), 10);
        for v in &values {
            assert!(*v >= 0.0 && *v < 1.0, "valeur hors bornes : {v}");
        }
    }

    #[test]
    fn generate_uniform_deterministic() {
        let mut rng1 = DeterministicRng::from_seed([42u8; 32]);
        let mut rng2 = DeterministicRng::from_seed([42u8; 32]);
        let v1 = generate_uniform(0.0, 1.0, 100, &mut rng1).unwrap();
        let v2 = generate_uniform(0.0, 1.0, 100, &mut rng2).unwrap();
        assert_eq!(v1, v2);
    }

    #[test]
    fn generate_uniform_invalid_bounds() {
        let mut rng = DeterministicRng::from_seed([42u8; 32]);
        let err_eq = generate_uniform(1.0, 1.0, 10, &mut rng).unwrap_err();
        let err_inv = generate_uniform(1.0, 0.0, 10, &mut rng).unwrap_err();

        // Vérifie que les erreurs sont bien de type MathError::InvalidParameter
        assert!(
            matches!(err_eq, MathError::InvalidParameter(_)),
            "erreur pour a=b doit être InvalidParameter"
        );
        assert!(
            matches!(err_inv, MathError::InvalidParameter(_)),
            "erreur pour b<a doit être InvalidParameter"
        );
    }

    #[test]
    fn generate_uniform_different_ranges() {
        let mut rng = DeterministicRng::from_seed([42u8; 32]);
        let values = generate_uniform(-5.0, 5.0, 1000, &mut rng).unwrap();
        for v in &values {
            assert!(*v >= -5.0 && *v < 5.0, "valeur hors bornes : {v}");
        }
    }
}
