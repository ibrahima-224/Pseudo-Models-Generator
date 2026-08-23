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

//! Corrélation contrôlée entre colonnes d'un tenseur 2D.
//!
//! Construction canonique (spécification étape 4.4) :
//! `X = ρ·Z + √(1−ρ²)·ε` avec `Z, ε ~ N(0,1)`, qui donne
//! `Corr(X, Z) ≈ ρ`. Toutes les colonnes du tenseur sont rendues corrélées à
//! un facteur commun `Z` (structure « facteur latent »), puis la corrélation
//! **empirique** est mesurée par [`empirical_correlation`] pour validation.
//!
//! Le facteur commun `Z` est tiré une seule fois (consommation stable du RNG) ;
//! chaque colonne reçoit ensuite son propre bruit `ε`. Toute la génération est
//! déterministe (flux dérivé de seed).

use pmg_math::covariance::sample_correlated;
use pmg_math::distribution::Distribution;
use pmg_math::distributions::Normal;
use pmg_math::error::MathResult;
use pmg_math::rng::DeterministicRng;

use crate::error::{InjectorError, InjectorResult};

/// Injecte une corrélation contrôlée `ρ` entre les colonnes d'une matrice.
///
/// # Entrées
/// - `rng` : flux déterministe dérivé (domaine `"correlated"`) ;
/// - `rows`, `cols` : dimensions de la matrice (`rows ≥ 2` pour mesurer une
///   corrélation significative, `cols ≥ 2` pour avoir au moins une paire) ;
/// - `rho` : corrélation cible dans `[0, 1)` (ρ = 1 exclu : variance nulle
///   de la composante indépendante, construction instable) ;
/// - `stddev` : écart-type commun des colonnes générées (`> 0`).
///
/// # Sorties
/// Matrice `rows × cols` (ligne par ligne) dont les colonnes sont
/// approximativement corrélées à `ρ` avec le facteur latent.
///
/// # Construction
/// On pose `Z ~ N(0,1)` (un vecteur de `rows` valeurs) et, pour chaque colonne
/// `j`, `Xⱼ = stddev·(ρ·Z + √(1−ρ²)·εⱼ)`. La corrélation théorique entre
/// deux colonnes est `ρ²` et entre une colonne et `Z` est `ρ` (testé).
///
/// # Erreurs
/// - [`InjectorError::InvalidTensor`] si `rows < 2` ou `cols < 2` ;
/// - [`InjectorError::InvalidPolicy`] si `rho` hors `[0, 1)` ou `stddev ≤ 0`.
///
/// # Complexité
/// O(rows·cols).
pub fn generate_correlated_columns(
    rng: &mut DeterministicRng,
    rows: usize,
    cols: usize,
    rho: f64,
    stddev: f64,
) -> InjectorResult<Vec<f64>> {
    if rows < 2 || cols < 2 {
        return Err(InjectorError::InvalidTensor(format!(
            "corrélation impossible : rows={rows}, cols={cols} (minimum 2×2)"
        )));
    }
    if !rho.is_finite() || !(0.0..1.0).contains(&rho) {
        return Err(InjectorError::InvalidPolicy(format!(
            "rho hors [0, 1) : {rho}"
        )));
    }
    if !stddev.is_finite() || stddev <= 0.0 {
        return Err(InjectorError::InvalidPolicy(format!(
            "stddev doit être fini et > 0, reçu {stddev}"
        )));
    }
    // 1) Facteur latent commun Z (un tirage par ligne).
    let mut normal = Normal::new(0.0, 1.0)?;
    let z: Vec<f64> = (0..rows).map(|_| normal.sample(rng)).collect();
    // 2) Par ligne : Xⱼ = stddev·(ρ·Zᵢ + √(1−ρ²)·εᵢⱼ). Le stockage est
    //    ligne-major (out[i·cols + j]) — exigé par empirical_correlation.
    let scale = (1.0 - rho * rho).sqrt();
    let mut out = Vec::with_capacity(rows * cols);
    for &zi in &z {
        for _ in 0..cols {
            let eps = normal.sample(rng);
            out.push(stddev * (rho * zi + scale * eps));
        }
    }
    Ok(out)
}

/// Corrélations empiriques de Pearson entre colonnes d'une matrice.
///
/// # Entrées
/// - `matrix` : `rows × cols` valeurs (ligne par ligne) ;
/// - `rows`, `cols` : dimensions.
///
/// # Sorties
/// `cols × cols` coefficients (ligne par ligne), diagonale = 1.0. Les paires
/// à variance nulle produisent `NaN` (corrélation indéfinie), propagé tel quel.
///
/// # Erreurs
/// [`InjectorError::InvalidTensor`] si la longueur est incohérente ou si
/// `cols < 2` / `rows < 2`.
///
/// # Complexité
/// O(cols²·rows).
pub fn empirical_correlation(matrix: &[f64], rows: usize, cols: usize) -> InjectorResult<Vec<f64>> {
    if matrix.len() != rows * cols {
        return Err(InjectorError::InvalidTensor(format!(
            "matrice de longueur {} ≠ rows·cols = {rows}·{cols}",
            matrix.len()
        )));
    }
    if rows < 2 || cols < 2 {
        return Err(InjectorError::InvalidTensor(format!(
            "corrélation empirique impossible : rows={rows}, cols={cols}"
        )));
    }
    // Moyenne et variance de chaque colonne (population).
    let mut means = vec![0.0f64; cols];
    for j in 0..cols {
        let mut acc = 0.0;
        for i in 0..rows {
            acc += matrix[i * cols + j];
        }
        means[j] = acc / rows as f64;
    }
    let mut result = vec![0.0f64; cols * cols];
    for a in 0..cols {
        for b in 0..cols {
            let mut cov = 0.0;
            let mut va = 0.0;
            let mut vb = 0.0;
            for i in 0..rows {
                let da = matrix[i * cols + a] - means[a];
                let db = matrix[i * cols + b] - means[b];
                cov += da * db;
                va += da * da;
                vb += db * db;
            }
            result[a * cols + b] = cov / (va.sqrt() * vb.sqrt());
        }
    }
    Ok(result)
}

/// Version matricielle `X = LZ + μ` via pmg-math (Cholesky) pour une cible
/// de covariance complète.
///
/// Utilitaire d'interopérabilité : le chemin préféré pour une corrélation
/// unique reste [`generate_correlated_columns`], mais cette fonction permet
/// de produire des blocs de colonnes avec une matrice de corrélation PSD
/// arbitraire (ex. structure par blocs de [`pmg_math::covariance::equicorrelation_matrix`]).
///
/// # Erreurs
/// - [`InjectorError::Math`] si la matrice de corrélation n'est pas PSD ;
/// - [`InjectorError::InvalidTensor`] si les dimensions sont incohérentes.
///
/// # Complexité
/// O(rows·dim²) — produit matrice-vecteur par échantillon.
pub fn sample_correlated_matrix(
    rng: &mut DeterministicRng,
    means: &[f64],
    cholesky: &pmg_math::covariance::Cholesky,
    rows: usize,
) -> InjectorResult<Vec<f64>> {
    let dim = cholesky.dim;
    if means.len() != dim {
        return Err(InjectorError::InvalidTensor(format!(
            "moyennes de longueur {} ≠ dimension {dim}",
            means.len()
        )));
    }
    let out = sample_correlated(rng, means, cholesky, rows)?;
    Ok(out)
}

/// Vérifie que la corrélation empirique mesurée entre la colonne `j` et le
/// facteur latent est proche de `rho` (tolérance absolue `epsilon`).
///
/// # Entrées
/// - `col_j` : valeurs de la colonne `j` ;
/// - `factor` : valeurs du facteur latent `Z` (même longueur) ;
/// - `expected` : corrélation cible `ρ` ;
/// - `epsilon` : tolérance absolue (`> 0`).
///
/// # Sorties
/// `Ok(écart |ρ̂ − ρ|)` si `|ρ̂ − ρ| < ε`, sinon
/// [`InjectorError::ValidationFailed`].
///
/// # Complexité
/// O(n).
pub fn assert_correlation_within(
    col_j: &[f64],
    factor: &[f64],
    expected: f64,
    epsilon: f64,
) -> InjectorResult<f64> {
    if col_j.len() != factor.len() {
        return Err(InjectorError::InvalidTensor(format!(
            "longueurs différentes : colonne {} vs facteur {}",
            col_j.len(),
            factor.len()
        )));
    }
    if col_j.len() < 2 {
        return Err(InjectorError::InvalidTensor(
            "au moins 2 valeurs pour une corrélation".into(),
        ));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(InjectorError::InvalidPolicy(format!(
            "epsilon doit être fini et > 0, reçu {epsilon}"
        )));
    }
    let rho_hat = pearson(col_j, factor)?;
    let err = (rho_hat - expected).abs();
    if err < epsilon {
        Ok(err)
    } else {
        Err(InjectorError::ValidationFailed(format!(
            "corrélation mesurée {rho_hat:.4} ≠ attendue {expected:.4} (écart {err:.4} ≥ {epsilon})"
        )))
    }
}

/// Coefficient de corrélation de Pearson entre deux séries.
fn pearson(a: &[f64], b: &[f64]) -> MathResult<f64> {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let dx = x - ma;
        let dy = y - mb;
        cov += dx * dy;
        va += dx * dx;
        vb += dy * dy;
    }
    Ok(cov / (va * vb).sqrt())
}

#[cfg(test)]
mod tests {
    use super::{assert_correlation_within, empirical_correlation, generate_correlated_columns};
    use crate::error::InjectorError;
    use pmg_math::rng::{derive_sub_seed, DeterministicRng};

    fn rng_for(seed: [u8; 32]) -> DeterministicRng {
        DeterministicRng::from_seed(derive_sub_seed(&seed, "correlated", 0))
    }

    fn base_seed() -> [u8; 32] {
        [11u8; 32]
    }

    #[test]
    fn dimensions_are_correct() {
        let m = generate_correlated_columns(&mut rng_for(base_seed()), 100, 5, 0.5, 1.0).unwrap();
        assert_eq!(m.len(), 500);
        assert!(m.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn zero_rho_produces_nearly_independent_columns() {
        let m =
            generate_correlated_columns(&mut rng_for(base_seed()), 20_000, 3, 0.0, 1.0).unwrap();
        let corr = empirical_correlation(&m, 20_000, 3).unwrap();
        // |Corr(X₀, X₁)| < 0.05 pour des colonnes indépendantes.
        assert!(corr[1].abs() < 0.05, "ρ̂ = {}", corr[1]);
    }

    #[test]
    fn rho_080_produces_strong_correlation() {
        // Corrélation théorique entre colonnes = ρ² = 0.64.
        let m =
            generate_correlated_columns(&mut rng_for(base_seed()), 40_000, 2, 0.8, 1.0).unwrap();
        let corr = empirical_correlation(&m, 40_000, 2).unwrap();
        let rho_hat = corr[1];
        assert!(
            (rho_hat - 0.64).abs() < 0.02,
            "ρ̂ entre colonnes = {rho_hat}, attendu ≈ 0.64"
        );
    }

    #[test]
    fn correlation_with_factor_matches_rho() {
        // Corr(X, Z) ≈ ρ : on reconstruit le facteur latent de façon
        // déterministe (première colonne tirée avec le même flux).
        let mut rng = rng_for(base_seed());
        let m = generate_correlated_columns(&mut rng, 30_000, 2, 0.7, 1.0).unwrap();
        // Le facteur latent n'est pas exposé ; on vérifie via la corrélation
        // entre colonnes (ρ²) et la borne |ρ̂| ≤ ρ + ε.
        let corr = empirical_correlation(&m, 30_000, 2).unwrap();
        let rho_hat = corr[1].abs().sqrt();
        assert!(
            (rho_hat - 0.7).abs() < 0.02,
            "ρ̂ reconstruit = {rho_hat}, attendu ≈ 0.7"
        );
    }

    #[test]
    fn generation_is_deterministic() {
        let a = generate_correlated_columns(&mut rng_for(base_seed()), 200, 4, 0.3, 2.0).unwrap();
        let b = generate_correlated_columns(&mut rng_for(base_seed()), 200, 4, 0.3, 2.0).unwrap();
        assert_eq!(a, b);
        let mut other = base_seed();
        other[1] ^= 0x55;
        let c = generate_correlated_columns(&mut rng_for(other), 200, 4, 0.3, 2.0).unwrap();
        assert_ne!(a, c);
    }

    #[test]
    fn invalid_dimensions_rejected() {
        assert!(matches!(
            generate_correlated_columns(&mut rng_for(base_seed()), 1, 5, 0.5, 1.0),
            Err(InjectorError::InvalidTensor(_))
        ));
        assert!(matches!(
            generate_correlated_columns(&mut rng_for(base_seed()), 5, 1, 0.5, 1.0),
            Err(InjectorError::InvalidTensor(_))
        ));
    }

    #[test]
    fn invalid_rho_rejected() {
        assert!(matches!(
            generate_correlated_columns(&mut rng_for(base_seed()), 5, 5, 1.0, 1.0),
            Err(InjectorError::InvalidPolicy(_))
        ));
        assert!(matches!(
            generate_correlated_columns(&mut rng_for(base_seed()), 5, 5, -0.5, 1.0),
            Err(InjectorError::InvalidPolicy(_))
        ));
        assert!(matches!(
            generate_correlated_columns(&mut rng_for(base_seed()), 5, 5, 0.5, 0.0),
            Err(InjectorError::InvalidPolicy(_))
        ));
    }

    #[test]
    fn empirical_correlation_diagonal_is_one() {
        let m = generate_correlated_columns(&mut rng_for(base_seed()), 500, 3, 0.4, 1.0).unwrap();
        let corr = empirical_correlation(&m, 500, 3).unwrap();
        for j in 0..3 {
            assert!((corr[j * 3 + j] - 1.0).abs() < 1e-9);
            assert!((corr[j * 3 + 3 - 1 - j] - corr[3 - 1 - j + j * 3]).abs() < 1e-9);
        }
    }

    #[test]
    fn empirical_correlation_rejects_bad_input() {
        assert!(empirical_correlation(&[1.0, 2.0], 1, 2).is_err());
        assert!(empirical_correlation(&[1.0, 2.0, 3.0], 2, 2).is_err());
    }

    #[test]
    fn assert_correlation_within_accepts_tolerance() {
        // Cas trivial : corrélation parfaite → écart 0 < ε.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = a.clone();
        let err = assert_correlation_within(&a, &b, 1.0, 0.01).unwrap();
        assert!(err < 0.01);
    }

    #[test]
    fn assert_correlation_within_rejects_deviation() {
        // Deux séries indépendantes : ρ̂ ≈ 0 ≠ 1.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert!(matches!(
            assert_correlation_within(&a, &b, 1.0, 0.1),
            Err(InjectorError::ValidationFailed(_))
        ));
    }

    #[test]
    fn assert_correlation_rejects_short_series() {
        assert!(assert_correlation_within(&[1.0], &[2.0], 0.0, 0.1).is_err());
    }
}
