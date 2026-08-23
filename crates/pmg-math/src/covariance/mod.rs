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

//! Covariance PSD et génération corrélée `X = LZ + μ`.
//!
//! Conformité : `docs/architecture/04-moteurs-math-injection-generation.md`
//! §3. `Σ = Σᵀ ⪰ 0` est exigée ; la factorisation `LLᵀ = Σ` (Cholesky) est
//! vérifiée à une tolérance ; toute matrice non PSD produit une erreur typée
//! [`MathError::NotPsd`] — **jamais** de pseudo-inverse ou de correction
//! silencieuse.

use crate::distribution::Distribution;
use crate::distributions::Normal;
use crate::error::{MathError, MathResult};
use crate::rng::DeterministicRng;

/// Tolérance relative de reconstruction `‖Σ − LLᵀ‖` pour valider la factorisation.
pub const CHOLESKY_TOLERANCE: f64 = 1e-9;

/// Factorisation de Cholesky `L` telle que `L Lᵀ = Σ` (matrice triangulaire
/// inférieure, `dim × dim`).
///
/// # Invariant
/// Vérifié par [`Covariance::reconstruction_error`] ≤ tolérance relative.
#[derive(Debug, Clone, PartialEq)]
pub struct Cholesky {
    /// Matrice triangulaire inférieure (stockée ligne par ligne).
    pub l: Vec<f64>,
    /// Dimension (nombre de lignes/colonnes de `Σ`).
    pub dim: usize,
}

/// Covariance cible validée : matrice PSD + sa factorisation de Cholesky.
///
/// # Contrat
/// - `Σ = Σᵀ` (symétrie vérifiée) ;
/// - `Σ ⪰ 0` (semi-définie positive, vérifiée par la factorisation) ;
/// - `L Lᵀ = Σ` à [`CHOLESKY_TOLERANCE`] près (reconstruction vérifiée).
#[derive(Debug, Clone)]
pub struct Covariance {
    sigma: Vec<f64>,
    cholesky: Cholesky,
}

impl Covariance {
    /// Construit une covariance PSD à partir d'une matrice carrée `dim × dim`
    /// stockée ligne par ligne.
    ///
    /// # Entrées
    /// - `sigma` : `dim²` valeurs, `sigma[i * dim + j]` ;
    /// - `dim` : dimension.
    ///
    /// # Erreurs
    /// - [`MathError::InvalidParameter`] : matrice non carrée ou dimensions nulles ;
    /// - [`MathError::NotPsd`] : asymétrie ou pivot négatif pendant le Cholesky
    ///   (matrice non semi-définie positive).
    ///
    /// # Complexité
    /// O(dim³) — factorisation puis reconstruction.
    ///
    /// # Limites
    /// La vérification de PSD est la factorisation elle-même : les pivots
    /// légèrement négatifs (bruit numérique) échouent explicitement.
    pub fn new(sigma: Vec<f64>, dim: usize) -> MathResult<Self> {
        validate_square(&sigma, dim)?;
        if !is_symmetric(&sigma, dim) {
            return Err(MathError::NotPsd(format!(
                "matrice {dim}×{dim} non symétrique : Σᵢⱼ ≠ Σⱼᵢ"
            )));
        }
        let l = cholesky_factor(&sigma, dim)?;
        let cholesky = Cholesky { l, dim };
        let recon_err = reconstruction_error(&cholesky, &sigma);
        let scale = frobenius_norm(&sigma);
        let rel = recon_err / scale.max(f64::MIN_POSITIVE);
        if rel > CHOLESKY_TOLERANCE {
            return Err(MathError::NotPsd(format!(
                "reconstruction LLᵀ ≠ Σ : erreur relative {rel:e} > {CHOLESKY_TOLERANCE:e}"
            )));
        }
        Ok(Self { sigma, cholesky })
    }

    /// Accès à la factorisation de Cholesky.
    pub fn cholesky(&self) -> &Cholesky {
        &self.cholesky
    }

    /// Accès à la matrice de covariance (lecture seule).
    pub fn sigma(&self) -> &[f64] {
        &self.sigma
    }

    /// Erreur relative de reconstruction `‖Σ − LLᵀ‖_F / ‖Σ‖_F`.
    pub fn reconstruction_error(&self) -> f64 {
        let recon_err = reconstruction_error(&self.cholesky, &self.sigma);
        recon_err / frobenius_norm(&self.sigma).max(f64::MIN_POSITIVE)
    }

    /// Construit une covariance **diagonale** `diag(variances)` (cas fréquent
    /// des structures simplifiées `Σ = D`, spécification doc 4 §3.1).
    ///
    /// # Erreurs
    /// [`MathError::InvalidParameter`] si une variance ≤ 0.
    ///
    /// # Complexité
    /// O(dim) — construction directe de `L = diag(√vᵢ)`.
    pub fn diagonal(variances: &[f64]) -> MathResult<Self> {
        if variances.is_empty() {
            return Err(MathError::InvalidParameter(
                "covariance diagonale vide".into(),
            ));
        }
        let dim = variances.len();
        let mut sigma = vec![0.0; dim * dim];
        let mut l = vec![0.0; dim * dim];
        for (i, &v) in variances.iter().enumerate() {
            if !v.is_finite() || v <= 0.0 {
                return Err(MathError::InvalidParameter(format!(
                    "variance diagonale {v} ≤ 0 à l'index {i}"
                )));
            }
            sigma[i * dim + i] = v;
            l[i * dim + i] = v.sqrt();
        }
        Ok(Self {
            sigma,
            cholesky: Cholesky { l, dim },
        })
    }

    /// Construit une matrice de corrélation PSD à partir d'un vecteur de
    /// corrélations `ρᵢⱼ` entre paires (`rho.len() = dim·(dim−1)/2`, ordre
    /// lexicographique `(i, j)` avec `i < j`).
    ///
    /// # Contrat
    /// La matrice résultante a une diagonale unité et est PSD **si** les
    /// corrélations fournies forment une matrice PSD (vérifiée par la
    /// factorisation). Une combinaison invalide produit [`MathError::NotPsd`].
    ///
    /// # Complexité
    /// O(dim³) — construction + Cholesky.
    pub fn from_pairwise_correlations(dim: usize, rho: &[f64]) -> MathResult<Self> {
        let expected = dim.saturating_mul(dim.saturating_sub(1)) / 2;
        if rho.len() != expected {
            return Err(MathError::InvalidParameter(format!(
                "attendu {expected} corrélations pour dim={dim}, reçu {}",
                rho.len()
            )));
        }
        let mut sigma = vec![0.0; dim * dim];
        for (i, v) in sigma.iter_mut().enumerate() {
            if i % (dim + 1) == 0 {
                *v = 1.0;
            }
        }
        let mut k = 0;
        for i in 0..dim {
            for j in (i + 1)..dim {
                let r = rho[k];
                if !r.is_finite() || !(-1.0..=1.0).contains(&r) {
                    return Err(MathError::InvalidParameter(format!(
                        "corrélation hors [−1, 1] : {r}"
                    )));
                }
                sigma[i * dim + j] = r;
                sigma[j * dim + i] = r;
                k += 1;
            }
        }
        Self::new(sigma, dim)
    }
}

/// Échantillonne `n` vecteurs corrélés `X = LZ + μ` avec `Z ~ N(0, I)`.
///
/// # Entrées
/// - `rng` : flux déterministe ;
/// - `means` : vecteur de moyennes (dimension `dim`) ;
/// - `cholesky` : factorisation de `Σ` ;
/// - `n` : nombre d'échantillons.
///
/// # Sorties
/// `n × dim` valeurs (ligne par ligne) — `sample[i * dim + j]`.
///
/// # Complexité
/// O(n · dim²) — produit matrice-vecteur `Lz` pour chaque échantillon.
///
/// # Limites
/// `n · dim` allocations bornées par la taille du buffer.
pub fn sample_correlated(
    rng: &mut DeterministicRng,
    means: &[f64],
    cholesky: &Cholesky,
    n: usize,
) -> MathResult<Vec<f64>> {
    let dim = cholesky.dim;
    if means.len() != dim {
        return Err(MathError::InvalidParameter(format!(
            "moyennes de longueur {} ≠ dimension {}",
            means.len(),
            dim
        )));
    }
    let mut normal = Normal::new(0.0, 1.0)?;
    let mut out = Vec::with_capacity(n * dim);
    let mut z = vec![0.0f64; dim];
    for _ in 0..n {
        for zj in z.iter_mut() {
            *zj = normal.sample(rng);
        }
        // x = L z + μ : chaque composante i = Σⱼ L[i][j] z[j] + μ[i].
        for (i, mean) in means.iter().enumerate().take(dim) {
            let mut acc = *mean;
            for (j, z_val) in z.iter().enumerate().take(i + 1) {
                acc += cholesky.l[i * dim + j] * *z_val;
            }
            out.push(acc);
        }
    }
    Ok(out)
}

/// Génère une matrice de corrélation PSD valide à partir d'une structure
/// simple « équicorrélée par blocs » : `Σ = (1−ρ) I + ρ J` sur chaque bloc.
///
/// # Entrées
/// - `dims` : tailles des blocs ;
/// - `rhos` : corrélation intra-bloc par bloc (chaque `ρ ∈ [−1, 1]`, avec
///   validité PSD vérifiée : `ρ ≥ −1/(d−1)` pour un bloc de taille `d`).
///
/// # Sorties
/// Matrice de corrélation `n×n` (n = Σ dims), PSD par construction.
///
/// # Complexité
/// O(n²) — remplissage direct, puis factorisation par l'appelant.
pub fn equicorrelation_matrix(dims: &[usize], rhos: &[f64]) -> MathResult<Vec<f64>> {
    if dims.is_empty() || dims.len() != rhos.len() {
        return Err(MathError::InvalidParameter(
            "dims et rhos doivent être non vides et de même longueur".into(),
        ));
    }
    let n: usize = dims.iter().sum();
    let mut sigma = vec![0.0f64; n * n];
    let mut offset = 0;
    for (block, (&d, &rho)) in dims.iter().zip(rhos.iter()).enumerate() {
        if d == 0 {
            return Err(MathError::InvalidParameter(format!(
                "bloc {block} de taille nulle"
            )));
        }
        if !rho.is_finite() || !(-1.0..=1.0).contains(&rho) {
            return Err(MathError::InvalidParameter(format!(
                "corrélation du bloc {block} hors [−1, 1] : {rho}"
            )));
        }
        // PSD : ρ ≥ −1/(d−1) pour une matrice équicorrélée de taille d.
        let min_rho = -1.0 / (d as f64 - 1.0);
        if rho < min_rho {
            return Err(MathError::NotPsd(format!(
                "bloc {block} : ρ = {rho} < {min_rho} (borne PSD équicorrélée)"
            )));
        }
        for i in 0..d {
            for j in 0..d {
                let v = if i == j { 1.0 } else { rho };
                sigma[(offset + i) * n + (offset + j)] = v;
            }
        }
        offset += d;
    }
    // Hors blocs : corrélation nulle (structure par blocs).
    Ok(sigma)
}

/// Valide qu'une matrice est carrée et de dimension positive.
fn validate_square(sigma: &[f64], dim: usize) -> MathResult<()> {
    if dim == 0 {
        return Err(MathError::InvalidParameter(
            "dimension nulle pour une matrice de covariance".into(),
        ));
    }
    if sigma.len() != dim * dim {
        return Err(MathError::InvalidParameter(format!(
            "matrice de {} valeurs ≠ carrée {dim}×{dim}",
            sigma.len()
        )));
    }
    Ok(())
}

/// Vérifie la symétrie `Σᵢⱼ = Σⱼᵢ` (tolérance absolue 1e-12 relative).
fn is_symmetric(sigma: &[f64], dim: usize) -> bool {
    for i in 0..dim {
        for j in 0..i {
            let a = sigma[i * dim + j];
            let b = sigma[j * dim + i];
            if (a - b).abs() > 1e-12 * (1.0 + a.abs().max(b.abs())) {
                return false;
            }
        }
    }
    true
}

/// Factorisation de Cholesky sans pivot : `L Lᵀ = Σ`.
///
/// # Erreurs
/// [`MathError::NotPsd`] dès qu'un pivot `≤ 0` apparaît (matrice non PSD).
///
/// # Complexité
/// O(dim³/3).
fn cholesky_factor(sigma: &[f64], dim: usize) -> MathResult<Vec<f64>> {
    let mut l = vec![0.0f64; dim * dim];
    for i in 0..dim {
        for j in 0..=i {
            let mut sum = sigma[i * dim + j];
            for k in 0..j {
                sum -= l[i * dim + k] * l[j * dim + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(MathError::NotPsd(format!(
                        "pivot diagonal {i} = {sum} ≤ 0 — matrice non semi-définie positive"
                    )));
                }
                l[i * dim + j] = sum.sqrt();
            } else {
                l[i * dim + j] = sum / l[j * dim + j];
            }
        }
    }
    Ok(l)
}

/// Erreur absolue de reconstruction `‖Σ − LLᵀ‖_F` (norme de Frobenius).
///
/// # Complexité
/// O(dim³).
fn reconstruction_error(cholesky: &Cholesky, sigma: &[f64]) -> f64 {
    let dim = cholesky.dim;
    let mut acc = 0.0;
    for i in 0..dim {
        for j in 0..dim {
            // (LLᵀ)[i][j] = Σₖ L[i][k] L[j][k], k ≤ min(i, j).
            let mut recon = 0.0;
            let kmax = i.min(j);
            for k in 0..=kmax {
                recon += cholesky.l[i * dim + k] * cholesky.l[j * dim + k];
            }
            let diff = recon - sigma[i * dim + j];
            acc += diff * diff;
        }
    }
    acc.sqrt()
}

/// Norme de Frobenius d'une matrice carrée.
fn frobenius_norm(sigma: &[f64]) -> f64 {
    sigma.iter().map(|v| v * v).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests;
