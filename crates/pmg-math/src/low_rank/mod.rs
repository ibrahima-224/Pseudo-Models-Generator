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

//! Structures bas-rang : `W = α·U·Vᵀ`.
//!
//! Conformité : `docs/architecture/04-moteurs-math-injection-generation.md`
//! §4. `U ∈ ℝ^{m×r}`, `V ∈ ℝ^{n×r}` avec `r ≤ min(m, n)`, α = amplitude
//! contrôlée. Le produit par blocs de lignes ([`low_rank_block`]) garantit une
//! mémoire `O(bloc×n + m×r + n×r)`, jamais `O(m×n)` pour le calcul.

use crate::distribution::Distribution;
use crate::distributions::Normal;
use crate::error::{MathError, MathResult};
use crate::rng::DeterministicRng;

/// Spécification d'une composante bas-rang.
///
/// # Invariants (vérifiés par [`LowRankSpec::validate`])
/// - `rank ≥ 1` ;
/// - `rank ≤ min(m, n)` ;
/// - `alpha`, `uv_factor` finis et > 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LowRankSpec {
    /// Rang cible `r` (nombre de colonnes de `U`/`V`).
    pub rank: usize,
    /// Amplitude α multipliant le produit `UVᵀ`.
    pub alpha: f64,
    /// Facteur de variance des gaussiennes de `U` et `V` (écart-type).
    pub uv_factor: f64,
}

impl LowRankSpec {
    /// Construit une spécification valide pour des dimensions `m × n`.
    ///
    /// # Erreurs
    /// [`MathError::InvalidRank`] si `rank < 1` ou `rank > min(m, n)` ;
    /// [`MathError::InvalidParameter`] si `alpha`/`uv_factor` invalides.
    ///
    /// # Complexité
    /// O(1).
    pub fn new(rank: usize, alpha: f64, uv_factor: f64, m: usize, n: usize) -> MathResult<Self> {
        let spec = Self {
            rank,
            alpha,
            uv_factor,
        };
        spec.validate(m, n)?;
        Ok(spec)
    }

    /// Valide la spécification contre les dimensions `m × n`.
    ///
    /// # Erreurs
    /// [`MathError::InvalidRank`] si `rank` est hors bornes ;
    /// [`MathError::InvalidParameter`] si `alpha`/`uv_factor` invalides.
    pub fn validate(&self, m: usize, n: usize) -> MathResult<()> {
        let max_rank = m.min(n);
        if self.rank < 1 {
            return Err(MathError::InvalidRank(format!("rank {} < 1", self.rank)));
        }
        if self.rank > max_rank {
            return Err(MathError::InvalidRank(format!(
                "rank {} > min(m, n) = {max_rank}",
                self.rank
            )));
        }
        if !self.alpha.is_finite() || self.alpha <= 0.0 {
            return Err(MathError::InvalidParameter(format!(
                "alpha doit être fini et > 0, reçu {}",
                self.alpha
            )));
        }
        if !self.uv_factor.is_finite() || self.uv_factor <= 0.0 {
            return Err(MathError::InvalidParameter(format!(
                "uv_factor doit être fini et > 0, reçu {}",
                self.uv_factor
            )));
        }
        Ok(())
    }
}

/// Génère la composante bas-rang complète `W = α·U·Vᵀ` (`m × n`).
///
/// # Entrées
/// - `rng` : flux déterministe ;
/// - `m`, `n` : dimensions de la matrice cible ;
/// - `rank`, `alpha` : rang et amplitude (facteur de variance unitaire).
///
/// # Sorties
/// `m × n` valeurs (ligne par ligne).
///
/// # Complexité
/// O(m·n·r) — matérialisation complète (petites matrices ou tests).
///
/// # Erreurs
/// [`MathError::InvalidRank`] si la spécification est invalide pour `m × n`.
pub fn low_rank_contribution(
    rng: &mut DeterministicRng,
    m: usize,
    n: usize,
    rank: usize,
    alpha: f64,
) -> MathResult<Vec<f64>> {
    let spec = LowRankSpec::new(rank, alpha, 1.0, m, n)?;
    generate_low_rank(rng, m, n, &spec)
}

/// Génère `W = α·U·Vᵀ` à partir d'une spécification complète.
///
/// # Complexité
/// O(m·n·r).
pub fn generate_low_rank(
    rng: &mut DeterministicRng,
    m: usize,
    n: usize,
    spec: &LowRankSpec,
) -> MathResult<Vec<f64>> {
    let (u, v) = generate_factors(rng, m, n, spec)?;
    Ok(low_rank_from_factors(&u, &v, m, n, spec.alpha))
}

/// Génère les facteurs gaussiens `U` (`m × r`) et `V` (`n × r`).
///
/// Ordre de tirage canonique : **`U` puis `V`** (déterministe, indépendant de
/// l'ordre d'appel — chaque tenseur possède son propre flux dérivé).
///
/// # Sorties
/// `(u, v)` plats (ligne par ligne).
///
/// # Complexité
/// O((m + n)·r) — tirages gaussiens.
pub fn generate_factors(
    rng: &mut DeterministicRng,
    m: usize,
    n: usize,
    spec: &LowRankSpec,
) -> MathResult<(Vec<f64>, Vec<f64>)> {
    spec.validate(m, n)?;
    if m == 0 || n == 0 {
        return Err(MathError::InvalidParameter(format!(
            "dimensions nulles : m={m}, n={n}"
        )));
    }
    let r = spec.rank;
    let factor = spec.uv_factor;
    let mut normal = Normal::new(0.0, 1.0)?;
    let mut u = vec![0.0f64; m * r];
    for v in u.iter_mut() {
        *v = factor * normal.sample(rng);
    }
    let mut v = vec![0.0f64; n * r];
    for vv in v.iter_mut() {
        *vv = factor * normal.sample(rng);
    }
    Ok((u, v))
}

/// Calcule le produit complet `α·U·Vᵀ` à partir des facteurs.
///
/// # Complexité
/// O(m·n·r) — produit pleinement matérialisé.
pub fn low_rank_from_factors(u: &[f64], v: &[f64], m: usize, n: usize, alpha: f64) -> Vec<f64> {
    let r = u.len() / m;
    let mut w = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..r {
                acc += u[i * r + k] * v[j * r + k];
            }
            w[i * n + j] = alpha * acc;
        }
    }
    w
}

/// Calcule un **bloc de lignes** `W_block = α·U_block·Vᵀ` (streaming).
///
/// # Entrées
/// - `u` : facteur complet `m × r` ;
/// - `v` : facteur complet `n × r` ;
/// - `m`, `n` : dimensions totales ;
/// - `row_start`, `row_end` : bornes du bloc (`0 ≤ row_start < row_end ≤ m`) ;
/// - `alpha` : amplitude.
///
/// # Sorties
/// `(row_end − row_start) × n` valeurs.
///
/// # Complexité
/// O(bloc·n·r), mémoire `O(bloc×n + n×r)` — jamais `O(m×n)`.
///
/// # Invariant de streaming
/// La concaténation des blocs consécutifs (dans l'ordre) est **exactement**
/// égale à [`low_rank_from_factors`] avec les mêmes facteurs (testé).
///
/// # Erreurs
/// [`MathError::InvalidParameter`] si les facteurs sont incohérents ou les
/// bornes invalides.
pub fn low_rank_block(
    u: &[f64],
    v: &[f64],
    m: usize,
    n: usize,
    row_start: usize,
    row_end: usize,
    alpha: f64,
) -> MathResult<Vec<f64>> {
    if m == 0 || n == 0 {
        return Err(MathError::InvalidParameter(format!(
            "dimensions nulles : m={m}, n={n}"
        )));
    }
    if u.len() % m != 0 {
        return Err(MathError::InvalidParameter(format!(
            "u de longueur {} non multiple de m={m}",
            u.len()
        )));
    }
    let r = u.len() / m;
    if r < 1 {
        return Err(MathError::InvalidParameter(
            "rang nul dans les facteurs".into(),
        ));
    }
    if v.len() != n * r {
        return Err(MathError::InvalidParameter(format!(
            "v de longueur {} ≠ n·r = {}",
            v.len(),
            n * r
        )));
    }
    if row_start >= row_end || row_end > m {
        return Err(MathError::InvalidParameter(format!(
            "bornes de bloc invalides : [{row_start}, {row_end}) hors [0, {m})"
        )));
    }
    let rows = row_end - row_start;
    let mut w = vec![0.0f64; rows * n];
    for i in 0..rows {
        let gi = row_start + i;
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..r {
                acc += u[gi * r + k] * v[j * r + k];
            }
            w[i * n + j] = alpha * acc;
        }
    }
    Ok(w)
}

/// Estimation du **rang effectif** d'une matrice par énergie spectrale.
///
/// Méthode documentée et volontairement simple : les valeurs singulières sont
/// estimées par les racines carrées des valeurs propres de `W Wᵀ` (dimension
/// `m`, supposée petite) via la méthode de la puissance **avec déflation
/// exacte et ré-orthogonalisation de Gram-Schmidt** du vecteur de départ
/// (base canonique déterministe). Le rang effectif est le plus petit `k` tel
/// que `Σ_{i≤k} σᵢ² ≥ ratio · Σ σᵢ²`.
///
/// # Avertissement (limite documentée)
/// Approche **approximative** : précision limitée pour les valeurs singulières
/// très rapprochées ou les matrices de grande dimension `m`. Pour une
/// estimation robuste, utiliser `pmg-validate` (énergie spectrale, doc 6).
///
/// # Entrées
/// - `w` : matrice `m × n` (ligne par ligne) ;
/// - `m`, `n` : dimensions ;
/// - `energy_ratio` : fraction d'énergie à conserver (défaut 0.99).
///
/// # Sorties
/// Rang effectif estimé dans `[0, min(m, n)]` (0 pour une matrice nulle).
///
/// # Complexité
/// O(itérations · m² · n) — borné à 200 itérations par valeur singulière.
pub fn effective_rank(w: &[f64], m: usize, n: usize, energy_ratio: f64) -> MathResult<usize> {
    if m == 0 || n == 0 || w.len() != m * n {
        return Err(MathError::InvalidParameter(format!(
            "matrice invalide : m={m}, n={n}, len={}",
            w.len()
        )));
    }
    if !(0.0..=1.0).contains(&energy_ratio) {
        return Err(MathError::InvalidParameter(format!(
            "energy_ratio hors [0, 1] : {energy_ratio}"
        )));
    }
    // A = W Wᵀ (m×m, symétrique semi-définie positive).
    let mut a = vec![0.0f64; m * m];
    for i in 0..m {
        for j in 0..m {
            let mut acc = 0.0;
            for k in 0..n {
                acc += w[i * n + k] * w[j * n + k];
            }
            a[i * m + j] = acc;
        }
    }
    let singulars = power_iteration_spectrum(&mut a, m)?;
    let total: f64 = singulars.iter().map(|s| s * s).sum();
    if total == 0.0 {
        // Matrice nulle : rang effectif 0.
        return Ok(0);
    }
    let mut cum = 0.0;
    for (k, s) in singulars.iter().enumerate() {
        cum += s * s;
        if cum / total >= energy_ratio {
            return Ok(k + 1);
        }
    }
    Ok(singulars.len())
}

/// Spectre de valeurs singulières estimé de `W` via `W Wᵀ`.
///
/// Méthode de la puissance avec **déflation** : après chaque valeur propre
/// trouvée, sa contribution `λ q qᵀ` est retirée de `A`, et le vecteur de
/// départ suivant (base canonique) est ré-orthogonalisé par Gram-Schmidt
/// contre les vecteurs propres déjà trouvés.
///
/// # Complexité
/// O(200 · m² · m) — borné.
fn power_iteration_spectrum(a: &mut [f64], m: usize) -> MathResult<Vec<f64>> {
    let mut found: Vec<Vec<f64>> = Vec::new();
    let mut singulars = Vec::new();
    for step in 0..m {
        // Vecteur de départ déterministe : base canonique.
        let mut v = vec![0.0f64; m];
        if step < m {
            v[step] = 1.0;
        }
        for _ in 0..200 {
            // Ré-orthogonalisation contre les vecteurs déjà trouvés.
            for q in &found {
                let dot: f64 = v.iter().zip(q.iter()).map(|(x, y)| x * y).sum();
                for (x, y) in v.iter_mut().zip(q.iter()) {
                    *x -= dot * y;
                }
            }
            let mut next = vec![0.0f64; m];
            for i in 0..m {
                next[i] = (0..m).map(|j| a[i * m + j] * v[j]).sum();
            }
            let norm = next.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-14 {
                v = next;
                break;
            }
            for x in next.iter_mut() {
                *x /= norm;
            }
            v = next;
        }
        // Normalisation finale avec orthogonalisation.
        for q in &found {
            let dot: f64 = v.iter().zip(q.iter()).map(|(x, y)| x * y).sum();
            for (x, y) in v.iter_mut().zip(q.iter()) {
                *x -= dot * y;
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            // Plus d'énergie résiduelle.
            break;
        }
        for x in v.iter_mut() {
            *x /= norm;
        }
        // Quotient de Rayleigh : λ = vᵀ A v.
        let mut eigen = 0.0;
        for i in 0..m {
            let av: f64 = (0..m).map(|j| a[i * m + j] * v[j]).sum();
            eigen += v[i] * av;
        }
        if eigen <= 1e-14 {
            break;
        }
        singulars.push(eigen.sqrt());
        // Déflation : A ← A − λ v vᵀ.
        for i in 0..m {
            for j in 0..m {
                a[i * m + j] -= eigen * v[i] * v[j];
            }
        }
        found.push(v);
    }
    Ok(singulars)
}

/// Estime les valeurs singulières d'une matrice via la méthode de la puissance.
///
/// Cette fonction retourne les valeurs singulières estimées de la matrice `w`
/// de dimensions `m × n`, en utilisant la décomposition en valeurs singulières (SVD)
/// approximative via la méthode de la puissance avec déflation.
///
/// # Avertissement (limite documentée)
/// Approche **approximative** : précision limitée pour les valeurs singulières
/// très rapprochées ou les matrices de grande dimension `m`. Pour une
/// estimation robuste, utiliser `pmg-validate` (énergie spectrale, doc 6).
///
/// # Entrées
/// - `w` : matrice `m × n` (ligne par ligne) ;
/// - `m`, `n` : dimensions.
///
/// # Sorties
/// Un vecteur de valeurs singulières triées par ordre décroissant.
///
/// # Erreurs
/// [`MathError::InvalidParameter`] si la matrice est invalide.
pub fn singular_values(w: &[f64], m: usize, n: usize) -> MathResult<Vec<f64>> {
    if m == 0 || n == 0 || w.len() != m * n {
        return Err(MathError::InvalidParameter(format!(
            "matrice invalide : m={m}, n={n}, len={}",
            w.len()
        )));
    }

    // A = W Wᵀ (m×m, symétrique semi-définie positive).
    let mut a = vec![0.0f64; m * m];
    for i in 0..m {
        for j in 0..m {
            let mut acc = 0.0;
            for k in 0..n {
                acc += w[i * n + k] * w[j * n + k];
            }
            a[i * m + j] = acc;
        }
    }

    // Estimation des valeurs propres de A (qui sont les carrés des valeurs singulières)
    let eigenvalues = power_iteration_spectrum(&mut a, m)?;

    // Conversion en valeurs singulières (racine carrée des valeurs propres)
    let singulars: Vec<f64> = eigenvalues.iter().map(|e| e.sqrt()).collect();

    Ok(singulars)
}

#[cfg(test)]
mod tests;
