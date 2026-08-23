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

//! Injection bas-rang : `W' = W + α·U·Vᵀ` avec `r ≪ min(m, n)`.
//!
//! Cette étape applique une composante bas-rang sur un tenseur **existant**
//! (contrairement à [`pmg_math::low_rank`] qui génère une matrice complète) :
//! les facteurs `U ∈ ℝ^{m×r}` et `V ∈ ℝ^{n×r}` sont tirés gaussiennement
//! (flux dérivé de seed, ordre canonique `U` puis `V` — même convention que
//! `pmg-math`), puis la contribution `α·UVᵀ` est ajoutée au tenseur.
//!
//! Conformité : `docs/documents/CAHIER DE PLAN DEVELOPPEMENT SPRINT_0_6.md`
//! étape 4.5 (paramètres `alpha`, `rank`, `seed`, `distribution`).

use pmg_math::error::MathError;
use pmg_math::low_rank::{effective_rank, generate_factors, low_rank_from_factors, LowRankSpec};
use pmg_math::rng::DeterministicRng;

use crate::error::{InjectorError, InjectorResult};

/// Paramètres de l'injection bas-rang sur un tenseur existant.
///
/// # Invariants (vérifiés par [`LowRankInjection::new`])
/// - `rank ≥ 1` et `rank ≤ min(m, n)` ;
/// - `alpha > 0` (amplitude de la composante) ;
/// - `uv_factor > 0` (écart-type des facteurs gaussiens).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LowRankInjection {
    /// Rang cible `r` de la composante.
    pub rank: usize,
    /// Amplitude α multipliant `UVᵀ`.
    pub alpha: f64,
    /// Écart-type des gaussiennes des facteurs `U`/`V`.
    pub uv_factor: f64,
}

impl LowRankInjection {
    /// Construit des paramètres valides pour des dimensions `m × n`.
    ///
    /// # Erreurs
    /// - [`InjectorError::InvalidPolicy`] si `alpha`/`uv_factor` invalides ;
    /// - [`InjectorError::InvalidTensor`] si `rank` hors `[1, min(m, n)]`.
    ///
    /// # Complexité
    /// O(1).
    pub fn new(
        rank: usize,
        alpha: f64,
        uv_factor: f64,
        m: usize,
        n: usize,
    ) -> InjectorResult<Self> {
        let spec = LowRankSpec::new(rank, alpha, uv_factor, m, n).map_err(map_rank_err)?;
        Ok(Self {
            rank: spec.rank,
            alpha: spec.alpha,
            uv_factor: spec.uv_factor,
        })
    }
}

/// Injecte la composante bas-rang `W' = W + α·U·Vᵀ` sur place.
///
/// # Entrées
/// - `buffer` : tenseur `m × n` (ligne par ligne), modifié en place ;
/// - `rows`, `cols` : dimensions de la matrice ;
/// - `injection` : rang, amplitude et facteur de variance ;
/// - `rng` : flux déterministe dérivé (domaine `"low_rank"`).
///
/// # Garanties
/// - déterministe : mêmes entrées ⇒ mêmes facteurs et même contribution ;
/// - la contribution ajoutée a un rang exactement ≤ `rank` ;
/// - les valeurs restent finies (facteurs gaussiens bornés par construction
///   de l'algorithme).
///
/// # Erreurs
/// - [`InjectorError::InvalidTensor`] si `buffer.len() != rows·cols` ou si
///   `rows == 0 || cols == 0` ;
/// - [`InjectorError::Math`] / [`InjectorError::InvalidTensor`] si le rang
///   est incohérent avec les dimensions.
///
/// # Complexité
/// O(rows·cols·rank) — matérialisation complète de la contribution.
pub fn inject_low_rank(
    buffer: &mut [f64],
    rows: usize,
    cols: usize,
    injection: &LowRankInjection,
    rng: &mut DeterministicRng,
) -> InjectorResult<()> {
    if rows == 0 || cols == 0 {
        return Err(InjectorError::InvalidTensor(format!(
            "dimensions nulles : rows={rows}, cols={cols}"
        )));
    }
    if buffer.len() != rows * cols {
        return Err(InjectorError::InvalidTensor(format!(
            "buffer de longueur {} ≠ rows·cols = {rows}·{cols}",
            buffer.len()
        )));
    }
    let spec = LowRankSpec::new(
        injection.rank,
        injection.alpha,
        injection.uv_factor,
        rows,
        cols,
    )
    .map_err(map_rank_err)?;
    let (u, v) = generate_factors(rng, rows, cols, &spec)?;
    let contribution = low_rank_from_factors(&u, &v, rows, cols, spec.alpha);
    for (b, c) in buffer.iter_mut().zip(contribution.iter()) {
        *b += c;
    }
    Ok(())
}

/// Génère une matrice complète `W = α·U·Vᵀ` (délégation `pmg-math`).
///
/// Utile pour construire un tenseur de base directement structuré, sans
/// passer par l'ajout sur un tenseur existant.
///
/// # Erreurs
/// [`InjectorError::Math`] si le rang est incohérent avec `m × n`.
pub fn generate_low_rank_matrix(
    rng: &mut DeterministicRng,
    rows: usize,
    cols: usize,
    rank: usize,
    alpha: f64,
) -> InjectorResult<Vec<f64>> {
    let spec = LowRankSpec::new(rank, alpha, 1.0, rows, cols).map_err(map_rank_err)?;
    pmg_math::low_rank::generate_low_rank(rng, rows, cols, &spec).map_err(Into::into)
}

/// Estime le rang effectif du buffer par énergie spectrale (délégation
/// `pmg-math`). `energy_ratio` est la fraction d'énergie à conserver.
///
/// # Erreurs
/// [`InjectorError::Math`] si le buffer est incohérent avec `m × n` ou si
/// `energy_ratio` est hors `[0, 1]`.
pub fn estimate_effective_rank(
    buffer: &[f64],
    rows: usize,
    cols: usize,
    energy_ratio: f64,
) -> InjectorResult<usize> {
    effective_rank(buffer, rows, cols, energy_ratio).map_err(Into::into)
}

/// Convertit une erreur de rang `pmg-math` en erreur injector cohérente :
/// rang nul / dépassement → [`InjectorError::InvalidTensor`] (dimensions),
/// paramètres invalides (alpha, uv_factor) → [`InjectorError::InvalidPolicy`].
fn map_rank_err(err: MathError) -> InjectorError {
    match err {
        MathError::InvalidRank(msg) => InjectorError::InvalidTensor(msg),
        MathError::InvalidParameter(msg) => InjectorError::InvalidPolicy(msg),
        other => InjectorError::Math(other),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        estimate_effective_rank, generate_low_rank_matrix, inject_low_rank, LowRankInjection,
    };
    use crate::error::InjectorError;
    use pmg_math::rng::{derive_sub_seed, DeterministicRng};

    fn rng_for(seed: [u8; 32]) -> DeterministicRng {
        DeterministicRng::from_seed(derive_sub_seed(&seed, "low_rank", 0))
    }

    fn base_seed() -> [u8; 32] {
        [17u8; 32]
    }

    #[test]
    fn injection_adds_contribution_in_place() {
        let mut buf = vec![1.0f64; 16]; // 4×4
        let inj = LowRankInjection::new(2, 0.5, 1.0, 4, 4).unwrap();
        inject_low_rank(&mut buf, 4, 4, &inj, &mut rng_for(base_seed())).unwrap();
        // Toutes les valeurs ont changé (contribution non nulle).
        assert!(buf.iter().any(|&x| (x - 1.0).abs() > 1e-9));
        assert!(buf.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn effective_rank_after_injection_is_bounded_by_rank_plus_base() {
        // Tenseur de base constant (rang 1) + contribution de rang r :
        // le rang effectif estimé reste borné (r + 1), bien inférieur à 4×4.
        let mut buf = vec![0.5f64; 64]; // 8×8
        let inj = LowRankInjection::new(2, 1.0, 1.0, 8, 8).unwrap();
        inject_low_rank(&mut buf, 8, 8, &inj, &mut rng_for(base_seed())).unwrap();
        let rank = estimate_effective_rank(&buf, 8, 8, 0.99).unwrap();
        assert!(rank <= 3, "rang effectif {rank} trop élevé");
    }

    #[test]
    fn low_rank_matrix_is_full_rank_bound() {
        let m = generate_low_rank_matrix(&mut rng_for(base_seed()), 6, 8, 3, 1.0).unwrap();
        assert_eq!(m.len(), 48);
        let rank = estimate_effective_rank(&m, 6, 8, 0.99).unwrap();
        assert!(rank <= 3, "rang effectif {rank} > rang cible 3");
        assert!(rank >= 2, "rang effectif {rank} trop faible");
    }

    #[test]
    fn rank_one_produces_rank_one_contribution() {
        let m = generate_low_rank_matrix(&mut rng_for(base_seed()), 5, 7, 1, 1.0).unwrap();
        let rank = estimate_effective_rank(&m, 5, 7, 0.999).unwrap();
        assert_eq!(rank, 1);
    }

    #[test]
    fn injection_is_deterministic() {
        let inj = LowRankInjection::new(2, 0.5, 1.0, 4, 4).unwrap();
        let mut a = vec![0.0f64; 16];
        let mut b = vec![0.0f64; 16];
        inject_low_rank(&mut a, 4, 4, &inj, &mut rng_for(base_seed())).unwrap();
        inject_low_rank(&mut b, 4, 4, &inj, &mut rng_for(base_seed())).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn invalid_rank_rejected() {
        // rank 0.
        assert!(matches!(
            LowRankInjection::new(0, 1.0, 1.0, 4, 4),
            Err(InjectorError::InvalidTensor(_))
        ));
        // rank > min(m, n).
        assert!(matches!(
            LowRankInjection::new(9, 1.0, 1.0, 4, 4),
            Err(InjectorError::InvalidTensor(_))
        ));
    }

    #[test]
    fn invalid_alpha_rejected() {
        assert!(matches!(
            LowRankInjection::new(2, 0.0, 1.0, 4, 4),
            Err(InjectorError::InvalidPolicy(_))
        ));
        assert!(matches!(
            LowRankInjection::new(2, -1.0, 1.0, 4, 4),
            Err(InjectorError::InvalidPolicy(_))
        ));
    }

    #[test]
    fn buffer_length_mismatch_rejected() {
        let inj = LowRankInjection::new(1, 1.0, 1.0, 4, 4).unwrap();
        let mut buf = vec![0.0f64; 15];
        assert!(matches!(
            inject_low_rank(&mut buf, 4, 4, &inj, &mut rng_for(base_seed())),
            Err(InjectorError::InvalidTensor(_))
        ));
    }

    #[test]
    fn rank_one_matrix_square_1x1() {
        // Matrice 1×1 : rang maximal = 1.
        let inj = LowRankInjection::new(1, 0.3, 1.0, 1, 1).unwrap();
        let mut buf = vec![0.0f64; 1];
        inject_low_rank(&mut buf, 1, 1, &inj, &mut rng_for(base_seed())).unwrap();
        assert!(buf[0].is_finite());
        // rank > min = 1 rejeté.
        assert!(LowRankInjection::new(2, 1.0, 1.0, 1, 1).is_err());
    }

    #[test]
    fn zero_dimensions_rejected() {
        let inj = LowRankInjection::new(1, 1.0, 1.0, 4, 4).unwrap();
        assert!(matches!(
            inject_low_rank(&mut [], 0, 4, &inj, &mut rng_for(base_seed())),
            Err(InjectorError::InvalidTensor(_))
        ));
    }
}
