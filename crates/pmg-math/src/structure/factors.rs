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

//! Générateur de facteurs pour la décomposition bas-rang L = UVᵀ.
//!
//! Conformité : `docs/architecture/04-moteurs-math-injection-generation.md` §5.2.
//! Ce module génère les matrices U et V utilisées dans la décomposition bas-rang.
//! Il utilise les distributions du Sprint 7 pour la génération des facteurs.
//!
//! ## Propriétés
//!
//! - U ∈ ℝ^{m×r}, V ∈ ℝ^{n×r} avec r ≤ min(m, n) ;
//! - Éléments générés selon une distribution normale centrée réduite ;
//! - Reproductibilité garantie par le RNG déterministe.

use crate::distribution::Distribution;
use crate::distributions::Normal;
use crate::error::{MathError, MathResult};
use crate::rng::DeterministicRng;

/// Générateur de facteurs pour la décomposition bas-rang.
///
/// Génère les matrices U et V pour L = UVᵀ.
#[derive(Debug, Clone)]
pub struct FactorGenerator {
    /// Rang cible r (nombre de colonnes de U et V).
    rank: usize,
    /// Écart-type des éléments de U et V.
    std_dev: f64,
}

impl FactorGenerator {
    /// Crée un nouveau générateur de facteurs.
    ///
    /// # Entrées
    /// - `rank` : rang cible r ;
    /// - `std_dev` : écart-type des éléments (défaut 1.0).
    ///
    /// # Erreurs
    /// [`MathError::InvalidParameter`] si `rank == 0` ou `std_dev <= 0`.
    pub fn new(rank: usize, std_dev: f64) -> MathResult<Self> {
        if rank == 0 {
            return Err(MathError::InvalidParameter(
                "le rang doit être supérieur à 0".into(),
            ));
        }
        if !std_dev.is_finite() || std_dev <= 0.0 {
            return Err(MathError::InvalidParameter(format!(
                "l'écart-type doit être fini et > 0, reçu {std_dev}"
            )));
        }
        Ok(Self { rank, std_dev })
    }

    /// Génère la matrice U (m × r).
    ///
    /// # Entrées
    /// - `rng` : flux déterministe ;
    /// - `m` : nombre de lignes.
    ///
    /// # Sorties
    /// Vecteur plat de taille m × r.
    pub fn generate_u(&self, rng: &mut DeterministicRng, m: usize) -> MathResult<Vec<f64>> {
        if m == 0 {
            return Err(MathError::InvalidParameter(
                "le nombre de lignes doit être > 0".into(),
            ));
        }
        let mut normal = Normal::new(0.0, self.std_dev)?;
        let mut u = vec![0.0; m * self.rank];
        for element in u.iter_mut() {
            *element = normal.sample(rng);
        }
        Ok(u)
    }

    /// Génère la matrice V (n × r).
    ///
    /// # Entrées
    /// - `rng` : flux déterministe ;
    /// - `n` : nombre de lignes.
    ///
    /// # Sorties
    /// Vecteur plat de taille n × r.
    pub fn generate_v(&self, rng: &mut DeterministicRng, n: usize) -> MathResult<Vec<f64>> {
        if n == 0 {
            return Err(MathError::InvalidParameter(
                "le nombre de lignes doit être > 0".into(),
            ));
        }
        let mut normal = Normal::new(0.0, self.std_dev)?;
        let mut v = vec![0.0; n * self.rank];
        for element in v.iter_mut() {
            *element = normal.sample(rng);
        }
        Ok(v)
    }

    /// Génère les deux matrices U et V.
    ///
    /// # Entrées
    /// - `rng` : flux déterministe ;
    /// - `m`, `n` : dimensions.
    ///
    /// # Sorties
    /// Tuple (U, V) sous forme de vecteurs plats.
    pub fn generate_both(
        &self,
        rng: &mut DeterministicRng,
        m: usize,
        n: usize,
    ) -> MathResult<(Vec<f64>, Vec<f64>)> {
        let u = self.generate_u(rng, m)?;
        let v = self.generate_v(rng, n)?;
        Ok((u, v))
    }

    /// Retourne le rang cible.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Retourne l'écart-type utilisé.
    pub fn std_dev(&self) -> f64 {
        self.std_dev
    }
}

/// Calcule le produit matriciel L = α * U * Vᵀ.
///
/// # Entrées
/// - `u` : matrice U (m × r) ;
/// - `v` : matrice V (n × r) ;
/// - `m`, `n` : dimensions ;
/// - `alpha` : facteur d'amplitude.
///
/// # Sorties
/// Matrice L (m × n) sous forme de vecteur plat.
///
/// # Complexité
/// O(m × n × r).
pub fn matrix_product(
    u: &[f64],
    v: &[f64],
    m: usize,
    n: usize,
    alpha: f64,
) -> MathResult<Vec<f64>> {
    let r = u.len() / m;
    if u.len() != m * r {
        return Err(MathError::InvalidParameter(format!(
            "U de longueur {} ≠ m × r = {}",
            u.len(),
            m * r
        )));
    }
    if v.len() != n * r {
        return Err(MathError::InvalidParameter(format!(
            "V de longueur {} ≠ n × r = {}",
            v.len(),
            n * r
        )));
    }
    if m == 0 || n == 0 {
        return Err(MathError::InvalidParameter("dimensions nulles".into()));
    }

    let mut l = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..r {
                acc += u[i * r + k] * v[j * r + k];
            }
            l[i * n + j] = alpha * acc;
        }
    }
    Ok(l)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::DeterministicRng;

    #[test]
    fn factor_generator_new_valid() {
        let gen = FactorGenerator::new(5, 1.0);
        assert!(gen.is_ok());
    }

    #[test]
    fn factor_generator_new_invalid_rank() {
        let gen = FactorGenerator::new(0, 1.0);
        assert!(gen.is_err());
    }

    #[test]
    fn factor_generator_new_invalid_std_dev() {
        let gen = FactorGenerator::new(5, -1.0);
        assert!(gen.is_err());
    }

    #[test]
    fn factor_generator_generate_u() {
        let gen = FactorGenerator::new(3, 1.0).unwrap();
        let mut rng = DeterministicRng::from_seed([42u8; 32]);
        let u = gen.generate_u(&mut rng, 4).unwrap();
        assert_eq!(u.len(), 4 * 3);
        for &x in &u {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn factor_generator_generate_v() {
        let gen = FactorGenerator::new(3, 1.0).unwrap();
        let mut rng = DeterministicRng::from_seed([42u8; 32]);
        let v = gen.generate_v(&mut rng, 5).unwrap();
        assert_eq!(v.len(), 5 * 3);
        for &x in &v {
            assert!(x.is_finite());
        }
    }

    #[test]
    fn factor_generator_generate_both() {
        let gen = FactorGenerator::new(2, 1.0).unwrap();
        let mut rng = DeterministicRng::from_seed([42u8; 32]);
        let (u, v) = gen.generate_both(&mut rng, 3, 4).unwrap();
        assert_eq!(u.len(), 3 * 2);
        assert_eq!(v.len(), 4 * 2);
    }

    #[test]
    fn matrix_product_basic() {
        let u = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let v = vec![5.0, 6.0, 7.0, 8.0]; // 2x2
        let l = matrix_product(&u, &v, 2, 2, 1.0).unwrap();
        assert_eq!(l.len(), 4);
        // Vérification manuelle : L = U * Vᵀ
        // U = [[1,2],[3,4]], V = [[5,6],[7,8]], Vᵀ = [[5,7],[6,8]]
        // L[0,0] = 1*5 + 2*6 = 17
        // L[0,1] = 1*7 + 2*8 = 23
        // L[1,0] = 3*5 + 4*6 = 39
        // L[1,1] = 3*7 + 4*8 = 53
        assert!((l[0] - 17.0).abs() < 1e-10);
        assert!((l[1] - 23.0).abs() < 1e-10);
        assert!((l[2] - 39.0).abs() < 1e-10);
        assert!((l[3] - 53.0).abs() < 1e-10);
    }

    #[test]
    fn matrix_product_with_alpha() {
        let u = vec![1.0, 0.0, 0.0, 1.0]; // Identité 2x2
        let v = vec![1.0, 0.0, 0.0, 1.0]; // Identité 2x2
        let l = matrix_product(&u, &v, 2, 2, 2.0).unwrap();
        assert!((l[0] - 2.0).abs() < 1e-10);
        assert!((l[1] - 0.0).abs() < 1e-10);
        assert!((l[2] - 0.0).abs() < 1e-10);
        assert!((l[3] - 2.0).abs() < 1e-10);
    }
}
