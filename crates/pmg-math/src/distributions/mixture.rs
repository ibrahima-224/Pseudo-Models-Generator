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

//! Mélange de distributions : `X ~ Σ πᵢ fᵢ`.
//!
//! Échantillonnage : tirage catégoriel par les poids `πᵢ`, puis tirage de la
//! composante sélectionnée. Densité : somme pondérée des densités.
//!
//! # Validation des poids
//! Les poids doivent être dans `[0, 1]` et leur somme égale à 1 à la
//! tolérance documentée [`WEIGHT_TOLERANCE`]. Une violation produit une
//! erreur typée (jamais de renormalisation silencieuse).

use crate::distribution::Distribution;
use crate::error::{MathError, MathResult};
use crate::rng::DeterministicRng;

/// Tolérance sur la somme des poids d'un mélange (spécification doc 4 §2.2 :
/// « mixture Σπ = 1 »).
///
/// # Cas limites
///
/// - La somme des poids doit être égale à `1.0` à cette tolérance près.
/// - Si `|somme - 1.0| > WEIGHT_TOLERANCE`, une erreur `MathError::InvalidMixtureWeights` est levée.
/// - Cette tolérance évite les erreurs d'arrondi flottant lors de la sommation.
/// - Valeur recommandée : `1e-9` (suffisant pour la précision `f64`).
pub const WEIGHT_TOLERANCE: f64 = 1e-9;

/// Mélange de distributions pondérées.
///
/// # Invariants (vérifiés à la construction)
/// - au moins une composante ;
/// - chaque poids dans `[0, 1]` et fini ;
/// - somme des poids = 1 à [`WEIGHT_TOLERANCE`] près.
///
/// # Cas limites
///
/// - **Poids nuls** : Acceptés (composante jamais sélectionnée).
/// - **Poids égaux** : Chaque composante a la même probabilité.
/// - **Grand nombre de composantes** : Peut impacter les performances.
/// - **Poids très petits** (< `WEIGHT_TOLERANCE`) : Acceptés, mais composante rarement sélectionnée.
pub struct Mixture {
    components: Vec<(f64, Box<dyn Distribution>)>,
}

impl Mixture {
    /// Construit un mélange et valide les poids.
    ///
    /// # Erreurs
    /// [`MathError::InvalidMixtureWeights`] si les poids sont invalides.
    ///
    /// # Complexité
    /// O(C) — validation + copie des poids.
    pub fn new(components: Vec<(f64, Box<dyn Distribution>)>) -> MathResult<Self> {
        validate_weights(&components)?;
        Ok(Self { components })
    }

    /// Nombre de composantes.
    pub fn len(&self) -> usize {
        self.components.len()
    }

    /// Vrai si le mélange est vide (impossible par construction).
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Poids des composantes (copie).
    pub fn weights(&self) -> Vec<f64> {
        self.components.iter().map(|(w, _)| *w).collect()
    }

    /// Sélectionne l'index d'une composante selon les poids (tirage catégoriel).
    ///
    /// # Complexité
    /// O(C) — parcours cumulatif des poids.
    pub fn pick_component(&self, rng: &mut DeterministicRng) -> usize {
        let u = rng.next_f64();
        let mut acc = 0.0;
        for (i, (w, _)) in self.components.iter().enumerate() {
            acc += w;
            if u < acc {
                return i;
            }
        }
        // Sécurité : u ≈ 1 (borné par arrondi) → dernière composante.
        self.components.len() - 1
    }
}

/// Valide les poids d'un mélange : non vides, finis, dans `[0, 1]`, somme = 1.
///
/// # Erreurs
/// [`MathError::InvalidMixtureWeights`] avec un message explicite.
pub fn validate_weights(components: &[(f64, Box<dyn Distribution>)]) -> MathResult<()> {
    if components.is_empty() {
        return Err(MathError::InvalidMixtureWeights(
            "au moins une composante est requise".into(),
        ));
    }
    let mut sum = 0.0;
    for (i, (w, _)) in components.iter().enumerate() {
        if !w.is_finite() || *w < 0.0 || *w > 1.0 {
            return Err(MathError::InvalidMixtureWeights(format!(
                "poids {w} hors de [0, 1] à l'index {i}"
            )));
        }
        sum += w;
    }
    if (sum - 1.0).abs() > WEIGHT_TOLERANCE {
        return Err(MathError::InvalidMixtureWeights(format!(
            "somme des poids {sum} ≠ 1 (tolérance {WEIGHT_TOLERANCE})"
        )));
    }
    Ok(())
}

impl Distribution for Mixture {
    /// Tirage catégoriel puis tirage de la composante sélectionnée.
    ///
    /// # Complexité
    /// O(C) — sélection + tirage composante.
    fn sample(&mut self, rng: &mut DeterministicRng) -> f64 {
        let idx = self.pick_component(rng);
        self.components[idx].1.sample(rng)
    }

    /// Densité : somme pondérée `Σ πᵢ fᵢ(x)`.
    ///
    /// # Complexité
    /// O(C).
    fn pdf(&self, x: f64) -> f64 {
        self.components.iter().map(|(w, d)| w * d.pdf(x)).sum()
    }

    /// Fonction de répartition : somme pondérée `Σ πᵢ Fᵢ(x)`.
    ///
    /// Retourne `None` si l'une des composantes n'a pas de cdf définie.
    ///
    /// # Complexité
    /// O(C).
    fn cdf(&self, x: f64) -> Option<f64> {
        let mut acc = 0.0;
        for (w, d) in &self.components {
            acc += w * d.cdf(x)?;
        }
        Some(acc)
    }

    /// Espérance : `Σ πᵢ μᵢ`, `None` si l'une n'est pas définie.
    fn mean(&self) -> Option<f64> {
        let mut acc = 0.0;
        for (w, d) in &self.components {
            acc += w * d.mean()?;
        }
        Some(acc)
    }

    /// Variance : `Σ πᵢ (σᵢ² + (μᵢ − μ)²)`, `None` si un moment manque.
    fn variance(&self) -> Option<f64> {
        let mu = self.mean()?;
        let mut acc = 0.0;
        for (w, d) in &self.components {
            let m = d.mean()?;
            let v = d.variance()?;
            acc += w * (v + (m - mu) * (m - mu));
        }
        Some(acc)
    }

    fn name(&self) -> &'static str {
        "mixture"
    }
}

#[cfg(test)]
mod tests {
    use super::{Mixture, WEIGHT_TOLERANCE};
    use crate::distribution::Distribution;
    use crate::rng::DeterministicRng;

    use crate::distributions::{Laplace, Normal, Pareto};

    fn normal(mean: f64, std: f64) -> Box<dyn Distribution> {
        Box::new(Normal::new(mean, std).unwrap())
    }

    fn rng() -> DeterministicRng {
        DeterministicRng::from_seed([19u8; 32])
    }

    #[test]
    fn well_formed_mixture_accepted() {
        let mix = Mixture::new(vec![(0.7, normal(0.0, 1.0)), (0.3, normal(10.0, 1.0))]).unwrap();
        assert_eq!(mix.len(), 2);
        assert_eq!(mix.weights(), vec![0.7, 0.3]);
    }

    #[test]
    fn invalid_weights_rejected() {
        // Somme ≠ 1.
        let bad_sum = Mixture::new(vec![(0.5, normal(0.0, 1.0)), (0.4, normal(0.0, 1.0))]);
        assert!(bad_sum.is_err());
        // Poids négatif.
        let neg = Mixture::new(vec![(-0.1, normal(0.0, 1.0)), (1.1, normal(0.0, 1.0))]);
        assert!(neg.is_err());
        // Poids > 1.
        let over = Mixture::new(vec![(1.5, normal(0.0, 1.0))]);
        assert!(over.is_err());
        // Composante vide.
        let empty = Mixture::new(vec![]);
        assert!(empty.is_err());
    }

    #[test]
    fn sum_of_weights_is_one() {
        let mix = Mixture::new(vec![
            (0.25, normal(0.0, 1.0)),
            (0.25, normal(0.0, 1.0)),
            (0.5, normal(0.0, 1.0)),
        ])
        .unwrap();
        let s: f64 = mix.weights().iter().sum();
        assert!((s - 1.0).abs() <= WEIGHT_TOLERANCE);
    }

    #[test]
    fn sampling_hits_expected_supports() {
        // Composantes bien séparées : N(0,1) à 90 %, N(100,1) à 10 %.
        let mut mix =
            Mixture::new(vec![(0.9, normal(0.0, 1.0)), (0.1, normal(100.0, 1.0))]).unwrap();
        let mut rng = rng();
        let samples: Vec<f64> = (0..100_000).map(|_| mix.sample(&mut rng)).collect();
        let near_zero = samples.iter().filter(|&&x| x.abs() < 5.0).count();
        let near_hundred = samples.iter().filter(|&&x| (x - 100.0).abs() < 5.0).count();
        // Proportions ≈ 0.9 / 0.1 (tolérances larges, propriété de support).
        let p_zero = near_zero as f64 / samples.len() as f64;
        let p_hundred = near_hundred as f64 / samples.len() as f64;
        assert!((p_zero - 0.9).abs() < 0.02, "p_zero={p_zero}");
        assert!((p_hundred - 0.1).abs() < 0.02, "p_hundred={p_hundred}");
    }

    #[test]
    fn pdf_is_weighted_sum() {
        let mix = Mixture::new(vec![(0.7, normal(0.0, 1.0)), (0.3, normal(0.0, 1.0))]).unwrap();
        // Deux normales identiques : pdf = 1.0 × f(x).
        let n = Normal::new(0.0, 1.0).unwrap();
        for x in [-1.0, 0.0, 1.0, 2.0] {
            assert!((mix.pdf(x) - n.pdf(x)).abs() < 1e-12, "x={x}");
        }
    }

    #[test]
    fn cdf_is_weighted_sum() {
        let mix = Mixture::new(vec![(0.5, normal(0.0, 1.0)), (0.5, normal(2.0, 1.0))]).unwrap();
        let n1 = Normal::new(0.0, 1.0).unwrap();
        let n2 = Normal::new(2.0, 1.0).unwrap();
        for x in [-1.0, 0.0, 1.0, 3.0] {
            let expected = 0.5 * n1.cdf(x).unwrap() + 0.5 * n2.cdf(x).unwrap();
            assert!((mix.cdf(x).unwrap() - expected).abs() < 1e-9, "x={x}");
        }
    }

    #[test]
    fn cdf_none_if_component_has_no_cdf() {
        // Toutes nos distributions ont une cdf ; on vérifie le contrat sur une
        // composante artificielle sans cdf via un double trait objet.
        struct NoCdf;
        impl Distribution for NoCdf {
            fn sample(&mut self, _: &mut DeterministicRng) -> f64 {
                0.0
            }
            fn pdf(&self, _: f64) -> f64 {
                0.0
            }
            fn cdf(&self, _: f64) -> Option<f64> {
                None
            }
            fn mean(&self) -> Option<f64> {
                Some(0.0)
            }
            fn variance(&self) -> Option<f64> {
                Some(0.0)
            }
            fn name(&self) -> &'static str {
                "nocdf"
            }
        }
        let mix = Mixture::new(vec![(1.0, Box::new(NoCdf))]).unwrap();
        assert_eq!(mix.cdf(0.0), None);
    }

    #[test]
    fn mixture_is_deterministic() {
        let mut a = Mixture::new(vec![(0.5, normal(0.0, 1.0)), (0.5, normal(5.0, 1.0))]).unwrap();
        let mut b = Mixture::new(vec![(0.5, normal(0.0, 1.0)), (0.5, normal(5.0, 1.0))]).unwrap();
        let mut rng_a = rng();
        let mut rng_b = rng();
        for _ in 0..10_000 {
            assert_eq!(a.sample(&mut rng_a), b.sample(&mut rng_b));
        }
    }

    #[test]
    fn mixed_families_supported() {
        // Mélange de familles différentes : normale + Laplace + Pareto.
        let mix = Mixture::new(vec![
            (0.5, normal(0.0, 1.0)),
            (0.3, Box::new(Laplace::new(0.0, 1.0).unwrap())),
            (0.2, Box::new(Pareto::new(1.0, 3.0).unwrap())),
        ])
        .unwrap();
        let mut dist = mix;
        let mut rng = rng();
        for _ in 0..10_000 {
            let x = dist.sample(&mut rng);
            assert!(x.is_finite());
        }
    }
}
