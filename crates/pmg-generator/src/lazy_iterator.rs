//! Itérateur lazy pour la génération de distributions de base.
//!
//! Ce module implémente le pattern `LazyBaseDistribution` qui permet de générer
//! des distributions de valeurs flottantes sans allocation massive en mémoire.
//! L'itérateur ne produit les valeurs qu'à la demande, ce qui garantit une
//! consommation mémoire constante O(1) quelle que soit la taille de la distribution.
//!
//! # Objectif
//! Respecter la contrainte d'optimisation mémoire : < 1 Go de RAM pour toutes
//! les commandes CLI, même pour des modèles de grande taille (centaines de Go).
//!
//! # Exemple d'utilisation
//!
//! ```rust,ignore
//! use pmg_generator::lazy_iterator::LazyBaseDistribution;
//! use pmg_core::rng_trait::DeterministicRng;
//!
//! // Création d'un itérateur pour 1 million d'éléments
//! // Le RNG doit être possédé (Box<dyn DeterministicRng>)
//! // let rng: Box<dyn DeterministicRng> = /* votre RNG déterministe */;
//! // let iter = LazyBaseDistribution::new(1_000_000, rng);
//!
//! // Consommation séquentielle sans allocation massive
//! // for value in iter {
//! //     // Traitement de chaque valeur...
//! // }
//! ```

use std::fmt;

use pmg_core::rng_trait::DeterministicRng;

/// Itérateur lazy pour la génération de distributions de base.
///
/// Génère des valeurs `f64` déterministes via un RNG, une par une,
/// sans jamais allouer de `Vec` contenant l'ensemble des valeurs.
///
/// # Propriétés
/// - **Déterminisme** : Même graine → même séquence de valeurs.
/// - **Mémoire constante** : O(1) quelle que soit la taille.
/// - **Évaluable à la demande** : Compatible avec les itérateurs Rust.
/// - **Taille connue** : Implémente `ExactSizeIterator` pour la pré-allocation optimisée.
///
/// # Ownership
/// Cet itérateur prend possession du RNG via `Box<dyn DeterministicRng>`.
/// Pour les fonctions existantes qui empruntent un RNG (`&mut dyn DeterministicRng`),
/// utilisez directement la collecte dans un `Vec` ou adaptez le code.
pub struct LazyBaseDistribution<'a> {
    /// Nombre d'éléments restants à générer.
    remaining: usize,
    /// Générateur de nombres aléatoires déterministe (possédé avec durée de vie).
    rng: Box<dyn DeterministicRng + 'a>,
}

impl<'a> LazyBaseDistribution<'a> {
    /// Crée un nouvel itérateur lazy pour la génération de distribution.
    ///
    /// # Arguments
    /// * `num_elements` - Nombre total d'éléments à générer.
    /// * `rng` - Générateur déterministe (propriété transférée via Box avec durée de vie).
    ///
    /// # Retourne
    /// Une instance de `LazyBaseDistribution` prête à être consommée.
    pub fn new(num_elements: usize, rng: Box<dyn DeterministicRng + 'a>) -> Self {
        Self {
            remaining: num_elements,
            rng,
        }
    }

    /// Retourne le nombre d'éléments restants à générer.
    pub fn remaining(&self) -> usize {
        self.remaining
    }
}

impl<'a> Iterator for LazyBaseDistribution<'a> {
    type Item = f64;

    /// Génère la valeur suivante de la distribution.
    ///
    /// # Retourne
    /// `Some(value)` si des éléments restent, `None` sinon.
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            None
        } else {
            self.remaining -= 1;
            Some(self.rng.next_f64())
        }
    }

    /// Fournit une estimation précise de la taille restante.
    ///
    /// Permet aux consommateurs de pré-allouer efficacement si nécessaire.
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a> ExactSizeIterator for LazyBaseDistribution<'a> {
    /// Retourne le nombre exact d'éléments restants.
    ///
    /// Contrairement à `size_hint()`, cette méthode est garantie précise.
    fn len(&self) -> usize {
        self.remaining
    }
}

impl<'a> fmt::Debug for LazyBaseDistribution<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LazyBaseDistribution")
            .field("remaining", &self.remaining)
            .field("rng", &"<DeterministicRng>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmg_core::rng_trait::DeterministicRng;

    /// RNG minimal pour les tests (constante simple).
    #[derive(Debug)]
    struct TestRng {
        state: u64,
    }

    impl TestRng {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
    }

    impl DeterministicRng for TestRng {
        fn next_u64(&mut self) -> u64 {
            // LCG simple pour tests
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.state
        }

        fn next_f64(&mut self) -> f64 {
            (self.next_u64() as f64) / (u64::MAX as f64)
        }
    }

    #[test]
    fn test_lazy_iterator_basic() {
        let rng = TestRng::new(12345);
        let mut iter = LazyBaseDistribution::new(5, Box::new(rng));

        assert_eq!(iter.len(), 5);
        assert_eq!(iter.remaining(), 5);

        let values: Vec<f64> = iter.by_ref().take(5).collect();
        assert_eq!(values.len(), 5);
        assert_eq!(iter.len(), 0);
        assert_eq!(iter.remaining(), 0);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_lazy_iterator_empty() {
        let rng = TestRng::new(12345);
        let mut iter = LazyBaseDistribution::new(0, Box::new(rng));

        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_lazy_iterator_determinism() {
        let rng1 = TestRng::new(42);
        let rng2 = TestRng::new(42);

        let iter1 = LazyBaseDistribution::new(100, Box::new(rng1));
        let iter2 = LazyBaseDistribution::new(100, Box::new(rng2));

        let values1: Vec<f64> = iter1.collect();
        let values2: Vec<f64> = iter2.collect();

        assert_eq!(values1, values2);
    }

    #[test]
    fn test_lazy_iterator_partial_consumption() {
        let rng = TestRng::new(12345);
        let mut iter = LazyBaseDistribution::new(10, Box::new(rng));

        // Consommer seulement 3 éléments
        let first_three: Vec<f64> = iter.by_ref().take(3).collect();
        assert_eq!(first_three.len(), 3);
        assert_eq!(iter.remaining(), 7);

        // Consommer le reste
        let remaining: Vec<f64> = iter.by_ref().collect();
        assert_eq!(remaining.len(), 7);
        assert_eq!(iter.remaining(), 0);
    }

    #[test]
    fn test_lazy_iterator_large_size() {
        let rng = TestRng::new(42);
        let iter = LazyBaseDistribution::new(1_000_000, Box::new(rng));

        assert_eq!(iter.len(), 1_000_000);
        assert_eq!(iter.remaining(), 1_000_000);
    }

    #[test]
    fn test_lazy_iterator_debug() {
        let rng = TestRng::new(12345);
        let iter = LazyBaseDistribution::new(100, Box::new(rng));
        let debug_str = format!("{:?}", iter);
        assert!(debug_str.contains("LazyBaseDistribution"));
        assert!(debug_str.contains("remaining: 100"));
    }
}
