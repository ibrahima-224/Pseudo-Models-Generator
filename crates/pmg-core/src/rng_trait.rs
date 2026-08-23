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

//! Trait abstrait pour le générateur de nombres aléatoires déterministe.
//!
//! Ce module définit le trait [`DeterministicRng`] qui permet l'injection
//! de dépendance sans cycler les crates. L'implémentation réelle (ChaCha12)
//! est fournie par `pmg_math::rng::DeterministicRng` qui implémente ce trait.
//!
//! Conformité : Sprint 12, étape 12.1 « Intégration RNG ChaCha12 ».

/// Trait abstrait pour le générateur de nombres aléatoires déterministe.
///
/// Ce trait définit l'interface minimale nécessaire pour la génération
/// de nombres aléatoires déterministes dans le pipeline de génération.
/// L'implémentation réelle est fournie par `pmg_math` via le type
/// `pmg_math::rng::DeterministicRng` qui utilise ChaCha12.
///
/// # Exemple
///
/// ```rust
/// use pmg_core::rng_trait::DeterministicRng;
///
/// fn generate_data(rng: &mut dyn DeterministicRng, n: usize) -> Vec<f64> {
///     (0..n).map(|_| rng.next_f64()).collect()
/// }
/// ```
pub trait DeterministicRng: std::fmt::Debug {
    /// Génère un entier non signé 64 bits aléatoire.
    ///
    /// # Retourne
    /// Un `u64` uniformément distribué.
    fn next_u64(&mut self) -> u64;

    /// Génère un flottant 64 bits aléatoire dans [0, 1).
    ///
    /// # Retourne
    /// Un `f64` uniformément distribué dans [0, 1).
    fn next_f64(&mut self) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// RNG de test simple pour vérifier le fonctionnement du trait.
    #[derive(Debug)]
    struct MockRng {
        state: u64,
    }

    impl MockRng {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
    }

    impl DeterministicRng for MockRng {
        fn next_u64(&mut self) -> u64 {
            // LCG simple pour les tests
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
    fn test_deterministic_trait() {
        let mut rng1 = MockRng::new(42);
        let mut rng2 = MockRng::new(42);

        // Vérifie la reproductibilité
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_f64_range() {
        let mut rng = MockRng::new(123);
        for _ in 0..1000 {
            let value = rng.next_f64();
            assert!((0.0..1.0).contains(&value), "valeur hors bornes: {value}");
        }
    }
}
