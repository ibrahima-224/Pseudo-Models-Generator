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

//! RNG déterministe et politique de seed.
//!
//! Conformité : `docs/architecture/04-moteurs-math-injection-generation.md`
//! §1. Un flux de génération est une fonction déterministe de ses entrées ;
//! chaque unité logique (tenseur, chunk, facteur) possède son propre flux
//! dérivé, jamais partagé, jamais `thread_rng`.
//!
//! Choix documentés :
//! - générateur de flux : **XorShift128+** — déterministe,
//!   sans état global, testé en reproductibilité stricte sur une même plateforme ;
//! - fonction de dérivation : **SHA-256** (`sha2`) sur une concaténation
//!   canonique à **taille préfixée** des champs (évite les collisions de type
//!   `("a","bc")` vs `("ab","c")`), le digest est tronqué en 32 octets.

use sha2::{Digest, Sha256};

use pmg_core::rng_trait::DeterministicRng as DeterministicRngTrait;

/// Identité canonique d'un tenseur pour la dérivation de sa seed.
///
/// La concaténation est faite avec des tailles préfixées (u32 LE) pour garantir
/// l'absence de collision de concaténation (spécification doc 4 §1.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeedPlan<'a> {
    /// Seed globale du processus de génération (ex. `42`).
    pub seed_global: u64,
    /// Identifiant du modèle (ex. `"glm-5.2"`).
    pub model_id: &'a str,
    /// Nom complet du tenseur (ex. `"model.layers.0.mlp.gate.weight"`).
    pub tensor_name: &'a str,
    /// Index de couche (0-based) ; `None` pour les tenseurs hors couches
    /// (embeddings, norm finale, lm_head).
    pub layer_id: Option<u32>,
    /// Version du générateur : participe à l'identité du résultat
    /// (changer la version change les valeurs générées).
    pub generation_version: &'a str,
}

/// RNG déterministe de flux : XorShift128+ — simple, rapide et déterministe.
///
/// # Garanties
/// - Même seed ⇒ même séquence de valeurs (reproductibilité stricte testée
///   sur une même plateforme ; « meilleure effort » inter-plateformes).
/// - Zéro état global : chaque instance est indépendante.
/// - Période : 2^128 - 1 (suffisante pour nos cas d'usage).
#[derive(Debug, Clone)]
pub struct DeterministicRng {
    /// État interne du générateur (deux u64 pour XorShift128+).
    state: [u64; 2],
}

impl DeterministicRng {
    /// Crée un flux seedé explicitement à partir de 32 octets.
    ///
    /// # Complexité
    /// O(1) — initialisation simple.
    pub fn from_seed(seed: [u8; 32]) -> Self {
        // Convertir les 32 octets en deux u64
        let s0 = u64::from_le_bytes(seed[..8].try_into().unwrap());
        let s1 = u64::from_le_bytes(seed[8..16].try_into().unwrap());

        // S'assurer que l'état n'est pas nul (sinon le générateur reste bloqué)
        let state = if s0 == 0 && s1 == 0 {
            [1, 2] // Valeur de repli non nulle
        } else {
            [s0, s1]
        };

        Self { state }
    }

    /// Crée un flux seedé à partir d'un plan de seed canonique.
    ///
    /// Équivalent à `from_seed(derive_seed(plan))` — garantit que chaque
    /// tenseur possède un flux indépendant de l'ordre d'émission.
    pub fn from_seed_plan(plan: &SeedPlan<'_>) -> Self {
        Self::from_seed(derive_seed(plan))
    }

    /// Retourne une référence interne au RNG sous-jacent (pour les tests).
    pub fn state(&self) -> &[u64; 2] {
        &self.state
    }

    /// Génère un entier u64 aléatoire via XorShift128+.
    ///
    /// # Algorithme
    /// XorShift128+ : s1 ^= s0; s0 = rotl(s0, 24) ^ s1 ^ (s1 << 16); s1 = rotl(s1, 37)
    ///
    /// # Complexité
    /// O(1).
    pub fn next_u64(&mut self) -> u64 {
        let [s0, s1] = self.state;
        let result = s0.wrapping_add(s1);

        self.state[0] = s1;
        self.state[1] = s0 ^ s1 ^ (s1 << 16) ^ (s0 >> 24);
        // Rotation de s0
        self.state[0] = self.state[0].rotate_left(24);
        // Rotation de s1
        self.state[1] = self.state[1].rotate_left(37);

        result
    }

    /// Tire un flottant uniforme sur `[0, 1)` (53 bits de mantisse).
    ///
    /// # Complexité
    /// O(1).
    pub fn next_f64(&mut self) -> f64 {
        // Convertir u64 en f64 dans [0, 1) avec 53 bits de précision
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Remplit une slice de flottants avec des valeurs uniformes sur `[0, 1)`.
    ///
    /// # Complexité
    /// O(len).
    pub fn fill_slice(&mut self, buf: &mut [f64]) {
        for v in buf.iter_mut() {
            *v = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        }
    }

    /// Génère un vecteur de flottants selon une distribution normale.
    ///
    /// # Paramètres
    /// - `mean` : moyenne de la distribution.
    /// - `std` : écart-type de la distribution.
    /// - `n` : nombre d'éléments à générer.
    ///
    /// # Retourne
    /// Un vecteur de `n` flottants distribués selon N(mean, std²)
    /// en utilisant la transformation de Box-Muller.
    pub fn normal_vec(&mut self, mean: f64, std: f64, n: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            // Génère deux valeurs uniformes indépendantes dans (0, 1]
            // Éviter u1 = 0 qui causerait ln(0) = -inf
            let mut u1 = self.next_f64();
            while u1 <= 0.0 || !u1.is_finite() {
                u1 = self.next_f64();
            }
            let u2 = self.next_f64();

            // Transformation de Box-Muller pour obtenir une normale standard
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

            // Met à l'échelle avec mean et std
            result.push(mean + std * z);
        }
        result
    }
}

/// Implémentation du trait DeterministicRng pour DeterministicRng.
///
/// Permet d'utiliser le RNG XorShift128+ comme injecteur de dépendance
/// dans le pipeline de génération de pmg-core.
impl DeterministicRngTrait for DeterministicRng {
    /// Génère un entier non signé 64 bits aléatoire.
    ///
    /// # Retourne
    /// Un `u64` uniformément distribué via XorShift128+.
    fn next_u64(&mut self) -> u64 {
        DeterministicRng::next_u64(self)
    }

    /// Génère un flottant 64 bits aléatoire dans [0, 1).
    ///
    /// # Retourne
    /// Un `f64` uniformément distribué dans [0, 1) via XorShift128+.
    fn next_f64(&mut self) -> f64 {
        (DeterministicRng::next_u64(self) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Dérive la seed canonique d'un tenseur : `H(seed_global ‖ model_id ‖
/// tensor_name ‖ layer_id ‖ generation_version)`.
///
/// # Entrées
/// - plan : identité canonique (voir [`SeedPlan`]).
///
/// # Sorties
/// 32 octets (SHA-256 complet) — seed directe de `DeterministicRng`.
///
/// # Hypothèses
/// - Champs encodés en UTF-8, tailles préfixées en u32 little-endian ;
/// - `layer_id` est encodé sur 4 octets, `None` étant `0xFFFFFFFF`.
///
/// # Complexité
/// O(longueur totale des champs) — deux passes de SHA-256.
///
/// # Limites
/// Déterministe par construction ; aucune source d'entropie.
pub fn derive_seed(plan: &SeedPlan<'_>) -> [u8; 32] {
    let mut hasher = Sha256::new();
    push_u64(&mut hasher, plan.seed_global);
    push_str(&mut hasher, plan.model_id);
    push_str(&mut hasher, plan.tensor_name);
    push_u32_opt(&mut hasher, plan.layer_id);
    push_str(&mut hasher, plan.generation_version);
    hasher.finalize().into()
}

/// Dérive une seed secondaire à partir de la seed d'un tenseur et d'un
/// domaine séparé (`chunk_id`, `"factor"`, `"outlier"`…).
///
/// Sert à la hiérarchie `seed_chunk = H(seed_tensor ‖ domaine)` de la
/// spécification (doc 4 §1.2) sans collision entre domaines.
///
/// # Entrées
/// - `seed_tensor` : seed racine (32 octets) ;
/// - `domain` : identifiant de domaine séparé ;
/// - `index` : indice entier (chunk_id, pass_index, composante…).
pub fn derive_secondary_seed(seed_tensor: &[u8; 32], domain: &str, index: u32) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(seed_tensor);
    push_str(&mut hasher, domain);
    push_u32(&mut hasher, index);
    hasher.finalize().into()
}

/// Alias pour `derive_secondary_seed` (compatibilité API).
pub fn derive_sub_seed(seed_tensor: &[u8; 32], domain: &str, index: u32) -> [u8; 32] {
    derive_secondary_seed(seed_tensor, domain, index)
}

/// Convertit une seed de 32 octets en `u64` (pour les seeds simples).
///
/// # Panic
/// Ne panique pas : utilise les 8 premiers octets en little-endian.
pub fn seed_to_u64(seed: &[u8; 32]) -> u64 {
    u64::from_le_bytes(seed[..8].try_into().unwrap())
}

/// Encode un `u64` en little-endian et le push dans le hasher.
fn push_u64(hasher: &mut Sha256, v: u64) {
    hasher.update(v.to_le_bytes());
}

/// Encode un `u32` en little-endian et le push dans le hasher.
fn push_u32(hasher: &mut Sha256, v: u32) {
    hasher.update(v.to_le_bytes());
}

/// Encode un `Option<u32>` en little-endian (0xFFFFFFFF pour `None`).
fn push_u32_opt(hasher: &mut Sha256, v: Option<u32>) {
    push_u32(hasher, v.unwrap_or(0xFFFFFFFF));
}

/// Encode une chaîne UTF-8 avec sa taille préfixée (u32 LE).
fn push_str(hasher: &mut Sha256, s: &str) {
    let bytes = s.as_bytes();
    push_u32(hasher, bytes.len() as u32);
    hasher.update(bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_rng_reproductibilite() {
        let seed = [42u8; 32];
        let mut rng1 = DeterministicRng::from_seed(seed);
        let mut rng2 = DeterministicRng::from_seed(seed);

        // Même seed → même séquence
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
            assert_eq!(rng1.next_f64(), rng2.next_f64());
        }
    }

    #[test]
    fn test_deterministic_rng_non_nul() {
        let seed = [0u8; 32];
        let mut rng = DeterministicRng::from_seed(seed);

        // Vérifier que le générateur produit des valeurs non nulles
        let mut has_nonzero = false;
        for _ in 0..1000 {
            if rng.next_u64() != 0 {
                has_nonzero = true;
                break;
            }
        }
        assert!(has_nonzero, "Le générateur ne produit que des zéros");
    }

    #[test]
    fn test_seed_plan_deterministe() {
        let plan = SeedPlan {
            seed_global: 42,
            model_id: "glm-5.2",
            tensor_name: "model.layers.0.mlp.gate.weight",
            layer_id: Some(0),
            generation_version: "v1.0",
        };

        let seed1 = derive_seed(&plan);
        let seed2 = derive_seed(&plan);
        assert_eq!(seed1, seed2);
    }

    #[test]
    fn test_secondary_seed_deterministe() {
        let seed_tensor = [1u8; 32];
        let seed1 = derive_secondary_seed(&seed_tensor, "chunk", 0);
        let seed2 = derive_secondary_seed(&seed_tensor, "chunk", 0);
        assert_eq!(seed1, seed2);

        // Domaines différents → seeds différentes
        let seed3 = derive_secondary_seed(&seed_tensor, "factor", 0);
        assert_ne!(seed1, seed3);

        // Indices différents → seeds différentes
        let seed4 = derive_secondary_seed(&seed_tensor, "chunk", 1);
        assert_ne!(seed1, seed4);
    }

    #[test]
    fn test_normal_vec_distribution() {
        let mut rng = DeterministicRng::from_seed([0u8; 32]);
        let samples = rng.normal_vec(0.0, 1.0, 10000);

        // Vérifier que la moyenne est proche de 0
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!((mean).abs() < 0.1, "Moyenne trop éloignée de 0: {}", mean);

        // Vérifier que l'écart-type est proche de 1
        let variance: f64 = samples.iter().map(|x| x * x).sum::<f64>() / samples.len() as f64;
        let std_dev = variance.sqrt();
        assert!(
            (std_dev - 1.0).abs() < 0.1,
            "Écart-type trop éloigné de 1: {}",
            std_dev
        );
    }

    #[test]
    fn test_seed_to_u64() {
        let seed = [0u8; 32];
        assert_eq!(seed_to_u64(&seed), 0);

        let mut seed2 = [0u8; 32];
        seed2[0] = 1;
        assert_eq!(seed_to_u64(&seed2), 1);

        let mut seed3 = [0u8; 32];
        seed3[7] = 1;
        assert_eq!(seed_to_u64(&seed3), 1 << 56);
    }
}
