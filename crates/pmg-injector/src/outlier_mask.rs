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

//! Masque d'outliers : déterminer les positions affectées, sans modifier de
//! valeur.
//!
//! [`OutlierMask`] est une matrice booléenne (stockée en `Vec<bool>` compact)
//! qui marque les éléments d'un tenseur qui recevront un traitement de
//! super-poids. Sa génération est purement déterministe : elle consomme
//! exclusivement un flux dérivé de seed ([`pmg_math::rng::DeterministicRng`] — jamais de
//! `thread_rng`).
//!
//! Conformité : `docs/documents/CAHIER DE PLAN DEVELOPPEMENT SPRINT_0_6.md`
//! étape 4.2. Le ratio réellement mesuré (`[`OutlierMask::measured_ratio`]`)
//! peut différer de la probabilité demandée pour les petits tenseurs : c'est
//! précisément cet écart que [`crate::injection_validator`] contrôle.

use serde::{Deserialize, Serialize};

use pmg_math::rng::DeterministicRng;

use crate::error::{InjectorError, InjectorResult};

/// Masque booléen des positions d'outliers d'un tenseur.
///
/// # Invariants
/// - `len()` est le nombre d'éléments du tenseur cible ;
/// - `count()` compte les éléments marqués ;
/// - `measured_ratio()` = `count() / len()` (0 si le tenseur est vide).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutlierMask {
    flags: Vec<bool>,
    count: usize,
}

impl OutlierMask {
    /// Construit un masque à partir d'une liste de drapeaux.
    ///
    /// # Erreurs
    /// [`InjectorError::InvalidTensor`] si le vecteur est vide.
    ///
    /// # Complexité
    /// O(n).
    pub fn from_flags(flags: Vec<bool>) -> InjectorResult<Self> {
        if flags.is_empty() {
            return Err(InjectorError::InvalidTensor(
                "masque d'outliers vide : au moins un élément requis".into(),
            ));
        }
        let count = flags.iter().filter(|&&f| f).count();
        Ok(Self { flags, count })
    }

    /// Masque entièrement vide de `len` positions (aucun outlier).
    ///
    /// # Erreurs
    /// [`InjectorError::InvalidTensor`] si `len == 0`.
    pub fn empty(len: usize) -> InjectorResult<Self> {
        Self::from_flags(vec![false; len])
    }

    /// Génère déterministiquement le masque par tirage de Bernoulli
    /// `p` pour chaque position, dans l'ordre canonique du tenseur.
    ///
    /// # Entrées
    /// - `rng` : flux déterministe dérivé (domaine `"outlier"`) ;
    /// - `len` : nombre d'éléments du tenseur ;
    /// - `probability` : probabilité cible dans `[0, 1]`.
    ///
    /// # Erreurs
    /// - [`InjectorError::InvalidTensor`] si `len == 0` ;
    /// - [`InjectorError::InvalidPolicy`] si `probability` hors `[0, 1]`.
    ///
    /// # Complexité
    /// O(len).
    pub fn bernoulli(
        rng: &mut DeterministicRng,
        len: usize,
        probability: f64,
    ) -> InjectorResult<Self> {
        if len == 0 {
            return Err(InjectorError::InvalidTensor(
                "taille de masque nulle".into(),
            ));
        }
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(InjectorError::InvalidPolicy(format!(
                "probabilité d'outlier hors [0, 1] : {probability}"
            )));
        }
        let mut flags = Vec::with_capacity(len);
        let mut count = 0usize;
        for _ in 0..len {
            let u = rng.next_f64();
            let hit = u < probability;
            if hit {
                count += 1;
            }
            flags.push(hit);
        }
        Ok(Self { flags, count })
    }

    /// Nombre d'éléments couverts par le masque.
    pub fn len(&self) -> usize {
        self.flags.len()
    }

    /// Vrai si le masque ne couvre aucun élément (impossible par construction).
    pub fn is_empty(&self) -> bool {
        self.flags.is_empty()
    }

    /// Nombre de positions marquées.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Ratio réellement mesuré `count / len` (0 si le tenseur est vide).
    pub fn measured_ratio(&self) -> f64 {
        if self.flags.is_empty() {
            0.0
        } else {
            self.count as f64 / self.flags.len() as f64
        }
    }

    /// Valeur du drapeau à la position `index`.
    ///
    /// # Erreurs
    /// [`InjectorError::InvalidTensor`] si `index` est hors bornes.
    pub fn get(&self, index: usize) -> InjectorResult<bool> {
        self.flags.get(index).copied().ok_or_else(|| {
            InjectorError::InvalidTensor(format!(
                "index {index} hors du masque de longueur {}",
                self.flags.len()
            ))
        })
    }

    /// Accès immuable aux drapeaux (ordre canonique du tenseur).
    pub fn flags(&self) -> &[bool] {
        &self.flags
    }

    /// Applique le masque à un buffer : remplace par `value` les positions
    /// marquées. Le buffer est modifié sur place (aucune allocation).
    ///
    /// # Erreurs
    /// [`InjectorError::InvalidTensor`] si les longueurs diffèrent.
    ///
    /// # Complexité
    /// O(len).
    pub fn apply_values(&self, buffer: &mut [f64], value: f64) -> InjectorResult<()> {
        if buffer.len() != self.flags.len() {
            return Err(InjectorError::InvalidTensor(format!(
                "buffer de longueur {} ≠ masque de longueur {}",
                buffer.len(),
                self.flags.len()
            )));
        }
        for (b, &f) in buffer.iter_mut().zip(self.flags.iter()) {
            if f {
                *b = value;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::OutlierMask;
    use crate::error::InjectorError;
    use pmg_math::rng::{derive_sub_seed, DeterministicRng};

    fn rng_for(seed: [u8; 32]) -> DeterministicRng {
        DeterministicRng::from_seed(derive_sub_seed(&seed, "outlier", 0))
    }

    fn base_seed() -> [u8; 32] {
        [7u8; 32]
    }

    #[test]
    fn empty_mask_marks_nothing() {
        let mask = OutlierMask::empty(10).unwrap();
        assert_eq!(mask.len(), 10);
        assert_eq!(mask.count(), 0);
        assert_eq!(mask.measured_ratio(), 0.0);
        assert!(!mask.is_empty());
    }

    #[test]
    fn zero_length_mask_rejected() {
        assert!(matches!(
            OutlierMask::empty(0),
            Err(InjectorError::InvalidTensor(_))
        ));
        assert!(OutlierMask::bernoulli(&mut rng_for(base_seed()), 0, 0.1).is_err());
    }

    #[test]
    fn bernoulli_zero_and_one_probabilities() {
        let mut rng = rng_for(base_seed());
        let none = OutlierMask::bernoulli(&mut rng, 64, 0.0).unwrap();
        assert_eq!(none.count(), 0);
        assert_eq!(none.measured_ratio(), 0.0);

        let mut rng = rng_for(base_seed());
        let all = OutlierMask::bernoulli(&mut rng, 64, 1.0).unwrap();
        assert_eq!(all.count(), 64);
        assert_eq!(all.measured_ratio(), 1.0);
    }

    #[test]
    fn probability_out_of_bounds_rejected() {
        let mut rng = rng_for(base_seed());
        assert!(matches!(
            OutlierMask::bernoulli(&mut rng, 8, 1.5),
            Err(InjectorError::InvalidPolicy(_))
        ));
        assert!(matches!(
            OutlierMask::bernoulli(&mut rng, 8, -0.1),
            Err(InjectorError::InvalidPolicy(_))
        ));
    }

    #[test]
    fn measured_ratio_matches_bernoulli_law() {
        // Avec p = 0.5, le ratio mesuré est proche de 0.5 (tolérance large).
        let mut rng = rng_for(base_seed());
        let mask = OutlierMask::bernoulli(&mut rng, 20_000, 0.5).unwrap();
        let ratio = mask.measured_ratio();
        assert!(
            (ratio - 0.5).abs() < 0.02,
            "ratio mesuré {ratio} trop éloigné de 0.5"
        );
    }

    #[test]
    fn generation_is_deterministic() {
        // Même seed ⇒ même masque, bit à bit.
        let a = OutlierMask::bernoulli(&mut rng_for(base_seed()), 512, 0.05).unwrap();
        let b = OutlierMask::bernoulli(&mut rng_for(base_seed()), 512, 0.05).unwrap();
        assert_eq!(a, b);
        // Seeds différentes ⇒ masques différents (quasi certainement).
        let mut other = base_seed();
        other[0] ^= 0xFF;
        let c = OutlierMask::bernoulli(&mut rng_for(other), 512, 0.05).unwrap();
        assert_ne!(a, c);
    }

    #[test]
    fn get_returns_flag_and_bounds_check() {
        let mask = OutlierMask::from_flags(vec![true, false, true]).unwrap();
        assert!(mask.get(0).unwrap());
        assert!(!mask.get(1).unwrap());
        assert!(mask.get(3).is_err());
    }

    #[test]
    fn apply_values_modifies_only_marked_positions() {
        let mask = OutlierMask::from_flags(vec![true, false, true]).unwrap();
        let mut buf = vec![1.0, 2.0, 3.0];
        mask.apply_values(&mut buf, -9.0).unwrap();
        assert_eq!(buf, vec![-9.0, 2.0, -9.0]);
        // Longueurs incohérentes → erreur typée.
        let mut bad = vec![1.0];
        assert!(matches!(
            mask.apply_values(&mut bad, 0.0),
            Err(InjectorError::InvalidTensor(_))
        ));
    }

    #[test]
    fn serde_roundtrip_preserves_mask() {
        let mask = OutlierMask::from_flags(vec![true, false, true, true]).unwrap();
        let json = serde_json::to_string(&mask).unwrap();
        let back: OutlierMask = serde_json::from_str(&json).unwrap();
        assert_eq!(mask, back);
        assert_eq!(back.count(), 3);
    }
}
