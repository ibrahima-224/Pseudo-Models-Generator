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

//! Shape d'un tenseur : vecteur de dimensions strictement positives.
//!
//! Conforme au contrat de `docs/architecture/03-modeles-de-donnees.md` §2.2 :
//! - dimension nulle → [`CoreError::InvalidDimension`] (format Safetensors) ;
//! - `num_elements()` via `checked_mul` → [`CoreError::Overflow`] (jamais de wrap) ;
//! - shape vide `[]` = scalaire (1 élément), convention Safetensors.
//!
//! # Exemple
//!
//! ```
//! use pmg_core::Shape;
//!
//! // Création d'une shape 2D valide.
//! let shape = Shape::new(vec![6144, 6144]).unwrap();
//! assert_eq!(shape.rank(), 2);
//! assert_eq!(shape.num_elements().unwrap(), 6144 * 6144);
//!
//! // Création d'un scalaire.
//! let scalar = Shape::scalar();
//! assert!(scalar.is_scalar());
//! assert_eq!(scalar.num_elements().unwrap(), 1);
//!
//! // Dimension nulle rejetée.
//! assert!(Shape::new(vec![0, 10]).is_err());
//! ```

use serde::{Deserialize, Serialize};

use crate::error::{CoreError, CoreResult};

/// Forme (shape) d'un tenseur : liste ordonnée de dimensions.
///
/// Les dimensions sont stockées en `u64` (le format Safetensors les exprime
/// ainsi) et sont toutes strictement positives, sauf le cas du scalaire `[]`.
///
/// # Exemple
///
/// ```
/// use pmg_core::Shape;
///
/// let shape = Shape::new(vec![12, 34, 56]).unwrap();
/// assert_eq!(shape.rank(), 3);
/// assert_eq!(shape.dims(), &[12, 34, 56]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Shape {
    dimensions: Vec<u64>,
}

impl Shape {
    /// Construit une shape à partir de dimensions **strictement positives**.
    ///
    /// # Erreurs
    /// - [`CoreError::InvalidDimension`] si une dimension vaut zéro.
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_core::{Shape, CoreError};
    ///
    /// // Dimensions valides.
    /// let shape = Shape::new(vec![10, 20, 30]).unwrap();
    /// assert_eq!(shape.rank(), 3);
    ///
    /// // Dimension nulle → erreur.
    /// let err = Shape::new(vec![10, 0, 30]).unwrap_err();
    /// assert!(matches!(err, CoreError::InvalidDimension));
    /// ```
    pub fn new(dims: Vec<u64>) -> CoreResult<Shape> {
        if dims.contains(&0) {
            // Le format Safetensors interdit les dimensions nulles : on rejette
            // la shape avant toute utilisation (jamais de dimension 0 stockée).
            return Err(CoreError::InvalidDimension);
        }
        Ok(Shape { dimensions: dims })
    }

    /// Construit une shape scalaire (dimensions vides, 1 élément).
    pub fn scalar() -> Shape {
        Shape {
            dimensions: Vec::new(),
        }
    }

    /// Accès immuable aux dimensions (dans l'ordre).
    pub fn dims(&self) -> &[u64] {
        &self.dimensions
    }

    /// Nombre de dimensions (rang).
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }

    /// Nombre total d'éléments, produit vérifié des dimensions.
    ///
    /// Une shape vide `[]` (scalaire) vaut 1 élément. Tout débordement du
    /// produit retourne [`CoreError::Overflow`] — jamais de wrap silencieux.
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_core::Shape;
    ///
    /// let shape = Shape::new(vec![4, 8, 16]).unwrap();
    /// assert_eq!(shape.num_elements().unwrap(), 4 * 8 * 16);
    /// ```
    pub fn num_elements(&self) -> CoreResult<u64> {
        self.dimensions.iter().try_fold(1u64, |acc, &d| {
            acc.checked_mul(d).ok_or_else(|| {
                CoreError::Overflow(format!(
                    "produit des dimensions de la shape {:?} dépasse u64::MAX",
                    self.dimensions
                ))
            })
        })
    }

    /// Nombre total d'éléments sur `usize` (confort d'API), vérifié.
    ///
    /// # Erreurs
    /// - [`CoreError::Overflow`] si le produit dépasse `usize::MAX` (32 bits).
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_core::Shape;
    ///
    /// let shape = Shape::new(vec![100, 200]).unwrap();
    /// let n = shape.num_elements_usize().unwrap();
    /// assert_eq!(n, 20000);
    /// ```
    pub fn num_elements_usize(&self) -> CoreResult<usize> {
        let n = self.num_elements()?;
        usize::try_from(n)
            .map_err(|_| CoreError::Overflow(format!("{} éléments ne tiennent pas dans usize", n)))
    }

    /// Vrai si la shape est un scalaire (aucune dimension).
    pub fn is_scalar(&self) -> bool {
        self.dimensions.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::Shape;
    use crate::error::CoreError;

    #[test]
    fn shape_examples_in_doc() {
        // Vérifie les exemples de la doc.
        let shape = Shape::new(vec![6144, 6144]).unwrap();
        assert_eq!(shape.rank(), 2);
        assert_eq!(shape.num_elements().unwrap(), 6144 * 6144);
    }

    #[test]
    fn scalar_shape_has_one_element() {
        let s = Shape::scalar();
        assert_eq!(s.rank(), 0);
        assert!(s.is_scalar());
        assert_eq!(s.num_elements().unwrap(), 1);
        assert!(s.dims().is_empty());
    }

    #[test]
    fn product_of_dimensions() {
        let s = Shape::new(vec![2, 3, 4]).unwrap();
        assert_eq!(s.rank(), 3);
        assert_eq!(s.num_elements().unwrap(), 24);
        assert_eq!(s.dims(), &[2, 3, 4]);
    }

    #[test]
    fn zero_dimension_is_rejected() {
        // Le format Safetensors interdit les shapes à dimension nulle.
        for bad in [vec![0], vec![2, 0, 4], vec![0, 0]] {
            let err = Shape::new(bad).unwrap_err();
            assert_eq!(err, CoreError::InvalidDimension);
        }
    }

    #[test]
    fn product_overflow_is_explicit() {
        // 2^40 * 2^24 = 2^64 → débordement, jamais de wrap.
        let s = Shape::new(vec![1 << 40, 1 << 24]).unwrap();
        let err = s.num_elements().unwrap_err();
        assert!(matches!(err, CoreError::Overflow(_)), "obtenu {err}");
    }

    #[test]
    fn large_but_valid_product() {
        // Cas limite : produit exactement u64::MAX est impossible (2^64-1
        // n'est pas un produit de deux puissances de deux distinctes) ; on
        // vérifie un grand produit valide.
        let s = Shape::new(vec![1 << 40, 1 << 23]).unwrap();
        assert_eq!(s.num_elements().unwrap(), 1 << 63);
    }

    #[test]
    fn usize_conversion_fails_on_32bit_overflow() {
        // Sur plateformes 64 bits ce test ne s'applique pas : on construit
        // simplement une shape dont le produit dépasse usize sur 32 bits.
        let s = Shape::new(vec![1 << 40]).unwrap();
        if usize::BITS < 64 {
            assert!(s.num_elements_usize().is_err());
        } else {
            assert_eq!(s.num_elements_usize().unwrap(), 1usize << 40);
        }
    }

    #[test]
    fn serde_roundtrip() {
        let s = Shape::new(vec![4, 16]).unwrap();
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(serde_json::from_str::<Shape>(&json).unwrap(), s);
    }

    #[test]
    fn shape_derives_equality_and_hash() {
        assert_eq!(Shape::new(vec![2]).unwrap(), Shape::new(vec![2]).unwrap());
        assert_ne!(Shape::new(vec![2]).unwrap(), Shape::new(vec![3]).unwrap());
    }
}
