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

//! Métadonnées d'un tenseur (`TensorMetadata`) — sans aucune donnée.
//!
//! **Invariant architectural** : `TensorMetadata` décrit un tenseur sans
//! contenir ses valeurs. Aucune API ne permet de stocker des données ici.
//! Les données générées transitent uniquement par des `TensorChunk`
//! (voir moteurs). Cette séparation est le socle de Zero-Payload
//! (`docs/architecture/03-modeles-de-donnees.md` §2.3).
//!
//! # Exemple
//!
//! ```
//! use pmg_core::{TensorMetadata, Shape, DType};
//!
//! // Création de métadonnées pour un tenseur 2D.
//! let meta = TensorMetadata::new("model.embed_tokens.weight", Shape::new(vec![6144, 6144]).unwrap(), DType::Bf16).unwrap();
//! assert_eq!(meta.byte_size().unwrap(), Some(6144 * 6144 * 2));
//!
//! // Scalaire.
//! let scalar_meta = TensorMetadata::new("scale", Shape::scalar(), DType::F32).unwrap();
//! assert_eq!(scalar_meta.byte_size().unwrap(), Some(4));
//! ```

use serde::{Deserialize, Serialize};

use crate::dtype::DType;
use crate::error::{CoreError, CoreResult};
use crate::shape::Shape;

/// Métadonnées d'un tenseur : identité, forme, dtype — pas de valeurs.
///
/// Peut décrire soit un tenseur « logique » (produit par le planner), soit un
/// tenseur « physique » observé dans un shard (offsets de stockage).
///
/// # Exemple
///
/// ```
/// use pmg_core::{TensorMetadata, Shape, DType};
///
/// let meta = TensorMetadata::new("a.weight", Shape::new(vec![10, 20]).unwrap(), DType::F32).unwrap();
/// assert_eq!(meta.name, "a.weight");
/// assert_eq!(meta.shape, Shape::new(vec![10, 20]).unwrap());
/// assert_eq!(meta.dtype, DType::F32);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorMetadata {
    /// Nom unique du tenseur (ex. `model.layers.0.self_attn.o_proj.weight`).
    pub name: String,
    /// Forme (dimensions strictement positives, `[]` = scalaire).
    pub shape: Shape,
    /// Type de données.
    pub dtype: DType,
    /// Nom du shard Safetensors contenant le tenseur (si observé).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shard: Option<String>,
    /// Offset de début dans le payload du shard (si observé).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset_start: Option<u64>,
    /// Offset de fin (exclusif) dans le payload du shard (si observé).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset_end: Option<u64>,
    /// Taille en octets **déclarée/observée** (extension) : dérivée de
    /// `shape × dtype` à la construction, ou lue dans un header.
    ///
    /// NB : le champ sérialisé reste `byte_size` (compatibilité JSON) ; la
    /// valeur calculée est accessible via [`TensorMetadata::byte_size`].
    #[serde(rename = "byte_size", default, skip_serializing_if = "Option::is_none")]
    pub byte_size_declared: Option<u64>,
}

impl TensorMetadata {
    /// Construit des métadonnées minimales (nom, shape, dtype).
    ///
    /// La taille en octets est calculée automatiquement lorsque le dtype est
    /// émissible ; pour un dtype non émissible (F4/F6*/F8E8M0) `byte_size`
    /// reste `None` (écriture refusée, lecture/validation seulement).
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_core::{TensorMetadata, Shape, DType, CoreError};
    ///
    /// // Métadonnées valides.
    /// let meta = TensorMetadata::new("weight", Shape::new(vec![10, 20]).unwrap(), DType::F32).unwrap();
    /// assert_eq!(meta.byte_size().unwrap(), Some(800)); // 10*20*4
    ///
    /// // Nom vide → erreur.
    /// let err = TensorMetadata::new("", Shape::scalar(), DType::F32).unwrap_err();
    /// assert!(matches!(err, CoreError::InvalidShape(_)));
    /// ```
    pub fn new(name: impl Into<String>, shape: Shape, dtype: DType) -> CoreResult<TensorMetadata> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(CoreError::invalid_shape(
                "le nom d'un tenseur ne peut pas être vide".to_string(),
            ));
        }
        // Taille en octets = éléments × octets/élément, calcul vérifié.
        // Dtype non émissible (F4/F6*/F8E8M0) → byte_size reste None.
        let byte_size = match dtype.size_bytes() {
            Some(bytes) => {
                let n = shape.num_elements()?;
                let size = n.checked_mul(bytes).ok_or_else(|| {
                    CoreError::Overflow(format!(
                        "taille en octets du tenseur '{name}' dépasse u64::MAX"
                    ))
                })?;
                Some(size)
            },
            None => None,
        };
        Ok(TensorMetadata {
            name,
            shape,
            dtype,
            shard: None,
            offset_start: None,
            offset_end: None,
            byte_size_declared: byte_size,
        })
    }

    /// Nombre d'éléments du tenseur (produit vérifié des dimensions).
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_core::{TensorMetadata, Shape, DType};
    ///
    /// let meta = TensorMetadata::new("tensor", Shape::new(vec![4, 8, 16]).unwrap(), DType::F16).unwrap();
    /// assert_eq!(meta.num_elements().unwrap(), 4 * 8 * 16);
    /// ```
    pub fn num_elements(&self) -> CoreResult<u64> {
        self.shape.num_elements()
    }

    /// Taille en octets du tenseur, dérivée de `shape × size_bytes(dtype)`.
    ///
    /// Retourne `None` si le dtype n'est pas émissible (`size_bytes = None`),
    /// conformément au contrat §2.3 : jamais de valeur inventée.
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_core::{TensorMetadata, Shape, DType};
    ///
    /// // Dtype émissible.
    /// let meta = TensorMetadata::new("w", Shape::new(vec![10, 20]).unwrap(), DType::F32).unwrap();
    /// assert_eq!(meta.byte_size().unwrap(), Some(800));
    ///
    /// // Dtype non émissible.
    /// let meta_q = TensorMetadata::new("q", Shape::new(vec![10, 20]).unwrap(), DType::F4).unwrap();
    /// assert_eq!(meta_q.byte_size().unwrap(), None);
    /// ```
    pub fn byte_size(&self) -> CoreResult<Option<u64>> {
        match self.dtype.size_bytes() {
            Some(bytes) => {
                let n = self.num_elements()?;
                n.checked_mul(bytes).map(Some).ok_or_else(|| {
                    CoreError::Overflow(format!(
                        "taille en octets du tenseur '{}' dépasse u64::MAX",
                        self.name
                    ))
                })
            },
            None => Ok(None),
        }
    }

    /// Valide la cohérence interne des métadonnées.
    ///
    /// Vérifie : nom non vide, dimensions strictement positives (via
    /// `Shape::new`), et — si des offsets sont présents — leur contiguïté
    /// avec `byte_size` (`offset_end - offset_start == byte_size`).
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_core::{TensorMetadata, Shape, DType};
    ///
    /// let mut meta = TensorMetadata::new("valid", Shape::new(vec![10, 20]).unwrap(), DType::F32).unwrap();
    /// meta.offset_start = Some(0);
    /// meta.offset_end = Some(800);
    /// assert!(meta.validate().is_ok());
    /// ```
    pub fn validate(&self) -> CoreResult<()> {
        if self.name.trim().is_empty() {
            return Err(CoreError::invalid_shape(
                "le nom d'un tenseur ne peut pas être vide".to_string(),
            ));
        }
        // Reconstruit la shape pour rejouer les invariants de dimensions.
        Shape::new(self.shape.dims().to_vec())?;
        if let Some(byte_size) = self.byte_size_declared {
            let calculated = self.byte_size()?;
            if calculated != Some(byte_size) {
                return Err(CoreError::Validation(format!(
                    "taille déclarée ({byte_size} octets) incohérente avec shape × dtype \
                     ({} octets) pour le tenseur '{}'",
                    calculated.map_or_else(|| "N/A".to_string(), |v| v.to_string()),
                    self.name
                )));
            }
        }
        // Contiguïté des offsets si tous deux présents (sinon rien à vérifier).
        if let (Some(start), Some(end)) = (self.offset_start, self.offset_end) {
            if start > end {
                return Err(CoreError::Validation(format!(
                    "offsets inversés [start={start}, end={end}) pour '{}'",
                    self.name
                )));
            }
            if let Some(size) = self.byte_size_declared {
                if end - start != size {
                    return Err(CoreError::Validation(format!(
                        "intervalle [{start}, {end}) ne correspond pas à la taille \
                         {size} octets pour '{}'",
                        self.name
                    )));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::TensorMetadata;
    use crate::dtype::DType;
    use crate::error::CoreError;
    use crate::shape::Shape;

    #[test]
    fn tensor_metadata_examples_in_doc() {
        // Vérifie les exemples de la doc.
        let meta = TensorMetadata::new(
            "model.embed_tokens.weight",
            Shape::new(vec![6144, 6144]).unwrap(),
            DType::Bf16,
        )
        .unwrap();
        assert_eq!(meta.byte_size().unwrap(), Some(6144 * 6144 * 2));
    }

    #[test]
    fn byte_size_is_calculated() {
        // [4, 16] en F32 = 4×16×4 = 256 octets.
        let meta =
            TensorMetadata::new("a.weight", Shape::new(vec![4, 16]).unwrap(), DType::F32).unwrap();
        assert_eq!(meta.byte_size_declared, Some(256));
        assert_eq!(meta.byte_size().unwrap(), Some(256));
        assert_eq!(meta.num_elements().unwrap(), 64);
    }

    #[test]
    fn non_emittable_dtype_yields_none() {
        // F4 : taille en octets inconnue → None (jamais de valeur inventée).
        let meta =
            TensorMetadata::new("q.weight", Shape::new(vec![4, 16]).unwrap(), DType::F4).unwrap();
        assert_eq!(meta.byte_size_declared, None);
        assert_eq!(meta.byte_size().unwrap(), None);
    }

    #[test]
    fn scalar_byte_size() {
        let meta = TensorMetadata::new("scale", Shape::scalar(), DType::F32).unwrap();
        assert_eq!(meta.byte_size_declared, Some(4));
    }

    #[test]
    fn byte_size_overflow_is_explicit() {
        // 2^40 × 2^24 × 8 octets > u64::MAX → Overflow, jamais de wrap.
        let shape = Shape::new(vec![1 << 40, 1 << 24]).unwrap();
        let err = TensorMetadata::new("huge.weight", shape, DType::F64).unwrap_err();
        assert!(matches!(err, CoreError::Overflow(_)), "obtenu {err}");
    }

    #[test]
    fn empty_name_is_rejected() {
        assert!(TensorMetadata::new("", Shape::scalar(), DType::F32).is_err());
        assert!(TensorMetadata::new("   ", Shape::scalar(), DType::F32).is_err());
    }

    #[test]
    fn validate_accepts_coherent_metadata() {
        let mut meta =
            TensorMetadata::new("o.weight", Shape::new(vec![4, 16]).unwrap(), DType::F16).unwrap();
        // [4,16] en F16 = 128 octets.
        meta.offset_start = Some(100);
        meta.offset_end = Some(228);
        assert_eq!(meta.byte_size_declared, Some(128));
        assert!(meta.validate().is_ok());
    }

    #[test]
    fn validate_rejects_incoherent_offsets() {
        let mut meta =
            TensorMetadata::new("o.weight", Shape::new(vec![4, 16]).unwrap(), DType::F16).unwrap();
        // Intervalle de 130 octets ≠ 128 → Validation.
        meta.offset_start = Some(100);
        meta.offset_end = Some(230);
        assert!(meta.validate().is_err());

        // Offsets inversés → Validation.
        let mut meta2 =
            TensorMetadata::new("p.weight", Shape::new(vec![4]).unwrap(), DType::F32).unwrap();
        meta2.offset_start = Some(20);
        meta2.offset_end = Some(10);
        assert!(meta2.validate().is_err());
    }

    #[test]
    fn no_data_field_exists_by_design() {
        // Test de conception : TensorMetadata ne contient AUCUN champ de données.
        // (Vérification structurelle : les champs sont exclusivement des
        // métadonnées — nom, shape, dtype, shard, offsets, byte_size.)
        let meta = TensorMetadata::new("x", Shape::scalar(), DType::Bool).unwrap();
        let json = serde_json::to_string(&meta).unwrap();
        assert!(
            !json.contains("data"),
            "aucun payload ne doit être sérialisé"
        );
    }

    #[test]
    fn serde_roundtrip() {
        let meta = TensorMetadata::new("w", Shape::new(vec![2, 2]).unwrap(), DType::Bf16).unwrap();
        let json = serde_json::to_string(&meta).unwrap();
        assert_eq!(serde_json::from_str::<TensorMetadata>(&json).unwrap(), meta);
    }
}
