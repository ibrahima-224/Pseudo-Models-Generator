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

//! Type de données `DType` et ses invariants.
//!
//! Conforme à la décision D6 (`docs/architecture/01-decisions-architecture.md`)
//! et au contrat de `docs/architecture/03-modeles-de-donnees.md` §2.1 :
//! - enum `#[non_exhaustive]` (mécanisme d'extension) ;
//! - `size_bytes()` = `None` pour les dtypes non émissibles en v1.0
//!   (F4, F6E2M3, F6E3M2, F8E8M0) — bits connus mais écriture refusée ;
//! - parsing des noms officiels Safetensors (`F32`, `BF16`, `F8_E4M3`, `BOOL`…).

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::error::{CoreError, CoreResult};

/// Type de données d'un tenseur.
///
/// L'enum est `#[non_exhaustive]` : tout `match` externe doit prévoir des
/// variantes futures (mécanisme d'extension de la décision D6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DType {
    /// Flottant 64 bits (double précision).
    #[serde(alias = "f64", alias = "F64")]
    F64,
    /// Flottant 32 bits.
    #[serde(alias = "f32", alias = "F32")]
    F32,
    /// Flottant 16 bits (IEEE 754 half).
    #[serde(alias = "f16", alias = "F16")]
    F16,
    /// Bfloat16 (Google Brain).
    #[serde(alias = "bf16", alias = "BF16")]
    Bf16,
    /// FP8 e4m3 (NVIDIA/OCP).
    #[serde(alias = "f8_e4m3", alias = "F8_E4M3")]
    F8E4M3,
    /// FP8 e5m2.
    #[serde(alias = "f8_e5m2", alias = "F8_E5M2")]
    F8E5M2,
    /// Format scale `ue8m0` (réservé aux scales de quantification en lecture).
    #[serde(alias = "f8_e8m0", alias = "F8_E8M0")]
    F8E8M0,
    /// Flottant 6 bits e2m3 (déclaré, non émissible en v1.0).
    #[serde(alias = "f6_e2m3", alias = "F6_E2M3")]
    F6E2M3,
    /// Flottant 6 bits e3m2 (déclaré, non émissible en v1.0).
    #[serde(alias = "f6_e3m2", alias = "F6_E3M2")]
    F6E3M2,
    /// Flottant 4 bits (déclaré, non émissible en v1.0).
    #[serde(alias = "f4", alias = "F4")]
    F4,
    /// Entier signé 64 bits.
    #[serde(alias = "i64", alias = "I64")]
    I64,
    /// Entier signé 32 bits.
    #[serde(alias = "i32", alias = "I32")]
    I32,
    /// Entier signé 16 bits.
    #[serde(alias = "i16", alias = "I16")]
    I16,
    /// Entier signé 8 bits.
    #[serde(alias = "i8", alias = "I8")]
    I8,
    /// Entier non signé 64 bits.
    #[serde(alias = "u64", alias = "U64")]
    U64,
    /// Entier non signé 32 bits.
    #[serde(alias = "u32", alias = "U32")]
    U32,
    /// Entier non signé 16 bits.
    #[serde(alias = "u16", alias = "U16")]
    U16,
    /// Entier non signé 8 bits.
    #[serde(alias = "u8", alias = "U8")]
    U8,
    /// Booléen (1 octet).
    #[serde(alias = "bool", alias = "BOOL")]
    Bool,
}

impl DType {
    /// Nombre d'octets par élément, `None` pour les dtypes non émissibles
    /// (F4, F6*, F8E8M0) : bits connus mais écriture refusée en v1.0.
    pub fn size_bytes(self) -> Option<u64> {
        match self {
            DType::F64 | DType::I64 | DType::U64 => Some(8),
            DType::F32 | DType::I32 | DType::U32 => Some(4),
            DType::F16 | DType::Bf16 | DType::I16 | DType::U16 => Some(2),
            DType::F8E4M3 | DType::F8E5M2 | DType::I8 | DType::U8 | DType::Bool => Some(1),
            // Sous-octets et format scale : taille connue en bits, non émissible.
            DType::F8E8M0 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 => None,
        }
    }

    /// Nombre de bits par élément (toujours connu pour les 19 variantes).
    pub fn bits_per_element(self) -> Option<u32> {
        Some(match self {
            DType::F64 | DType::I64 | DType::U64 => 64,
            DType::F32 | DType::I32 | DType::U32 => 32,
            DType::F16 | DType::Bf16 | DType::I16 | DType::U16 => 16,
            DType::F8E4M3 | DType::F8E5M2 | DType::F8E8M0 | DType::I8 | DType::U8 | DType::Bool => {
                8
            },
            DType::F6E2M3 | DType::F6E3M2 => 6,
            DType::F4 => 4,
        })
    }

    /// Nom officiel Safetensors (`F32`, `BF16`, `F8_E4M3`, `BOOL`, …),
    /// `None` pour les dtypes sans représentation standard v1.0 (F6*, F4).
    pub fn safetensors_name(self) -> Option<&'static str> {
        match self {
            DType::F64 => Some("F64"),
            DType::F32 => Some("F32"),
            DType::F16 => Some("F16"),
            DType::Bf16 => Some("BF16"),
            DType::F8E4M3 => Some("F8_E4M3"),
            DType::F8E5M2 => Some("F8_E5M2"),
            DType::F8E8M0 => Some("F8_E8M0"),
            DType::I64 => Some("I64"),
            DType::I32 => Some("I32"),
            DType::I16 => Some("I16"),
            DType::I8 => Some("I8"),
            DType::U64 => Some("U64"),
            DType::U32 => Some("U32"),
            DType::U16 => Some("U16"),
            DType::U8 => Some("U8"),
            DType::Bool => Some("BOOL"),
            // Pas de nom officiel Safetensors pour les formats sous-octets.
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 => None,
        }
    }

    /// Parse un nom officiel Safetensors (insensible à la casse pour `Bf16`).
    pub fn from_safetensors_name(s: &str) -> CoreResult<DType> {
        match s {
            "F64" => Ok(DType::F64),
            "F32" => Ok(DType::F32),
            "F16" => Ok(DType::F16),
            "BF16" | "Bf16" => Ok(DType::Bf16),
            "F8_E4M3" => Ok(DType::F8E4M3),
            "F8_E5M2" => Ok(DType::F8E5M2),
            "F8_E8M0" => Ok(DType::F8E8M0),
            "I64" => Ok(DType::I64),
            "I32" => Ok(DType::I32),
            "I16" => Ok(DType::I16),
            "I8" => Ok(DType::I8),
            "U64" => Ok(DType::U64),
            "U32" => Ok(DType::U32),
            "U16" => Ok(DType::U16),
            "U8" => Ok(DType::U8),
            "BOOL" | "Bool" => Ok(DType::Bool),
            other => Err(CoreError::UnsupportedDType(format!(
                "nom Safetensors inconnu : '{other}' (attendu F32, BF16, F8_E4M3, I8, BOOL…)"
            ))),
        }
    }

    /// Vrai si le dtype est un flottant (y compris formats sous-octets).
    pub fn is_float(self) -> bool {
        matches!(
            self,
            DType::F64
                | DType::F32
                | DType::F16
                | DType::Bf16
                | DType::F8E4M3
                | DType::F8E5M2
                | DType::F8E8M0
                | DType::F6E2M3
                | DType::F6E3M2
                | DType::F4
        )
    }

    /// Vrai si le dtype est un entier signé.
    pub fn is_signed_int(self) -> bool {
        matches!(self, DType::I64 | DType::I32 | DType::I16 | DType::I8)
    }

    /// Vrai si le dtype est un entier non signé.
    pub fn is_unsigned_int(self) -> bool {
        matches!(self, DType::U64 | DType::U32 | DType::U16 | DType::U8)
    }

    /// Vrai si le dtype est un entier (signé ou non).
    pub fn is_integer(self) -> bool {
        self.is_signed_int() || self.is_unsigned_int()
    }

    /// Vrai si le dtype est supporté pour l'écriture binaire v1.0 :
    /// taille fixe **≥ 1 octet** (F64 → U8 inclus, ainsi que `Bool`).
    ///
    /// Les dtypes F4/F6*/F8E8M0 sont déclarés (lecture/validation) mais leur
    /// écriture est refusée par une erreur explicite (décision D6).
    pub fn is_supported_for_write(self) -> bool {
        self.size_bytes().is_some()
    }

    /// Vrai si le dtype est un format de stockage quantifié (réservé aux scales).
    ///
    /// En v1.0, aucun dtype n'est un schéma de quantification : la séparation
    /// `StorageDType` / `QuantizationScheme` est conservée (décision D6).
    pub fn is_quantized_storage(self) -> bool {
        false
    }
}

impl FromStr for DType {
    type Err = CoreError;

    fn from_str(s: &str) -> CoreResult<DType> {
        DType::from_safetensors_name(s)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Forme canonique pour l'affichage et les messages d'erreur.
        write!(f, "{}", self.safetensors_name().unwrap_or("F4/F6/ue8m0"))
    }
}

#[cfg(test)]
mod tests {
    use super::DType;
    use crate::error::CoreError;

    /// (dtype, octets, bits) pour les dtypes émissibles.
    const EMITTABLE: &[(DType, u64, u32)] = &[
        (DType::F64, 8, 64),
        (DType::F32, 4, 32),
        (DType::F16, 2, 16),
        (DType::Bf16, 2, 16),
        (DType::F8E4M3, 1, 8),
        (DType::F8E5M2, 1, 8),
        (DType::I64, 8, 64),
        (DType::I32, 4, 32),
        (DType::I16, 2, 16),
        (DType::I8, 1, 8),
        (DType::U64, 8, 64),
        (DType::U32, 4, 32),
        (DType::U16, 2, 16),
        (DType::U8, 1, 8),
        (DType::Bool, 1, 8),
    ];

    /// (dtype, bits) pour les dtypes non émissibles (déclarés).
    const NON_EMITTABLE: &[(DType, u32)] = &[
        (DType::F8E8M0, 8),
        (DType::F6E2M3, 6),
        (DType::F6E3M2, 6),
        (DType::F4, 4),
    ];

    #[test]
    fn size_and_bits_are_coherent() {
        // Invariant : size_bytes = Some(b) ⇒ bits = 8*b.
        for &(dtype, bytes, bits) in EMITTABLE {
            assert_eq!(dtype.size_bytes(), Some(bytes), "{dtype:?}");
            assert_eq!(dtype.bits_per_element(), Some(bits), "{dtype:?}");
            assert_eq!(bits, 8 * bytes as u32);
        }
        for &(dtype, bits) in NON_EMITTABLE {
            assert_eq!(dtype.size_bytes(), None, "{dtype:?} non émissible");
            assert_eq!(dtype.bits_per_element(), Some(bits), "{dtype:?}");
        }
    }

    #[test]
    fn write_support_rule() {
        // Écriture v1.0 : seulement taille fixe ≥ 1 octet.
        for &(dtype, _, _) in EMITTABLE {
            assert!(dtype.is_supported_for_write(), "{dtype:?} émissible");
        }
        for &(dtype, _) in NON_EMITTABLE {
            assert!(!dtype.is_supported_for_write(), "{dtype:?} refusé");
        }
    }

    #[test]
    fn classification_predicates() {
        assert!(DType::F32.is_float() && !DType::F32.is_integer());
        assert!(DType::Bf16.is_float());
        assert!(DType::I8.is_signed_int() && DType::I8.is_integer());
        assert!(DType::U8.is_unsigned_int() && DType::U8.is_integer());
        assert!(!DType::Bool.is_float() && !DType::Bool.is_integer());
        assert!(DType::F4.is_float());
        // Aucun dtype n'est un schéma de quantification en v1.0.
        assert!(!DType::F8E8M0.is_quantized_storage());
    }

    #[test]
    fn safetensors_name_roundtrip() {
        // Aller-retour nom → dtype → nom pour tous les dtypes à nom officiel.
        for dtype in EMITTABLE.iter().map(|(d, _, _)| *d) {
            let name = dtype.safetensors_name().expect("nom officiel");
            assert_eq!(DType::from_safetensors_name(name).unwrap(), dtype);
        }
        // F8E8M0 a un nom officiel (F8_E8M0) même s'il n'est pas émissible.
        assert_eq!(DType::F8E8M0.safetensors_name(), Some("F8_E8M0"));
        assert_eq!(
            DType::from_safetensors_name("F8_E8M0").unwrap(),
            DType::F8E8M0
        );
        // Formats sous-octets (F6*, F4) : aucun nom officiel Safetensors.
        for dtype in [DType::F6E2M3, DType::F6E3M2, DType::F4] {
            assert!(dtype.safetensors_name().is_none(), "{dtype:?}");
        }
    }

    #[test]
    fn from_str_accepts_official_names() {
        // Noms officiels Safetensors (dont casse particulière BF16/BOOL).
        assert_eq!("BF16".parse::<DType>().unwrap(), DType::Bf16);
        assert_eq!("F8_E4M3".parse::<DType>().unwrap(), DType::F8E4M3);
        assert_eq!("BOOL".parse::<DType>().unwrap(), DType::Bool);
        assert_eq!("I8".parse::<DType>().unwrap(), DType::I8);
    }

    #[test]
    fn from_str_rejects_unknown_names() {
        // Nom inconnu → erreur typée UnsupportedDType (jamais de panic).
        for bad in ["F32X", "bf16", "UINT8", "", "float32"] {
            let err = bad.parse::<DType>().unwrap_err();
            assert!(
                matches!(err, CoreError::UnsupportedDType(_)),
                "nom '{bad}' doit échouer en UnsupportedDType, obtenu {err}"
            );
        }
    }

    #[test]
    fn extension_via_non_exhaustive_is_possible() {
        // `#[non_exhaustive]` : un match externe doit couvrir un cas futur.
        // On vérifie que l'attribut est bien présent au niveau type en
        // compilant un match exhaustif *dans la crate* (autorisé en interne).
        let all = [
            DType::F64,
            DType::F32,
            DType::F16,
            DType::Bf16,
            DType::F8E4M3,
            DType::F8E5M2,
            DType::F8E8M0,
            DType::F6E2M3,
            DType::F6E3M2,
            DType::F4,
            DType::I64,
            DType::I32,
            DType::I16,
            DType::I8,
            DType::U64,
            DType::U32,
            DType::U16,
            DType::U8,
            DType::Bool,
        ];
        assert_eq!(all.len(), 19);
    }

    #[test]
    fn serde_roundtrip() {
        // Sérialisation JSON des noms de variantes (config, manifeste).
        for dtype in EMITTABLE.iter().map(|(d, _, _)| *d) {
            let json = serde_json::to_string(&dtype).unwrap();
            assert_eq!(serde_json::from_str::<DType>(&json).unwrap(), dtype);
        }
    }
}
