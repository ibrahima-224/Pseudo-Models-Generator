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

//! Séparation `StorageDType` / `QuantizationScheme` (décision D6).
//!
//! Les schémas de quantification (NF4, GPTQ, AWQ, FP8 e4m3 block-wise…) ne
//! sont **pas** des `DType` : ils décrivent *comment* des poids sont encodés
//! au-delà du dtype de stockage. En v1.0 la quantification est **non
//! émissible** ; ces types servent à la lecture/validation de modèles
//! quantifiés (DeepSeek FP8/FP4).

use serde::{Deserialize, Serialize};

/// Dtype de stockage déclaré (`StorageDType`) — alias sémantique de
/// [`DType`](crate::DType) utilisé pour insister sur le rôle « stockage »
/// dans les configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StorageDType(pub crate::dtype::DType);

/// Schéma de quantification déclaré dans un `config.json` source.
///
/// En v1.0 aucun schéma n'est émissible : la génération refuse toute
/// quantification (erreur `UnsupportedDType` au moment du plan).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantizationScheme {
    /// NF4 (QLoRA, 4 bits normalisés).
    Nf4,
    /// GPTQ (quantification par lignes avec calibration).
    Gptq,
    /// AWQ (activation-aware weight quantization).
    Awq,
    /// FP8 e4m3 block-wise (DeepSeek-V4-Flash).
    Fp8E4M3,
    /// FP4 (DeepSeek expert_dtype, non émissible).
    Fp4,
}

impl QuantizationScheme {
    /// Nom canonique du schéma (forme sérialisée).
    pub fn name(self) -> &'static str {
        match self {
            QuantizationScheme::Nf4 => "nf4",
            QuantizationScheme::Gptq => "gptq",
            QuantizationScheme::Awq => "awq",
            QuantizationScheme::Fp8E4M3 => "fp8_e4m3",
            QuantizationScheme::Fp4 => "fp4",
        }
    }

    /// Vrai si le schéma peut être émis en v1.0 (toujours `false`).
    ///
    /// La quantification reste déclarée dans les métadonnées mais aucune
    /// écriture n'est possible : invariant d'honnêteté (décision D6).
    pub fn is_emittable(self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::{QuantizationScheme, StorageDType};
    use crate::dtype::DType;

    #[test]
    fn quantization_schemes_are_never_emittable_in_v1() {
        for scheme in [
            QuantizationScheme::Nf4,
            QuantizationScheme::Gptq,
            QuantizationScheme::Awq,
            QuantizationScheme::Fp8E4M3,
            QuantizationScheme::Fp4,
        ] {
            assert!(!scheme.is_emittable(), "{}", scheme.name());
        }
    }

    #[test]
    fn storage_dtype_is_distinct_from_quantization() {
        // La séparation est structurelle : StorageDType ⊂ DType, la
        // quantification est un type orthogonal.
        let s = StorageDType(DType::Bf16);
        assert_eq!(s.0, DType::Bf16);
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(serde_json::from_str::<StorageDType>(&json).unwrap(), s);
    }

    #[test]
    fn quantization_serde_roundtrip() {
        for scheme in [
            QuantizationScheme::Nf4,
            QuantizationScheme::Fp8E4M3,
            QuantizationScheme::Fp4,
        ] {
            let json = serde_json::to_string(&scheme).unwrap();
            assert_eq!(
                serde_json::from_str::<QuantizationScheme>(&json).unwrap(),
                scheme
            );
        }
    }
}
