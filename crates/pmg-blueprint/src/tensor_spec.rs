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

//! Spécification de génération d'un tenseur (`TensorSpec`).
//!
//! `TensorSpec` décrit **quoi générer** (nom, shape, dtype, rôle, politique de
//! distribution) sans contenir aucune valeur — c'est l'élément de base du
//! blueprint, consommé par le planner puis le générateur.
//! Référence : `docs/architecture/03-modeles-de-donnees.md` §3.3.

use serde::{Deserialize, Serialize};

use pmg_core::{CoreError, CoreResult, DType, Shape, TensorRole};

/// Famille de distribution statistique cible pour un tenseur.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum DistributionFamily {
    /// Distribution normale (défaut pour les poids).
    Normal,
    /// T-distribution de Student (queues lourdes).
    StudentT,
    /// Distribution de Laplace.
    Laplace,
    /// Log-normale (valeurs positives).
    LogNormal,
    /// Weibull.
    Weibull,
    /// Pareto (queues très lourdes).
    Pareto,
    /// Mélange de distributions.
    Mixture,
    /// Uniforme (embeddings, routeurs).
    Uniform,
}

impl DistributionFamily {
    /// Nom canonique (forme sérialisée).
    pub fn name(self) -> &'static str {
        match self {
            DistributionFamily::Normal => "normal",
            DistributionFamily::StudentT => "student-t",
            DistributionFamily::Laplace => "laplace",
            DistributionFamily::LogNormal => "log-normal",
            DistributionFamily::Weibull => "weibull",
            DistributionFamily::Pareto => "pareto",
            DistributionFamily::Mixture => "mixture",
            DistributionFamily::Uniform => "uniform",
        }
    }
}

/// Composante d'un mélange de distributions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MixtureComponent {
    /// Poids de la composante (doit sommer à 1.0 avec les autres).
    pub weight: f64,
    /// Distribution de la composante.
    pub distribution: DistributionSpec,
}

/// Spécification de distribution : famille + paramètres normalisés.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistributionSpec {
    /// Famille de distribution.
    pub family: DistributionFamily,
    /// Moyenne cible.
    pub mean: f64,
    /// Écart-type cible.
    pub stddev: f64,
    /// Composantes du mélange (uniquement si family == Mixture).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mixture_components: Option<Vec<MixtureComponent>>,
}

impl DistributionSpec {
    /// Distribution normale standard (défaut conservateur).
    pub fn standard() -> DistributionSpec {
        DistributionSpec {
            family: DistributionFamily::Normal,
            mean: 0.0,
            stddev: 1.0,
            mixture_components: None,
        }
    }

    /// Distribution uniforme `U(min, max)`.
    pub fn uniform(min: f64, max: f64) -> DistributionSpec {
        DistributionSpec {
            family: DistributionFamily::Uniform,
            mean: min,
            stddev: max,
            mixture_components: None,
        }
    }

    /// Mélange de distributions avec composantes pondérées.
    ///
    /// # Erreurs
    /// Aucune validation des poids ici (validée à la construction de la distribution).
    pub fn mixture(components: Vec<MixtureComponent>) -> DistributionSpec {
        DistributionSpec {
            family: DistributionFamily::Mixture,
            mean: 0.0,
            stddev: 1.0,
            mixture_components: Some(components),
        }
    }
}

/// Spécification de structure (bas-rang, corrélation, sparse, blocs).
///
/// Les paramètres détaillés relèvent de `pmg-math`/`pmg-injector` (sprints 3-9) ;
/// cette structure en pose le contrat minimal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct StructureSpec {
    /// Force de structure dans [0, 1] (0 = i.i.d., 1 = totalement structuré).
    pub strength: f64,
    /// Rang cible pour les composantes bas-rang (optionnel).
    pub low_rank: Option<u64>,
    /// Densité cible pour les structures sparse (optionnel).
    pub sparsity: Option<f64>,
}

impl StructureSpec {
    /// Structure neutre (aucune structure imposée).
    pub fn none() -> StructureSpec {
        StructureSpec {
            strength: 0.0,
            low_rank: None,
            sparsity: None,
        }
    }
}

/// Spécification d'outliers (super-poids).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct OutlierSpec {
    /// Probabilité d'un élément outlier.
    pub probability: f64,
    /// Amplitude relative des outliers (multiplicateur de l'écart-type).
    pub amplitude: f64,
    /// Localité des outliers (`row`, `column`, `block`, `scatter`).
    pub locality: String,
}

impl OutlierSpec {
    /// Aucun outlier (politique par défaut).
    pub fn none() -> OutlierSpec {
        OutlierSpec {
            probability: 0.0,
            amplitude: 0.0,
            locality: "scatter".to_string(),
        }
    }
}

/// Spécification de génération d'un tenseur (aucune donnée).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorSpec {
    /// Nom du tenseur conforme à l'index (ex. `model.layers.0.mlp.gate.weight`).
    pub name: String,
    /// Shape du tenseur.
    pub shape: Shape,
    /// Dtype de stockage cible.
    pub dtype: DType,
    /// Rôle fonctionnel (pilote distribution/injection).
    pub role: TensorRole,
    /// Index de couche parente (None pour les tenseurs non-couches).
    pub layer_id: Option<u64>,
    /// Index d'expert (None pour non-experts).
    pub expert_id: Option<u64>,
    /// Distribution statistique cible.
    pub distribution: DistributionSpec,
    /// Structure imposée.
    pub structure: StructureSpec,
    /// Outliers.
    pub outlier: OutlierSpec,
    /// Provenance de la spécification (`EXACT` si shape/dtype observés).
    pub provenance: pmg_core::Origin,
}

impl TensorSpec {
    /// Construit une spécification minimale avec les politiques par défaut.
    pub fn new(
        name: impl Into<String>,
        shape: Shape,
        dtype: DType,
        role: TensorRole,
    ) -> CoreResult<TensorSpec> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(CoreError::invalid_shape(
                "le nom d'une TensorSpec ne peut pas être vide".to_string(),
            ));
        }
        Ok(TensorSpec {
            name,
            shape,
            dtype,
            role,
            layer_id: None,
            expert_id: None,
            distribution: DistributionSpec::standard(),
            structure: StructureSpec::none(),
            outlier: OutlierSpec::none(),
            provenance: pmg_core::Origin::Derived,
        })
    }

    /// Nombre d'éléments du tenseur (produit vérifié).
    pub fn num_elements(&self) -> CoreResult<u64> {
        self.shape.num_elements()
    }

    /// Taille en octets si le dtype est émissible (`None` sinon).
    pub fn byte_size(&self) -> CoreResult<Option<u64>> {
        match self.dtype.size_bytes() {
            Some(bytes) => {
                let n = self.num_elements()?;
                n.checked_mul(bytes).map(Some).ok_or_else(|| {
                    CoreError::Overflow(format!(
                        "taille en octets de '{}' dépasse u64::MAX",
                        self.name
                    ))
                })
            },
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DistributionFamily, DistributionSpec, OutlierSpec, StructureSpec, TensorSpec};
    use pmg_core::{DType, Shape, TensorRole};

    #[test]
    fn default_specs_are_neutral() {
        assert_eq!(
            DistributionSpec::standard().family,
            DistributionFamily::Normal
        );
        assert_eq!(StructureSpec::none().strength, 0.0);
        assert_eq!(OutlierSpec::none().probability, 0.0);
    }

    #[test]
    fn tensor_spec_byte_size() {
        let spec = TensorSpec::new(
            "model.embed_tokens.weight",
            Shape::new(vec![154880, 6144]).unwrap(),
            DType::Bf16,
            TensorRole::Embedding,
        )
        .unwrap();
        assert_eq!(spec.byte_size().unwrap(), Some(154880 * 6144 * 2));
        assert_eq!(spec.role, TensorRole::Embedding);
        assert_eq!(spec.provenance, pmg_core::Origin::Derived);
    }

    #[test]
    fn non_emittable_dtype_byte_size_is_none() {
        let spec = TensorSpec::new(
            "e.weight",
            Shape::new(vec![4]).unwrap(),
            DType::F4,
            TensorRole::Other,
        )
        .unwrap();
        assert_eq!(spec.byte_size().unwrap(), None);
    }

    #[test]
    fn empty_name_is_rejected() {
        assert!(TensorSpec::new("", Shape::scalar(), DType::F32, TensorRole::Other).is_err());
    }

    #[test]
    fn serde_roundtrip() {
        let spec = TensorSpec::new(
            "layers.0.ffn.gate.weight",
            Shape::new(vec![4096, 256]).unwrap(),
            DType::F32,
            TensorRole::MoeRouter,
        )
        .unwrap();
        let json = serde_json::to_string(&spec).unwrap();
        assert_eq!(serde_json::from_str::<TensorSpec>(&json).unwrap(), spec);
    }

    #[test]
    fn mixture_component_serde_roundtrip() {
        use super::MixtureComponent;
        let comp1 = MixtureComponent {
            weight: 0.7,
            distribution: DistributionSpec::standard(),
        };
        let comp2 = MixtureComponent {
            weight: 0.3,
            distribution: DistributionSpec::uniform(0.0, 1.0),
        };
        let dist = DistributionSpec::mixture(vec![comp1, comp2]);
        let json = serde_json::to_string(&dist).unwrap();
        let back: DistributionSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(dist, back);
    }

    #[test]
    fn mixture_tensor_spec_serde_roundtrip() {
        let mut spec = TensorSpec::new(
            "model.mlp.gate.weight",
            Shape::new(vec![64, 32]).unwrap(),
            DType::F32,
            TensorRole::MlpGate,
        )
        .unwrap();
        spec.distribution = DistributionSpec::mixture(vec![
            super::MixtureComponent {
                weight: 0.7,
                distribution: DistributionSpec::standard(),
            },
            super::MixtureComponent {
                weight: 0.3,
                distribution: DistributionSpec::uniform(-1.0, 1.0),
            },
        ]);
        let json = serde_json::to_string(&spec).unwrap();
        let back: TensorSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(spec, back);
    }
}
