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

//! Générateur de tenseurs individuels.
//!
//! Ce module responsable de la génération des valeurs initiales d'un seul tenseur
//! à partir de sa spécification (`TensorSpec`) et d'une seed déterministe.
//!
//! # Pipeline
//!
//! ```text
//! TensorSpec
//!    ↓
//! DistributionSpec → DistributionConfig
//!    ↓
//! RNG déterministe (seed dérivée)
//!    ↓
//! valeurs (Vec<f64>)
//! ```
//!
//! Le générateur ne modifie pas les valeurs : il les produit uniquement.
//! Les injections structurelles sont gérées par `pmg-injector`.

use std::io::Write;

use pmg_blueprint::tensor_spec::{DistributionFamily, TensorSpec};
use pmg_core::distribution_config::{DistributionConfig, DistributionKind};
use pmg_math::distribution::from_config;

use crate::error::{GeneratorError, GeneratorResult};
use crate::seed_plan::GeneratorSeedPlan;

/// Générateur de tenseurs individuels.
///
/// Génère les valeurs initiales d'un tenseur selon sa spécification
/// et une seed déterministe.
pub struct TensorGenerator {
    /// Spécification du tenseur.
    spec: TensorSpec,
    /// Plan de seed pour la dérivation.
    seed_plan: GeneratorSeedPlan,
    /// Budget en octets pour ce tenseur (None = pas de limite).
    budget: Option<u64>,
}

impl TensorGenerator {
    /// Crée un nouveau générateur pour un tenseur donné.
    ///
    /// # Paramètres
    /// - `spec` : spécification du tenseur (blueprint)
    /// - `seed_plan` : plan de seed global
    /// - `budget` : budget en octets pour ce tenseur (None = pas de limite)
    ///
    /// # Retourne
    /// Un générateur prêt à produire les valeurs du tenseur.
    pub fn new(spec: TensorSpec, seed_plan: GeneratorSeedPlan, budget: Option<u64>) -> Self {
        Self {
            spec,
            seed_plan,
            budget,
        }
    }

    /// Génère les valeurs initiales du tenseur.
    ///
    /// # Retourne
    /// Un vecteur de `f64` contenant les valeurs générées, de longueur
    /// égale au nombre d'éléments du tenseur.
    ///
    /// # Erreurs
    /// Retourne une erreur si la distribution est invalide ou si le
    /// nombre d'éléments ne peut être calculé.
    pub fn generate(&self) -> GeneratorResult<Vec<f64>> {
        let n = self.spec.num_elements()? as usize;
        let dist_config = self.distribution_config()?;
        let mut dist = from_config(&dist_config).map_err(GeneratorError::Math)?;

        // Dériver la seed du tenseur
        let layer_id = self.spec.layer_id.map(|l| l as u32);
        let mut rng = self.seed_plan.tensor_rng(&self.spec.name, layer_id);

        // Calculer le nombre d'éléments selon le budget
        let n = if let Some(budget) = self.budget {
            let element_size = self.spec.dtype.size_bytes().unwrap_or(4);
            let max_elements = (budget / element_size) as usize;
            if max_elements == 0 {
                return Err(GeneratorError::BudgetExceeded {
                    tensor_name: self.spec.name.clone(),
                    budget,
                    required_bytes: element_size,
                });
            }
            n.min(max_elements)
        } else {
            n
        };

        // Générer les valeurs
        let mut values = Vec::with_capacity(n);
        for _ in 0..n {
            values.push(dist.sample(&mut rng));
        }

        Ok(values)
    }

    /// Génère les valeurs du tenseur et les écrit directement dans un writer.
    ///
    /// Cette méthode évite l'allocation massive d'un `Vec<f64>` contenant toutes
    /// les valeurs. Elle génère par chunks et écrit directement sur le disque,
    /// ce qui garantit une consommation mémoire bornée à O(chunk_size).
    ///
    /// # Paramètres
    /// - `writer` : writer binaire dans lequel écrire les valeurs (format little-endian).
    ///
    /// # Retourne
    /// `Ok(())` si l'écriture réussit.
    ///
    /// # Erreurs
    /// Retourne une erreur si la distribution est invalide, si le budget est dépassé,
    /// ou si l'écriture échoue.
    pub fn generate_to_writer<W: Write>(&self, writer: &mut W) -> GeneratorResult<()> {
        let n = self.spec.num_elements()? as usize;
        let dist_config = self.distribution_config()?;
        let mut dist = from_config(&dist_config).map_err(GeneratorError::Math)?;

        // Dériver la seed du tenseur
        let layer_id = self.spec.layer_id.map(|l| l as u32);
        let mut rng = self.seed_plan.tensor_rng(&self.spec.name, layer_id);

        // Calculer le nombre d'éléments selon le budget
        let n = if let Some(budget) = self.budget {
            let element_size = self.spec.dtype.size_bytes().unwrap_or(4);
            let max_elements = (budget / element_size) as usize;
            if max_elements == 0 {
                return Err(GeneratorError::BudgetExceeded {
                    tensor_name: self.spec.name.clone(),
                    budget,
                    required_bytes: element_size,
                });
            }
            n.min(max_elements)
        } else {
            n
        };

        // Taille du chunk : 8 Mo de f64 = 1 million d'éléments
        const CHUNK_SIZE: usize = 8 * 1024 * 1024 / std::mem::size_of::<f64>();

        // Génération par chunks avec écriture directe
        let mut remaining = n;
        while remaining > 0 {
            let chunk_len = remaining.min(CHUNK_SIZE);

            // Générer un chunk de valeurs
            let chunk: Vec<f64> = (0..chunk_len).map(|_| dist.sample(&mut rng)).collect();

            // Écriture directe en little-endian
            for &value in &chunk {
                writer
                    .write_all(&value.to_le_bytes())
                    .map_err(|e| GeneratorError::Internal(format!("erreur d'écriture : {e}")))?;
            }

            remaining -= chunk_len;
        }

        Ok(())
    }

    /// Convertit la DistributionSpec du blueprint en DistributionConfig pour pmg-math.
    fn distribution_config(&self) -> GeneratorResult<DistributionConfig> {
        let spec = &self.spec.distribution;
        let family = match spec.family {
            DistributionFamily::Normal => DistributionKind::Normal,
            DistributionFamily::StudentT => DistributionKind::StudentT,
            DistributionFamily::Laplace => DistributionKind::Laplace,
            DistributionFamily::LogNormal => DistributionKind::LogNormal,
            DistributionFamily::Weibull => DistributionKind::Weibull,
            DistributionFamily::Pareto => DistributionKind::Pareto,
            DistributionFamily::Mixture => DistributionKind::Mixture,
            DistributionFamily::Uniform => DistributionKind::Uniform,
            _ => {
                // Fallback pour les futures familles non exhaustives
                DistributionKind::Normal
            },
        };

        // Mapping des paramètres selon la famille
        let (p1, p2) = match family {
            DistributionKind::Normal => (spec.mean, Some(spec.stddev)),
            DistributionKind::StudentT => (spec.stddev, None), // df = stddev comme approximation
            DistributionKind::Laplace => (spec.mean, Some(spec.stddev)),
            DistributionKind::LogNormal => (spec.mean, Some(spec.stddev)),
            DistributionKind::Weibull => (spec.stddev, Some(spec.mean)), // scale=stddev, shape=mean
            DistributionKind::Pareto => (spec.mean, Some(spec.stddev)),  // xm=mean, alpha=stddev
            DistributionKind::Uniform => (spec.mean, Some(spec.stddev)), // min=mean, max=stddev
            DistributionKind::Mixture => (0.0, None),                    // Sera géré séparément
        };

        // Construction des composantes de mélange si nécessaire
        let mixture_components = if family == DistributionKind::Mixture {
            match &spec.mixture_components {
                Some(components) => {
                    // Convertir chaque MixtureComponent en (f64, DistributionConfig)
                    let mut converted = Vec::with_capacity(components.len());
                    for comp in components {
                        let sub_config = self.distribution_config_for_spec(&comp.distribution)?;
                        converted.push((comp.weight, sub_config));
                    }
                    converted
                },
                None => {
                    // Retourner un mélange vide par défaut (comportement rétrocompatible)
                    Vec::new()
                },
            }
        } else {
            Vec::new()
        };

        Ok(DistributionConfig {
            kind: family,
            p1,
            p2,
            mixture_components,
        })
    }

    /// Retourne la spécification du tenseur.
    pub fn spec(&self) -> &TensorSpec {
        &self.spec
    }

    /// Convertit une DistributionSpec en DistributionConfig (récursif pour les mélanges).
    fn distribution_config_for_spec(
        &self,
        spec: &pmg_blueprint::tensor_spec::DistributionSpec,
    ) -> GeneratorResult<DistributionConfig> {
        let family = match spec.family {
            DistributionFamily::Normal => DistributionKind::Normal,
            DistributionFamily::StudentT => DistributionKind::StudentT,
            DistributionFamily::Laplace => DistributionKind::Laplace,
            DistributionFamily::LogNormal => DistributionKind::LogNormal,
            DistributionFamily::Weibull => DistributionKind::Weibull,
            DistributionFamily::Pareto => DistributionKind::Pareto,
            DistributionFamily::Mixture => DistributionKind::Mixture,
            DistributionFamily::Uniform => DistributionKind::Uniform,
            _ => DistributionKind::Normal,
        };

        let (p1, p2) = match family {
            DistributionKind::Normal => (spec.mean, Some(spec.stddev)),
            DistributionKind::StudentT => (spec.stddev, None),
            DistributionKind::Laplace => (spec.mean, Some(spec.stddev)),
            DistributionKind::LogNormal => (spec.mean, Some(spec.stddev)),
            DistributionKind::Weibull => (spec.stddev, Some(spec.mean)),
            DistributionKind::Pareto => (spec.mean, Some(spec.stddev)),
            DistributionKind::Uniform => (spec.mean, Some(spec.stddev)),
            DistributionKind::Mixture => (0.0, None),
        };

        let mixture_components = if family == DistributionKind::Mixture {
            match &spec.mixture_components {
                Some(components) => {
                    let mut converted = Vec::with_capacity(components.len());
                    for comp in components {
                        let sub_config = self.distribution_config_for_spec(&comp.distribution)?;
                        converted.push((comp.weight, sub_config));
                    }
                    converted
                },
                None => Vec::new(),
            }
        } else {
            Vec::new()
        };

        Ok(DistributionConfig {
            kind: family,
            p1,
            p2,
            mixture_components,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmg_core::{DType, Shape, TensorRole};

    fn test_spec() -> TensorSpec {
        TensorSpec::new(
            "model.layers.0.mlp.gate.weight",
            Shape::new(vec![64, 32]).unwrap(),
            DType::F32,
            TensorRole::MlpGate,
        )
        .unwrap()
    }

    fn test_seed_plan() -> GeneratorSeedPlan {
        GeneratorSeedPlan::new(42, "glm-5.2", "1.0.0")
    }

    #[test]
    fn generator_creation() {
        let spec = test_spec();
        let seed_plan = test_seed_plan();
        let gen = TensorGenerator::new(spec.clone(), seed_plan, None);
        assert_eq!(gen.spec().name, spec.name);
    }

    #[test]
    fn generate_produces_correct_length() {
        let spec = test_spec();
        let seed_plan = test_seed_plan();
        let gen = TensorGenerator::new(spec, seed_plan, None);
        let values = gen.generate().unwrap();
        assert_eq!(values.len(), 64 * 32);
    }

    #[test]
    fn generate_deterministic() {
        let spec = test_spec();
        let seed_plan = test_seed_plan();
        let gen1 = TensorGenerator::new(spec.clone(), seed_plan.clone(), None);
        let gen2 = TensorGenerator::new(spec, seed_plan, None);
        let values1 = gen1.generate().unwrap();
        let values2 = gen2.generate().unwrap();
        assert_eq!(values1, values2);
    }

    #[test]
    fn generate_different_seeds_different_values() {
        let spec = test_spec();
        let seed_plan1 = GeneratorSeedPlan::new(42, "glm-5.2", "1.0.0");
        let seed_plan2 = GeneratorSeedPlan::new(43, "glm-5.2", "1.0.0");
        let gen1 = TensorGenerator::new(spec.clone(), seed_plan1, None);
        let gen2 = TensorGenerator::new(spec, seed_plan2, None);
        let values1 = gen1.generate().unwrap();
        let values2 = gen2.generate().unwrap();
        assert_ne!(values1, values2);
    }

    #[test]
    fn distribution_config_mapping() {
        let spec = test_spec();
        let seed_plan = test_seed_plan();
        let gen = TensorGenerator::new(spec, seed_plan, None);
        let config = gen.distribution_config().unwrap();
        assert_eq!(config.kind, DistributionKind::Normal);
        assert_eq!(config.p1, 0.0); // mean par défaut
        assert_eq!(config.p2, Some(1.0)); // stddev par défaut
    }

    #[test]
    fn uniform_distribution_mapping() {
        // Créer un spec avec family Uniform
        let mut spec = test_spec();
        spec.distribution.family = DistributionFamily::Uniform;
        spec.distribution.mean = 0.0;
        spec.distribution.stddev = 1.0;
        let seed_plan = test_seed_plan();
        let gen = TensorGenerator::new(spec, seed_plan, None);
        let config = gen.distribution_config().unwrap();
        assert_eq!(config.kind, DistributionKind::Uniform);
        assert_eq!(config.p1, 0.0); // min
        assert_eq!(config.p2, Some(1.0)); // max
    }

    #[test]
    fn mixture_distribution_with_components() {
        use pmg_blueprint::tensor_spec::{DistributionSpec, MixtureComponent};
        let mut spec = test_spec();
        spec.distribution = DistributionSpec::mixture(vec![
            MixtureComponent {
                weight: 0.7,
                distribution: DistributionSpec::standard(),
            },
            MixtureComponent {
                weight: 0.3,
                distribution: DistributionSpec::uniform(0.0, 1.0),
            },
        ]);
        let seed_plan = test_seed_plan();
        let gen = TensorGenerator::new(spec, seed_plan, None);
        let config = gen.distribution_config().unwrap();
        assert_eq!(config.kind, DistributionKind::Mixture);
        assert_eq!(config.mixture_components.len(), 2);
        assert!((config.mixture_components[0].0 - 0.7).abs() < 1e-10);
        assert_eq!(
            config.mixture_components[0].1.kind,
            DistributionKind::Normal
        );
        assert!((config.mixture_components[1].0 - 0.3).abs() < 1e-10);
        assert_eq!(
            config.mixture_components[1].1.kind,
            DistributionKind::Uniform
        );
    }

    #[test]
    fn mixture_distribution_empty_components() {
        // Comportement rétrocompatible : mélange sans composantes
        let mut spec = test_spec();
        spec.distribution.family = DistributionFamily::Mixture;
        spec.distribution.mixture_components = None;
        let seed_plan = test_seed_plan();
        let gen = TensorGenerator::new(spec, seed_plan, None);
        let config = gen.distribution_config().unwrap();
        assert_eq!(config.kind, DistributionKind::Mixture);
        assert!(config.mixture_components.is_empty());
    }

    #[test]
    fn generate_with_budget() {
        let spec = test_spec();
        let seed_plan = test_seed_plan();
        // Budget pour 1000 éléments (4 octets chacun) = 4000 octets
        let budget = Some(4000);
        let gen = TensorGenerator::new(spec, seed_plan, budget);
        let values = gen.generate().unwrap();
        // Le nombre d'éléments doit être min(64*32, 4000/4) = min(2048, 1000) = 1000
        assert_eq!(values.len(), 1000);
    }

    #[test]
    fn budget_insufficient() {
        let spec = test_spec();
        let seed_plan = test_seed_plan();
        // Budget trop petit pour un seul élément (4 octets)
        let budget = Some(3);
        let gen = TensorGenerator::new(spec, seed_plan, budget);
        let result = gen.generate();
        assert!(result.is_err());
        // Vérifier le type d'erreur
        match result.unwrap_err() {
            GeneratorError::BudgetExceeded { .. } => {},
            other => panic!("Erreur attendue BudgetExceeded, reçue {:?}", other),
        }
    }

    #[test]
    fn budget_none_no_limit() {
        let spec = test_spec();
        let seed_plan = test_seed_plan();
        let gen = TensorGenerator::new(spec, seed_plan, None);
        let values = gen.generate().unwrap();
        assert_eq!(values.len(), 64 * 32);
    }
}
