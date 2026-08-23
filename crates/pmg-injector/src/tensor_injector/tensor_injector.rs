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

//! Sous-module contenant l'orchestrateur d'injection de tenseurs.

use pmg_blueprint::tensor_spec::TensorSpec;
use pmg_math::distribution::Distribution;
use pmg_math::distributions::StudentT;
use pmg_math::rng::{derive_seed, derive_sub_seed, DeterministicRng, SeedPlan};

use crate::correlated::generate_correlated_columns;
use crate::distribution_mapping::distribution_from_family;
use crate::error::{InjectorError, InjectorResult};
use crate::injection_policy::InjectionPolicy;
use crate::low_rank::{inject_low_rank, LowRankInjection};
use crate::outlier_mask::OutlierMask;
use crate::sparse_structure::generate_sparse_structure;

use super::helpers::{matrix_dims, sparse_spec_from_policy};
use super::injection_stage::InjectionStage;

/// Orchestrateur d'injection d'un tenseur.
///
/// Construit depuis une [`TensorSpec`] (blueprint), une [`InjectionPolicy`]
/// et une seed racine (plan ou seed brute). Toutes les étapes sont dérivées de
/// cette seed : mêmes entrées ⇒ même tenseur final, bit à bit.
#[derive(Debug, Clone)]
pub struct TensorInjector {
    spec: TensorSpec,
    policy: InjectionPolicy,
    seed: [u8; 32],
}

impl TensorInjector {
    /// Construit l'injecteur depuis un plan de seed canonique.
    ///
    /// # Complexité
    /// O(longueur des champs du plan) — dérivation SHA-256.
    pub fn from_seed_plan(spec: &TensorSpec, policy: InjectionPolicy, plan: &SeedPlan<'_>) -> Self {
        Self::from_seed(spec, policy, derive_seed(plan))
    }

    /// Construit l'injecteur depuis une seed brute de 32 octets.
    pub fn from_seed(spec: &TensorSpec, policy: InjectionPolicy, seed: [u8; 32]) -> Self {
        Self {
            spec: spec.clone(),
            policy,
            seed,
        }
    }

    /// Politique d'injection utilisée (lecture seule).
    pub fn policy(&self) -> &InjectionPolicy {
        &self.policy
    }

    /// Spécification du tenseur cible (lecture seule).
    pub fn spec(&self) -> &TensorSpec {
        &self.spec
    }

    /// Génère le tenseur complet : distribution de base puis pipeline
    /// canonique [`InjectionStage::ORDER`].
    ///
    /// # Erreurs
    /// [`InjectorError::InvalidTensor`] si la shape est inadmissible ;
    /// [`InjectorError::InvalidPolicy`] si la distribution n'est pas
    /// mappable.
    ///
    /// # Complexité
    /// O(n) pour la génération + coûts des étapes (O(n) à O(n·r)).
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_blueprint::tensor_spec::TensorSpec;
    /// use pmg_core::{DType, Shape, TensorRole};
    /// use pmg_injector::injection_policy::InjectionPolicy;
    /// use pmg_injector::tensor_injector::TensorInjector;
    /// use pmg_math::rng::SeedPlan;
    ///
    /// let spec = TensorSpec::new(
    ///     "model.layers.0.mlp.gate.weight",
    ///     Shape::new(vec![64, 32]).unwrap(),
    ///     DType::F32,
    ///     TensorRole::Other,
    /// ).unwrap();
    ///
    /// let plan = SeedPlan {
    ///     seed_global: 42,
    ///     model_id: "glm-5.2",
    ///     tensor_name: &spec.name,
    ///     layer_id: Some(0),
    ///     generation_version: "1.0.0",
    /// };
    ///
    /// let injector = TensorInjector::from_seed_plan(&spec, InjectionPolicy::default(), &plan);
    /// let tensor = injector.inject().expect("injection valide");
    /// assert_eq!(tensor.len(), 64 * 32);
    /// ```
    pub fn inject(&self) -> InjectorResult<Vec<f64>> {
        let n = self.spec.shape.num_elements_usize()?;
        let mut buffer = Vec::with_capacity(n);
        let mut dist = distribution_from_family(
            self.spec.distribution.family,
            self.spec.distribution.mean,
            self.spec.distribution.stddev,
        )?;
        let mut rng = self.stage_rng(InjectionStage::Distribution);
        for _ in 0..n {
            buffer.push(dist.sample(&mut rng));
        }
        self.apply_to(&mut buffer)?;
        Ok(buffer)
    }

    /// Applique le pipeline d'injection (hors distribution) sur un tenseur de
    /// base déjà fourni, dans l'ordre canonique.
    ///
    /// Le buffer est modifié sur place. La longueur doit correspondre au
    /// nombre d'éléments du spec.
    ///
    /// # Erreurs
    /// [`InjectorError::InvalidTensor`] si la longueur est incohérente.
    pub fn apply_to(&self, buffer: &mut [f64]) -> InjectorResult<()> {
        let expected = self.spec.shape.num_elements_usize()?;
        if buffer.len() != expected {
            return Err(InjectorError::InvalidTensor(format!(
                "buffer de longueur {} ≠ nombre d'éléments du spec {}",
                buffer.len(),
                expected
            )));
        }
        if let Some((rows, cols)) = matrix_dims(&self.spec.shape) {
            self.apply_structure(buffer, rows, cols)?;
            self.apply_correlation(buffer, rows, cols)?;
            self.apply_low_rank(buffer, rows, cols)?;
        }
        self.apply_super_weights(buffer)?;
        Ok(())
    }

    /// Étape 2 — structure sparse localisée (shape 2D uniquement).
    fn apply_structure(&self, buffer: &mut [f64], rows: usize, cols: usize) -> InjectorResult<()> {
        let mut rng = self.stage_rng(InjectionStage::Structure);
        if rng.next_f64() < self.policy.sparse_structure_probability {
            let spec = sparse_spec_from_policy(&self.policy, rows, cols);
            let structured = generate_sparse_structure(&mut rng, rows, cols, &spec, 1.0)?;
            for (b, s) in buffer.iter_mut().zip(structured.iter()) {
                if *s == 0.0 {
                    *b = 0.0;
                }
            }
        }
        Ok(())
    }

    /// Étape 3 — corrélation contrôlée entre colonnes (shape 2D, ρ > 0).
    fn apply_correlation(
        &self,
        buffer: &mut [f64],
        rows: usize,
        cols: usize,
    ) -> InjectorResult<()> {
        let rho = self.policy.correlation_strength;
        if rho == 0.0 || rows < 2 || cols < 2 {
            return Ok(());
        }
        // La corrélation remplace les colonnes : on préserve la moyenne et
        // l'écart-type du tenseur courant (stabilité statistique documentée).
        let mean = pmg_math::statistics::mean(buffer)?;
        let std = pmg_math::statistics::std_population(buffer)?;
        let std = if std == 0.0 { 1.0 } else { std };
        let mut rng = self.stage_rng(InjectionStage::Correlation);
        let correlated = generate_correlated_columns(&mut rng, rows, cols, rho, std)?;
        for (b, c) in buffer.iter_mut().zip(correlated.iter()) {
            *b = mean + c;
        }
        Ok(())
    }

    /// Étape 4 — composante bas-rang `α·UVᵀ` (shape 2D).
    fn apply_low_rank(&self, buffer: &mut [f64], rows: usize, cols: usize) -> InjectorResult<()> {
        let mut rng = self.stage_rng(InjectionStage::LowRank);
        if rng.next_f64() < self.policy.low_rank_probability {
            let rank = self.policy.low_rank_rank.min(rows.min(cols));
            let injection =
                LowRankInjection::new(rank, self.policy.low_rank_alpha, 1.0, rows, cols)?;
            inject_low_rank(buffer, rows, cols, &injection, &mut rng)?;
        }
        Ok(())
    }

    /// Étape 5 — outliers/super-poids (toute shape).
    ///
    /// Pour chaque position marquée par le masque Bernoulli, la stratégie est
    /// tirée : multiplicative `w' = s·w` avec probabilité
    /// `1 − heavy_tail_probability`, statistique Student-t `t(df)` sinon.
    fn apply_super_weights(&self, buffer: &mut [f64]) -> InjectorResult<()> {
        let mut rng = self.stage_rng(InjectionStage::SuperWeights);
        let mask = OutlierMask::bernoulli(&mut rng, buffer.len(), self.policy.outlier_frequency)?;
        if mask.count() == 0 {
            return Ok(());
        }
        let mut student = StudentT::new(self.policy.heavy_tail_df)?;
        for (v, &f) in buffer.iter_mut().zip(mask.flags()) {
            if f {
                if rng.next_f64() < self.policy.heavy_tail_probability {
                    *v = student.sample(&mut rng);
                } else {
                    *v *= self.policy.outlier_scale;
                }
            }
        }
        Ok(())
    }

    /// Flux déterministe dérivé de la seed racine pour une étape.
    fn stage_rng(&self, stage: InjectionStage) -> DeterministicRng {
        DeterministicRng::from_seed(derive_sub_seed(&self.seed, stage.domain(), 0))
    }
}
