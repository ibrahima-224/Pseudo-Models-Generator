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

//! Modèle d'outlier : abstraction pour les stratégies de transformation des super-poids.
//!
//! Ce module définit les primitives pour appliquer des transformations aux valeurs
//! d'origine afin de produire des anomalies contrôlées. Deux modèles principaux :
//!
//! - **Additif** : `W' = W + O` où `O` est un offset contrôlé ;
//! - **Multiplicatif** : `W' = W ⊙ M` où `M` est un facteur multiplicatif.
//!
//! # Conformité
//!
//! Spécification Sprint 9, étape 5.1 : « Modèle d'outlier — abstraction pour les
//! différentes stratégies ».

use crate::error::{MathError, MathResult};

/// Stratégie de transformation d'un outlier.
///
/// Chaque variante encode la règle de calcul de la nouvelle valeur à partir
/// de la valeur d'origine et d'une amplitude prédéterminée.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutlierStrategy {
    /// Transformation additive : `W' = W + signe × amplitude`.
    /// Le signe est tiré aléatoirement (ou fixé) et l'amplitude est un scalaire.
    Additive,
    /// Transformation multiplicative : `W' = W × (1 + signe × amplitude)`.
    /// Conserve le signe d'origine si l'amplitude est < 1, sinon peut inverser.
    Multiplicative,
}

/// Spécification complète d'un outlier à injecter.
///
/// Contient la stratégie, l'amplitude calculée et le signe décidé.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutlierSpec {
    /// Stratégie de transformation (additive ou multiplicative).
    pub strategy: OutlierStrategy,
    /// Amplitude de l'anomalie (toujours ≥ 0).
    pub amplitude: f64,
    /// Signe de l'anomalie (`true` = positif, `false` = négatif).
    pub positive: bool,
}

impl OutlierSpec {
    /// Crée un spec avec validation.
    ///
    /// # Erreurs
    /// [`MathError::InvalidParameter`] si l'amplitude est négative ou non finie.
    pub fn new(strategy: OutlierStrategy, amplitude: f64, positive: bool) -> MathResult<Self> {
        if !amplitude.is_finite() || amplitude < 0.0 {
            return Err(MathError::InvalidParameter(format!(
                "amplitude d'outlier doit être finie et ≥ 0, reçu {amplitude}"
            )));
        }
        Ok(Self {
            strategy,
            amplitude,
            positive,
        })
    }
}

/// Modèle d'outlier encapsulant la logique de transformation.
///
/// Utilisation typique :
/// ```ignore
/// let model = OutlierModel::new(OutlierStrategy::Multiplicative);
/// let spec = OutlierSpec::new(OutlierStrategy::Multiplicative, 2.0, true)?;
/// let new_val = model.apply(original_val, &spec);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OutlierModel {
    strategy: OutlierStrategy,
}

impl OutlierModel {
    /// Crée un modèle avec la stratégie spécifiée.
    pub fn new(strategy: OutlierStrategy) -> Self {
        Self { strategy }
    }

    /// Applique la transformation à une valeur selon le spec.
    ///
    /// # Comportement
    /// - Additif : `value + signe × amplitude`
    /// - Multiplicatif : `value × (1 + signe × amplitude)`
    ///
    /// # Panique
    /// Ne panique pas, toutes les opérations sont sur `f64` et gèrent l'inf.
    pub fn apply(&self, value: f64, spec: &OutlierSpec) -> f64 {
        let sign = if spec.positive { 1.0 } else { -1.0 };
        match spec.strategy {
            OutlierStrategy::Additive => value + sign * spec.amplitude,
            OutlierStrategy::Multiplicative => value * (1.0 + sign * spec.amplitude),
        }
    }

    /// Applique la transformation à un tableau entier aux positions marquées.
    ///
    /// # Arguments
    /// - `buffer` : tableau mutable des valeurs du tenseur ;
    /// - `mask` : vecteur booléen indiquant les positions d'outliers ;
    /// - `specs` : spécifications pour chaque outlier (même ordre que `mask`) ;
    /// - `rng` : flux déterministe (non utilisé ici, mais requis pour la cohérence API).
    ///
    /// # Erreurs
    /// [`MathError::InvalidParameter`] si les longueurs sont incohérentes.
    pub fn apply_to_buffer(
        &self,
        buffer: &mut [f64],
        mask: &[bool],
        specs: &[OutlierSpec],
    ) -> MathResult<()> {
        if buffer.len() != mask.len() {
            return Err(MathError::InvalidParameter(format!(
                "buffer (len={}) et mask (len={}) doivent avoir la même longueur",
                buffer.len(),
                mask.len()
            )));
        }
        let outlier_count = mask.iter().filter(|&&m| m).count();
        if specs.len() != outlier_count {
            return Err(MathError::InvalidParameter(format!(
                "nombre de specs ({}) doit correspondre au nombre d'outliers ({})",
                specs.len(),
                outlier_count
            )));
        }

        let mut spec_idx = 0;
        for (val, &is_outlier) in buffer.iter_mut().zip(mask.iter()) {
            if is_outlier {
                *val = self.apply(*val, &specs[spec_idx]);
                spec_idx += 1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outlier_spec_validation() {
        assert!(OutlierSpec::new(OutlierStrategy::Additive, 1.0, true).is_ok());
        assert!(OutlierSpec::new(OutlierStrategy::Additive, -1.0, true).is_err());
        assert!(OutlierSpec::new(OutlierStrategy::Multiplicative, f64::NAN, true).is_err());
    }

    #[test]
    fn test_apply_additive() {
        let model = OutlierModel::new(OutlierStrategy::Additive);
        let spec = OutlierSpec::new(OutlierStrategy::Additive, 2.0, true).unwrap();
        assert_eq!(model.apply(5.0, &spec), 7.0);
        let spec_neg = OutlierSpec::new(OutlierStrategy::Additive, 2.0, false).unwrap();
        assert_eq!(model.apply(5.0, &spec_neg), 3.0);
    }

    #[test]
    fn test_apply_multiplicative() {
        let model = OutlierModel::new(OutlierStrategy::Multiplicative);
        let spec = OutlierSpec::new(OutlierStrategy::Multiplicative, 0.5, true).unwrap();
        assert_eq!(model.apply(10.0, &spec), 15.0); // 10 * 1.5
        let spec_neg = OutlierSpec::new(OutlierStrategy::Multiplicative, 0.5, false).unwrap();
        assert_eq!(model.apply(10.0, &spec_neg), 5.0); // 10 * 0.5
    }

    #[test]
    fn test_apply_to_buffer() {
        let model = OutlierModel::new(OutlierStrategy::Additive);
        let mut buffer = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![false, true, false, true];
        let specs = vec![
            OutlierSpec::new(OutlierStrategy::Additive, 10.0, true).unwrap(),
            OutlierSpec::new(OutlierStrategy::Additive, 5.0, false).unwrap(),
        ];
        assert!(model.apply_to_buffer(&mut buffer, &mask, &specs).is_ok());
        assert_eq!(buffer, vec![1.0, 12.0, 3.0, -1.0]); // 2+10, 4-5
    }
}
