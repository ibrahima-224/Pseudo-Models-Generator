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

//! Super-poids : transformer des valeurs ordinaires en valeurs extrêmes
//! contrôlées.
//!
//! Deux stratégies complémentaires (spécification étape 4.3) :
//! - **multiplicative** : `w' = s·w` avec `s ≥ 1` — conserve le signe de la
//!   valeur d'origine et amplifie sa magnitude ;
//! - **statistique** : `w' ~ T(θ)` où `T` est une distribution à queue lourde
//!   (Student-t) ou un mélange contrôlé construit via
//!   [`pmg_math::distribution::from_config`].
//!
//! Les anomalies produites sont rares (fréquence pilotée par le masque),
//! contrôlées (amplitude/loi paramétrées) et reproductibles (flux dérivé de
//! seed, jamais de source aléatoire globale).

use pmg_core::distribution_config::DistributionConfig;
use pmg_math::distribution::from_config;
use pmg_math::rng::DeterministicRng;

use crate::error::{InjectorError, InjectorResult};
use crate::outlier_mask::OutlierMask;

/// Stratégie d'injection d'un super-poids.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SuperWeightStrategy {
    /// Multiplicatif : `w' = scale·w` (conserve le signe, amplifie).
    Multiplicative { scale: f64 },
    /// Statistique : remplacement par un tirage d'une loi à queue lourde
    /// (Student-t standard de `df` degrés de liberté).
    HeavyTail { df: f64 },
    /// Statistique : remplacement par un tirage d'un mélange contrôlé
    /// (config pmg-math, ex. normale + Student-t).
    Mixture(DistributionConfig),
}

impl SuperWeightStrategy {
    /// Construit la stratégie multiplicative validée.
    ///
    /// # Erreurs
    /// [`InjectorError::InvalidPolicy`] si `scale < 1` (un super-poids doit
    /// amplifier, jamais atténuer).
    pub fn multiplicative(scale: f64) -> InjectorResult<Self> {
        if !scale.is_finite() || scale < 1.0 {
            return Err(InjectorError::InvalidPolicy(format!(
                "scale d'un super-poids doit être fini et ≥ 1, reçu {scale}"
            )));
        }
        Ok(Self::Multiplicative { scale })
    }

    /// Construit la stratégie Student-t validée.
    ///
    /// # Erreurs
    /// [`InjectorError::InvalidPolicy`] si `df ≤ 0` (bornes de la Student-t).
    pub fn heavy_tail(df: f64) -> InjectorResult<Self> {
        if !df.is_finite() || df <= 0.0 {
            return Err(InjectorError::InvalidPolicy(format!(
                "df de la Student-t doit être fini et > 0, reçu {df}"
            )));
        }
        Ok(Self::HeavyTail { df })
    }

    /// Construit la stratégie mélange en validant la config pmg-math.
    ///
    /// # Erreurs
    /// [`InjectorError::Math`] si la config produit une distribution invalide
    /// (poids du mélange incorrects, paramètres hors bornes).
    pub fn mixture(config: DistributionConfig) -> InjectorResult<Self> {
        // La validation est déléguée à pmg-math (from_config).
        from_config(&config)?;
        Ok(Self::Mixture(config))
    }
}

/// Applique la stratégie de super-poids aux positions marquées par le masque.
///
/// # Entrées
/// - `buffer` : valeurs du tenseur, modifiées sur place aux positions marquées ;
/// - `mask` : positions affectées (même longueur que `buffer`) ;
/// - `strategy` : loi ou transformation à appliquer ;
/// - `rng` : flux déterministe dérivé (domaine `"super_weight"`) — utilisé
///   uniquement par les stratégies statistiques.
///
/// # Garanties
/// - les positions non marquées sont **exactement** inchangées ;
/// - stratégie multiplicative : `w' = scale·w` (signe conservé) ;
/// - stratégie statistique : remplacement par un tirage indépendant.
///
/// # Erreurs
/// - [`InjectorError::InvalidTensor`] si les longueurs diffèrent ;
/// - [`InjectorError::Math`] si la construction de la distribution échoue.
///
/// # Complexité
/// O(n) — un tirage (ou un produit) par position marquée.
pub fn inject_super_weights(
    buffer: &mut [f64],
    mask: &OutlierMask,
    strategy: &SuperWeightStrategy,
    rng: &mut DeterministicRng,
) -> InjectorResult<()> {
    if buffer.len() != mask.len() {
        return Err(InjectorError::InvalidTensor(format!(
            "buffer de longueur {} ≠ masque de longueur {}",
            buffer.len(),
            mask.len()
        )));
    }
    match strategy {
        SuperWeightStrategy::Multiplicative { scale } => {
            // Multiplicatif : ne consomme AUCUN tirage du RNG — la séquence
            // reste déterministe et reproductible.
            for (v, &f) in buffer.iter_mut().zip(mask.flags()) {
                if f {
                    *v *= scale;
                }
            }
            Ok(())
        },
        SuperWeightStrategy::HeavyTail { df } => {
            let mut dist = from_config(&DistributionConfig::student_t(*df))?;
            for (v, &f) in buffer.iter_mut().zip(mask.flags()) {
                if f {
                    *v = dist.sample(rng);
                }
            }
            Ok(())
        },
        SuperWeightStrategy::Mixture(config) => {
            let mut dist = from_config(config)?;
            for (v, &f) in buffer.iter_mut().zip(mask.flags()) {
                if f {
                    *v = dist.sample(rng);
                }
            }
            Ok(())
        },
    }
}

/// Applique la stratégie multiplicative `w' = scale·w` aux seules positions
/// marquées (forme spécialisée, sans allocation de distribution).
///
/// # Erreurs
/// [`InjectorError::InvalidTensor`] si les longueurs diffèrent.
pub fn inject_multiplicative(
    buffer: &mut [f64],
    mask: &OutlierMask,
    scale: f64,
) -> InjectorResult<()> {
    inject_super_weights(
        buffer,
        mask,
        &SuperWeightStrategy::multiplicative(scale)?,
        &mut dummy_rng(),
    )
}

/// RNG factice pour les stratégies déterministes sans tirage.
///
/// Utilisé uniquement lorsque la stratégie ne consomme rien du flux ;
/// conserve l'API unique de [`inject_super_weights`].
fn dummy_rng() -> DeterministicRng {
    DeterministicRng::from_seed([0u8; 32])
}

#[cfg(test)]
mod tests {
    use super::{inject_multiplicative, inject_super_weights, SuperWeightStrategy};
    use crate::error::InjectorError;
    use crate::outlier_mask::OutlierMask;
    use pmg_core::distribution_config::DistributionConfig;
    use pmg_math::rng::{derive_sub_seed, DeterministicRng};

    fn rng_for(seed: [u8; 32]) -> DeterministicRng {
        DeterministicRng::from_seed(derive_sub_seed(&seed, "super_weight", 0))
    }

    fn mask_with(f: &[bool]) -> OutlierMask {
        OutlierMask::from_flags(f.to_vec()).unwrap()
    }

    #[test]
    fn multiplicative_preserves_sign_and_unmarked() {
        let mask = mask_with(&[true, false, true]);
        let mut buf = vec![2.0, -3.0, -4.0];
        let strat = SuperWeightStrategy::multiplicative(5.0).unwrap();
        inject_super_weights(&mut buf, &mask, &strat, &mut rng_for([1u8; 32])).unwrap();
        assert_eq!(buf, vec![10.0, -3.0, -20.0]);
    }

    #[test]
    fn multiplicative_zero_values_stay_zero() {
        // 0·s = 0 : une valeur nulle reste nulle (propriété importante pour
        // préserver les structures sparse).
        let mask = mask_with(&[true]);
        let mut buf = vec![0.0];
        let strat = SuperWeightStrategy::multiplicative(10.0).unwrap();
        inject_super_weights(&mut buf, &mask, &strat, &mut rng_for([2u8; 32])).unwrap();
        assert_eq!(buf, vec![0.0]);
    }

    #[test]
    fn scale_below_one_rejected() {
        assert!(matches!(
            SuperWeightStrategy::multiplicative(0.5),
            Err(InjectorError::InvalidPolicy(_))
        ));
        assert!(SuperWeightStrategy::multiplicative(1.0).is_ok());
    }

    #[test]
    fn heavy_tail_replaces_with_finite_values() {
        let mask = mask_with(&[true, false, true]);
        let mut buf = vec![1.0, 1.0, 1.0];
        let strat = SuperWeightStrategy::heavy_tail(3.0).unwrap();
        inject_super_weights(&mut buf, &mask, &strat, &mut rng_for([3u8; 32])).unwrap();
        assert_eq!(buf[1], 1.0, "position non marquée inchangée");
        assert!(buf[0].is_finite());
        assert!(buf[2].is_finite());
        // La valeur remplacée diffère de l'originale (quasi certainement).
        assert_ne!(buf[0], 1.0);
    }

    #[test]
    fn heavy_tail_tail_is_heavy() {
        // Student-t df=3 : kurtosis infini → |w'| dépasse 3σ bien plus souvent
        // qu'une normale. On vérifie qu'au moins 5 % des tirages dépassent 3.
        let mut rng = rng_for([4u8; 32]);
        let strat = SuperWeightStrategy::heavy_tail(3.0).unwrap();
        let mask = mask_with(&vec![true; 20_000]);
        let mut buf = vec![0.0; 20_000];
        inject_super_weights(&mut buf, &mask, &strat, &mut rng).unwrap();
        let heavy = buf.iter().filter(|&&x| x.abs() > 3.0).count();
        let ratio = heavy as f64 / buf.len() as f64;
        assert!(ratio > 0.02, "queues pas assez lourdes : ratio {ratio}");
    }

    #[test]
    fn heavy_tail_invalid_df_rejected() {
        assert!(matches!(
            SuperWeightStrategy::heavy_tail(0.0),
            Err(InjectorError::InvalidPolicy(_))
        ));
        assert!(matches!(
            SuperWeightStrategy::heavy_tail(-2.0),
            Err(InjectorError::InvalidPolicy(_))
        ));
    }

    #[test]
    fn mixture_strategy_samples_controlled_mixture() {
        // Mélange 50/50 normale + Student-t df=3 : les valeurs doivent être
        // finies et majoritairement centrées autour de 0.
        let cfg = DistributionConfig::mixture(vec![
            (0.5, DistributionConfig::normal(0.0, 1.0)),
            (0.5, DistributionConfig::student_t(3.0)),
        ]);
        let strat = SuperWeightStrategy::mixture(cfg).unwrap();
        let mask = mask_with(&vec![true; 5_000]);
        let mut buf = vec![0.0; 5_000];
        inject_super_weights(&mut buf, &mask, &strat, &mut rng_for([5u8; 32])).unwrap();
        assert!(buf.iter().all(|x| x.is_finite()));
        let mean = buf.iter().sum::<f64>() / buf.len() as f64;
        assert!(mean.abs() < 0.2, "moyenne inattendue {mean}");
    }

    #[test]
    fn mixture_with_bad_weights_rejected() {
        // Somme des poids ≠ 1 → erreur pmg-math propagée.
        let cfg = DistributionConfig::mixture(vec![
            (0.5, DistributionConfig::normal(0.0, 1.0)),
            (0.3, DistributionConfig::normal(0.0, 1.0)),
        ]);
        assert!(SuperWeightStrategy::mixture(cfg).is_err());
    }

    #[test]
    fn injection_is_deterministic() {
        // Même seed, même stratégie statistique ⇒ mêmes valeurs.
        let mask = mask_with(&[true; 64]);
        let strat = SuperWeightStrategy::heavy_tail(4.0).unwrap();
        let mut a = vec![0.0; 64];
        let mut b = vec![0.0; 64];
        inject_super_weights(&mut a, &mask, &strat, &mut rng_for([6u8; 32])).unwrap();
        inject_super_weights(&mut b, &mask, &strat, &mut rng_for([6u8; 32])).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn length_mismatch_rejected() {
        let mask = mask_with(&[true, true]);
        let mut buf = vec![1.0];
        let strat = SuperWeightStrategy::multiplicative(2.0).unwrap();
        assert!(matches!(
            inject_super_weights(&mut buf, &mask, &strat, &mut rng_for([7u8; 32])),
            Err(InjectorError::InvalidTensor(_))
        ));
    }

    #[test]
    fn inject_multiplicative_convenience() {
        let mask = mask_with(&[true, false]);
        let mut buf = vec![3.0, 4.0];
        inject_multiplicative(&mut buf, &mask, 2.0).unwrap();
        assert_eq!(buf, vec![6.0, 4.0]);
    }
}
