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

//! Implémentation des étapes du pipeline de génération.
//!
//! Ce module contient les fonctions d'application réelles pour chaque étape
//! du pipeline, en déléguant aux modules mathématiques appropriés de `pmg-math`.

use std::collections::HashMap;

use pmg_math::low_rank_analysis::compute_singular_values;
use pmg_math::outlier_analysis::detect_by_threshold;
use pmg_math::rng::DeterministicRng;
use pmg_math::structure::correlation::{Correlation, CorrelationConfig as MathCorrelationConfig};

use crate::error::{GeneratorError, GeneratorResult};
use crate::pipeline::{PipelineStep, StepResult};
use crate::pipeline_config::{
    CorrelationConfig, LowRankConfig, OutlierReplacementMode, OutliersConfig, SuperWeightsConfig,
};

/// Applique l'étape de distribution aux valeurs.
///
/// Cette étape génère de nouvelles valeurs selon une distribution normale
/// avec les paramètres spécifiés dans la configuration.
///
/// # Paramètres
/// - `values` : slice de valeurs à transformer (modifiée en place)
/// - `config` : configuration de la distribution
/// - `rng` : générateur de nombres aléatoires déterministe
///
/// # Retourne
/// Un `StepResult` contenant les métriques de l'opération.
pub fn apply_distribution(
    values: &mut [f64],
    config: &crate::pipeline_config::DistributionConfig,
    rng: &mut DeterministicRng,
) -> GeneratorResult<StepResult> {
    let n = values.len();
    if n == 0 {
        return Ok(StepResult {
            step_name: PipelineStep::Distribution.name().to_string(),
            elements_modified: 0,
            metrics: HashMap::new(),
        });
    }

    // Génère de nouvelles valeurs selon N(mean, std²)
    let new_values = rng.normal_vec(config.mean, config.std, n);

    // Remplace les valeurs existantes
    values.copy_from_slice(&new_values);

    let mut metrics = HashMap::new();
    metrics.insert("distribution_type".to_string(), 1.0); // 1 = normale
    metrics.insert("mean".to_string(), config.mean);
    metrics.insert("std".to_string(), config.std);

    Ok(StepResult {
        step_name: PipelineStep::Distribution.name().to_string(),
        elements_modified: n,
        metrics,
    })
}

/// Applique l'étape de corrélation aux valeurs.
///
/// Cette étape applique une corrélation structurelle entre les éléments
/// en utilisant une matrice de covariance spécifiée.
///
/// # Paramètres
/// - `values` : slice de valeurs à transformer (modifiée en place)
/// - `config` : configuration de la corrélation
/// - `rng` : générateur de nombres aléatoires déterministe
///
/// # Retourne
/// Un `StepResult` contenant les métriques de l'opération.
pub fn apply_correlation(
    values: &mut [f64],
    config: &CorrelationConfig,
    rng: &mut DeterministicRng,
) -> GeneratorResult<StepResult> {
    let n = values.len();
    if n == 0 {
        return Ok(StepResult {
            step_name: PipelineStep::Correlation.name().to_string(),
            elements_modified: 0,
            metrics: HashMap::new(),
        });
    }

    // Si aucune matrice de covariance n'est spécifiée, retourne les valeurs inchangées
    let sigma = match &config.sigma {
        Some(s) => s.clone(),
        None => {
            // Matrice identité (pas de corrélation)
            let dim = config.dim;
            let mut identity = vec![0.0; dim * dim];
            for i in 0..dim {
                identity[i * dim + i] = 1.0;
            }
            identity
        },
    };

    // Crée la configuration de corrélation pour pmg-math
    let math_config =
        MathCorrelationConfig::new(sigma, config.dim).map_err(GeneratorError::Math)?;

    let correlation = Correlation::new(math_config).map_err(GeneratorError::Math)?;

    // Nombre d'échantillons à générer (chaque échantillon a `dim` éléments)
    let n_samples = n / config.dim;

    if n_samples == 0 {
        return Ok(StepResult {
            step_name: PipelineStep::Correlation.name().to_string(),
            elements_modified: 0,
            metrics: HashMap::new(),
        });
    }

    // Génère des échantillons corrélés
    let correlated_samples = correlation
        .generate(rng, n_samples)
        .map_err(GeneratorError::Math)?;

    // Copie les échantillons générés dans le tableau de valeurs
    let copy_len = std::cmp::min(correlated_samples.len(), n);
    values[..copy_len].copy_from_slice(&correlated_samples[..copy_len]);

    let mut metrics = HashMap::new();
    metrics.insert("correlation_strength".to_string(), 1.0);
    metrics.insert("dim".to_string(), config.dim as f64);
    metrics.insert("samples_generated".to_string(), n_samples as f64);

    Ok(StepResult {
        step_name: PipelineStep::Correlation.name().to_string(),
        elements_modified: copy_len,
        metrics,
    })
}

/// Applique l'étape de bas-rang aux valeurs.
///
/// Cette étape effectue une réduction de rang sur les données en utilisant
/// la décomposition en valeurs singulières (SVD).
///
/// # Paramètres
/// - `values` : slice de valeurs à transformer (modifiée en place)
/// - `config` : configuration du bas-rang
///
/// # Retourne
/// Un `StepResult` contenant les métriques de l'opération.
pub fn apply_low_rank(values: &mut [f64], config: &LowRankConfig) -> GeneratorResult<StepResult> {
    let n = values.len();
    if n == 0 {
        return Ok(StepResult {
            step_name: PipelineStep::LowRank.name().to_string(),
            elements_modified: 0,
            metrics: HashMap::new(),
        });
    }

    // Trouve les dimensions de la matrice les plus proches d'un carré
    // tel que rows * cols <= n
    let rows = (n as f64).sqrt() as usize;
    if rows == 0 {
        return Ok(StepResult {
            step_name: PipelineStep::LowRank.name().to_string(),
            elements_modified: 0,
            metrics: HashMap::new(),
        });
    }

    let cols = n / rows;
    let actual_size = rows * cols;

    // Si la taille n'est pas exacte, on tronque les données
    if actual_size < n {
        // On travaille uniquement sur la partie complète de la matrice
        let singular_values = compute_singular_values(&values[..actual_size], rows, cols)
            .map_err(GeneratorError::Math)?;

        if singular_values.is_empty() {
            return Ok(StepResult {
                step_name: PipelineStep::LowRank.name().to_string(),
                elements_modified: 0,
                metrics: HashMap::new(),
            });
        }

        // Applique la réduction de rang
        let target_rank = (singular_values.len() as f64 * config.rank_reduction) as usize;
        let target_rank = std::cmp::max(1, target_rank);

        let mut modified_count = 0;
        for (i, _sv) in singular_values.iter().enumerate() {
            if i >= target_rank {
                let start_idx = i * cols;
                let end_idx = std::cmp::min(start_idx + cols, actual_size);
                for value in values.iter_mut().take(end_idx).skip(start_idx) {
                    *value *= 0.1;
                    modified_count += 1;
                }
            }
        }

        let total_energy: f64 = singular_values.iter().map(|x| x * x).sum();

        let mut metrics = HashMap::new();
        metrics.insert("rank_reduction".to_string(), config.rank_reduction);
        metrics.insert("original_rank".to_string(), singular_values.len() as f64);
        metrics.insert("target_rank".to_string(), target_rank as f64);
        metrics.insert("total_energy".to_string(), total_energy);
        metrics.insert("matrix_rows".to_string(), rows as f64);
        metrics.insert("matrix_cols".to_string(), cols as f64);

        return Ok(StepResult {
            step_name: PipelineStep::LowRank.name().to_string(),
            elements_modified: modified_count,
            metrics,
        });
    }

    // Calcule les valeurs singulières
    let singular_values =
        compute_singular_values(values, rows, cols).map_err(GeneratorError::Math)?;

    if singular_values.is_empty() {
        return Ok(StepResult {
            step_name: PipelineStep::LowRank.name().to_string(),
            elements_modified: 0,
            metrics: HashMap::new(),
        });
    }

    // Calcule l'énergie totale
    let total_energy: f64 = singular_values.iter().map(|x| x * x).sum();

    // Applique la réduction de rang en mettant à zéro les petites valeurs singulières
    let target_rank = (singular_values.len() as f64 * config.rank_reduction) as usize;
    let target_rank = std::cmp::max(1, target_rank);

    let mut modified_count = 0;
    for (i, _sv) in singular_values.iter().enumerate() {
        if i >= target_rank {
            // Met à zéro les éléments correspondants dans la matrice
            // Pour simplifier, on met à zéro les éléments de la matrice
            // qui contribuent principalement à cette valeur singulière
            let start_idx = i * cols;
            let end_idx = std::cmp::min(start_idx + cols, n);
            for value in values.iter_mut().take(end_idx).skip(start_idx) {
                *value *= 0.1; // Réduction partielle plutôt que mise à zéro complète
                modified_count += 1;
            }
        }
    }

    let mut metrics = HashMap::new();
    metrics.insert("rank_reduction".to_string(), config.rank_reduction);
    metrics.insert("original_rank".to_string(), singular_values.len() as f64);
    metrics.insert("target_rank".to_string(), target_rank as f64);
    metrics.insert("total_energy".to_string(), total_energy);
    metrics.insert("matrix_rows".to_string(), rows as f64);
    metrics.insert("matrix_cols".to_string(), cols as f64);

    Ok(StepResult {
        step_name: PipelineStep::LowRank.name().to_string(),
        elements_modified: modified_count,
        metrics,
    })
}

/// Applique l'étape d'outliers aux valeurs.
///
/// Cette étape détecte et traite les valeurs aberrantes selon la configuration.
///
/// # Paramètres
/// - `values` : slice de valeurs à transformer (modifiée en place)
/// - `config` : configuration des outliers
///
/// # Retourne
/// Un `StepResult` contenant les métriques de l'opération.
pub fn apply_outliers(values: &mut [f64], config: &OutliersConfig) -> GeneratorResult<StepResult> {
    let n = values.len();
    if n == 0 {
        return Ok(StepResult {
            step_name: PipelineStep::Outliers.name().to_string(),
            elements_modified: 0,
            metrics: HashMap::new(),
        });
    }

    // Détecte les outliers
    let analysis = detect_by_threshold(values, config.threshold_k).map_err(GeneratorError::Math)?;

    let mut modified_count = 0;

    match config.replacement_mode {
        OutlierReplacementMode::ClipToThreshold => {
            // Calcule mean et std pour le clipping
            let mean = values.iter().sum::<f64>() / n as f64;
            let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            let std_dev = variance.sqrt();

            if std_dev > 0.0 {
                let lower_bound = mean - config.threshold_k * std_dev;
                let upper_bound = mean + config.threshold_k * std_dev;

                for value in values.iter_mut() {
                    if *value < lower_bound {
                        *value = lower_bound;
                        modified_count += 1;
                    } else if *value > upper_bound {
                        *value = upper_bound;
                        modified_count += 1;
                    }
                }
            }
        },
        OutlierReplacementMode::ReplaceWithMean => {
            let mean = values.iter().sum::<f64>() / n as f64;
            for outlier in &analysis.outliers {
                if outlier.index < values.len() {
                    values[outlier.index] = mean;
                    modified_count += 1;
                }
            }
        },
        OutlierReplacementMode::Remove => {
            // Pour cet exemple, on ne supprime pas vraiment les éléments
            // car cela changerait la taille du tableau
            // On les marque à 0.0 à la place
            for outlier in &analysis.outliers {
                if outlier.index < values.len() {
                    values[outlier.index] = 0.0;
                    modified_count += 1;
                }
            }
        },
    }

    let mut metrics = HashMap::new();
    metrics.insert("outlier_count".to_string(), analysis.outlier_count as f64);
    metrics.insert("outlier_ratio".to_string(), analysis.outlier_ratio);
    metrics.insert("threshold_k".to_string(), config.threshold_k);

    Ok(StepResult {
        step_name: PipelineStep::Outliers.name().to_string(),
        elements_modified: modified_count,
        metrics,
    })
}

/// Applique l'étape de super-poids aux valeurs.
///
/// Cette étape identifie et amplifie les valeurs très élevées (super-poids).
///
/// # Paramètres
/// - `values` : slice de valeurs à transformer (modifiée en place)
/// - `config` : configuration des super-poids
///
/// # Retourne
/// Un `StepResult` contenant les métriques de l'opération.
pub fn apply_super_weights(
    values: &mut [f64],
    config: &SuperWeightsConfig,
) -> GeneratorResult<StepResult> {
    let n = values.len();
    if n == 0 {
        return Ok(StepResult {
            step_name: PipelineStep::SuperWeights.name().to_string(),
            elements_modified: 0,
            metrics: HashMap::new(),
        });
    }

    // Calcule mean et std
    let mean = values.iter().sum::<f64>() / n as f64;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        return Ok(StepResult {
            step_name: PipelineStep::SuperWeights.name().to_string(),
            elements_modified: 0,
            metrics: HashMap::new(),
        });
    }

    // Identifie les super-poids (valeurs > kσ)
    let threshold = config.threshold_k * std_dev;
    let mut super_weight_count = 0;
    let max_super_weights = (n as f64 * config.max_proportion) as usize;

    for value in values.iter_mut() {
        if (*value - mean).abs() > threshold && super_weight_count < max_super_weights {
            // Amplifie le super-poids
            *value *= config.multiplier;
            super_weight_count += 1;
        }
    }

    let mut metrics = HashMap::new();
    metrics.insert("super_weight_count".to_string(), super_weight_count as f64);
    metrics.insert("threshold_k".to_string(), config.threshold_k);
    metrics.insert("multiplier".to_string(), config.multiplier);
    metrics.insert("max_proportion".to_string(), config.max_proportion);

    Ok(StepResult {
        step_name: PipelineStep::SuperWeights.name().to_string(),
        elements_modified: super_weight_count,
        metrics,
    })
}
