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

//! Sous-module contenant les statistiques descriptives de base.

use crate::error::{MathError, MathResult};

/// Statistiques descriptives d'un échantillon.
#[derive(Debug, Clone, PartialEq)]
pub struct SummaryStats {
    pub count: usize,
    pub mean: f64,
    pub variance_sample: f64,
    pub variance_population: f64,
    pub std_sample: f64,
    pub std_population: f64,
    pub min: f64,
    pub max: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

/// Calcule la somme des éléments.
pub fn sum(data: &[f64]) -> MathResult<f64> {
    require_non_empty(data, "sum")?;
    Ok(data.iter().sum())
}

/// Calcule la moyenne arithmétique.
pub fn mean(data: &[f64]) -> MathResult<f64> {
    require_non_empty(data, "mean")?;
    Ok(data.iter().sum::<f64>() / data.len() as f64)
}

/// Calcule la variance d'échantillon (diviseur `n − 1`, sans biais).
pub fn variance_sample(data: &[f64]) -> MathResult<f64> {
    variance_welford(data, false)
}

/// Calcule la variance de population (diviseur `n`).
pub fn variance_population(data: &[f64]) -> MathResult<f64> {
    variance_welford(data, true)
}

/// Calcule l'écart-type d'échantillon (racine de [`variance_sample`]).
pub fn std_sample(data: &[f64]) -> MathResult<f64> {
    Ok(variance_sample(data)?.sqrt())
}

/// Calcule l'écart-type de population (racine de [`variance_population`]).
pub fn std_population(data: &[f64]) -> MathResult<f64> {
    Ok(variance_population(data)?.sqrt())
}

/// Retourne le minimum et le maximum de la slice.
pub fn min_max(data: &[f64]) -> MathResult<(f64, f64)> {
    require_non_empty(data, "min_max")?;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in data {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    Ok((min, max))
}

/// Calcule l'asymétrie (skewness) — moment centré d'ordre 3 normalisé.
pub fn skewness(data: &[f64]) -> MathResult<f64> {
    let m = mean(data)?;
    let n = data.len() as f64;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    for &v in data {
        let d = v - m;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= n;
    m3 /= n;
    if m2 == 0.0 {
        return Ok(0.0);
    }
    Ok(m3 / m2.powf(1.5))
}

/// Calcule le kurtosis excédentaire (kurtosis − 3), moment centré d'ordre 4 normalisé.
pub fn kurtosis(data: &[f64]) -> MathResult<f64> {
    let m = mean(data)?;
    let n = data.len() as f64;
    let mut m2 = 0.0;
    let mut m4 = 0.0;
    for &v in data {
        let d = v - m;
        let d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    m2 /= n;
    m4 /= n;
    if m2 == 0.0 {
        return Ok(0.0);
    }
    Ok(m4 / m2.powi(2) - 3.0)
}

/// Calcule la médiane.
pub fn median(data: &[f64]) -> MathResult<f64> {
    let q = quantiles(data, &[0.5])?;
    Ok(q[0])
}

/// Calcule les quantiles demandés.
pub fn quantiles(data: &[f64], qs: &[f64]) -> MathResult<Vec<f64>> {
    require_non_empty(data, "quantiles")?;
    for &q in qs {
        if !(0.0..=1.0).contains(&q) {
            return Err(MathError::InvalidParameter(format!(
                "quantile hors [0,1] : {q}"
            )));
        }
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let mut result = Vec::with_capacity(qs.len());
    for &q in qs {
        let index = q * (n - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        if lower == upper {
            result.push(sorted[lower]);
        } else {
            let weight = index - lower as f64;
            result.push(sorted[lower] * (1.0 - weight) + sorted[upper] * weight);
        }
    }
    Ok(result)
}

/// Calcule le résumé complet des statistiques.
pub fn summary(data: &[f64]) -> MathResult<SummaryStats> {
    let count = data.len();
    let mean = mean(data)?;
    let variance_sample = variance_sample(data)?;
    let variance_population = variance_population(data)?;
    let std_sample = variance_sample.sqrt();
    let std_population = variance_population.sqrt();
    let (min, max) = min_max(data)?;
    let skewness = skewness(data)?;
    let kurtosis = kurtosis(data)?;

    Ok(SummaryStats {
        count,
        mean,
        variance_sample,
        variance_population,
        std_sample,
        std_population,
        min,
        max,
        skewness,
        kurtosis,
    })
}

/// Vérifie que la slice n'est pas vide.
fn require_non_empty(data: &[f64], func_name: &str) -> MathResult<()> {
    if data.is_empty() {
        Err(MathError::EmptyData(format!(
            "{func_name} nécessite une slice non vide"
        )))
    } else {
        Ok(())
    }
}

/// Algorithme de Welford pour la variance (numériquement stable).
fn variance_welford(data: &[f64], population: bool) -> MathResult<f64> {
    require_non_empty(data, "variance")?;
    let n = data.len() as f64;
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for (i, &x) in data.iter().enumerate() {
        let delta = x - mean;
        mean += delta / (i as f64 + 1.0);
        let delta2 = x - mean;
        m2 += delta * delta2;
    }
    if population {
        Ok(m2 / n)
    } else {
        Ok(m2 / (n - 1.0))
    }
}
