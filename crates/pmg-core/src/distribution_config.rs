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

//! Configuration des distributions de probabilité.
//!
//! Ce module définit une configuration sérialisable (JSON/TOML) pour les
//! distributions, indépendante de l'implémentation mathématique. La
//! configuration est utilisée par [`pmg_math::distribution::from_config`]
//! pour construire des distributions concrètes.
//!
//! Conformité : `docs/architecture/04-moteurs-math-injection-generation.md`

use serde::{Deserialize, Serialize};

/// Type de distribution, sérialisable en snake_case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistributionKind {
    Normal,
    StudentT,
    Laplace,
    LogNormal,
    Weibull,
    Pareto,
    Uniform,
    Mixture,
}

impl DistributionKind {
    /// Nom lisible en français (pour les rapports CLI).
    pub fn display_name(self) -> &'static str {
        match self {
            DistributionKind::Normal => "normale",
            DistributionKind::StudentT => "student-t",
            DistributionKind::Laplace => "laplace",
            DistributionKind::LogNormal => "log-normale",
            DistributionKind::Weibull => "weibull",
            DistributionKind::Pareto => "pareto",
            DistributionKind::Uniform => "uniforme",
            DistributionKind::Mixture => "mélange",
        }
    }
}

/// Configuration d'une distribution, sérialisable.
///
/// Exemple JSON :
/// ```json
/// {
///     "type": "student_t",
///     "degrees_of_freedom": 5.0,
///     "location": 0.0,
///     "scale": 0.02
/// }
/// ```
///
/// Les champs non pertinents pour une famille sont ignorés. La validation
/// stricte est faite à la construction de la distribution concrète.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistributionConfig {
    /// Type de distribution.
    pub kind: DistributionKind,
    /// Première valeur scalaire (μ pour normale/log-normale, μ pour Laplace,
    /// k pour Weibull, x_m pour Pareto, df pour Student-t).
    pub p1: f64,
    /// Deuxième valeur scalaire (σ pour normale/log-normale/Laplace, λ pour
    /// Weibull, α pour Pareto, `None` pour Student-t standard).
    pub p2: Option<f64>,
    /// Composantes du mélange (poids, sous-config) — vide hors mélange.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mixture_components: Vec<(f64, DistributionConfig)>,
}

impl DistributionConfig {
    /// Construit une config normale `N(mean, std)`.
    pub fn normal(mean: f64, std: f64) -> Self {
        Self {
            kind: DistributionKind::Normal,
            p1: mean,
            p2: Some(std),
            mixture_components: Vec::new(),
        }
    }

    /// Construit une config Student-t standard `t(df)` (location 0, scale 1).
    pub fn student_t(df: f64) -> Self {
        Self {
            kind: DistributionKind::StudentT,
            p1: df,
            p2: None,
            mixture_components: Vec::new(),
        }
    }

    /// Construit une config Laplace `L(μ, b)`.
    pub fn laplace(location: f64, scale: f64) -> Self {
        Self {
            kind: DistributionKind::Laplace,
            p1: location,
            p2: Some(scale),
            mixture_components: Vec::new(),
        }
    }

    /// Construit une config log-normale `LogN(μ_log, σ_log)`.
    pub fn log_normal(mu_log: f64, sigma_log: f64) -> Self {
        Self {
            kind: DistributionKind::LogNormal,
            p1: mu_log,
            p2: Some(sigma_log),
            mixture_components: Vec::new(),
        }
    }

    /// Construit une config Weibull `W(λ, k)` (scale, shape).
    pub fn weibull(scale: f64, shape: f64) -> Self {
        Self {
            kind: DistributionKind::Weibull,
            p1: scale,
            p2: Some(shape),
            mixture_components: Vec::new(),
        }
    }

    /// Construit une config Pareto `P(x_m, α)`.
    pub fn pareto(scale: f64, shape: f64) -> Self {
        Self {
            kind: DistributionKind::Pareto,
            p1: scale,
            p2: Some(shape),
            mixture_components: Vec::new(),
        }
    }

    /// Construit une config mélange à partir de composantes pondérées.
    pub fn mixture(components: Vec<(f64, DistributionConfig)>) -> Self {
        Self {
            kind: DistributionKind::Mixture,
            p1: 0.0,
            p2: None,
            mixture_components: components,
        }
    }

    /// Construit une config uniforme `U(min, max)`.
    ///
    /// # Paramètres
    /// - `min` : borne inférieure (incluse)
    /// - `max` : borne supérieure (incluse)
    ///
    /// # Erreurs
    /// La validation est déléguée à la construction de la distribution concrète.
    pub fn uniform(min: f64, max: f64) -> Self {
        Self {
            kind: DistributionKind::Uniform,
            p1: min,
            p2: Some(max),
            mixture_components: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_round_trip() {
        let config = DistributionConfig::student_t(5.0);
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"kind\":\"student_t\""), "json={json}");
        let back: DistributionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, back);
    }

    #[test]
    fn serde_mixture_with_components() {
        let config = DistributionConfig::mixture(vec![
            (0.7, DistributionConfig::normal(0.0, 1.0)),
            (0.3, DistributionConfig::laplace(0.0, 0.5)),
        ]);
        let json = serde_json::to_string(&config).unwrap();
        let back: DistributionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, back);
    }

    #[test]
    fn display_names() {
        assert_eq!(DistributionKind::Normal.display_name(), "normale");
        assert_eq!(DistributionKind::StudentT.display_name(), "student-t");
        assert_eq!(DistributionKind::Laplace.display_name(), "laplace");
        assert_eq!(DistributionKind::LogNormal.display_name(), "log-normale");
        assert_eq!(DistributionKind::Weibull.display_name(), "weibull");
        assert_eq!(DistributionKind::Pareto.display_name(), "pareto");
        assert_eq!(DistributionKind::Uniform.display_name(), "uniforme");
        assert_eq!(DistributionKind::Mixture.display_name(), "mélange");
    }

    #[test]
    fn serde_round_trip_uniform() {
        let config = DistributionConfig::uniform(0.0, 1.0);
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"kind\":\"uniform\""), "json={json}");
        let back: DistributionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, back);
    }
}
