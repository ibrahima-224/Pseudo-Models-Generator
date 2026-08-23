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

//! Rapport de génération structuré.
//!
//! Ce module produit un résumé de la génération d'un pseudo-modèle,
//! incluant les statistiques de distribution, les injections et les
//! métadonnées de la génération.
//!
//! # Format attendu
//!
//! ```text
//! PMG — Rapport de génération
//!
//! Modèle       : ExempleTransformer
//! Couches      : 32
//! Tenseurs     : 418
//! Paramètres   : 7.1B
//! Seed         : 42
//!
//! Distribution :
//!   Normale     94.1 %
//!   Student-t    4.2 %
//!   Pareto       0.7 %
//!   Autres       1.0 %
//!
//! Injection :
//!   Outliers    : 0.83 %
//!   Low-rank    : 12 couches
//!   Corrélation : activée
//! ```

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

/// Rapport de génération d'un pseudo-modèle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationReport {
    /// Nom du modèle.
    pub model_name: String,
    /// Nombre de couches.
    pub num_layers: u64,
    /// Nombre total de tenseurs.
    pub num_tensors: u64,
    /// Nombre total de paramètres.
    pub parameter_count: u64,
    /// Seed globale utilisée.
    pub seed: u64,
    /// Statistiques de distribution.
    pub distribution_stats: DistributionStats,
    /// Statistiques d'injection.
    pub injection_stats: InjectionStats,
    /// Métadonnées supplémentaires.
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

/// Statistiques de distribution des tenseurs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionStats {
    /// Pourcentage de tenseurs avec distribution normale.
    pub normal_pct: f64,
    /// Pourcentage de tenseurs avec distribution Student-t.
    pub student_t_pct: f64,
    /// Pourcentage de tenseurs avec distribution Pareto.
    pub pareto_pct: f64,
    /// Pourcentage de tenseurs avec autres distributions.
    pub other_pct: f64,
    /// Nombre total de tenseurs analysés.
    pub total_analyzed: u64,
}

/// Statistiques d'injection structurelle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionStats {
    /// Pourcentage de tenseurs avec outliers.
    pub outlier_pct: f64,
    /// Nombre de couches avec injection bas-rang.
    pub low_rank_layers: u64,
    /// Corrélation activée (oui/non).
    pub correlation_enabled: bool,
    /// Nombre total de tenseurs analysés.
    pub total_analyzed: u64,
}

impl GenerationReport {
    /// Crée un nouveau rapport vide.
    pub fn new(model_name: impl Into<String>, seed: u64) -> Self {
        Self {
            model_name: model_name.into(),
            num_layers: 0,
            num_tensors: 0,
            parameter_count: 0,
            seed,
            distribution_stats: DistributionStats::default(),
            injection_stats: InjectionStats::default(),
            metadata: BTreeMap::new(),
        }
    }

    /// Formate le rapport en texte lisible.
    pub fn format_text(&self) -> String {
        let mut output = String::new();
        output.push_str("PMG — Rapport de génération\n\n");

        output.push_str(&format!("Modèle       : {}\n", self.model_name));
        output.push_str(&format!("Couches      : {}\n", self.num_layers));
        output.push_str(&format!("Tenseurs     : {}\n", self.num_tensors));
        output.push_str(&format!(
            "Paramètres   : {}\n",
            format_parameter_count(self.parameter_count)
        ));
        output.push_str(&format!("Seed         : {}\n", self.seed));

        output.push_str("\nDistribution :\n");
        output.push_str(&format!(
            "  Normale     {:.1} %\n",
            self.distribution_stats.normal_pct
        ));
        output.push_str(&format!(
            "  Student-t   {:.1} %\n",
            self.distribution_stats.student_t_pct
        ));
        output.push_str(&format!(
            "  Pareto      {:.1} %\n",
            self.distribution_stats.pareto_pct
        ));
        output.push_str(&format!(
            "  Autres      {:.1} %\n",
            self.distribution_stats.other_pct
        ));

        output.push_str("\nInjection :\n");
        output.push_str(&format!(
            "  Outliers    : {:.2} %\n",
            self.injection_stats.outlier_pct
        ));
        output.push_str(&format!(
            "  Low-rank    : {} couches\n",
            self.injection_stats.low_rank_layers
        ));
        output.push_str(&format!(
            "  Corrélation : {}\n",
            if self.injection_stats.correlation_enabled {
                "activée"
            } else {
                "désactivée"
            }
        ));

        output
    }
}

impl fmt::Display for GenerationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format_text())
    }
}

impl Default for DistributionStats {
    fn default() -> Self {
        Self {
            normal_pct: 0.0,
            student_t_pct: 0.0,
            pareto_pct: 0.0,
            other_pct: 0.0,
            total_analyzed: 0,
        }
    }
}

impl Default for InjectionStats {
    fn default() -> Self {
        Self {
            outlier_pct: 0.0,
            low_rank_layers: 0,
            correlation_enabled: false,
            total_analyzed: 0,
        }
    }
}

/// Formate un nombre de paramètres en format lisible (K, M, B, T).
fn format_parameter_count(count: u64) -> String {
    if count >= 1_000_000_000_000 {
        format!("{:.1}T", count as f64 / 1_000_000_000_000.0)
    } else if count >= 1_000_000_000 {
        format!("{:.1}B", count as f64 / 1_000_000_000.0)
    } else if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_creation() {
        let report = GenerationReport::new("test-model", 42);
        assert_eq!(report.model_name, "test-model");
        assert_eq!(report.seed, 42);
    }

    #[test]
    fn report_display() {
        let mut report = GenerationReport::new("ExempleTransformer", 42);
        report.num_layers = 32;
        report.num_tensors = 418;
        report.parameter_count = 7_100_000_000;
        report.distribution_stats = DistributionStats {
            normal_pct: 94.1,
            student_t_pct: 4.2,
            pareto_pct: 0.7,
            other_pct: 1.0,
            total_analyzed: 418,
        };
        report.injection_stats = InjectionStats {
            outlier_pct: 0.83,
            low_rank_layers: 12,
            correlation_enabled: true,
            total_analyzed: 418,
        };

        let text = report.format_text();
        assert!(text.contains("Modèle       : ExempleTransformer"));
        assert!(text.contains("Couches      : 32"));
        assert!(text.contains("Tenseurs     : 418"));
        assert!(text.contains("Paramètres   : 7.1B"));
        assert!(text.contains("Seed         : 42"));
        assert!(text.contains("Normale     94.1 %"));
        assert!(text.contains("Student-t   4.2 %"));
        assert!(text.contains("Pareto      0.7 %"));
        assert!(text.contains("Outliers    : 0.83 %"));
        assert!(text.contains("Low-rank    : 12 couches"));
        assert!(text.contains("Corrélation : activée"));
    }

    #[test]
    fn format_parameter_count_variants() {
        assert_eq!(format_parameter_count(42), "42");
        assert_eq!(format_parameter_count(1_500), "1.5K");
        assert_eq!(format_parameter_count(2_500_000), "2.5M");
        assert_eq!(format_parameter_count(7_100_000_000), "7.1B");
        assert_eq!(format_parameter_count(1_200_000_000_000), "1.2T");
    }

    #[test]
    fn serde_roundtrip() {
        let report = GenerationReport::new("test-model", 42);
        let json = serde_json::to_string(&report).unwrap();
        let back: GenerationReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model_name, report.model_name);
        assert_eq!(back.seed, report.seed);
    }
}
