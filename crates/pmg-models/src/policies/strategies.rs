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

//! Sous-module contenant les énumérations de stratégies pour les politiques.

use serde::{Deserialize, Serialize};

/// Stratégie de dérivation de la seed pour un tenseur.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SeedStrategy {
    /// Seed globale partagée par tous les tenseurs.
    Global,
    /// Seed dérivée par tenseur (hash du nom).
    PerTensor,
    /// Seed dérivée par couche + tenseur.
    PerLayer,
    /// Seed dérivée par expert + tenseur.
    PerExpert,
}

impl SeedStrategy {
    /// Libellé français pour l'affichage.
    pub fn label_fr(self) -> &'static str {
        match self {
            SeedStrategy::Global => "globale",
            SeedStrategy::PerTensor => "par tenseur",
            SeedStrategy::PerLayer => "par couche",
            SeedStrategy::PerExpert => "par expert",
        }
    }
}

/// Stratégie de sérialisation des outliers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutlierStrategy {
    /// Super-poids multiplicatifs (w' = scale·w).
    Multiplicative,
    /// Super-poids additifs (w' = w + offset).
    Additive,
    /// Remplacement par une valeur fixe.
    Replacement,
    /// Mélange à queues lourdes (Student-t).
    HeavyTail,
}

impl OutlierStrategy {
    /// Libellé français pour l'affichage.
    pub fn label_fr(self) -> &'static str {
        match self {
            OutlierStrategy::Multiplicative => "multiplicatif",
            OutlierStrategy::Additive => "additif",
            OutlierStrategy::Replacement => "remplacement",
            OutlierStrategy::HeavyTail => "queues lourdes",
        }
    }
}

/// Stratégie de corrélation entre colonnes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CorrelationStrategy {
    /// Corrélation de Pearson (linéaire).
    Pearson,
    /// Corrélation de Spearman (basée sur les rangs).
    Spearman,
    /// Corrélation de Kendall (concordance).
    Kendall,
    /// Corrélation par blocs (matrices de corrélation).
    Block,
}

impl CorrelationStrategy {
    /// Libellé français pour l'affichage.
    pub fn label_fr(self) -> &'static str {
        match self {
            CorrelationStrategy::Pearson => "Pearson",
            CorrelationStrategy::Spearman => "Spearman",
            CorrelationStrategy::Kendall => "Kendall",
            CorrelationStrategy::Block => "par blocs",
        }
    }
}

/// Stratégie de décomposition bas-rang.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LowRankStrategy {
    /// Décomposition en valeurs singulières (SVD).
    Svd,
    /// Factorisation NMF (Non-negative Matrix Factorization).
    Nmf,
    /// Factorisation par valeurs aléatoires (randomized).
    Randomized,
    /// Composantes principales (PCA).
    Pca,
}

impl LowRankStrategy {
    /// Libellé français pour l'affichage.
    pub fn label_fr(self) -> &'static str {
        match self {
            LowRankStrategy::Svd => "SVD",
            LowRankStrategy::Nmf => "NMF",
            LowRankStrategy::Randomized => "aléatoire",
            LowRankStrategy::Pca => "ACP",
        }
    }
}

/// Stratégie de compression pour la sérialisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompressionStrategy {
    /// Pas de compression.
    None,
    /// Compression Zstandard.
    Zstd,
    /// Compression LZ4.
    Lz4,
    /// Compression Gzip.
    Gzip,
}

impl CompressionStrategy {
    /// Libellé français pour l'affichage.
    pub fn label_fr(self) -> &'static str {
        match self {
            CompressionStrategy::None => "aucune",
            CompressionStrategy::Zstd => "zstd",
            CompressionStrategy::Lz4 => "lz4",
            CompressionStrategy::Gzip => "gzip",
        }
    }
}
