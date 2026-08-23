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

//! Sous-module contenant l'énumération des étapes du pipeline d'injection.

/// Étape du pipeline d'injection canonique.
///
/// L'ordre est fixé par [`InjectionStage::ORDER`] ; chaque étape possède un
/// domaine de seed séparé ([`InjectionStage::domain`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InjectionStage {
    /// Génération du tenseur de base selon la distribution du spec.
    Distribution,
    /// Structure sparse localisée (blocs, bande, lignes/colonnes).
    Structure,
    /// Corrélation contrôlée entre colonnes.
    Correlation,
    /// Composante bas-rang `α·UVᵀ`.
    LowRank,
    /// Outliers / super-poids.
    SuperWeights,
}

impl InjectionStage {
    /// Ordre canonique d'injection (spécification §4.8) — testé dans ce module.
    pub const ORDER: [InjectionStage; 5] = [
        InjectionStage::Distribution,
        InjectionStage::Structure,
        InjectionStage::Correlation,
        InjectionStage::LowRank,
        InjectionStage::SuperWeights,
    ];

    /// Domaine de dérivation de seed propre à l'étape (séparation des flux).
    pub const fn domain(self) -> &'static str {
        match self {
            InjectionStage::Distribution => "distribution",
            InjectionStage::Structure => "structure",
            InjectionStage::Correlation => "correlation",
            InjectionStage::LowRank => "low_rank",
            InjectionStage::SuperWeights => "super_weights",
        }
    }
}
