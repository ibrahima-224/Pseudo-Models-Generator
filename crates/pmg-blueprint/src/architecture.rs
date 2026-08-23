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

//! Types d'architecture (`ArchitectureKind`) et métadonnées associées.
//!
//! Le blueprint décrit l'architecture d'un pseudo-modèle de façon abstraite :
//! la famille architecturale pilote le planner (quels tenseurs, quels motifs
//! de nommage, quelles structures MoE/attention).

use serde::{Deserialize, Serialize};

/// Famille architecturale d'un pseudo-modèle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ArchitectureKind {
    /// Transformeur dense (attention + MLP dense), sans MoE.
    DenseTransformer,
    /// Transformeur à experts (GLM-5.2-like, DeepSeek-like).
    MoETransformer,
    /// Transformeur à attention hybride (DSA + MLA + hyper-connections).
    HybridAttentionTransformer,
}

impl ArchitectureKind {
    /// Nom canonique (forme sérialisée stable).
    pub fn name(self) -> &'static str {
        match self {
            ArchitectureKind::DenseTransformer => "dense-transformer",
            ArchitectureKind::MoETransformer => "moe-transformer",
            ArchitectureKind::HybridAttentionTransformer => "hybrid-attention-transformer",
        }
    }

    /// Vrai si l'architecture comporte des experts MoE.
    pub fn has_moe(self) -> bool {
        matches!(
            self,
            ArchitectureKind::MoETransformer | ArchitectureKind::HybridAttentionTransformer
        )
    }
}

/// Métadonnées d'architecture décrivant le style d'attention et de norme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArchitectureMeta {
    /// Style d'attention dominant (`dense`, `gqa`, `mla`, `dsa`, `hybrid`).
    pub attention_style: AttentionStyle,
    /// Type de normalisation (`rmsnorm` par défaut pour GLM/DeepSeek).
    pub norm_style: NormStyle,
    /// Hyper-connections présentes (DeepSeek `hc_*`).
    pub hyper_connections: bool,
}

impl ArchitectureMeta {
    /// Métadonnées de référence pour un GLM-5.2 (DSA + indexeur).
    pub fn glm52() -> ArchitectureMeta {
        ArchitectureMeta {
            attention_style: AttentionStyle::Dsa,
            norm_style: NormStyle::RmsNorm,
            hyper_connections: false,
        }
    }

    /// Métadonnées de référence pour un DeepSeek-V4-Flash (MLA + hc).
    pub fn deepseek_v4_flash() -> ArchitectureMeta {
        ArchitectureMeta {
            attention_style: AttentionStyle::Mla,
            norm_style: NormStyle::RmsNorm,
            hyper_connections: true,
        }
    }
}

/// Style d'attention d'un modèle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AttentionStyle {
    /// Attention dense classique.
    Dense,
    /// Grouped-Query Attention.
    Gqa,
    /// Multi-head Latent Attention.
    Mla,
    /// DeepSeek Sparse Attention (indexeur).
    Dsa,
    /// Hybride dense + sparse.
    Hybrid,
}

/// Style de normalisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum NormStyle {
    /// RMSNorm (GLM, DeepSeek).
    RmsNorm,
    /// LayerNorm.
    LayerNorm,
}

#[cfg(test)]
mod tests {
    use super::{ArchitectureKind, ArchitectureMeta, AttentionStyle, NormStyle};

    #[test]
    fn moe_detection() {
        assert!(ArchitectureKind::MoETransformer.has_moe());
        assert!(ArchitectureKind::HybridAttentionTransformer.has_moe());
        assert!(!ArchitectureKind::DenseTransformer.has_moe());
    }

    #[test]
    fn canonical_names() {
        assert_eq!(
            ArchitectureKind::DenseTransformer.name(),
            "dense-transformer"
        );
        assert_eq!(ArchitectureKind::MoETransformer.name(), "moe-transformer");
        assert_eq!(
            ArchitectureKind::HybridAttentionTransformer.name(),
            "hybrid-attention-transformer"
        );
    }

    #[test]
    fn reference_metadata() {
        let glm = ArchitectureMeta::glm52();
        assert_eq!(glm.attention_style, AttentionStyle::Dsa);
        assert!(!glm.hyper_connections);
        let ds = ArchitectureMeta::deepseek_v4_flash();
        assert_eq!(ds.attention_style, AttentionStyle::Mla);
        assert!(ds.hyper_connections);
        assert_eq!(ds.norm_style, NormStyle::RmsNorm);
    }

    #[test]
    fn serde_roundtrip() {
        let meta = ArchitectureMeta::glm52();
        let json = serde_json::to_string(&meta).unwrap();
        assert_eq!(
            serde_json::from_str::<ArchitectureMeta>(&json).unwrap(),
            meta
        );
    }
}
