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

//! Structures sparse contrôlées : blocs denses dans une matrice clairsemée,
//! patterns de blocs, lignes/colonnes structurées.
//!
//! Le but n'est pas de produire des zéros « au hasard », mais de représenter
//! une **structure contrôlée** (spécification étape 4.6) : un tenseur sparse
//! dont les éléments non nuls sont concentrés en blocs localisés, plutôt que
//! dispersés uniformément. C'est le comportement observé dans les poids réels
//! (structured sparsity des modèles compressés).
//!
//! Toutes les structures sont produites de façon **déterministe** : les
//! décisions (position des blocs, densité intra-bloc) passent par un flux
//! dérivé de seed, jamais par une source aléatoire globale.

use serde::{Deserialize, Serialize};

use pmg_math::distribution::Distribution;
use pmg_math::distributions::Normal;
use pmg_math::rng::DeterministicRng;

use crate::error::{InjectorError, InjectorResult};

/// Type de structure localisée à injecter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum BlockPattern {
    /// Un seul bloc dense centré aléatoirement dans la matrice.
    SingleBlock,
    /// Plusieurs blocs denses disposés en grille régulière.
    Grid,
    /// Lignes entières denses (structure par lignes).
    Rows,
    /// Colonnes entières denses (structure par colonnes).
    Columns,
    /// Motif en diagonale (bande dense autour de la diagonale).
    DiagonalBand,
}

/// Paramètres de construction d'une structure sparse contrôlée.
///
/// # Invariants (vérifiés par [`SparseStructureSpec::validate`])
/// - `density ∈ (0, 1]` : fraction des éléments qui restent non nuls ;
/// - `block_rows`, `block_cols ≥ 1` pour les patterns de blocs ;
/// - `band_width ≥ 1` pour [`BlockPattern::DiagonalBand`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SparseStructureSpec {
    /// Pattern de localisation.
    pub pattern: BlockPattern,
    /// Densité cible des éléments non nuls dans `(0, 1]`.
    pub density: f64,
    /// Hauteur des blocs (patterns `SingleBlock`/`Grid`).
    pub block_rows: usize,
    /// Largeur des blocs (patterns `SingleBlock`/`Grid`).
    pub block_cols: usize,
    /// Nombre de blocs par dimension pour `Grid`.
    pub grid_blocks: usize,
    /// Demi-largeur de la bande diagonale pour `DiagonalBand`.
    pub band_width: usize,
}

impl SparseStructureSpec {
    /// Construit une spécification valide.
    ///
    /// # Erreurs
    /// [`InjectorError::InvalidPolicy`] si un paramètre viole ses bornes.
    pub fn new(
        pattern: BlockPattern,
        density: f64,
        block_rows: usize,
        block_cols: usize,
        grid_blocks: usize,
        band_width: usize,
    ) -> InjectorResult<Self> {
        let spec = Self {
            pattern,
            density,
            block_rows,
            block_cols,
            grid_blocks,
            band_width,
        };
        spec.validate()?;
        Ok(spec)
    }

    /// Valide les invariants de la spécification.
    ///
    /// # Erreurs
    /// [`InjectorError::InvalidPolicy`] avec le nom du champ fautif.
    pub fn validate(&self) -> InjectorResult<()> {
        if !self.density.is_finite() || self.density <= 0.0 || self.density > 1.0 {
            return Err(InjectorError::InvalidPolicy(format!(
                "density doit être dans (0, 1], reçu {}",
                self.density
            )));
        }
        if (self.pattern == BlockPattern::SingleBlock || self.pattern == BlockPattern::Grid)
            && (self.block_rows == 0 || self.block_cols == 0)
        {
            return Err(InjectorError::InvalidPolicy(format!(
                "block_rows={} et block_cols={} doivent être ≥ 1",
                self.block_rows, self.block_cols
            )));
        }
        if self.pattern == BlockPattern::Grid && self.grid_blocks == 0 {
            return Err(InjectorError::InvalidPolicy(
                "grid_blocks doit être ≥ 1 pour le pattern Grid".into(),
            ));
        }
        if self.pattern == BlockPattern::DiagonalBand && self.band_width == 0 {
            return Err(InjectorError::InvalidPolicy(
                "band_width doit être ≥ 1 pour le pattern DiagonalBand".into(),
            ));
        }
        Ok(())
    }
}

/// Construit une matrice sparse structurée `rows × cols` : les éléments
/// sélectionnés par le pattern reçoivent un tirage gaussien `N(0, stddev)`,
/// les autres valent exactement `0.0`.
///
/// # Entrées
/// - `rng` : flux déterministe dérivé (domaine `"sparse"`) ;
/// - `rows`, `cols` : dimensions ;
/// - `spec` : pattern + densité ;
/// - `stddev` : écart-type des éléments non nuls (`> 0`).
///
/// # Sorties
/// Matrice `rows × cols` (ligne par ligne), sparse **structurée**.
///
/// # Erreurs
/// - [`InjectorError::InvalidTensor`] si `rows == 0 || cols == 0` ;
/// - [`InjectorError::InvalidPolicy`] si la spécification ou `stddev` est
///   invalide.
///
/// # Complexité
/// O(rows·cols).
pub fn generate_sparse_structure(
    rng: &mut DeterministicRng,
    rows: usize,
    cols: usize,
    spec: &SparseStructureSpec,
    stddev: f64,
) -> InjectorResult<Vec<f64>> {
    if rows == 0 || cols == 0 {
        return Err(InjectorError::InvalidTensor(format!(
            "dimensions nulles : rows={rows}, cols={cols}"
        )));
    }
    if !stddev.is_finite() || stddev <= 0.0 {
        return Err(InjectorError::InvalidPolicy(format!(
            "stddev doit être fini et > 0, reçu {stddev}"
        )));
    }
    spec.validate()?;
    let mut mask = vec![false; rows * cols];
    let mut normal = Normal::new(0.0, stddev)?;
    match spec.pattern {
        BlockPattern::SingleBlock => {
            // Bloc unique : position de départ tirée déterministiquement.
            let max_r = rows.saturating_sub(spec.block_rows) + 1;
            let max_c = cols.saturating_sub(spec.block_cols) + 1;
            if max_r == 0 || max_c == 0 {
                // Bloc plus grand que la matrice : on le clamp (densité 1).
                for row in 0..rows {
                    for col in 0..cols {
                        mask[row * cols + col] = true;
                    }
                }
            } else {
                let start_r = (rng.next_f64() * max_r as f64).floor() as usize;
                let start_c = (rng.next_f64() * max_c as f64).floor() as usize;
                for row in start_r..(start_r + spec.block_rows) {
                    for col in start_c..(start_c + spec.block_cols) {
                        mask[row * cols + col] = true;
                    }
                }
            }
        },
        BlockPattern::Grid => {
            // Grille régulière de blocs : positions déterministes (pas de tirage).
            let g = spec.grid_blocks.max(1);
            for gi in 0..g {
                let start_r = (rows as f64 * gi as f64 / g as f64).floor() as usize;
                let start_c = (cols as f64 * gi as f64 / g as f64).floor() as usize;
                let end_r = (start_r + spec.block_rows).min(rows);
                let end_c = (start_c + spec.block_cols).min(cols);
                for row in start_r..end_r {
                    for col in start_c..end_c {
                        mask[row * cols + col] = true;
                    }
                }
            }
        },
        BlockPattern::Rows => {
            // Lignes denses : sélection déterministe par pas régulier (le
            // nombre de lignes suit la densité ; chaque ligne est unique).
            let n_rows = (rows as f64 * spec.density).round() as usize;
            let n_rows = n_rows.max(1).min(rows);
            let step = (rows as f64 / n_rows as f64).ceil() as usize;
            for i in 0..n_rows {
                let idx = (i * step).min(rows - 1);
                for col in 0..cols {
                    mask[idx * cols + col] = true;
                }
            }
        },
        BlockPattern::Columns => {
            // Colonnes denses : sélection déterministe par pas régulier.
            let n_cols = (cols as f64 * spec.density).round() as usize;
            let n_cols = n_cols.max(1).min(cols);
            let step = (cols as f64 / n_cols as f64).ceil() as usize;
            for i in 0..n_cols {
                let idx = (i * step).min(cols - 1);
                for row in 0..rows {
                    mask[row * cols + idx] = true;
                }
            }
        },
        BlockPattern::DiagonalBand => {
            // Bande autour de la diagonale : |i − j| < band_width.
            let w = spec.band_width.max(1);
            for row in 0..rows {
                let lo = row.saturating_sub(w.saturating_sub(1));
                let hi = (row + w).min(cols);
                for col in lo..hi {
                    mask[row * cols + col] = true;
                }
            }
        },
    }
    // Remplissage gaussien des positions sélectionnées.
    let mut out = vec![0.0f64; rows * cols];
    for (o, &m) in out.iter_mut().zip(mask.iter()) {
        if m {
            *o = normal.sample(rng);
        }
    }
    Ok(out)
}

/// Applique une structure sparse contrôlée **sur un tenseur existant** :
/// les positions hors structure sont mises à zéro, celles de la structure
/// sont conservées (sans modification de leurs valeurs).
///
/// # Erreurs
/// [`InjectorError::InvalidTensor`] si `buffer.len() != rows·cols`.
///
/// # Complexité
/// O(rows·cols).
pub fn apply_sparse_mask(
    buffer: &mut [f64],
    rows: usize,
    cols: usize,
    spec: &SparseStructureSpec,
) -> InjectorResult<()> {
    if buffer.len() != rows * cols {
        return Err(InjectorError::InvalidTensor(format!(
            "buffer de longueur {} ≠ rows·cols = {rows}·{cols}",
            buffer.len()
        )));
    }
    spec.validate()?;
    // Masque de structure : mêmes décisions que la génération, sans tirage
    // (le remplissage gaussien est inutile ici).
    let mut mask = vec![false; rows * cols];
    match spec.pattern {
        BlockPattern::SingleBlock => {
            let max_r = rows.saturating_sub(spec.block_rows) + 1;
            let max_c = cols.saturating_sub(spec.block_cols) + 1;
            if max_r == 0 || max_c == 0 {
                return Ok(()); // Tout est dans la structure : rien à mettre à zéro.
            }
            // Position déterministe : reprise du premier tirage du flux n'est
            // pas possible ici (pas de RNG) — on centre le bloc. Choix
            // documenté : pour l'application sur tenseur existant, le bloc est
            // centré (déterministe, sans tirage).
            let start_r = rows / 2 - spec.block_rows / 2;
            let start_c = cols / 2 - spec.block_cols / 2;
            for row in start_r..(start_r + spec.block_rows).min(rows) {
                for col in start_c..(start_c + spec.block_cols).min(cols) {
                    mask[row * cols + col] = true;
                }
            }
        },
        BlockPattern::Grid => {
            let g = spec.grid_blocks.max(1);
            for gi in 0..g {
                let start_r = (rows as f64 * gi as f64 / g as f64).floor() as usize;
                let start_c = (cols as f64 * gi as f64 / g as f64).floor() as usize;
                let end_r = (start_r + spec.block_rows).min(rows);
                let end_c = (start_c + spec.block_cols).min(cols);
                for row in start_r..end_r {
                    for col in start_c..end_c {
                        mask[row * cols + col] = true;
                    }
                }
            }
        },
        BlockPattern::Rows => {
            // Lignes paires (choix déterministe sans tirage).
            for row in (0..rows).step_by(2) {
                for col in 0..cols {
                    mask[row * cols + col] = true;
                }
            }
        },
        BlockPattern::Columns => {
            for col in (0..cols).step_by(2) {
                for row in 0..rows {
                    mask[row * cols + col] = true;
                }
            }
        },
        BlockPattern::DiagonalBand => {
            let w = spec.band_width.max(1);
            for row in 0..rows {
                let lo = row.saturating_sub(w.saturating_sub(1));
                let hi = (row + w).min(cols);
                for col in lo..hi {
                    mask[row * cols + col] = true;
                }
            }
        },
    }
    for (v, &m) in buffer.iter_mut().zip(mask.iter()) {
        if !m {
            *v = 0.0;
        }
    }
    Ok(())
}

/// Mesure la densité réelle (fraction de valeurs non nulles) d'un buffer.
///
/// # Entrées
/// - `buffer` : valeurs ;
/// - `epsilon` : seuil de « zéro » (une valeur `|x| ≤ epsilon` est considérée
///   nulle, défaut typique `1e-12`).
///
/// # Complexité
/// O(n).
pub fn measured_density(buffer: &[f64], epsilon: f64) -> f64 {
    if buffer.is_empty() {
        return 0.0;
    }
    let nonzero = buffer.iter().filter(|&&x| x.abs() > epsilon).count();
    nonzero as f64 / buffer.len() as f64
}

#[cfg(test)]
#[path = "sparse_structure_tests.rs"]
mod tests;
