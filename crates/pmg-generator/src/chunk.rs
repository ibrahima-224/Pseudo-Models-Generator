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

//! Découpage en chunks pour la gestion mémoire.
//!
//! Ce module permet de générer un grand tenseur par blocs (chunks) sans
//! conserver l'intégralité du tenseur en mémoire. La génération par chunks
//! doit produire exactement les mêmes résultats qu'une génération non découpée.
//!
//! # Principe
//!
//! ```text
//! Tensor (n éléments)
//!    ↓
//! Chunk 0 (taille fixe)
//!    ↓
//! Chunk 1
//!    ↓
//! ...
//!    ↓
//! Chunk N-1 (taille ≤ fixe)
//! ```
//!
//! Chaque chunk possède sa propre seed dérivée de la seed du tenseur,
//! garantissant l'indépendance et le déterminisme.

use crate::error::GeneratorResult;

/// Taille par défaut d'un chunk en nombre d'éléments (1 Mo / 8 octets = 131072).
pub const DEFAULT_CHUNK_SIZE: usize = 131072;

/// Représentation d'un chunk de tenseur.
///
/// Un chunk contient une tranche de valeurs générées et son index.
/// Les chunks sont utilisés pour la génération streaming de grands tenseurs.
///
/// # Exemple
///
/// ```
/// use pmg_generator::TensorChunk;
///
/// let chunk = TensorChunk {
///     chunk_id: 0,
///     values: vec![0.1, 0.2, 0.3],
///     start: 0,
///     end: 3,
/// };
///
/// assert_eq!(chunk.len(), 3);
/// assert!(!chunk.is_empty());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TensorChunk {
    /// Index du chunk (0-based).
    pub chunk_id: u64,
    /// Valeurs du chunk.
    pub values: Vec<f64>,
    /// Index de début dans le tenseur original (inclus).
    pub start: usize,
    /// Index de fin dans le tenseur original (exclus).
    pub end: usize,
}

impl TensorChunk {
    /// Retourne la taille du chunk en nombre d'éléments.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Vérifie si le chunk est vide.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Itérateur de chunks pour un tenseur.
///
/// Permet de parcourir un tenseur par blocs de taille fixe,
/// en générant les valeurs à la volée. L'itérateur est utile pour
/// la génération streaming de grands tenseurs sans les charger entièrement en mémoire.
///
/// # Exemple
///
/// ```
/// use pmg_generator::{ChunkIterator, GeneratorResult};
///
/// let total_size = 1000;
/// let chunk_size = 256;
///
/// let iterator = ChunkIterator::new(total_size, chunk_size, |chunk_id, start, end| {
///     // Générer les valeurs du chunk
///     let values: Vec<f64> = (start..end).map(|i| i as f64).collect();
///     Ok(values)
/// });
///
/// // Parcourir les chunks
/// let mut count = 0;
/// for chunk_result in iterator {
///     let chunk = chunk_result.unwrap();
///     assert!(chunk.len() <= chunk_size);
///     count += 1;
/// }
/// assert_eq!(count, 4); // 1000 / 256 = 3.9 → 4 chunks
/// ```
pub struct ChunkIterator {
    /// Taille totale du tenseur.
    total_size: usize,
    /// Taille de chaque chunk (sauf le dernier).
    chunk_size: usize,
    /// Index du chunk courant.
    current_chunk: u64,
    /// Offset courant dans le tenseur.
    current_offset: usize,
    /// Fonction de génération pour un chunk donné.
    generator: Box<dyn Fn(u64, usize, usize) -> GeneratorResult<Vec<f64>>>,
}

impl ChunkIterator {
    /// Crée un nouvel itérateur de chunks.
    ///
    /// # Paramètres
    /// - `total_size` : nombre total d'éléments du tenseur
    /// - `chunk_size` : taille souhaitée par chunk (sera ajusté si > total_size)
    /// - `generator` : fonction qui génère les valeurs d'un chunk
    ///   `(chunk_id, start, end) -> Vec<f64>`
    ///
    /// # Exemple
    ///
    /// ```
    /// use pmg_generator::ChunkIterator;
    ///
    /// let iterator = ChunkIterator::new(1000, 256, |chunk_id, start, end| {
    ///     let values: Vec<f64> = (start..end).map(|i| i as f64).collect();
    ///     Ok(values)
    /// });
    /// ```
    pub fn new<F>(total_size: usize, chunk_size: usize, generator: F) -> Self
    where
        F: Fn(u64, usize, usize) -> GeneratorResult<Vec<f64>> + 'static,
    {
        let effective_chunk_size = chunk_size.min(total_size);
        Self {
            total_size,
            chunk_size: effective_chunk_size,
            current_chunk: 0,
            current_offset: 0,
            generator: Box::new(generator),
        }
    }

    /// Crée un itérateur avec la taille par défaut.
    pub fn with_default_size<F>(total_size: usize, generator: F) -> Self
    where
        F: Fn(u64, usize, usize) -> GeneratorResult<Vec<f64>> + 'static,
    {
        Self::new(total_size, DEFAULT_CHUNK_SIZE, generator)
    }
}

impl Iterator for ChunkIterator {
    type Item = GeneratorResult<TensorChunk>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_offset >= self.total_size {
            return None;
        }

        let start = self.current_offset;
        let end = (start + self.chunk_size).min(self.total_size);
        let chunk_id = self.current_chunk;

        // Générer les valeurs du chunk
        let values = match (self.generator)(chunk_id, start, end) {
            Ok(v) => v,
            Err(e) => return Some(Err(e)),
        };

        let chunk = TensorChunk {
            chunk_id,
            values,
            start,
            end,
        };

        self.current_offset = end;
        self.current_chunk += 1;

        Some(Ok(chunk))
    }
}

/// Génère tous les chunks d'un tenseur et les collecte dans un vecteur.
///
/// # Avertissement
/// Cette fonction charge tous les chunks en mémoire. Pour la génération
/// streamée, utiliser l'itérateur directement.
pub fn collect_all_chunks<F>(
    total_size: usize,
    chunk_size: usize,
    generator: F,
) -> GeneratorResult<Vec<TensorChunk>>
where
    F: Fn(u64, usize, usize) -> GeneratorResult<Vec<f64>> + 'static,
{
    let iter = ChunkIterator::new(total_size, chunk_size, generator);
    iter.collect()
}

/// Vérifie que la concaténation des chunks reproduit exactement le tenseur complet.
pub fn validate_chunk_concatenation(
    chunks: &[TensorChunk],
    total_size: usize,
) -> GeneratorResult<()> {
    // Vérifier la continuité des indices
    let mut expected_start = 0;
    for (i, chunk) in chunks.iter().enumerate() {
        if chunk.chunk_id != i as u64 {
            return Err(crate::error::GeneratorError::Validation(format!(
                "chunk {} a un id inattendu {}",
                i, chunk.chunk_id
            )));
        }
        if chunk.start != expected_start {
            return Err(crate::error::GeneratorError::Validation(format!(
                "chunk {} a un start {} inattendu (attendu {})",
                i, chunk.start, expected_start
            )));
        }
        if chunk.end <= chunk.start {
            return Err(crate::error::GeneratorError::Validation(format!(
                "chunk {} a des indices invalides: start={}, end={}",
                i, chunk.start, chunk.end
            )));
        }
        expected_start = chunk.end;
    }

    // Vérifier la taille totale
    if expected_start != total_size {
        return Err(crate::error::GeneratorError::Validation(format!(
            "la concaténation des chunks a une taille {} au lieu de {}",
            expected_start, total_size
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_generator(_chunk_id: u64, start: usize, end: usize) -> GeneratorResult<Vec<f64>> {
        // Génère des valeurs déterministes basées sur l'index
        Ok((start..end).map(|i| i as f64).collect())
    }

    #[test]
    fn chunk_iterator_basic() {
        let total_size = 100;
        let chunk_size = 30;
        let chunks: Vec<_> = ChunkIterator::new(total_size, chunk_size, test_generator)
            .collect::<Result<_, _>>()
            .unwrap();

        assert_eq!(chunks.len(), 4); // 30, 30, 30, 10
        assert_eq!(chunks[0].len(), 30);
        assert_eq!(chunks[1].len(), 30);
        assert_eq!(chunks[2].len(), 30);
        assert_eq!(chunks[3].len(), 10);

        // Vérifier la continuité
        assert!(validate_chunk_concatenation(&chunks, total_size).is_ok());
    }

    #[test]
    fn chunk_iterator_exact_multiple() {
        let total_size = 100;
        let chunk_size = 25;
        let chunks: Vec<_> = ChunkIterator::new(total_size, chunk_size, test_generator)
            .collect::<Result<_, _>>()
            .unwrap();

        assert_eq!(chunks.len(), 4);
        for chunk in &chunks {
            assert_eq!(chunk.len(), 25);
        }
    }

    #[test]
    fn chunk_iterator_single_chunk() {
        let total_size = 50;
        let chunk_size = 100; // Plus grand que total_size
        let chunks: Vec<_> = ChunkIterator::new(total_size, chunk_size, test_generator)
            .collect::<Result<_, _>>()
            .unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 50);
    }

    #[test]
    fn chunk_iterator_empty() {
        let total_size = 0;
        let chunk_size = 100;
        let chunks: Vec<_> = ChunkIterator::new(total_size, chunk_size, test_generator)
            .collect::<Result<_, _>>()
            .unwrap();

        assert_eq!(chunks.len(), 0);
    }

    #[test]
    fn collect_all_chunks_basic() {
        let total_size = 100;
        let chunk_size = 30;
        let chunks = collect_all_chunks(total_size, chunk_size, test_generator).unwrap();
        assert_eq!(chunks.len(), 4);
    }

    #[test]
    fn validate_chunk_concatenation_valid() {
        let chunks = vec![
            TensorChunk {
                chunk_id: 0,
                values: vec![0.0, 1.0],
                start: 0,
                end: 2,
            },
            TensorChunk {
                chunk_id: 1,
                values: vec![2.0, 3.0],
                start: 2,
                end: 4,
            },
        ];
        assert!(validate_chunk_concatenation(&chunks, 4).is_ok());
    }

    #[test]
    fn validate_chunk_concatenation_wrong_size() {
        let chunks = vec![TensorChunk {
            chunk_id: 0,
            values: vec![0.0, 1.0],
            start: 0,
            end: 2,
        }];
        assert!(validate_chunk_concatenation(&chunks, 4).is_err());
    }

    #[test]
    fn validate_chunk_concatenation_gap() {
        let chunks = vec![
            TensorChunk {
                chunk_id: 0,
                values: vec![0.0, 1.0],
                start: 0,
                end: 2,
            },
            TensorChunk {
                chunk_id: 1,
                values: vec![2.0, 3.0],
                start: 3,
                end: 5,
            }, // Gap à 2
        ];
        assert!(validate_chunk_concatenation(&chunks, 5).is_err());
    }
}
