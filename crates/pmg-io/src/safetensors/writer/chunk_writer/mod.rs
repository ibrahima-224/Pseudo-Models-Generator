//! # ChunkWriter : Écriture SafeTensors par chunks adaptatifs
//!
//! Ce module implémente un writer SafeTensors optimisé pour l'écriture par chunks,
//! permettant de réduire la consommation mémoire à O(chunk_size) au lieu de O(model_size).
//!
//! ## Objectifs
//! - Écrire les tenseurs par morceaux (chunks) de taille configurable
//! - Réutiliser les buffers via un pool pour éviter les allocations successives
//! - Adapter la taille des chunks selon la taille des tenseurs
//! - Fournir des métriques de performance détaillées
//!
//! ## Organisation
//!
//! Le module est divisé en sous-modules pour respecter la limite de 500 lignes :
//! - `buffer_pool` : Pool de buffers réutilisables
//! - `chunk_writer_metrics` : Métriques et types de résultat
//! - `chunk_writer_tests` : Tests unitaires

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use crate::pool::{PoolConfig, UnifiedBufferPool};
use crate::safetensors::types::DType;

// Déclaration des sous-modules
pub mod buffer_pool;
pub mod chunk_writer_metrics;
#[cfg(test)]
pub mod chunk_writer_tests;

// Réexports pour rétrocompatibilité API
// On garde les anciennes constantes pour la rétrocompatibilité
pub use buffer_pool::{DEFAULT_MAX_POOL_MEMORY, MAX_CHUNK_SIZE, MIN_CHUNK_SIZE};
pub use chunk_writer_metrics::{ChunkWriteResult, ChunkWriterMetrics, DEFAULT_CHUNK_SIZE};

/// Writer SafeTensors avec écriture par chunks adaptatifs.
///
/// Ce writer découpe les tenseurs en chunks de taille configurable
/// et utilise un pool de buffers pour optimiser les performances.
pub struct ChunkWriter {
    /// Fichier de sortie avec buffer interne.
    file: BufWriter<File>,
    /// Réserve pour l'en-tête SafeTensors (non utilisée pour l'instant).
    _header_reserve: usize,
    /// Offset actuel dans le fichier.
    current_offset: usize,
    /// Taille du chunk courant (adaptative).
    chunk_size: usize,
    /// Pool de buffers unifié et thread-safe.
    buffer_pool: UnifiedBufferPool,
    /// Métriques de performance.
    metrics: ChunkWriterMetrics,
    /// Nombre de chunks écrits.
    chunks_written: usize,
    /// Chemin du fichier de sortie.
    path: std::path::PathBuf,
}

impl ChunkWriter {
    /// Crée un nouveau ChunkWriter avec taille de chunk par défaut (8 Mo).
    ///
    /// # Paramètres
    /// - `path` : chemin du fichier de sortie
    /// - `header_reserve` : taille réservée pour l'en-tête (en octets)
    ///
    /// # Retourne
    /// Un ChunkWriter prêt à l'emploi.
    ///
    /// # Erreurs
    /// Retourne une erreur si la création du fichier échoue.
    pub fn new(path: impl AsRef<Path>, header_reserve: usize) -> std::io::Result<Self> {
        Self::with_chunk_size(path, header_reserve, DEFAULT_CHUNK_SIZE)
    }

    /// Crée un ChunkWriter avec taille de chunk personnalisée.
    ///
    /// # Paramètres
    /// - `path` : chemin du fichier de sortie
    /// - `header_reserve` : taille réservée pour l'en-tête
    /// - `chunk_size` : taille des chunks (entre MIN_CHUNK_SIZE et MAX_CHUNK_SIZE)
    ///
    /// # Retourne
    /// Un ChunkWriter prêt à l'emploi.
    ///
    /// # Erreurs
    /// Retourne une erreur si :
    /// - Le fichier ne peut pas être créé
    /// - La taille du chunk est hors limites
    pub fn with_chunk_size(
        path: impl AsRef<Path>,
        header_reserve: usize,
        chunk_size: usize,
    ) -> std::io::Result<Self> {
        // Validation de la taille du chunk
        if !(MIN_CHUNK_SIZE..=MAX_CHUNK_SIZE).contains(&chunk_size) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "La taille du chunk doit être entre {} et {} octets",
                    MIN_CHUNK_SIZE, MAX_CHUNK_SIZE
                ),
            ));
        }

        // Création du fichier avec buffer interne
        let file = File::create(path.as_ref())?;
        let buffer = BufWriter::with_capacity(chunk_size, file);

        // Initialisation du pool de buffers unifié
        let pool_config = PoolConfig::new(
            DEFAULT_MAX_POOL_MEMORY, // Mémoire maximale
            MIN_CHUNK_SIZE,          // Taille minimale buffer
            MAX_CHUNK_SIZE,          // Taille maximale buffer
            true,                    // Métriques activées
        );
        let buffer_pool = UnifiedBufferPool::new(pool_config);

        Ok(Self {
            file: buffer,
            _header_reserve: header_reserve,
            current_offset: header_reserve, // On commence après la réservation
            chunk_size,
            buffer_pool,
            metrics: ChunkWriterMetrics::new(),
            chunks_written: 0,
            path: path.as_ref().to_path_buf(),
        })
    }

    /// Écrit un tenseur complet en le découpant en chunks.
    ///
    /// # Paramètres
    /// - `name` : nom du tenseur
    /// - `data` : données binaires du tenseur
    /// - `dtype` : type de donnée
    /// - `shape` : forme du tenseur
    ///
    /// # Retourne
    /// Le résultat de l'écriture avec métriques.
    ///
    /// # Comportement
    /// - Découpe `data` en chunks de taille `self.chunk_size`
    /// - Réutilise les buffers du pool pour éviter les allocations
    /// - Met à jour les métriques de performance
    pub fn write_tensor(
        &mut self,
        _name: &str,
        data: &[u8],
        _dtype: DType,
        _shape: &[u64],
    ) -> std::io::Result<ChunkWriteResult> {
        let start_time = Instant::now();
        let tensor_size = data.len();

        // Adaptation de la taille du chunk selon la taille du tenseur
        let chunk_size = self.adaptive_chunk_size(tensor_size);

        let mut total_bytes_written = 0;
        let mut chunks_used = 0;
        let mut offset = 0;

        // Écriture par chunks
        while offset < tensor_size {
            let current_chunk_size = (tensor_size - offset).min(chunk_size);
            let chunk_data = &data[offset..offset + current_chunk_size];

            // Obtention d'un buffer du pool unifié
            let mut buffer = self.buffer_pool.acquire_u8(current_chunk_size);
            buffer.extend_from_slice(chunk_data);

            // Écriture du chunk
            self.file.write_all(&buffer)?;
            total_bytes_written += current_chunk_size;
            chunks_used += 1;

            // Remise du buffer dans le pool unifié
            self.buffer_pool.release_u8(buffer);

            // Mise à jour des compteurs
            offset += current_chunk_size;
            self.current_offset += current_chunk_size;
        }

        // Calcul du temps d'écriture
        let write_time_ms = start_time.elapsed().as_millis() as u64;

        // Mise à jour des métriques
        self.metrics
            .update_write(total_bytes_written, chunks_used, write_time_ms);
        self.metrics
            .update_peak_memory(self.buffer_pool.memory_usage());
        self.chunks_written += chunks_used;

        Ok(ChunkWriteResult {
            bytes_written: total_bytes_written,
            chunks_used,
            write_time_ms,
        })
    }

    /// Écrit un chunk unique (pour le streaming par chunks).
    ///
    /// # Paramètres
    /// - `name` : nom du tenseur
    /// - `data` : données du chunk (pas le tenseur complet)
    /// - `offset` : offset du chunk dans le tenseur
    /// - `total_size` : taille totale du tenseur
    /// - `dtype` : type de donnée
    /// - `shape` : forme du tenseur
    ///
    /// # Erreurs
    /// Retourne une erreur si l'écriture échoue.
    pub fn write_tensor_chunk(
        &mut self,
        _name: &str,
        data: &[u8],
        _offset: usize,
        _total_size: usize,
        _dtype: DType,
        _shape: &[u64],
    ) -> std::io::Result<()> {
        let start_time = Instant::now();

        // Obtention d'un buffer du pool unifié
        let mut buffer = self.buffer_pool.acquire_u8(data.len());
        buffer.extend_from_slice(data);

        // Écriture du chunk
        self.file.write_all(&buffer)?;

        // Remise du buffer dans le pool unifié
        self.buffer_pool.release_u8(buffer);

        // Mise à jour des compteurs
        self.current_offset += data.len();
        self.chunks_written += 1;

        // Calcul du temps d'écriture
        let write_time_ms = start_time.elapsed().as_millis() as u64;

        // Mise à jour des métriques
        self.metrics.update_write(data.len(), 1, write_time_ms);
        self.metrics
            .update_peak_memory(self.buffer_pool.memory_usage());

        Ok(())
    }

    /// Adapte la taille du chunk selon la taille du tenseur.
    ///
    /// # Logique
    /// - Tenseur < 1 Mo : chunk = taille du tenseur
    /// - Tenseur 1-32 Mo : chunk = 1 Mo
    /// - Tenseur > 32 Mo : chunk = 8 Mo (défaut)
    fn adaptive_chunk_size(&self, tensor_size: usize) -> usize {
        if tensor_size < MIN_CHUNK_SIZE {
            // Pour les petits tenseurs, on utilise toute la taille
            tensor_size
        } else if tensor_size <= MAX_CHUNK_SIZE {
            // Pour les tenseurs moyens, on utilise 1 Mo
            MIN_CHUNK_SIZE
        } else {
            // Pour les grands tenseurs, on utilise la taille par défaut
            self.chunk_size
        }
    }

    /// Force le vidage du buffer et synchronise avec le disque.
    ///
    /// # Erreurs
    /// Retourne une erreur si le vidage échoue.
    pub fn flush_and_sync(&mut self) -> std::io::Result<()> {
        self.file.flush()?;
        self.file.get_ref().sync_all()?;
        Ok(())
    }

    /// Finalise l'écriture et retourne les métriques.
    ///
    /// # Retourne
    /// Les métriques de performance de l'écriture.
    pub fn finalize(mut self) -> std::io::Result<ChunkWriterMetrics> {
        // Vidage final du buffer
        self.file.flush()?;
        // Synchronisation avec le disque pour garantir l'écriture
        self.file.get_ref().sync_all()?;

        // Retourne les métriques
        Ok(self.metrics.clone())
    }

    /// Retourne les métriques de performance.
    ///
    /// # Retourne
    /// Une référence vers les métriques actuelles.
    pub fn metrics(&self) -> &ChunkWriterMetrics {
        &self.metrics
    }

    /// Retourne l'offset actuel dans le fichier.
    ///
    /// # Retourne
    /// L'offset actuel en octets.
    pub fn current_offset(&self) -> usize {
        self.current_offset
    }

    /// Retourne le chemin du fichier de sortie.
    ///
    /// # Retourne
    /// Une référence vers le chemin du fichier.
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// Retourne la taille du chunk configurée.
    ///
    /// # Retourne
    /// La taille du chunk en octets.
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Retourne la mémoire totale utilisée par le pool de buffers.
    ///
    /// # Retourne
    /// La mémoire utilisée en octets.
    pub fn pool_memory_usage(&self) -> usize {
        self.buffer_pool.memory_usage()
    }

    /// Retourne le nombre de buffers dans le pool.
    ///
    /// # Retourne
    /// Le nombre total de buffers disponibles.
    pub fn pool_buffer_count(&self) -> usize {
        self.buffer_pool.buffer_count()
    }

    /// Réinitialise le pool de buffers.
    pub fn clear_pool(&mut self) {
        self.buffer_pool.clear();
    }
}
