//! Compression/décompression en streaming
//!
//! Ce module fournit des compresseurs/décompresseurs en streaming avec
//! gestion mémoire bornée et vidage sur disque pour les gros volumes.

use crate::buffer_pool::BufferPool;
use crate::compressor::{CompressionConfig, Compressor};
use crate::error::{CompressionError, CompressionResult};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use tempfile::tempdir;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Compresseur en streaming avec batch writing et pool de buffers
pub struct StreamingCompressor<W: AsyncWrite + Unpin> {
    writer: W,
    compressor: Compressor,
    buffer: Vec<u8>,
    compressed_buffer: Vec<u8>,
    block_size: usize,
    batch_size: usize,
    buffer_pool: BufferPool,
}

impl<W: AsyncWrite + Unpin> StreamingCompressor<W> {
    /// Crée un nouveau compresseur en streaming
    ///
    /// # Arguments
    /// * `writer` - Le writer asynchrone de sortie
    /// * `config` - La configuration de compression
    pub fn new(writer: W, config: CompressionConfig) -> CompressionResult<Self> {
        let block_size = config.block_size;
        let batch_size = 4; // Par défaut, 4 blocs avant flush
        Ok(Self {
            writer,
            compressor: Compressor::new(config)?,
            buffer: Vec::with_capacity(block_size * 2),
            compressed_buffer: Vec::with_capacity(block_size * 2),
            block_size,
            batch_size,
            buffer_pool: BufferPool::default(),
        })
    }

    /// Crée un nouveau compresseur en streaming avec pool de buffers personnalisé
    ///
    /// # Arguments
    /// * `writer` - Le writer asynchrone de sortie
    /// * `config` - La configuration de compression
    /// * `buffer_pool` - Le pool de buffers à utiliser
    pub fn new_with_pool(
        writer: W,
        config: CompressionConfig,
        buffer_pool: BufferPool,
    ) -> CompressionResult<Self> {
        let block_size = config.block_size;
        let batch_size = 4;
        Ok(Self {
            writer,
            compressor: Compressor::new(config)?,
            buffer: Vec::with_capacity(block_size * 2),
            compressed_buffer: Vec::with_capacity(block_size * 2),
            block_size,
            batch_size,
            buffer_pool,
        })
    }

    /// Configure le nombre de blocs avant flush
    ///
    /// # Arguments
    /// * `batch_size` - Nombre de blocs à accumuler avant écriture
    pub fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = batch_size.max(1);
    }

    /// Retourne le nombre de blocs avant flush
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Retourne une référence au pool de buffers
    pub fn buffer_pool(&self) -> &BufferPool {
        &self.buffer_pool
    }

    /// Retourne une référence mutable au pool de buffers
    pub fn buffer_pool_mut(&mut self) -> &mut BufferPool {
        &mut self.buffer_pool
    }

    /// Écrit des données avec compression
    ///
    /// Utilise le batch writing pour accumuler les données compressées
    /// avant de les écrire en une seule opération.
    pub async fn write(&mut self, data: &[u8]) -> CompressionResult<()> {
        self.buffer.extend_from_slice(data);

        // Compresser par blocs
        while self.buffer.len() >= self.block_size {
            let chunk: Vec<u8> = self.buffer.drain(..self.block_size).collect();
            let compressed = self.compressor.compress(&chunk)?;

            // Écrire la taille du bloc compressé
            let len = compressed.len() as u32;
            self.compressed_buffer.extend_from_slice(&len.to_le_bytes());
            self.compressed_buffer.extend_from_slice(&compressed);

            // Flush si le batch est atteint
            if self.compressed_buffer.len() >= self.block_size * self.batch_size {
                self.flush_compressed_buffer().await?;
            }
        }

        Ok(())
    }

    /// Écrit le buffer compressé accumulé
    async fn flush_compressed_buffer(&mut self) -> CompressionResult<()> {
        if !self.compressed_buffer.is_empty() {
            self.writer.write_all(&self.compressed_buffer).await?;
            self.compressed_buffer.clear();
        }
        Ok(())
    }

    /// Finalise la compression
    ///
    /// Compress le reste du buffer et écrit le marqueur de fin.
    pub async fn finish(&mut self) -> CompressionResult<()> {
        // Compresser le reste du buffer
        if !self.buffer.is_empty() {
            let compressed = self.compressor.compress(&self.buffer)?;
            let len = compressed.len() as u32;
            self.compressed_buffer.extend_from_slice(&len.to_le_bytes());
            self.compressed_buffer.extend_from_slice(&compressed);
        }

        // Écrire un marqueur de fin
        self.compressed_buffer
            .extend_from_slice(&0u32.to_le_bytes());

        // Flush final
        self.flush_compressed_buffer().await?;
        self.writer.flush().await?;

        Ok(())
    }

    /// Récupère un buffer du pool ou crée un nouveau
    pub fn get_buffer_from_pool(&mut self) -> Vec<u8> {
        self.buffer_pool.get()
    }

    /// Remet un buffer dans le pool
    pub fn put_buffer_to_pool(&mut self, buffer: Vec<u8>) {
        self.buffer_pool.put(buffer);
    }
}

/// Décompresseur en streaming avec buffer borné
///
/// Gère la décompression en streaming avec une limite mémoire configurable.
/// Si le buffer dépasse la limite, les données sont écrites sur disque.
pub struct StreamingDecompressor<R: AsyncRead + Unpin> {
    /// Lecteur source des données compressées
    reader: R,
    /// Compresseur pour la décompression
    compressor: Compressor,
    /// Buffer de données décompressées
    buffer: Vec<u8>,
    /// Position actuelle de lecture dans le buffer
    pos: usize,
    /// Taille maximale du buffer en octets
    max_buffer_size: usize,
    /// Fichier temporaire pour le vidage sur disque
    temp_file: Option<BufWriter<File>>,
    /// Chemin du fichier temporaire
    temp_path: Option<PathBuf>,
    /// Positions des segments écrits sur disque
    file_positions: Vec<u64>,
}

impl<R: AsyncRead + Unpin> StreamingDecompressor<R> {
    /// Crée un nouveau décompresseur en streaming avec buffer borné
    ///
    /// # Arguments
    /// * `reader` - Le reader asynchrone d'entrée
    /// * `config` - La configuration de compression
    pub fn new(reader: R, config: CompressionConfig) -> CompressionResult<Self> {
        let max_buffer_size = config.max_buffer_size;
        Ok(Self {
            reader,
            compressor: Compressor::new(config)?,
            buffer: Vec::with_capacity(max_buffer_size.min(1024 * 1024)), // Capacité initiale raisonnable
            pos: 0,
            max_buffer_size,
            temp_file: None,
            temp_path: None,
            file_positions: Vec::new(),
        })
    }

    /// Lit des données décompressées
    pub async fn read(&mut self, buf: &mut [u8]) -> CompressionResult<usize> {
        // Remplir le buffer si nécessaire
        if self.pos >= self.buffer.len() {
            self.fill_buffer().await?;
        }

        if self.buffer.is_empty() {
            return Ok(0);
        }

        // Copier les données
        let available = &self.buffer[self.pos..];
        let to_copy = std::cmp::min(buf.len(), available.len());
        buf[..to_copy].copy_from_slice(&available[..to_copy]);
        self.pos += to_copy;

        Ok(to_copy)
    }

    /// Écrit le buffer actuel sur disque pour libérer de la mémoire
    ///
    /// Crée un fichier temporaire si nécessaire et écrit le contenu du buffer.
    fn flush_to_disk(&mut self) -> CompressionResult<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        // Créer un fichier temporaire si nécessaire
        if self.temp_file.is_none() {
            let temp_dir = tempdir().map_err(|e| {
                CompressionError::IoError(std::io::Error::other(format!(
                    "Impossible de créer le répertoire temporaire: {}",
                    e
                )))
            })?;
            let temp_path = temp_dir.keep().join("pmg_streaming_buffer.bin");
            let file = File::create(&temp_path).map_err(|e| {
                CompressionError::IoError(std::io::Error::other(format!(
                    "Impossible de créer le fichier temporaire: {}",
                    e
                )))
            })?;
            self.temp_file = Some(BufWriter::new(file));
            self.temp_path = Some(temp_path);
        }

        // Écrire le buffer sur disque
        if let Some(ref mut writer) = self.temp_file {
            writer.write_all(&self.buffer).map_err(|e| {
                CompressionError::IoError(std::io::Error::other(format!(
                    "Erreur d'écriture sur disque: {}",
                    e
                )))
            })?;
            writer.flush().map_err(|e| {
                CompressionError::IoError(std::io::Error::other(format!(
                    "Erreur de flush sur disque: {}",
                    e
                )))
            })?;

            // Enregistrer la position
            let position =
                self.file_positions.last().copied().unwrap_or(0) + self.buffer.len() as u64;
            self.file_positions.push(position);
        }

        // Vider le buffer
        self.buffer.clear();
        self.pos = 0;

        Ok(())
    }

    /// Remplit le buffer avec le prochain bloc décompressé
    async fn fill_buffer(&mut self) -> CompressionResult<()> {
        // Lire la taille du bloc
        let mut len_bytes = [0u8; 4];
        match self.reader.read_exact(&mut len_bytes).await {
            Ok(_bytes_read) => {},
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(());
            },
            Err(e) => return Err(e.into()),
        }

        let len = u32::from_le_bytes(len_bytes) as usize;

        // Marqueur de fin
        if len == 0 {
            return Ok(());
        }

        // Lire et décompresser le bloc
        let mut compressed = vec![0u8; len];
        self.reader.read_exact(&mut compressed).await?;

        // Décompresser le bloc
        let decompressed = self.compressor.decompress(&compressed)?;

        // Vérifier si le buffer va dépasser la limite
        if self.buffer.len() + decompressed.len() > self.max_buffer_size {
            // Écrire le buffer actuel sur disque avant d'ajouter les nouvelles données
            self.flush_to_disk()?;
        }

        // Ajouter les données décompressées au buffer
        self.buffer.extend_from_slice(&decompressed);
        self.pos = 0;

        Ok(())
    }
}

impl<R: AsyncRead + Unpin> Drop for StreamingDecompressor<R> {
    fn drop(&mut self) {
        // Nettoyer le fichier temporaire s'il existe
        if let Some(path) = &self.temp_path {
            if let Err(e) = fs::remove_file(path) {
                eprintln!(
                    "Attention: Impossible de supprimer le fichier temporaire {:?}: {}",
                    path, e
                );
            }
        }
    }
}

/// Interface de compression async
#[allow(async_fn_in_trait)]
pub trait AsyncCompression {
    /// Compresse des données de manière asynchrone
    async fn compress_async(&mut self, data: &[u8]) -> CompressionResult<Vec<u8>>;

    /// Décompresse des données de manière asynchrone
    async fn decompress_async(&mut self, data: &[u8]) -> CompressionResult<Vec<u8>>;
}

impl AsyncCompression for Compressor {
    async fn compress_async(&mut self, data: &[u8]) -> CompressionResult<Vec<u8>> {
        self.compress(data)
    }

    async fn decompress_async(&mut self, data: &[u8]) -> CompressionResult<Vec<u8>> {
        self.decompress(data)
    }
}
