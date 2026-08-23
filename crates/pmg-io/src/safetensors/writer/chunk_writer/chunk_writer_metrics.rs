//! Métriques et types pour le ChunkWriter
//!
//! Ce module contient les structures de métriques de performance et les types
//! de résultat pour l'écriture SafeTensors par chunks.

/// Taille par défaut du chunk (8 Mo).
pub const DEFAULT_CHUNK_SIZE: usize = 8 * 1024 * 1024;

/// Métriques de performance pour le ChunkWriter.
///
/// Ces métriques permettent de suivre l'efficacité de l'écriture
/// et d'identifier les goulots d'étranglement.
#[derive(Debug, Clone, Default)]
pub struct ChunkWriterMetrics {
    /// Nombre total d'octets écrits.
    pub bytes_written: u64,
    /// Nombre de chunks écrits.
    pub chunks_written: u64,
    /// Nombre de tensors écrits.
    pub tensors_written: u64,
    /// Temps total d'écriture en millisecondes.
    pub total_write_time_ms: u64,
    /// Nombre de réutilisations de buffers.
    pub buffer_reuses: u64,
    /// Mémoire maximale utilisée simultanément.
    pub peak_memory_usage: usize,
}

impl ChunkWriterMetrics {
    /// Crée des métriques vides.
    pub fn new() -> Self {
        Self::default()
    }

    /// Met à jour les métriques avec une nouvelle écriture.
    pub fn update_write(&mut self, bytes: usize, chunks: usize, time_ms: u64) {
        self.bytes_written += bytes as u64;
        self.chunks_written += chunks as u64;
        self.total_write_time_ms += time_ms;
    }

    /// Incrémente le compteur de réutilisations de buffers.
    pub fn increment_buffer_reuse(&mut self) {
        self.buffer_reuses += 1;
    }

    /// Met à jour la mémoire maximale si nécessaire.
    pub fn update_peak_memory(&mut self, current_memory: usize) {
        if current_memory > self.peak_memory_usage {
            self.peak_memory_usage = current_memory;
        }
    }
}

/// Résultat d'écriture d'un chunk.
///
/// Contient les informations sur l'écriture d'un tenseur ou d'un chunk.
#[derive(Debug, Clone)]
pub struct ChunkWriteResult {
    /// Nombre d'octets écrits.
    pub bytes_written: usize,
    /// Nombre de chunks utilisés.
    pub chunks_used: usize,
    /// Temps d'écriture en millisecondes.
    pub write_time_ms: u64,
}
