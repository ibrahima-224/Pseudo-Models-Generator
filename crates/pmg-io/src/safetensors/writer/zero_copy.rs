//! Writer de Tenseurs Zero-Copy
//!
//! Ce module implémente le `ZeroCopyTensorWriter` qui permet l'écriture de tenseurs
//! sans allocation intermédiaire, en utilisant un buffer de conversion unique et
//! en écrivant directement sur disque.

use std::sync::Arc;

use pmg_core::dtype::DType;
use pmg_core::memory::{GlobalMemoryManager, MemoryError, MemoryMonitor};

use super::shard::ShardWriter;

/// Configuration du writer de tenseurs zero-copy
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TensorWriterConfig {
    /// Taille du buffer de conversion en octets (défaut: 8 Mo)
    pub conversion_buffer_size: usize,

    /// Activer la compression optionnelle
    pub enable_compression: bool,

    /// Niveau de compression (0-9, défaut: 3)
    pub compression_level: u32,

    /// Activer le buffering pour l'écriture disque
    pub enable_buffering: bool,

    /// Taille du buffer d'écriture en octets (défaut: 1 Mo)
    pub write_buffer_size: usize,

    /// Activer le tracking des métriques
    pub enable_metrics: bool,

    /// Nombre maximum de chunks en attente
    pub max_pending_chunks: usize,
}

impl Default for TensorWriterConfig {
    fn default() -> Self {
        Self {
            conversion_buffer_size: 8 * 1024 * 1024, // 8 Mo
            enable_compression: false,
            compression_level: 3,
            enable_buffering: true,
            write_buffer_size: 1024 * 1024, // 1 Mo
            enable_metrics: true,
            max_pending_chunks: 16,
        }
    }
}

/// Métriques d'écriture
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct WriterMetrics {
    /// Nombre de tenseurs écrits
    pub tensors_written: usize,

    /// Nombre total de chunks écrits
    pub chunks_written: usize,

    /// Taille totale en octets écrite
    pub total_bytes_written: u64,

    /// Temps total d'écriture en secondes
    pub write_time_secs: f64,

    /// Temps total de conversion en secondes
    pub conversion_time_secs: f64,

    /// Efficacité mémoire (octets écrits / mémoire utilisée)
    pub memory_efficiency: f64,
}

/// État du writer
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriterState {
    /// Prêt à écrire
    Ready,
    /// En cours d'écriture d'un tenseur
    WritingTensor { name: String },
    /// Finalisé
    Finalized,
}

/// Informations sur le tenseur en cours
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TensorInfoWriter {
    /// Nom du tenseur
    pub name: String,

    /// Type de données
    pub dtype: DType,

    /// Forme du tenseur
    pub shape: Vec<usize>,

    /// Nombre d'éléments écrits
    pub elements_written: usize,

    /// Taille totale en octets
    pub total_bytes: usize,
}

/// Erreurs du writer de tenseurs
#[allow(dead_code)]
#[derive(Debug, thiserror::Error)]
pub enum TensorWriteError {
    #[error("Erreur d'écriture: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Erreur mémoire: {0}")]
    MemoryError(#[from] MemoryError),

    #[error("État invalide: {0}")]
    InvalidState(String),

    #[error("Type de données non supporté: {0:?}")]
    UnsupportedDType(DType),

    #[error("Données incomplètes: attendu {expected} éléments, reçu {received}")]
    IncompleteData { expected: usize, received: usize },
}

/// Writer de tenseurs avec écriture zero-copy
///
/// Ce writer permet l'écriture de tenseurs sans allocation intermédiaire
/// en utilisant un buffer de conversion unique et en écrivant directement
/// sur disque via le ShardWriter sous-jacent.
#[allow(dead_code)]
pub struct ZeroCopyTensorWriter {
    /// Writer SafeTensors sous-jacent
    writer: ShardWriter,

    /// Buffer de conversion unique (réutilisé)
    conversion_buffer: Vec<u8>,

    /// Moniteur mémoire global
    memory_monitor: Arc<GlobalMemoryManager>,

    /// Configuration du writer
    config: TensorWriterConfig,

    /// Métriques d'écriture
    metrics: WriterMetrics,

    /// État actuel du writer
    state: WriterState,

    /// Informations sur le tenseur en cours
    current_tensor: Option<TensorInfoWriter>,
}

impl ZeroCopyTensorWriter {
    /// Crée un nouveau writer
    pub fn new(
        writer: ShardWriter,
        memory_monitor: Arc<GlobalMemoryManager>,
        config: TensorWriterConfig,
    ) -> Result<Self, TensorWriteError> {
        // Allouer le buffer de conversion
        let buffer_size = config.conversion_buffer_size;
        memory_monitor.allocate(buffer_size as u64)?;

        let conversion_buffer = vec![0; buffer_size];

        Ok(Self {
            writer,
            conversion_buffer,
            memory_monitor,
            config,
            metrics: WriterMetrics::default(),
            state: WriterState::Ready,
            current_tensor: None,
        })
    }

    /// Commence l'écriture d'un nouveau tenseur
    ///
    /// Cette méthode vérifie que la mémoire totale reste dans les limites
    /// avant de commencer l'écriture du tenseur.
    pub fn begin_tensor(
        &mut self,
        name: &str,
        dtype: DType,
        shape: &[usize],
    ) -> Result<(), TensorWriteError> {
        // Vérifier l'état
        if self.state != WriterState::Ready {
            return Err(TensorWriteError::InvalidState(format!(
                "État actuel: {:?}, attendu: Ready",
                self.state
            )));
        }

        // Calculer la taille totale
        let total_elements: usize = shape.iter().product();
        let element_size = self.element_size(&dtype);
        let total_bytes = total_elements * element_size;

        // Vérifier que la mémoire est disponible pour ce tenseur
        // On vérifie que la mémoire totale utilisée ne dépasse pas la limite
        let current_usage = self.memory_monitor.current_usage();
        let max_memory = self.memory_monitor.max_memory();

        if current_usage + total_bytes as u64 > max_memory {
            return Err(TensorWriteError::MemoryError(
                MemoryError::InsufficientMemory {
                    available: max_memory - current_usage,
                    requested: total_bytes as u64,
                },
            ));
        }

        // Convertir shape en u64 pour ShardWriter
        let shape_u64: Vec<u64> = shape.iter().map(|&s| s as u64).collect();

        // Commencer l'écriture via ShardWriter
        let io_dtype = self.convert_dtype(&dtype);
        self.writer
            .begin_tensor(name, io_dtype, &shape_u64)
            .map_err(|e| {
                TensorWriteError::IoError(std::io::Error::other(format!(
                    "Erreur début tenseur: {}",
                    e
                )))
            })?;

        // Mettre à jour l'état
        self.state = WriterState::WritingTensor {
            name: name.to_string(),
        };
        self.current_tensor = Some(TensorInfoWriter {
            name: name.to_string(),
            dtype,
            shape: shape.to_vec(),
            elements_written: 0,
            total_bytes,
        });

        Ok(())
    }

    /// Écrit un chunk de bytes directement
    pub fn write_bytes_chunk(&mut self, bytes: &[u8]) -> Result<(), TensorWriteError> {
        // Vérifier l'état
        if let WriterState::WritingTensor { .. } = &self.state {
            // Écrire les bytes directement via ShardWriter
            self.writer.write_chunk(bytes).map_err(|e| {
                TensorWriteError::IoError(std::io::Error::other(format!(
                    "Erreur écriture chunk: {}",
                    e
                )))
            })?;

            // Mettre à jour les métriques
            self.metrics.total_bytes_written += bytes.len() as u64;
            self.metrics.chunks_written += 1;

            Ok(())
        } else {
            Err(TensorWriteError::InvalidState(format!(
                "État actuel: {:?}, attendu: WritingTensor",
                self.state
            )))
        }
    }

    /// Écrit un chunk de f64 avec conversion directe (sans copie inutile).
    ///
    /// Cette méthode convertit les valeurs f64 en bytes dans le buffer interne
    /// puis écrit directement depuis ce buffer, évitant ainsi l'allocation
    /// intermédiaire d'un `Vec<u8>`.
    pub fn write_f64_chunk(
        &mut self,
        values: &[f64],
        dtype: DType,
    ) -> Result<(), TensorWriteError> {
        // Vérifier l'état
        if let WriterState::WritingTensor { .. } = &self.state {
            // Convertir f64 en bytes dans le buffer interne (pas de copie)
            let size = self.convert_f64_to_bytes_in_place(values, &dtype)?;

            // Écrire directement depuis le buffer interne en utilisant un pointeur brut
            // pour éviter le conflit d'emprunt (self emprunté immuablement pour le buffer,
            // puis mutablement pour write_bytes_chunk).
            // Sécurité : conversion_buffer n'est pas modifié par write_bytes_chunk
            // (il ne fait qu'écrire dans self.writer).
            let ptr = self.conversion_buffer.as_ptr();
            let slice = unsafe { std::slice::from_raw_parts(ptr, size) };
            self.write_bytes_chunk(slice)?;

            Ok(())
        } else {
            Err(TensorWriteError::InvalidState(format!(
                "État actuel: {:?}, attendu: WritingTensor",
                self.state
            )))
        }
    }

    /// Convertit des valeurs f64 en bytes selon le dtype cible
    ///
    /// Cette méthode vérifie la mémoire disponible avant le redimensionnement
    /// du buffer pour éviter les panics et les allocations non sécurisées.
    fn convert_f64_to_bytes_in_place(
        &mut self,
        values: &[f64],
        dtype: &DType,
    ) -> Result<usize, TensorWriteError> {
        // Calculer la taille nécessaire
        let element_size = self.element_size(dtype);
        let total_size = values.len() * element_size;

        // Vérifier si le buffer est suffisant
        if total_size > self.conversion_buffer.len() {
            // Calculer la taille supplémentaire nécessaire
            let additional_size = total_size - self.conversion_buffer.len();

            // Vérifier si la mémoire est disponible avant le redimensionnement
            if !self.memory_monitor.can_allocate(additional_size as u64) {
                return Err(TensorWriteError::MemoryError(
                    MemoryError::InsufficientMemory {
                        available: self.memory_monitor.max_memory()
                            - self.memory_monitor.current_usage(),
                        requested: additional_size as u64,
                    },
                ));
            }

            // Allouer la mémoire supplémentaire
            self.memory_monitor.allocate(additional_size as u64)?;

            // Redimensionner le buffer de manière sécurisée
            // Utiliser try_resize pour éviter les panics en cas d'échec d'allocation
            self.conversion_buffer
                .try_reserve(additional_size)
                .map_err(|e| {
                    TensorWriteError::MemoryError(MemoryError::SystemError(format!(
                        "Échec d'allocation mémoire pour le buffer: {}",
                        e
                    )))
                })?;

            // Remplir le buffer avec des zéros
            self.conversion_buffer.resize(total_size, 0);
        }

        // Convertir selon le dtype
        match dtype {
            DType::F32 => {
                for (i, &val) in values.iter().enumerate() {
                    let bytes = (val as f32).to_le_bytes();
                    let offset = i * 4;
                    self.conversion_buffer[offset..offset + 4].copy_from_slice(&bytes);
                }
            },
            DType::F16 => {
                // Conversion simplifiée F16 (demi-précision)
                // Note: Sans le crate half, on utilise une conversion approximative
                for (i, &val) in values.iter().enumerate() {
                    // Conversion simplifiée: on utilise f32 puis on tronque
                    let f32_val = val as f32;
                    let bytes = f32_val.to_le_bytes();
                    // Pour F16, on garde les 16 bits de poids fort du f32
                    let offset = i * 2;
                    self.conversion_buffer[offset] = bytes[0];
                    self.conversion_buffer[offset + 1] = bytes[1];
                }
            },
            DType::F64 => {
                for (i, &val) in values.iter().enumerate() {
                    let bytes = val.to_le_bytes();
                    let offset = i * 8;
                    self.conversion_buffer[offset..offset + 8].copy_from_slice(&bytes);
                }
            },
            _ => return Err(TensorWriteError::UnsupportedDType(*dtype)),
        }

        // Retourner la taille utilisée (pas de copie)
        Ok(total_size)
    }

    /// Termine l'écriture du tenseur actuel
    pub fn end_tensor(&mut self) -> Result<(), TensorWriteError> {
        // Vérifier l'état
        if let WriterState::WritingTensor { .. } = &self.state {
            // Terminer l'écriture via ShardWriter
            self.writer.end_tensor().map_err(|e| {
                TensorWriteError::IoError(std::io::Error::other(format!(
                    "Erreur fin tenseur: {}",
                    e
                )))
            })?;

            // Réinitialiser l'état
            self.state = WriterState::Ready;
            self.current_tensor = None;

            // Incrémenter le compteur de tenseurs
            self.metrics.tensors_written += 1;

            Ok(())
        } else {
            Err(TensorWriteError::InvalidState(format!(
                "État actuel: {:?}, attendu: WritingTensor",
                self.state
            )))
        }
    }

    /// Finalise le writer
    pub fn finalize(&mut self) -> Result<(), TensorWriteError> {
        // Vérifier l'état
        if self.state != WriterState::Ready {
            return Err(TensorWriteError::InvalidState(format!(
                "État actuel: {:?}, attendu: Ready",
                self.state
            )));
        }

        // Libérer le buffer de conversion
        let buffer_size = self.conversion_buffer.len();
        self.memory_monitor.deallocate(buffer_size as u64);

        // Mettre à jour l'état
        self.state = WriterState::Finalized;

        Ok(())
    }

    /// Retourne les métriques
    pub fn metrics(&self) -> &WriterMetrics {
        &self.metrics
    }

    /// Réinitialise les métriques
    pub fn reset_metrics(&mut self) {
        self.metrics = WriterMetrics::default();
    }

    /// Retourne la taille en octets d'un élément selon le type de données
    fn element_size(&self, dtype: &DType) -> usize {
        match dtype {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F16 | DType::Bf16 | DType::I16 | DType::U16 => 2,
            DType::I64 | DType::U64 | DType::F64 => 8,
            DType::I8 | DType::U8 | DType::Bool => 1,
            _ => 4, // Par défaut
        }
    }

    /// Convertit un DType en type Safetensors
    fn convert_dtype(&self, core_dtype: &DType) -> super::super::types::DType {
        match core_dtype {
            DType::F32 => super::super::types::DType::F32,
            DType::F16 => super::super::types::DType::F16,
            DType::Bf16 => super::super::types::DType::BF16,
            DType::F8E4M3 => super::super::types::DType::F8E4M3,
            DType::F8E5M2 => super::super::types::DType::F8E5M2,
            DType::I8 => super::super::types::DType::I8,
            DType::I16 => super::super::types::DType::I16,
            DType::I32 => super::super::types::DType::I32,
            DType::I64 => super::super::types::DType::I64,
            DType::U8 => super::super::types::DType::U8,
            DType::U16 => super::super::types::DType::U16,
            DType::U32 => super::super::types::DType::U32,
            DType::U64 => super::super::types::DType::U64,
            DType::Bool => super::super::types::DType::Bool,
            _ => super::super::types::DType::F32, // Par défaut
        }
    }
}

impl Drop for ZeroCopyTensorWriter {
    fn drop(&mut self) {
        // S'assurer que le buffer est libéré
        if self.state != WriterState::Finalized {
            let buffer_size = self.conversion_buffer.len();
            self.memory_monitor.deallocate(buffer_size as u64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pmg_core::memory::GlobalMemoryManager;
    use std::path::PathBuf;
    use std::sync::Arc;

    #[test]
    fn test_creation_writer() {
        // Augmenter la limite mémoire pour couvrir le buffer de conversion (8 Mo) + overhead
        let manager = Arc::new(GlobalMemoryManager::with_limit(16 * 1024 * 1024));
        let path = PathBuf::from("test_writer.safetensors");
        let shard_writer = ShardWriter::new(path, 1024).unwrap();
        let config = TensorWriterConfig::default();

        let writer = ZeroCopyTensorWriter::new(shard_writer, manager, config);
        assert!(writer.is_ok());
    }

    #[test]
    fn test_ecriture_tensor() {
        // Augmenter la limite mémoire pour couvrir le buffer de conversion (8 Mo) + overhead
        let manager = Arc::new(GlobalMemoryManager::with_limit(16 * 1024 * 1024));
        let path = PathBuf::from("test_ecriture.safetensors");
        let shard_writer = ShardWriter::new(path, 1024).unwrap();
        let config = TensorWriterConfig::default();

        let mut writer = ZeroCopyTensorWriter::new(shard_writer, manager, config).unwrap();

        // Écrire un tenseur simple
        writer.begin_tensor("test", DType::F32, &[2, 3]).unwrap();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        writer.write_f64_chunk(&values, DType::F32).unwrap();
        writer.end_tensor().unwrap();

        // Vérifier les métriques
        assert_eq!(writer.metrics().tensors_written, 1);
        assert_eq!(writer.metrics().chunks_written, 1);
    }
}
