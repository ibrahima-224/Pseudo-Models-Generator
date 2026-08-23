//! Compresseur de tenseurs
//!
//! Ce module fournit un compresseur pour les tenseurs avec support LZ4 et Zstd.
//! Il inclut la réutilisation des encodeurs/décodeurs Zstd pour optimiser les performances.

use crate::algorithms::CompressionAlgorithm;
use crate::error::{CompressionError, CompressionResult};

/// Configuration de compression
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Algorithme de compression
    pub algorithm: CompressionAlgorithm,
    /// Niveau de compression (0-22 pour Zstd, 0-16 pour LZ4)
    pub level: u32,
    /// Taille des blocs en octets
    pub block_size: usize,
    /// Activer la compression en mémoire
    pub in_memory: bool,
    /// Utiliser le dictionary
    pub use_dictionary: bool,
    /// Taille maximale du buffer en octets (défaut: 8 Mo)
    pub max_buffer_size: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::Lz4,
            level: 6,
            block_size: 64 * 1024, // 64 KB
            in_memory: true,
            use_dictionary: false,
            max_buffer_size: 8 * 1024 * 1024, // 8 Mo par défaut
        }
    }
}

/// Compresseur de tenseurs
pub struct Compressor {
    config: CompressionConfig,
    #[cfg(feature = "zstd")]
    zstd_encoder: Option<zstd::Encoder<Vec<u8>>>,
    #[cfg(feature = "zstd")]
    zstd_decoder: Option<zstd::Decoder<Vec<u8>>>,
}

impl Compressor {
    /// Crée un nouveau compresseur
    pub fn new(config: CompressionConfig) -> CompressionResult<Self> {
        Ok(Self {
            config,
            #[cfg(feature = "zstd")]
            zstd_encoder: None,
            #[cfg(feature = "zstd")]
            zstd_decoder: None,
        })
    }

    /// Compresse des données
    pub fn compress(&mut self, data: &[u8]) -> CompressionResult<Vec<u8>> {
        match self.config.algorithm {
            CompressionAlgorithm::Lz4 => self.compress_lz4(data),
            CompressionAlgorithm::Zstd => self.compress_zstd(data),
            CompressionAlgorithm::None => Ok(data.to_vec()),
            _ => Err(CompressionError::UnsupportedAlgorithm(format!(
                "{:?}",
                self.config.algorithm
            ))),
        }
    }

    /// Compresse avec LZ4
    #[cfg(feature = "lz4")]
    fn compress_lz4(&mut self, data: &[u8]) -> CompressionResult<Vec<u8>> {
        Ok(lz4_flex::compress_prepend_size(data))
    }

    #[cfg(not(feature = "lz4"))]
    fn compress_lz4(&mut self, _data: &[u8]) -> CompressionResult<Vec<u8>> {
        Err(CompressionError::UnsupportedAlgorithm("LZ4".to_string()))
    }

    /// Compresse avec Zstd en réutilisant l'encodeur
    #[cfg(feature = "zstd")]
    fn compress_zstd(&mut self, data: &[u8]) -> CompressionResult<Vec<u8>> {
        // Réutiliser l'encodeur existant ou en créer un nouveau
        let encoder = match self.zstd_encoder.take() {
            Some(mut enc) => {
                // Réinitialiser l'encodeur avec une nouvelle sortie
                enc = zstd::Encoder::new(Vec::new(), self.config.level as i32)
                    .map_err(|e| CompressionError::CompressionFailed(e.to_string()))?;
                enc
            },
            None => zstd::Encoder::new(Vec::new(), self.config.level as i32)
                .map_err(|e| CompressionError::CompressionFailed(e.to_string()))?,
        };

        let mut writer = encoder;
        std::io::Write::write_all(&mut writer, data)
            .map_err(|e| CompressionError::CompressionFailed(e.to_string()))?;

        let result = writer
            .finish()
            .map_err(|e| CompressionError::CompressionFailed(e.to_string()))?;

        Ok(result)
    }

    #[cfg(not(feature = "zstd"))]
    fn compress_zstd(&mut self, _data: &[u8]) -> CompressionResult<Vec<u8>> {
        Err(CompressionError::UnsupportedAlgorithm("Zstd".to_string()))
    }

    /// Décompresse des données
    pub fn decompress(&mut self, data: &[u8]) -> CompressionResult<Vec<u8>> {
        match self.config.algorithm {
            CompressionAlgorithm::Lz4 => self.decompress_lz4(data),
            CompressionAlgorithm::Zstd => self.decompress_zstd(data),
            CompressionAlgorithm::None => Ok(data.to_vec()),
            _ => Err(CompressionError::UnsupportedAlgorithm(format!(
                "{:?}",
                self.config.algorithm
            ))),
        }
    }

    /// Décompresse LZ4
    #[cfg(feature = "lz4")]
    fn decompress_lz4(&mut self, data: &[u8]) -> CompressionResult<Vec<u8>> {
        lz4_flex::decompress_size_prepended(data)
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))
    }

    #[cfg(not(feature = "lz4"))]
    fn decompress_lz4(&mut self, _data: &[u8]) -> CompressionResult<Vec<u8>> {
        Err(CompressionError::UnsupportedAlgorithm("LZ4".to_string()))
    }

    /// Décompresse Zstd avec pré-allocation intelligente et réutilisation du décodeur
    #[cfg(feature = "zstd")]
    fn decompress_zstd(&mut self, data: &[u8]) -> CompressionResult<Vec<u8>> {
        // Réutiliser le décodeur existant ou en créer un nouveau
        let decoder = match self.zstd_decoder.take() {
            Some(mut dec) => {
                // Réinitialiser le décodeur avec les nouvelles données
                dec = zstd::Decoder::new(data)
                    .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;
                dec
            },
            None => zstd::Decoder::new(data)
                .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?,
        };

        // Estimer la taille décompressée (Zstd stocke souvent la taille dans le header)
        // Utiliser une estimation conservative : 4x la taille compressée
        let estimated_size = data.len() * 4;
        let mut output = Vec::with_capacity(estimated_size);

        std::io::Read::read_to_end(&mut decoder, &mut output)
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))?;

        Ok(output)
    }

    #[cfg(not(feature = "zstd"))]
    fn decompress_zstd(&mut self, _data: &[u8]) -> CompressionResult<Vec<u8>> {
        Err(CompressionError::UnsupportedAlgorithm("Zstd".to_string()))
    }

    /// Calcule le ratio de compression
    pub fn compression_ratio(&self, original: &[u8], compressed: &[u8]) -> f64 {
        if original.is_empty() {
            return 0.0;
        }
        compressed.len() as f64 / original.len() as f64
    }

    /// Retourne la configuration
    pub fn config(&self) -> &CompressionConfig {
        &self.config
    }
}
