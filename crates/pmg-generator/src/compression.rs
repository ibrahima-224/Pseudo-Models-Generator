//! Module de compression pour le générateur
//!
//! Ce module fournit l'intégration de la compression dans le pipeline de génération.

#[cfg(feature = "compression")]
use pmg_compression::{CompressionAlgorithm, CompressionConfig, Compressor};

use crate::error::GeneratorError;

/// Configuration de compression pour le générateur
#[derive(Debug, Clone, Default)]
pub struct GeneratorCompressionConfig {
    /// Activer la compression
    pub enabled: bool,
    /// Configuration de compression
    #[cfg(feature = "compression")]
    pub config: CompressionConfig,
}

/// Compresseur de tenseurs pour le générateur
pub struct GeneratorCompressor {
    #[cfg(feature = "compression")]
    compressor: Option<Compressor>,
}

impl GeneratorCompressor {
    /// Crée un nouveau compresseur pour le générateur
    pub fn new(_config: &GeneratorCompressionConfig) -> Result<Self, GeneratorError> {
        #[cfg(feature = "compression")]
        if config.enabled {
            let compressor = Compressor::new(config.config.clone())
                .map_err(|e| GeneratorError::GenerationFailed(e.to_string()))?;
            return Ok(Self {
                compressor: Some(compressor),
            });
        }

        Ok(Self {
            #[cfg(feature = "compression")]
            compressor: None,
        })
    }

    /// Compresse des données si la compression est activée
    pub fn compress(
        &mut self,
        data: &[u8],
        config: &GeneratorCompressionConfig,
    ) -> Result<Vec<u8>, GeneratorError> {
        if !config.enabled {
            return Ok(data.to_vec());
        }

        #[cfg(feature = "compression")]
        if let Some(ref mut compressor) = self.compressor {
            return compressor
                .compress(data)
                .map_err(|e| GeneratorError::GenerationFailed(e.to_string()));
        }

        Ok(data.to_vec())
    }

    /// Décompresse des données si la compression est activée
    pub fn decompress(
        &mut self,
        data: &[u8],
        config: &GeneratorCompressionConfig,
    ) -> Result<Vec<u8>, GeneratorError> {
        if !config.enabled {
            return Ok(data.to_vec());
        }

        #[cfg(feature = "compression")]
        if let Some(ref mut compressor) = self.compressor {
            return compressor
                .decompress(data)
                .map_err(|e| GeneratorError::GenerationFailed(e.to_string()));
        }

        Ok(data.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_compression_config_default() {
        let config = GeneratorCompressionConfig::default();
        assert!(!config.enabled);
    }

    #[test]
    fn test_generator_compressor_creation() {
        let config = GeneratorCompressionConfig::default();
        let compressor = GeneratorCompressor::new(&config);
        assert!(compressor.is_ok());
    }
}
