//! Algorithmes de compression

/// Algorithmes supportés
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CompressionAlgorithm {
    /// Aucune compression
    #[default]
    None,
    /// LZ4 (rapide)
    Lz4,
    /// Zstd (meilleur ratio)
    Zstd,
    /// GZIP
    Gzip,
}

impl CompressionAlgorithm {
    /// Retourne le nom de l'algorithme
    pub fn name(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Lz4 => "lz4",
            Self::Zstd => "zstd",
            Self::Gzip => "gzip",
        }
    }

    /// Retourne les algorithmes disponibles
    pub fn available() -> Vec<Self> {
        let mut algorithms = vec![Self::None];

        #[cfg(feature = "lz4")]
        algorithms.push(Self::Lz4);

        #[cfg(feature = "zstd")]
        algorithms.push(Self::Zstd);

        algorithms
    }

    /// Parse une chaîne en algorithme
    pub fn parse_from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" | "no" | "off" => Some(Self::None),
            "lz4" => Some(Self::Lz4),
            "zstd" | "zstandard" => Some(Self::Zstd),
            "gzip" | "gz" => Some(Self::Gzip),
            _ => None,
        }
    }
}

impl std::str::FromStr for CompressionAlgorithm {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse_from_str(s).ok_or_else(|| format!("Algorithme de compression inconnu: {}", s))
    }
}

/// Statistiques de compression
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Taille originale en octets
    pub original_size: usize,
    /// Taille compressée en octets
    pub compressed_size: usize,
    /// Temps de compression en millisecondes
    pub compression_time_ms: f64,
    /// Temps de décompression en millisecondes
    pub decompression_time_ms: f64,
    /// Algorithme utilisé
    pub algorithm: CompressionAlgorithm,
}

impl CompressionStats {
    /// Calcule le ratio de compression
    pub fn ratio(&self) -> f64 {
        if self.original_size == 0 {
            return 0.0;
        }
        self.compressed_size as f64 / self.original_size as f64
    }

    /// Calcule le taux de réduction
    pub fn reduction_percent(&self) -> f64 {
        (1.0 - self.ratio()) * 100.0
    }

    /// Calcule le débit de compression en MB/s
    pub fn compression_throughput_mbps(&self) -> f64 {
        if self.compression_time_ms == 0.0 {
            return 0.0;
        }
        (self.original_size as f64 / 1024.0 / 1024.0) / (self.compression_time_ms / 1000.0)
    }

    /// Calcule le débit de décompression en MB/s
    pub fn decompression_throughput_mbps(&self) -> f64 {
        if self.decompression_time_ms == 0.0 {
            return 0.0;
        }
        (self.compressed_size as f64 / 1024.0 / 1024.0) / (self.decompression_time_ms / 1000.0)
    }
}
