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

//! Hash des métadonnées canoniques d'un modèle (`MetadataHash`).
//!
//! `MetadataHash` permet de tracer l'origine des données sans exposer les
//! poids. Le hash porte sur la métadonnée canonique (config + index + header),
//! jamais sur les poids eux-mêmes.
//!
//! Conformité : `docs/architecture/03-modeles-de-donnees.md` §3.7.
//!
//! # Exemple
//!
//! ```
//! use pmg_core::{MetadataHash, HashAlgorithm};
//!
//! let hash = MetadataHash::compute(b"{\"model_type\":\"glm\"}");
//! assert_eq!(hash.algorithm(), HashAlgorithm::Sha256);
//! assert_eq!(hash.value().len(), 64); // SHA-256 → 64 caractères hex
//! ```

use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256, Sha512};

/// Algorithme de hachage supporté.
///
/// Seuls les algorithmes cryptographiquement sûrs sont autorisés.
/// L'enum est `#[non_exhaustive]` pour permettre l'ajout futur
/// d'algorithmes (ex. SHA-3) sans cassure ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[non_exhaustive]
pub enum HashAlgorithm {
    /// SHA-256 (256 bits, 64 caractères hexadécimaux).
    Sha256,
    /// SHA-512 (512 bits, 128 caractères hexadécimaux).
    Sha512,
}

impl HashAlgorithm {
    /// Libellé de l'algorithme pour l'affichage et la sérialisation.
    pub fn label(self) -> &'static str {
        match self {
            HashAlgorithm::Sha256 => "sha256",
            HashAlgorithm::Sha512 => "sha512",
        }
    }

    /// Longueur attendue de la valeur hexadécimale en caractères.
    pub fn hex_length(self) -> usize {
        match self {
            HashAlgorithm::Sha256 => 64,
            HashAlgorithm::Sha512 => 128,
        }
    }
}

impl fmt::Display for HashAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Hash des métadonnées canoniques d'un modèle.
///
/// Utilisé pour tracer l'origine des données sans exposer les poids.
/// Le hash porte sur la métadonnée canonique (config + index + header),
/// jamais sur les poids eux-mêmes.
///
/// Conformité : `docs/architecture/03-modeles-de-donnees.md` §3.7.
///
/// # Invariants
///
/// 1. `algorithm` est toujours un algorithme valide.
/// 2. `value` est une chaîne hexadécimale valide.
/// 3. La longueur de `value` correspond à `algorithm.hex_length()`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MetadataHash {
    /// Algorithme utilisé (ex. "sha256").
    algorithm: HashAlgorithm,
    /// Valeur du hash en hexadécimal.
    value: String,
}

impl MetadataHash {
    /// Crée un nouveau hash avec l'algorithme et la valeur spécifiés.
    ///
    /// # Paramètres
    ///
    /// * `algorithm` — Algorithme de hachage utilisé.
    /// * `value` — Valeur du hash en hexadécimal.
    ///
    /// # Erreurs
    ///
    /// Retourne une erreur si la valeur n'est pas un hash hexadécimal valide
    /// pour l'algorithme donné.
    pub fn new(algorithm: HashAlgorithm, value: impl Into<String>) -> Result<Self, String> {
        let value = value.into();
        if value.is_empty() {
            return Err("la valeur du hash ne peut pas être vide".to_string());
        }
        if value.len() != algorithm.hex_length() {
            return Err(format!(
                "longueur invalide : attendu {} caractères hex pour {}, obtenu {}",
                algorithm.hex_length(),
                algorithm.label(),
                value.len()
            ));
        }
        if !value.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err("la valeur du hash doit être en hexadécimal".to_string());
        }
        Ok(Self { algorithm, value })
    }

    /// Calcule un hash SHA-256 à partir de données brutes.
    ///
    /// # Paramètres
    ///
    /// * `data` — Données à hacher.
    ///
    /// # Retour
    ///
    /// Un `MetadataHash` utilisant SHA-256.
    pub fn compute(data: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        let value = hex_encode(&result);
        Self {
            algorithm: HashAlgorithm::Sha256,
            value,
        }
    }

    /// Calcule un hash avec l'algorithme spécifié.
    ///
    /// # Paramètres
    ///
    /// * `algorithm` — Algorithme à utiliser.
    /// * `data` — Données à hacher.
    ///
    /// # Retour
    ///
    /// Un `MetadataHash` avec l'algorithme choisi.
    pub fn compute_with(algorithm: HashAlgorithm, data: &[u8]) -> Self {
        let value = match algorithm {
            HashAlgorithm::Sha256 => {
                let mut hasher = Sha256::new();
                hasher.update(data);
                hex_encode(&hasher.finalize())
            },
            HashAlgorithm::Sha512 => {
                let mut hasher = Sha512::new();
                hasher.update(data);
                hex_encode(&hasher.finalize())
            },
        };
        Self { algorithm, value }
    }

    /// Vérifie que le hash correspond aux données fournies.
    ///
    /// # Paramètres
    ///
    /// * `data` — Données à vérifier.
    ///
    /// # Retour
    ///
    /// `true` si le hash correspond, `false` sinon.
    pub fn verify(&self, data: &[u8]) -> bool {
        let computed = Self::compute_with(self.algorithm, data);
        computed.value == self.value
    }

    /// Retourne l'algorithme utilisé.
    pub fn algorithm(&self) -> HashAlgorithm {
        self.algorithm
    }

    /// Retourne la valeur du hash en hexadécimal.
    pub fn value(&self) -> &str {
        &self.value
    }

    /// Formate le hash pour l'affichage (ex. "sha256:abc123...").
    pub fn display_short(&self) -> String {
        let prefix_len = 8.min(self.value.len());
        format!("{}:{}…", self.algorithm.label(), &self.value[..prefix_len])
    }
}

impl fmt::Display for MetadataHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.algorithm.label(), self.value)
    }
}

impl Default for MetadataHash {
    /// Hash par défaut (SHA-256 de chaîne vide) — principalement pour la
    /// désérialisation.
    fn default() -> Self {
        Self::compute(b"")
    }
}

/// Encode un tableau d'octets en chaîne hexadécimale minuscule.
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Teste la création d'un hash SHA-256 valide.
    #[test]
    fn test_metadata_hash_creation() {
        let hash = MetadataHash::new(
            HashAlgorithm::Sha256,
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        )
        .expect("hash valide");

        assert_eq!(hash.algorithm(), HashAlgorithm::Sha256);
        assert_eq!(hash.value().len(), 64);
    }

    /// Teste la création d'un hash SHA-512 valide.
    #[test]
    fn test_metadata_hash_creation_sha512() {
        let hex_val = "0".repeat(128);
        let hash = MetadataHash::new(HashAlgorithm::Sha512, &hex_val).expect("hash valide");
        assert_eq!(hash.algorithm(), HashAlgorithm::Sha512);
        assert_eq!(hash.value().len(), 128);
    }

    /// Teste la création d'un hash SHA-256 à partir de données.
    #[test]
    fn test_metadata_hash_compute() {
        let data = b"Hello, PMG!";
        let hash = MetadataHash::compute(data);

        assert_eq!(hash.algorithm(), HashAlgorithm::Sha256);
        assert_eq!(hash.value().len(), 64);

        // Vérification indépendante avec sha2
        let mut hasher = Sha256::new();
        hasher.update(data);
        let expected: Vec<String> = hasher
            .finalize()
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect();
        let expected_str: String = expected.concat();
        assert_eq!(hash.value(), &expected_str);
    }

    /// Teste la vérification d'un hash.
    #[test]
    fn test_metadata_hash_verify() {
        let data = b"test data";
        let hash = MetadataHash::compute(data);

        assert!(hash.verify(data));
        assert!(!hash.verify(b"other data"));
    }

    /// Teste le format d'affichage court.
    #[test]
    fn test_metadata_hash_display_short() {
        let hash = MetadataHash::compute(b"test");
        let display = hash.display_short();
        assert!(display.starts_with("sha256:"));
        assert!(display.ends_with('…'));
    }

    /// Teste le format d'affichage complet.
    #[test]
    fn test_metadata_hash_display_full() {
        let hash = MetadataHash::compute(b"test");
        let display = format!("{}", hash);
        assert!(display.starts_with("sha256:"));
        assert_eq!(display.len(), 71); // "sha256:" (7) + 64 hex chars = 71
    }

    /// Teste la rejection d'une valeur vide.
    #[test]
    fn test_metadata_hash_empty_value_rejected() {
        let result = MetadataHash::new(HashAlgorithm::Sha256, "");
        assert!(result.is_err());
    }

    /// Teste la rejection d'une longueur incorrecte.
    #[test]
    fn test_metadata_hash_wrong_length_rejected() {
        let result = MetadataHash::new(HashAlgorithm::Sha256, "abc123");
        assert!(result.is_err());
    }

    /// Teste la rejection d'un caractère non hexadécimal.
    #[test]
    fn test_metadata_hash_invalid_hex_rejected() {
        // Construire manuellement un hash avec un caractère non hex
        let mut hex_str = "0".repeat(64);
        // Remplacer le premier caractère par 'g' (non hexadécimal)
        let mut chars: Vec<char> = hex_str.chars().collect();
        chars[0] = 'g';
        hex_str = chars.into_iter().collect();
        let result = MetadataHash::new(HashAlgorithm::Sha256, &hex_str);
        assert!(result.is_err());
    }

    /// Teste le Default : SHA-256 de chaîne vide.
    #[test]
    fn test_metadata_hash_default() {
        let hash = MetadataHash::default();
        assert_eq!(hash.algorithm(), HashAlgorithm::Sha256);

        // Vérifie que c'est bien le hash de chaîne vide
        let expected = MetadataHash::compute(b"");
        assert_eq!(hash, expected);
    }

    /// Teste la sérialisation/désérialisation roundtrip.
    #[test]
    fn test_metadata_hash_roundtrip() {
        let hash = MetadataHash::compute(b"donnees importantes");
        let json = serde_json::to_string(&hash).expect("sérialisation OK");
        let deserialized: MetadataHash = serde_json::from_str(&json).expect("désérialisation OK");
        assert_eq!(hash, deserialized);
    }

    /// Teste compute_with avec SHA-512.
    #[test]
    fn test_metadata_hash_compute_with_sha512() {
        let data = b"test sha512";
        let hash = MetadataHash::compute_with(HashAlgorithm::Sha512, data);
        assert_eq!(hash.algorithm(), HashAlgorithm::Sha512);
        assert_eq!(hash.value().len(), 128);
        assert!(hash.verify(data));
    }

    /// Teste le format Display de HashAlgorithm.
    #[test]
    fn test_hash_algorithm_display() {
        assert_eq!(format!("{}", HashAlgorithm::Sha256), "sha256");
        assert_eq!(format!("{}", HashAlgorithm::Sha512), "sha512");
    }

    /// Teste hex_length.
    #[test]
    fn test_hash_algorithm_hex_length() {
        assert_eq!(HashAlgorithm::Sha256.hex_length(), 64);
        assert_eq!(HashAlgorithm::Sha512.hex_length(), 128);
    }
}
