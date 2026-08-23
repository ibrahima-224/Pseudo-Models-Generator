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

//! Génération de l'index model.safetensors.index.json.

use std::fs::File;
use std::io::Write;
use std::path::Path;

use super::types::{SafetensorsError, SafetensorsIndex, SafetensorsResult};

/// Écrit l'index Safetensors dans un fichier.
///
/// # Paramètres
/// - `index` : l'index à sérialiser.
/// - `path` : chemin du fichier model.safetensors.index.json.
///
/// # Comportement
/// Écrit le JSON en format compact avec une newline finale.
pub fn write_index(index: &SafetensorsIndex, path: &Path) -> SafetensorsResult<()> {
    let json = serde_json::to_string(index).map_err(SafetensorsError::Json)?;
    let mut file = File::create(path).map_err(SafetensorsError::Io)?;
    file.write_all(json.as_bytes())
        .map_err(SafetensorsError::Io)?;
    file.write_all(b"\n").map_err(SafetensorsError::Io)?;
    file.flush().map_err(SafetensorsError::Io)?;
    Ok(())
}

/// Construit un nom de shard à partir de son index et du nombre total.
///
/// # Paramètres
/// - `shard_index` : index du shard (1-indexed).
/// - `total_shards` : nombre total de shards.
///
/// # Retour
/// Le nom du shard au format "model-XXXXX-of-YYYYY.safetensors".
pub fn shard_name(shard_index: usize, total_shards: usize) -> String {
    format!(
        "model-{:05}-of-{:05}.safetensors",
        shard_index, total_shards
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use tempfile::tempdir;

    #[test]
    fn test_write_index_basic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.safetensors.index.json");

        let mut weight_map = BTreeMap::new();
        weight_map.insert(
            "model.layer.weight".to_string(),
            "model-00001-of-00002.safetensors".to_string(),
        );
        weight_map.insert(
            "model.layer.bias".to_string(),
            "model-00002-of-00002.safetensors".to_string(),
        );

        let index = SafetensorsIndex {
            metadata: super::super::types::IndexMetadata { total_size: 1024 },
            weight_map,
        };

        write_index(&index, &path).unwrap();

        // Vérifie que le fichier existe et est valide
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("\"total_size\":1024"));
        assert!(content.contains("\"model.layer.weight\""));

        // Vérifie que le JSON est valide
        let parsed: SafetensorsIndex = serde_json::from_str(&content).unwrap();
        assert_eq!(parsed.metadata.total_size, 1024);
    }

    #[test]
    fn test_shard_name() {
        assert_eq!(shard_name(1, 3), "model-00001-of-00003.safetensors");
        assert_eq!(shard_name(12, 100), "model-00012-of-00100.safetensors");
        assert_eq!(shard_name(1, 1), "model-00001-of-00001.safetensors");
    }
}
