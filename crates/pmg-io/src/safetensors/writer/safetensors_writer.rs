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

//! Sous-module contenant le writer Safetensors avec support du sharding automatique.

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::safetensors::header::estimate_header_reserve;
use crate::safetensors::types::{
    DType, IndexMetadata, SafetensorsError, SafetensorsIndex, SafetensorsResult, TensorInfo,
};

use super::shard::{ShardWriter, DEFAULT_CHUNK_SIZE};

/// Writer Safetensors avec support du sharding automatique.
///
/// Ce writer gère la création de plusieurs shards et de l'index global.
pub struct SafetensorsWriter {
    output_dir: PathBuf,
    max_shard_size: usize,
    current_shard: usize,
    current_shard_size: u64,
    current_writer: Option<ShardWriter>,
    tensors: Vec<TensorInfo>,
    shard_names: Vec<String>,
}

impl SafetensorsWriter {
    /// Crée un nouveau writer Safetensors.
    ///
    /// # Paramètres
    /// - `output_dir` : répertoire de sortie pour les fichiers.
    /// - `max_shard_size` : taille maximale par shard en octets (défaut: 5 Go).
    pub fn new(output_dir: PathBuf, max_shard_size: usize) -> Self {
        Self {
            output_dir,
            max_shard_size,
            current_shard: 0,
            current_shard_size: 0,
            current_writer: None,
            tensors: Vec::new(),
            shard_names: Vec::new(),
        }
    }

    /// Écrit un tenseur en streaming.
    ///
    /// # Paramètres
    /// - `name` : nom complet du tenseur.
    /// - `data` : données binaires du tenseur.
    /// - `dtype` : type de donnée.
    /// - `shape` : forme du tenseur.
    ///
    /// # Comportement
    /// - Gère automatiquement le sharding si le shard courant dépasse la taille maximale.
    /// - Ne charge jamais le tenseur complet en mémoire (streaming).
    pub fn write_tensor(
        &mut self,
        name: &str,
        data: &[u8],
        dtype: DType,
        shape: &[u64],
    ) -> SafetensorsResult<()> {
        let element_count = shape.iter().product::<u64>();
        let tensor_size = element_count
            .checked_mul(dtype.size_bytes() as u64)
            .ok_or_else(|| {
                SafetensorsError::Overflow("calcul de la taille du tenseur".to_string())
            })?;

        // Vérifie que le tenseur peut tenir dans un shard unique
        if tensor_size > self.max_shard_size as u64 {
            return Err(SafetensorsError::TensorTooLargeForShard { size: tensor_size });
        }

        // Vérifie si on doit créer un nouveau shard
        if self.current_writer.is_none()
            || self.current_shard_size + tensor_size > self.max_shard_size as u64
        {
            self.start_new_shard()?;
        }

        let writer = self.current_writer.as_mut().unwrap();

        // Écrit le tenseur en streaming
        writer.begin_tensor(name, dtype, shape)?;

        // Écrit les données par chunks
        let mut offset = 0;
        while offset < data.len() {
            let end = std::cmp::min(offset + DEFAULT_CHUNK_SIZE, data.len());
            writer.write_chunk(&data[offset..end])?;
            offset = end;
        }

        writer.end_tensor()?;

        // Enregistre les métadonnées pour l'index
        self.tensors.push(TensorInfo {
            name: name.to_string(),
            shard_index: self.shard_names.len() - 1,
            dtype,
            shape: shape.to_vec(),
            data_offsets: [
                self.current_shard_size,
                self.current_shard_size + tensor_size,
            ],
            generated_bytes: tensor_size,
        });

        self.current_shard_size += tensor_size;

        Ok(())
    }

    /// Démarre un nouveau shard.
    fn start_new_shard(&mut self) -> SafetensorsResult<()> {
        // Finalise le shard précédent s'il existe
        if let Some(writer) = self.current_writer.take() {
            let _result = writer.finalize()?;
            // Met à jour les tailles si nécessaire
        }

        self.current_shard += 1;
        self.current_shard_size = 0;

        // Génère le nom du shard
        let shard_name = format!(
            "model-{:05}-of-{:05}.safetensors",
            self.current_shard,
            self.estimate_total_shards()
        );

        let path = self.output_dir.join(&shard_name);

        // Estime la réserve pour le header (basé sur le nombre de tenseurs déjà écrits)
        let header_reserve = estimate_header_reserve(&[]);

        let writer = ShardWriter::new(path, header_reserve)?;
        self.current_writer = Some(writer);
        self.shard_names.push(shard_name);

        Ok(())
    }

    /// Estime le nombre total de shards (pour le nommage).
    fn estimate_total_shards(&self) -> usize {
        // Estimation simple : au moins le shard courant
        // Sera corrigé à la finalisation
        std::cmp::max(self.current_shard, 1)
    }

    /// Finalise tous les shards et génère l'index.
    ///
    /// # Retour
    /// L'index Safetensors à écrire dans model.safetensors.index.json.
    pub fn finish(mut self) -> SafetensorsResult<SafetensorsIndex> {
        // Finalise le dernier shard
        if let Some(writer) = self.current_writer.take() {
            let _result = writer.finalize()?;
        }

        // Met à jour les noms de shards avec le nombre réel
        let total_shards = self.shard_names.len();
        if total_shards > 1 {
            // Renomme les shards avec le nombre correct
            for (i, shard_name) in self.shard_names.iter_mut().enumerate() {
                let new_name = format!("model-{:05}-of-{:05}.safetensors", i + 1, total_shards);

                // Renomme le fichier
                let old_path = self.output_dir.join(&shard_name);
                let new_path = self.output_dir.join(&new_name);

                if old_path.exists() {
                    std::fs::rename(&old_path, &new_path)?;
                }

                *shard_name = new_name;
            }
        }

        // Construit l'index
        let mut weight_map = BTreeMap::new();
        let mut total_size = 0u64;

        for tensor_info in &self.tensors {
            let shard_name = &self.shard_names[tensor_info.shard_index];
            weight_map.insert(tensor_info.name.clone(), shard_name.clone());
            total_size += tensor_info.generated_bytes;
        }

        Ok(SafetensorsIndex {
            metadata: IndexMetadata { total_size },
            weight_map,
        })
    }
}
