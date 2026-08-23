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

//! Sous-module contenant le writer pour un shard Safetensors unique.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::PathBuf;

use crate::safetensors::header::{build_header, header_size_with_padding, pad_header};
use crate::safetensors::types::{
    DType, SafetensorsError, SafetensorsResult, Shape, ShardResult, TensorHeaderEntry,
};

/// Constantes de configuration.
pub const DEFAULT_MAX_SHARD_SIZE: usize = 5 * 1024 * 1024 * 1024; // 5 Go
pub const DEFAULT_CHUNK_SIZE: usize = 4 * 1024 * 1024; // 4 MiB

/// Writer pour un shard Safetensors unique.
///
/// Ce writer gère l'écriture d'un seul fichier .safetensors en streaming.
/// Les métadonnées des tenseurs doivent être connues à l'avance (passe A).
pub struct ShardWriter {
    writer: BufWriter<File>,
    file: File, // Pour le seek
    header: BTreeMap<String, TensorHeaderEntry>,
    buffer_pos: u64,
    header_reserve: u64,
    current_tensor: Option<CurrentTensor>,
    _finalized: bool,
    _path: PathBuf,
}

/// État du tenseur en cours d'écriture.
#[derive(Debug)]
struct CurrentTensor {
    name: String,
    dtype: DType,
    shape: Shape,
    expected_bytes: u64,
    written_bytes: u64,
}

impl ShardWriter {
    /// Crée un nouveau writer pour un shard.
    ///
    /// # Paramètres
    /// - `path` : chemin du fichier shard à écrire.
    /// - `header_reserve` : taille réservée pour le header (estimée à l'avance).
    ///
    /// # Erreurs
    /// Retourne une erreur si la création du fichier échoue.
    pub fn new(path: PathBuf, header_reserve: u64) -> SafetensorsResult<Self> {
        let file = File::create(&path).map_err(SafetensorsError::Io)?;
        let writer = BufWriter::with_capacity(
            DEFAULT_CHUNK_SIZE,
            file.try_clone().map_err(SafetensorsError::Io)?,
        );

        Ok(Self {
            writer,
            file,
            header: BTreeMap::new(),
            buffer_pos: 0,
            header_reserve,
            current_tensor: None,
            _finalized: false,
            _path: path,
        })
    }

    /// Démarre l'écriture d'un nouveau tenseur.
    ///
    /// # Paramètres
    /// - `name` : nom complet du tenseur (ex: "model.layers.0.weight").
    /// - `dtype` : type de donnée.
    /// - `shape` : forme du tenseur (slice d'entiers 64 bits non signés).
    ///
    /// # Erreurs
    /// Retourne une erreur si un tenseur est déjà en cours d'écriture
    /// ou si le nom est déjà utilisé dans ce shard.
    pub fn begin_tensor(
        &mut self,
        name: &str,
        dtype: DType,
        shape: &[u64],
    ) -> SafetensorsResult<()> {
        if self.current_tensor.is_some() {
            return Err(SafetensorsError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "un tenseur est déjà en cours d'écriture",
            )));
        }

        if self.header.contains_key(name) {
            return Err(SafetensorsError::DuplicateTensorName(name.to_string()));
        }

        // Calcule la taille attendue
        let element_count = shape.iter().product::<u64>();
        let expected_bytes = element_count
            .checked_mul(dtype.size_bytes() as u64)
            .ok_or_else(|| {
                SafetensorsError::Overflow("calcul de la taille du tenseur".to_string())
            })?;

        self.current_tensor = Some(CurrentTensor {
            name: name.to_string(),
            dtype,
            shape: shape.to_vec(),
            expected_bytes,
            written_bytes: 0,
        });

        Ok(())
    }

    /// Écrit un chunk de données pour le tenseur en cours.
    ///
    /// # Paramètres
    /// - `bytes` : données à écrire.
    ///
    /// # Erreurs
    /// Retourne une erreur si aucun tenseur n'est en cours ou si le chunk
    /// dépasse la taille attendue.
    pub fn write_chunk(&mut self, bytes: &[u8]) -> SafetensorsResult<()> {
        let tensor = self.current_tensor.as_mut().ok_or_else(|| {
            SafetensorsError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "aucun tenseur en cours d'écriture",
            ))
        })?;

        let new_written = tensor
            .written_bytes
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| SafetensorsError::Overflow("taille de données écrites".to_string()))?;

        if new_written > tensor.expected_bytes {
            return Err(SafetensorsError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "dépassement de la taille attendue pour le tenseur {}",
                    tensor.name
                ),
            )));
        }

        self.writer.write_all(bytes).map_err(SafetensorsError::Io)?;
        tensor.written_bytes = new_written;
        self.buffer_pos = self
            .buffer_pos
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| SafetensorsError::Overflow("position dans le buffer".to_string()))?;

        Ok(())
    }

    /// Termine l'écriture du tenseur en cours et vérifie la taille.
    ///
    /// # Erreurs
    /// Retourne une erreur si la taille écrite ne correspond pas à la taille attendue.
    pub fn end_tensor(&mut self) -> SafetensorsResult<()> {
        let tensor = self.current_tensor.take().ok_or_else(|| {
            SafetensorsError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "aucun tenseur en cours d'écriture",
            ))
        })?;

        if tensor.written_bytes != tensor.expected_bytes {
            return Err(SafetensorsError::PartialTensorWrite {
                name: tensor.name.clone(),
                written: tensor.written_bytes,
                expected: tensor.expected_bytes,
            });
        }

        let begin = self
            .buffer_pos
            .checked_sub(tensor.written_bytes)
            .ok_or_else(|| SafetensorsError::Overflow("calcul des offsets".to_string()))?;
        let end = self.buffer_pos;

        self.header.insert(
            tensor.name,
            TensorHeaderEntry {
                dtype: tensor.dtype,
                shape: tensor.shape,
                data_offsets: [begin, end],
            },
        );

        Ok(())
    }

    /// Finalise le shard en écrivant le header et en retournant les métadonnées.
    ///
    /// Cette méthode doit être appelée après avoir écrit tous les tenseurs.
    /// Le fichier est ensuite valide et refermable.
    pub fn finalize(mut self) -> SafetensorsResult<ShardResult> {
        if self.current_tensor.is_some() {
            return Err(SafetensorsError::ShardNotFinalized);
        }

        // Construit le header JSON
        let json = build_header(&self.header)?;
        let json_len = json.len();
        let padded_json = pad_header(&json);
        let header_size = header_size_with_padding(json_len);

        // Vérifie que le header ne dépasse pas la réserve
        if header_size > self.header_reserve {
            return Err(SafetensorsError::HeaderReserveExceeded {
                reserved: self.header_reserve,
                needed: header_size,
            });
        }

        // Se repositionne au début du fichier pour écrire le header
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(SafetensorsError::Io)?;

        // Écrit la taille du header (u64 LE)
        self.file
            .write_all(&header_size.to_le_bytes())
            .map_err(SafetensorsError::Io)?;

        // Écrit le header JSON (avec padding)
        self.file
            .write_all(padded_json.as_bytes())
            .map_err(SafetensorsError::Io)?;

        // Les données sont déjà écrites après le header, donc le fichier est valide.
        // On retourne les métadonnées du shard.
        let shard_size = header_size + self.buffer_pos;
        Ok(ShardResult {
            tensor_count: self.header.len(),
            buffer_size: self.buffer_pos,
            shard_size,
        })
    }
}
