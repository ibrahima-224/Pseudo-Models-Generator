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

//! Pipeline de génération asynchrone avec tokio.
//!
//! Génère les tenseurs indépendants en parallèle via `tokio::task::spawn_blocking`,
//! puis les écrit séquentiellement dans le fichier Safetensors.
//!
//! ## Principe
//!
//! ```text
//! Blueprint → Collecte TensorSpec (ordre stable)
//!        → Génération parallèle (N workers)
//!        → Channel mpsc → Écriture séquentielle ShardWriter
//!        → Fichier Safetensors final
//! ```
//!
//! ## Déterminisme
//!
//! Même seed = même sortie binaire, identique au mode synchron.

use std::path::PathBuf;
use std::sync::Arc;

use pmg_blueprint::tensor_spec::TensorSpec;
use pmg_blueprint::ModelBlueprint;
use pmg_io::safetensors::{DType as SafetensorsDType, ShardWriter};
use tokio::task;

use crate::error::{GeneratorError, GeneratorResult};
use crate::generation_stats::GenerationStats;
use crate::model_generator::streaming::{convert_dtype, convert_values_to_bytes};
use crate::streaming_pipeline::StreamingPipeline;

/// Taille par défaut des chunks pour l'écriture (64 Mo).
const DEFAULT_CHUNK_SIZE: usize = 64 * 1024 * 1024;

/// Callback de progression asynchrone.
pub type AsyncProgressCallback = Arc<dyn Fn(usize, usize, &str) + Send + Sync>;

/// Entrée décrivant un tenseur à générer.
#[derive(Debug, Clone)]
pub struct TensorJob {
    /// Spécification complète du tenseur.
    pub spec: TensorSpec,
    /// Catégorie (`embedding`, `attention`, `ffn`, `norm`, `lm_head`, `extra`).
    pub category: String,
    /// Index de la couche parente.
    pub layer_index: Option<usize>,
    /// Seed dérivé pour ce tenseur.
    pub seed: u64,
    /// Position ordonnée dans le blueprint.
    pub order: usize,
}

/// Résultat de la génération d'un tenseur (bytes avant écriture).
#[derive(Debug)]
pub struct TensorResult {
    /// Nom du tenseur.
    pub name: String,
    /// Données converties en bytes (little-endian).
    pub bytes: Vec<u8>,
    /// Type de données Safetensors.
    pub safetensors_dtype: SafetensorsDType,
    /// Forme du tenseur (u64).
    pub shape: Vec<u64>,
    /// Nombre d'éléments.
    pub element_count: u64,
    /// Position ordonnée pour l'écriture.
    pub order: usize,
}

/// Configuration du pipeline asynchrone.
#[derive(Debug, Clone)]
pub struct AsyncConfig {
    /// Nombre de workers parallèles (défaut : nb cœurs CPU).
    pub num_workers: usize,
    /// Taille des chunks en octets.
    pub chunk_size: usize,
    /// Seed de base.
    pub seed: u64,
    /// Chemin du fichier de sortie.
    pub output_path: PathBuf,
}

impl Default for AsyncConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus::get().max(1),
            chunk_size: DEFAULT_CHUNK_SIZE,
            seed: 42,
            output_path: PathBuf::from("model.safetensors"),
        }
    }
}

/// Pipeline asynchrone pour la génération de modèles.
pub struct AsyncPipeline {
    config: AsyncConfig,
}

impl AsyncPipeline {
    /// Crée un nouveau pipeline asynchrone.
    pub fn new(config: AsyncConfig) -> Self {
        Self { config }
    }

    /// Retourne la configuration.
    pub fn config(&self) -> &AsyncConfig {
        &self.config
    }

    /// Exécute la génération asynchrone complète.
    pub async fn generate_model(
        &self,
        blueprint: ModelBlueprint,
        progress_callback: Option<AsyncProgressCallback>,
    ) -> GeneratorResult<GenerationStats> {
        let jobs = self.collect_tensor_jobs(&blueprint);
        let total = jobs.len();
        if total == 0 {
            return Ok(GenerationStats::new());
        }

        // ShardWriter : la task d'écriture le possède directement (pas de Mutex)
        let header_reserve = crate::model_generator::streaming::estimate_header_size(total);
        let writer =
            ShardWriter::new(self.config.output_path.clone(), header_reserve).map_err(|e| {
                GeneratorError::Internal(format!(
                    "Erreur création fichier {}: {}",
                    self.config.output_path.display(),
                    e
                ))
            })?;

        // Channel pour les résultats générés
        let (tx, mut rx) = tokio::sync::mpsc::channel::<TensorResult>(total.min(16));

        // Task d'écriture séquentielle (possède le ShardWriter)
        let write_handle = task::spawn(async move {
            let mut stats = GenerationStats::new();
            let mut w = writer;
            let mut pending: Vec<TensorResult> = Vec::new();
            let mut expected_order = 0usize;

            // Recevoir tous les résultats puis les trier par ordre canonique
            while let Some(r) = rx.recv().await {
                pending.push(r);
            }

            // Trier par ordre pour garantir le déterminisme
            pending.sort_by_key(|r| r.order);

            // Écrire dans l'ordre canonique
            for r in pending {
                assert_eq!(
                    r.order, expected_order,
                    "Ordre inattendu: {} != {}",
                    r.order, expected_order
                );
                expected_order += 1;

                w.begin_tensor(&r.name, r.safetensors_dtype, &r.shape)
                    .map_err(|e| GeneratorError::Internal(format!("begin {}: {}", r.name, e)))?;
                for chunk in r.bytes.chunks(DEFAULT_CHUNK_SIZE) {
                    w.write_chunk(chunk).map_err(|e| {
                        GeneratorError::Internal(format!("write {}: {}", r.name, e))
                    })?;
                }
                w.end_tensor()
                    .map_err(|e| GeneratorError::Internal(format!("end {}: {}", r.name, e)))?;
                stats.tensor_count += 1;
                stats.parameter_count += r.element_count;
            }

            // Finaliser le shard (écrit l'en-tête et renomme)
            w.finalize()
                .map_err(|e| GeneratorError::Internal(format!("finalize: {}", e)))?;

            Ok::<GenerationStats, GeneratorError>(stats)
        });

        // Génération parallèle
        let nw = self.config.num_workers;
        let mut handles = Vec::with_capacity(total);

        for job in jobs {
            let tx = tx.clone();
            let cs = self.config.chunk_size;
            let cb = progress_callback.clone();

            let h = task::spawn_blocking(move || {
                if let Some(ref f) = cb {
                    f(job.order + 1, total, &job.spec.name);
                }
                let pl = StreamingPipeline::new();
                let r = generate_tensor_data(&job, &pl, cs)?;
                tx.blocking_send(r).map_err(|e| {
                    GeneratorError::Internal(format!("send {}: {}", job.spec.name, e))
                })?;
                Ok::<(), GeneratorError>(())
            });
            handles.push(h);

            if handles.len() >= nw {
                if let Some(h) = handles.first_mut() {
                    match h.await {
                        Ok(Ok(())) => {},
                        Ok(Err(e)) => return Err(e),
                        Err(e) => return Err(GeneratorError::Internal(e.to_string())),
                    }
                }
                handles.remove(0);
            }
        }

        for h in handles {
            match h.await {
                Ok(Ok(())) => {},
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(GeneratorError::Internal(e.to_string())),
            }
        }

        drop(tx);
        match write_handle.await {
            Ok(r) => r,
            Err(e) => Err(GeneratorError::Internal(e.to_string())),
        }
    }

    /// Collecte les jobs dans l'ordre canonique du blueprint.
    fn collect_tensor_jobs(&self, bp: &ModelBlueprint) -> Vec<TensorJob> {
        let mut jobs = Vec::new();
        let mut o = 0;
        for s in &bp.embeddings {
            jobs.push(self.job(s, "embedding", None, o));
            o += 1;
        }
        for (li, l) in bp.layers.iter().enumerate() {
            for s in &l.attention {
                jobs.push(self.job(s, "attention", Some(li), o));
                o += 1;
            }
            for s in &l.mlp {
                jobs.push(self.job(s, "mlp", Some(li), o));
                o += 1;
            }
            for s in &l.norms {
                jobs.push(self.job(s, "norm", Some(li), o));
                o += 1;
            }
        }
        for s in &bp.final_norm {
            jobs.push(self.job(s, "final_norm", None, o));
            o += 1;
        }
        for s in &bp.lm_head {
            jobs.push(self.job(s, "lm_head", None, o));
            o += 1;
        }
        for s in &bp.extra_tensors {
            jobs.push(self.job(s, "extra", None, o));
            o += 1;
        }
        jobs
    }

    fn job(&self, spec: &TensorSpec, cat: &str, li: Option<usize>, order: usize) -> TensorJob {
        TensorJob {
            spec: spec.clone(),
            category: cat.to_string(),
            layer_index: li,
            seed: self.config.seed.wrapping_add(order as u64),
            order,
        }
    }
}

/// Génère les données bytes d'un tenseur via le pipeline streaming.
fn generate_tensor_data(
    job: &TensorJob,
    pipeline: &StreamingPipeline,
    chunk_size: usize,
) -> GeneratorResult<TensorResult> {
    let dtype = convert_dtype(job.spec.dtype)?;
    let shape: Vec<u64> = job.spec.shape.dims().to_vec();
    let total: usize = shape.iter().map(|&x| x as usize).product();
    let bpe = job.spec.dtype.size_bytes().unwrap_or(4) as usize;
    let cse = chunk_size / bpe.max(1);
    let mut all = Vec::with_capacity(total * bpe);
    let mut done = 0;

    while done < total {
        let cur = cse.min(total - done);
        let _ = pipeline.execute_chunk(&job.spec, done, cur, job.seed)?;
        let vals = generate_deterministic_values(cur, job.seed.wrapping_add(done as u64));
        all.extend_from_slice(&convert_values_to_bytes(&vals, job.spec.dtype)?);
        done += cur;
    }

    Ok(TensorResult {
        name: job.spec.name.clone(),
        bytes: all,
        safetensors_dtype: dtype,
        shape,
        element_count: total as u64,
        order: job.order,
    })
}

/// Génère des valeurs déterministes (xorshift64 + Box-Muller, normale N(0,1)).
///
/// PRNG interne reproductible basé sur un mélange xorshift64. La distribution
/// cible est normale standard (μ=0, σ=1) via la transformée de Box-Muller.
pub fn generate_deterministic_values(count: usize, seed: u64) -> Vec<f64> {
    let mut out = Vec::with_capacity(count);
    let mut s = seed;
    for i in 0..count {
        // Mélange xorshift64
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        // Premier uniform via les 32 bits de poids faible
        let u1 = ((s & 0xFFFF_FFFF) as f64) / (u32::MAX as f64);
        // Deuxième uniform via un seed dérivé (wrapping pour éviter l'overflow)
        let s2 = s.wrapping_add((i as u64).wrapping_mul(6364136223846793005));
        let u2 = ((s2 & 0xFFFF_FFFF) as f64) / (u32::MAX as f64);
        // Box-Muller : conversion uniforms → normale N(0,1)
        let u1c = u1.max(1e-10); // éviter log(0)
        out.push((-2.0 * u1c.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos());
    }
    out
}

/// Point d'entrée asynchrone pour la génération.
pub async fn generate_model_async(
    config: &AsyncConfig,
    blueprint: ModelBlueprint,
    progress_callback: Option<AsyncProgressCallback>,
) -> GeneratorResult<GenerationStats> {
    AsyncPipeline::new(config.clone())
        .generate_model(blueprint, progress_callback)
        .await
}
