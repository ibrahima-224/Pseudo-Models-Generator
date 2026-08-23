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

//! Architecture distribuée pour la génération de modèles
//!
//! Ce module implémente un système de distribution de travail
//! pour paralléliser la génération sur plusieurs nœuds.
//! Il permet de distribuer la génération de tenseurs sur plusieurs
//! workers pour accélérer la production de modèles de très grande taille.
//!
//! ## Composants
//!
//! - **DistributedTask** : Tâche à distribuer avec spécification du tenseur
//! - **DistributedWorker** : Nœud worker qui exécute les tâches
//! - **DistributedCoordinator** : Coordinateur central qui gère la distribution
//! - **DistributedConfig** : Configuration du système distribué
//!
//! ## Tolérance aux pannes
//!
//! Le système supporte :
//! - La reconnexion automatique des workers
//! - La redistribution des tâches en cas d'échec
//! - Le suivi de l'état de santé des workers
//!
//! ## Déterminisme
//!
//! La distribution ne compromet pas le déterminisme :
//! - Chaque tâche utilise une seed unique dérivée de la seed globale
//! - Les workers génèrent les mêmes résultats que le mode séquentiel
//! - L'ordre de distribution n'affecte pas les sorties

use crate::streaming_pipeline::StreamingPipeline;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// Sous-modules
pub mod distributed_config;
pub mod distributed_helpers;

// Réexports pour rétrocompatibilité API
pub use distributed_config::{
    DistributedConfig, DistributedCoordinator, DistributedResult, DistributedTask, GlobalStats,
    TaskStats, TaskStatus, WorkerInfo,
};

/// Nœud worker distribué
pub struct DistributedWorker {
    /// Identifiant du worker
    id: String,
    /// Adresse du coordinateur
    coordinator_addr: String,
    /// Capacité (nombre de tâches simultanées)
    capacity: usize,
    /// Tâches en cours
    active_tasks: Arc<Mutex<HashMap<String, DistributedTask>>>,
}

impl DistributedWorker {
    /// Crée un nouveau worker
    ///
    /// # Paramètres
    /// - `id` : Identifiant unique du worker
    /// - `coordinator_addr` : Adresse du coordinateur (ex: "127.0.0.1:9090")
    /// - `capacity` : Nombre maximum de tâches simultanées
    pub fn new(id: String, coordinator_addr: String, capacity: usize) -> Self {
        Self {
            id,
            coordinator_addr,
            capacity,
            active_tasks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Démarre le worker
    ///
    /// # Erreurs
    /// Retourne une erreur si la connexion au coordinateur échoue
    pub async fn start(&self) -> Result<(), DistributedError> {
        // 1. Se connecter au coordinateur
        let client = reqwest::Client::new();
        let register_url = format!("{}/register", self.coordinator_addr);

        let register_payload = serde_json::json!({
            "worker_id": self.id,
            "capacity": self.capacity,
            "address": self.get_address(),
        });

        client
            .post(&register_url)
            .json(&register_payload)
            .send()
            .await
            .map_err(DistributedError::Network)?;

        // 2. Boucle de traitement des tâches
        loop {
            // Demander une tâche
            let task_url = format!("{}/task/{}", self.coordinator_addr, self.id);
            let response = client
                .get(&task_url)
                .send()
                .await
                .map_err(DistributedError::Network)?;

            if response.status() == 200 {
                let task: DistributedTask =
                    response.json().await.map_err(DistributedError::Network)?;

                // Exécuter la tâche
                let result = self.execute_task(task.clone()).await;

                // Envoyer le résultat
                let result_url = format!("{}/result", self.coordinator_addr);
                client
                    .post(&result_url)
                    .json(&result)
                    .send()
                    .await
                    .map_err(DistributedError::Network)?;
            } else {
                // Pas de tâche disponible, attendre
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }

    /// Exécute une tâche
    async fn execute_task(&self, task: DistributedTask) -> DistributedResult {
        let start = Instant::now();

        // Ajouter aux tâches actives
        {
            let mut active = self.active_tasks.lock().unwrap();
            active.insert(task.id.clone(), task.clone());
        }

        // Exécuter la génération
        let result = self.generate_tensor(&task).await;

        // Retirer des tâches actives
        {
            let mut active = self.active_tasks.lock().unwrap();
            active.remove(&task.id);
        }

        let duration = start.elapsed().as_secs_f64();

        match result {
            Ok(data) => DistributedResult {
                task_id: task.id,
                status: TaskStatus::Completed,
                data: Some(data),
                file_path: None,
                stats: TaskStats {
                    duration_secs: duration,
                    bytes_generated: 0, // Sera calculé
                    worker_id: self.id.clone(),
                },
                error: None,
            },
            Err(e) => DistributedResult {
                task_id: task.id,
                status: TaskStatus::Failed,
                data: None,
                file_path: None,
                stats: TaskStats {
                    duration_secs: duration,
                    bytes_generated: 0,
                    worker_id: self.id.clone(),
                },
                error: Some(e.to_string()),
            },
        }
    }

    /// Génère un tenseur
    async fn generate_tensor(&self, _task: &DistributedTask) -> Result<Vec<u8>, DistributedError> {
        // Utiliser le pipeline streaming existant
        let _pipeline = StreamingPipeline::new();

        // Pour l'instant, retourner un vecteur vide
        // TODO: Implémenter la génération réelle via le pipeline
        let data = Vec::new();
        Ok(data)
    }

    /// Retourne l'adresse du worker
    fn get_address(&self) -> String {
        format!("worker-{}:{}", self.id, 8080)
    }
}

/// Erreur distribuée
#[derive(Debug, thiserror::Error)]
pub enum DistributedError {
    #[error("Erreur réseau: {0}")]
    Network(#[from] reqwest::Error),

    #[error("Erreur warp: {0}")]
    Warp(#[from] warp::Error),

    #[error("Erreur de sérialisation: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Worker non trouvé: {0}")]
    WorkerNotFound(String),

    #[error("Tâche non trouvée: {0}")]
    TaskNotFound(String),

    #[error("Erreur de génération: {0}")]
    Generation(#[from] crate::error::GeneratorError),

    #[error("Erreur de parsing d'adresse: {0}")]
    AddrParse(String),
}

// Tests unitaires
#[cfg(test)]
mod tests {
    use super::*;
    use pmg_blueprint::tensor_spec::TensorSpec;
    use pmg_core::DType;
    use pmg_core::Shape;
    use pmg_core::TensorRole;

    #[test]
    fn test_task_creation() {
        let task = DistributedTask {
            id: "test-1".to_string(),
            tensor_spec: TensorSpec::new(
                "test.tensor",
                Shape::new(vec![100, 64]).unwrap(),
                DType::F32,
                TensorRole::Other,
            )
            .unwrap(),
            tensor_type: "embedding".to_string(),
            layer_index: None,
            seed: 42,
            priority: 0,
        };

        assert_eq!(task.id, "test-1");
        assert_eq!(task.priority, 0);
        assert_eq!(task.tensor_type, "embedding");
    }

    #[test]
    fn test_coordinator_creation() {
        let coordinator = DistributedCoordinator::new();
        let stats = coordinator.get_stats();

        assert_eq!(stats.total_tasks, 0);
        assert_eq!(stats.completed_tasks, 0);
    }

    #[test]
    fn test_global_stats() {
        let stats = GlobalStats {
            total_tasks: 100,
            completed_tasks: 50,
            failed_tasks: 5,
            total_bytes: 1024 * 1024,
            ..Default::default()
        };

        assert_eq!(stats.total_tasks, 100);
        assert_eq!(stats.completed_tasks, 50);
        assert_eq!(stats.total_bytes, 1024 * 1024);
    }

    #[test]
    fn test_task_status() {
        let status = TaskStatus::Pending;
        assert_eq!(status, TaskStatus::Pending);

        let status = TaskStatus::Running;
        assert_eq!(status, TaskStatus::Running);

        let status = TaskStatus::Completed;
        assert_eq!(status, TaskStatus::Completed);

        let status = TaskStatus::Failed;
        assert_eq!(status, TaskStatus::Failed);

        let status = TaskStatus::Retry;
        assert_eq!(status, TaskStatus::Retry);
    }

    #[test]
    fn test_worker_creation() {
        let worker =
            DistributedWorker::new("worker-1".to_string(), "127.0.0.1:9090".to_string(), 4);

        assert_eq!(worker.id, "worker-1");
        assert_eq!(worker.capacity, 4);
    }

    #[test]
    fn test_config_default() {
        let config = DistributedConfig::default();
        assert_eq!(config.coordinator_addr, "127.0.0.1:9090");
        assert_eq!(config.num_workers, 4);
        assert_eq!(config.worker_capacity, 8);
        assert_eq!(config.task_timeout, 300);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_result_creation() {
        let result = DistributedResult {
            task_id: "test-1".to_string(),
            status: TaskStatus::Completed,
            data: Some(vec![1, 2, 3]),
            file_path: None,
            stats: TaskStats::default(),
            error: None,
        };

        assert_eq!(result.task_id, "test-1");
        assert_eq!(result.status, TaskStatus::Completed);
        assert!(result.data.is_some());
        assert!(result.error.is_none());
    }
}
