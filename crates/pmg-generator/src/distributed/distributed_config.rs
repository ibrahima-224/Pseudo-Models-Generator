//! Configuration du système distribué.
//!
//! Ce module contient la configuration pour l'architecture distribuée
//! de génération de modèles, incluant les paramètres de connexion,
//! de capacité et de tolérance aux pannes.

use serde::{Deserialize, Serialize};

/// Configuration distribuée
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Adresse du coordinateur
    pub coordinator_addr: String,
    /// Nombre de workers
    pub num_workers: usize,
    /// Capacité par worker
    pub worker_capacity: usize,
    /// Timeout des tâches en secondes
    pub task_timeout: u64,
    /// Nombre maximum de tentatives
    pub max_retries: usize,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            coordinator_addr: "127.0.0.1:9090".to_string(),
            num_workers: 4,
            worker_capacity: 8,
            task_timeout: 300,
            max_retries: 3,
        }
    }
}

/// Tâche à distribuer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedTask {
    /// Identifiant unique de la tâche
    pub id: String,
    /// Spécification du tenseur
    pub tensor_spec: pmg_blueprint::tensor_spec::TensorSpec,
    /// Type de tenseur (embedding, attention, ffn, etc.)
    pub tensor_type: String,
    /// Index de couche (le cas échéant)
    pub layer_index: Option<usize>,
    /// Graine pour le déterminisme
    pub seed: u64,
    /// Priorité (0 = haute, 10 = basse)
    pub priority: u8,
}

/// Résultat d'une tâche distribuée
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedResult {
    /// Identifiant de la tâche
    pub task_id: String,
    /// Statut de la tâche
    pub status: TaskStatus,
    /// Données générées (Optionnel - peut être vide en streaming)
    pub data: Option<Vec<u8>>,
    /// Chemin du fichier généré (si écriture locale)
    pub file_path: Option<String>,
    /// Statistiques de génération
    pub stats: TaskStats,
    /// Message d'erreur (le cas échéant)
    pub error: Option<String>,
}

/// Statut d'une tâche
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Retry,
}

/// Statistiques d'une tâche
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TaskStats {
    /// Durée de génération en secondes
    pub duration_secs: f64,
    /// Taille des données en octets
    pub bytes_generated: u64,
    /// Nœud ayant exécuté la tâche
    pub worker_id: String,
}

/// Information sur un worker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub id: String,
    pub address: String,
    pub capacity: usize,
    pub active_tasks: usize,
    pub last_seen: u64, // Timestamp en secondes depuis l'époque
}

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Statistiques globales
#[derive(Debug, Clone, Default)]
pub struct GlobalStats {
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub failed_tasks: usize,
    pub total_bytes: u64,
    pub start_time: Option<std::time::Instant>,
}

/// Coordinateur distribué
pub struct DistributedCoordinator {
    /// Tâches en attente
    pub(crate) pending_tasks: Arc<Mutex<Vec<DistributedTask>>>,
    /// Workers enregistrés
    pub(crate) workers: Arc<Mutex<HashMap<String, WorkerInfo>>>,
    /// Résultats reçus
    pub(crate) results: Arc<Mutex<HashMap<String, DistributedResult>>>,
    /// Statistiques globales
    pub(crate) stats: Arc<Mutex<GlobalStats>>,
}

impl DistributedCoordinator {
    /// Crée un nouveau coordinateur
    pub fn new() -> Self {
        Self {
            pending_tasks: Arc::new(Mutex::new(Vec::new())),
            workers: Arc::new(Mutex::new(HashMap::new())),
            results: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(GlobalStats::default())),
        }
    }
}

impl Default for DistributedCoordinator {
    fn default() -> Self {
        Self::new()
    }
}
