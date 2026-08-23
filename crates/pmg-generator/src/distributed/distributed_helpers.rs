//! Fonctions utilitaires pour le système distribué.
//!
//! Ce module contient les méthodes auxiliaires pour le coordinateur
//! distribué, incluant la gestion des tâches, des workers et des statistiques.

use std::sync::Arc;
use std::time::Instant;

use super::distributed_config::{
    DistributedCoordinator, DistributedResult, DistributedTask, GlobalStats, TaskStatus, WorkerInfo,
};
use crate::distributed::DistributedError;

/// Méthodes auxiliaires pour le coordinateur distribué.
impl DistributedCoordinator {
    /// Ajoute des tâches à distribuer.
    pub fn add_tasks(&self, tasks: Vec<DistributedTask>) {
        let mut pending = self.pending_tasks.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        stats.total_tasks += tasks.len();
        if stats.start_time.is_none() {
            stats.start_time = Some(Instant::now());
        }

        pending.extend(tasks);

        // Trier par priorité (0 = haute)
        pending.sort_by_key(|t| t.priority);
    }

    /// Récupère une tâche pour un worker spécifique.
    pub fn get_task_for_worker(&self, worker_id: &str) -> Option<DistributedTask> {
        let mut pending = self.pending_tasks.lock().unwrap();
        let mut workers = self.workers.lock().unwrap();

        // Vérifier que le worker existe et a de la capacité
        if let Some(worker) = workers.get_mut(worker_id) {
            if worker.active_tasks < worker.capacity {
                // Retourner la tâche la plus prioritaire
                if let Some(task) = pending.pop() {
                    worker.active_tasks += 1;
                    return Some(task);
                }
            }
        }
        None
    }

    /// Met à jour le statut d'un worker.
    pub fn update_worker_status(&self, worker_id: &str, active_tasks: usize) {
        let mut workers = self.workers.lock().unwrap();
        if let Some(worker) = workers.get_mut(worker_id) {
            worker.active_tasks = active_tasks;
            worker.last_seen = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
        }
    }

    /// Récupère les résultats.
    pub fn get_results(&self) -> Vec<DistributedResult> {
        let results = self.results.lock().unwrap();
        results.values().cloned().collect()
    }

    /// Récupère les statistiques.
    pub fn get_stats(&self) -> GlobalStats {
        self.stats.lock().unwrap().clone()
    }

    /// Démarre le serveur HTTP
    pub async fn start_server(&self, addr: &str) -> Result<(), DistributedError> {
        use warp::Filter;

        let workers = self.workers.clone();
        let pending = self.pending_tasks.clone();
        let results = self.results.clone();
        let stats = self.stats.clone();

        // Route: Register worker
        let register = warp::post()
            .and(warp::path("register"))
            .and(warp::body::json())
            .map({
                // Cloner l'Arc pour le capturer dans la closure
                let workers = Arc::clone(&workers);
                move |worker: WorkerInfo| {
                    let mut workers = workers.lock().unwrap();
                    workers.insert(worker.id.clone(), worker);
                    warp::reply::json(&serde_json::json!({"status": "ok"}))
                }
            });

        // Route: Get task for worker
        let get_task = warp::get()
            .and(warp::path("task"))
            .and(warp::path::param::<String>())
            .map({
                // Cloner les Arc pour les capturer dans la closure
                let pending = Arc::clone(&pending);
                let workers = Arc::clone(&workers);
                move |_worker_id: String| {
                    let mut pending = pending.lock().unwrap();
                    let _workers = workers.lock().unwrap();

                    // Trouver une tâche disponible
                    if let Some(task) = pending.pop() {
                        warp::reply::with_status(
                            warp::reply::json(&task),
                            warp::http::StatusCode::OK,
                        )
                    } else {
                        // Retourner une réponse JSON vide avec statut 404
                        warp::reply::with_status(
                            warp::reply::json(&serde_json::json!({"error": "no tasks"})),
                            warp::http::StatusCode::NOT_FOUND,
                        )
                    }
                }
            });

        // Route: Submit result
        let submit_result = warp::post()
            .and(warp::path("result"))
            .and(warp::body::json())
            .map({
                // Cloner les Arc pour les capturer dans la closure
                let results = Arc::clone(&results);
                let stats = Arc::clone(&stats);
                move |result: DistributedResult| {
                    let mut results = results.lock().unwrap();
                    let mut stats = stats.lock().unwrap();

                    stats.completed_tasks += 1;
                    if result.status == TaskStatus::Completed {
                        stats.total_bytes += result.stats.bytes_generated;
                    }

                    results.insert(result.task_id.clone(), result);
                    warp::reply::json(&serde_json::json!({"status": "ok"}))
                }
            });

        let routes = register.or(get_task).or(submit_result);

        // Parser l'adresse SocketAddr
        let socket_addr: std::net::SocketAddr = addr
            .parse()
            .map_err(|e: std::net::AddrParseError| DistributedError::AddrParse(e.to_string()))?;

        warp::serve(routes).run(socket_addr).await;

        Ok(())
    }
}
