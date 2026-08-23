//! Logique de distribution pour la commande generate
//!
//! Ce module contient la logique de distribution pour la génération
//! sur plusieurs machines, incluant le mode coordinateur et worker.

use anyhow::Result;
use pmg_blueprint::ModelBlueprint;
use pmg_generator::{
    DistributedConfig, DistributedCoordinator, DistributedTask, DistributedWorker,
};

use crate::output;

/// Exécute la génération en mode distribué
///
/// # Arguments
///
/// * `blueprint` - Le blueprint du modèle à générer
/// * `coordinator_addr` - Adresse du coordinateur
/// * `num_workers` - Nombre de workers
/// * `worker_mode` - Si true, exécute en mode worker
/// * `worker_id` - Identifiant du worker (optionnel)
/// * `verbose` - Mode verbeux
/// * `seed` - Graine pour la génération
///
/// # Retour
///
/// Retourne Ok(()) si l'exécution est réussie.
pub fn execute_distributed(
    blueprint: &ModelBlueprint,
    coordinator_addr: &str,
    num_workers: usize,
    worker_mode: bool,
    worker_id: Option<String>,
    verbose: bool,
    seed: u64,
) -> Result<()> {
    if verbose {
        output::info("Mode distribué activé : génération sur plusieurs machines");
    }

    let config = DistributedConfig {
        coordinator_addr: coordinator_addr.to_string(),
        num_workers,
        worker_capacity: 8,
        task_timeout: 300,
        max_retries: 3,
    };

    if worker_mode {
        // Mode worker
        let worker_id = worker_id.unwrap_or_else(|| format!("worker-{}", std::process::id()));

        if verbose {
            output::info(&format!("Démarrage du worker: {}", worker_id));
        }

        // Créer et démarrer le worker
        let worker = DistributedWorker::new(
            worker_id,
            config.coordinator_addr.clone(),
            config.worker_capacity,
        );

        // Exécuter le worker (bloquant)
        tokio::runtime::Runtime::new()?.block_on(worker.start())?;

        // Retourner un résultat vide (le worker ne génère pas de modèle local)
        Ok(())
    } else {
        // Mode coordinateur
        let coordinator = DistributedCoordinator::new();

        // Créer les tâches à distribuer
        let mut tasks = Vec::new();
        let mut task_id = 0;

        // Ajouter les embeddings
        for spec in &blueprint.embeddings {
            tasks.push(DistributedTask {
                id: format!("task-{}", task_id),
                tensor_spec: spec.clone(),
                tensor_type: "embedding".to_string(),
                layer_index: None,
                seed: seed + task_id as u64,
                priority: 0,
            });
            task_id += 1;
        }

        // Ajouter les tenseurs des couches
        for (layer_idx, layer) in blueprint.layers.iter().enumerate() {
            for spec in &layer.attention {
                tasks.push(DistributedTask {
                    id: format!("task-{}", task_id),
                    tensor_spec: spec.clone(),
                    tensor_type: "attention".to_string(),
                    layer_index: Some(layer_idx),
                    seed: seed + task_id as u64,
                    priority: 1,
                });
                task_id += 1;
            }

            for spec in &layer.mlp {
                tasks.push(DistributedTask {
                    id: format!("task-{}", task_id),
                    tensor_spec: spec.clone(),
                    tensor_type: "mlp".to_string(),
                    layer_index: Some(layer_idx),
                    seed: seed + task_id as u64,
                    priority: 2,
                });
                task_id += 1;
            }
        }

        // Ajouter les tâches au coordinateur
        coordinator.add_tasks(tasks);

        if verbose {
            output::info(&format!("Nombre de tâches à distribuer: {}", task_id));
        }

        // Démarrer le serveur du coordinateur
        tokio::runtime::Runtime::new()?
            .block_on(coordinator.start_server(&config.coordinator_addr))?;

        // Retourner un résultat vide (le coordinateur ne génère pas de modèle local)
        Ok(())
    }
}
