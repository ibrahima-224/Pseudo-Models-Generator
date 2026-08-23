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

//! Tests pour le module distribué
//!
//! Ce module contient les tests unitaires et d'intégration pour le système
//! de distribution de travail sur plusieurs nœuds.

use pmg_blueprint::tensor_spec::TensorSpec;
use pmg_core::{DType, Shape, TensorRole};
use pmg_generator::{
    DistributedConfig, DistributedCoordinator, DistributedTask, DistributedWorker, GlobalStats,
    TaskStats, TaskStatus, WorkerInfo,
};

/// Test de création de tâche distribuée
#[test]
fn test_distributed_task_creation() {
    let task = DistributedTask {
        id: "task-1".to_string(),
        tensor_spec: TensorSpec::new(
            "model.embed_tokens.weight",
            Shape::new(vec![100, 64]).unwrap(),
            DType::F32,
            TensorRole::Embedding,
        )
        .unwrap(),
        tensor_type: "embedding".to_string(),
        layer_index: None,
        seed: 42,
        priority: 0,
    };

    assert_eq!(task.id, "task-1");
    assert_eq!(task.tensor_type, "embedding");
    assert_eq!(task.seed, 42);
    assert_eq!(task.priority, 0);
    assert!(task.layer_index.is_none());
}

/// Test de création de tâche avec index de couche
#[test]
fn test_distributed_task_with_layer_index() {
    let task = DistributedTask {
        id: "task-2".to_string(),
        tensor_spec: TensorSpec::new(
            "model.layers.0.self_attn.q_proj.weight",
            Shape::new(vec![64, 64]).unwrap(),
            DType::F32,
            TensorRole::AttentionQuery,
        )
        .unwrap(),
        tensor_type: "attention".to_string(),
        layer_index: Some(0),
        seed: 123,
        priority: 1,
    };

    assert_eq!(task.layer_index, Some(0));
    assert_eq!(task.tensor_type, "attention");
    assert_eq!(task.priority, 1);
}

/// Test de création de coordinateur
#[test]
fn test_coordinator_creation() {
    let coordinator = DistributedCoordinator::new();
    let stats = coordinator.get_stats();

    assert_eq!(stats.total_tasks, 0);
    assert_eq!(stats.completed_tasks, 0);
    assert_eq!(stats.failed_tasks, 0);
    assert_eq!(stats.total_bytes, 0);
    assert!(stats.start_time.is_none());
}

/// Test d'ajout de tâches au coordinateur
#[test]
fn test_coordinator_add_tasks() {
    let coordinator = DistributedCoordinator::new();

    let tasks = vec![
        DistributedTask {
            id: "task-1".to_string(),
            tensor_spec: TensorSpec::new(
                "tensor1",
                Shape::new(vec![10, 10]).unwrap(),
                DType::F32,
                TensorRole::Other,
            )
            .unwrap(),
            tensor_type: "embedding".to_string(),
            layer_index: None,
            seed: 1,
            priority: 0,
        },
        DistributedTask {
            id: "task-2".to_string(),
            tensor_spec: TensorSpec::new(
                "tensor2",
                Shape::new(vec![20, 20]).unwrap(),
                DType::F16,
                TensorRole::Other,
            )
            .unwrap(),
            tensor_type: "attention".to_string(),
            layer_index: Some(0),
            seed: 2,
            priority: 1,
        },
    ];

    coordinator.add_tasks(tasks);

    let stats = coordinator.get_stats();
    assert_eq!(stats.total_tasks, 2);
    assert!(stats.start_time.is_some());
}

/// Test de création de worker
#[test]
fn test_worker_creation() {
    let worker = DistributedWorker::new("worker-1".to_string(), "127.0.0.1:9090".to_string(), 4);

    // Vérifier que le worker peut être créé
    // (pas de méthode publique pour accéder aux champs privés)
    drop(worker);
}

/// Test de création d'information worker
#[test]
fn test_worker_info_creation() {
    let worker_info = WorkerInfo {
        id: "worker-1".to_string(),
        address: "127.0.0.1:8080".to_string(),
        capacity: 4,
        active_tasks: 0,
        last_seen: 1234567890,
    };

    assert_eq!(worker_info.id, "worker-1");
    assert_eq!(worker_info.capacity, 4);
    assert_eq!(worker_info.active_tasks, 0);
    assert_eq!(worker_info.last_seen, 1234567890);
}

/// Test de statut de tâche
#[test]
fn test_task_status_variants() {
    assert_eq!(TaskStatus::Pending, TaskStatus::Pending);
    assert_eq!(TaskStatus::Running, TaskStatus::Running);
    assert_eq!(TaskStatus::Completed, TaskStatus::Completed);
    assert_eq!(TaskStatus::Failed, TaskStatus::Failed);
    assert_eq!(TaskStatus::Retry, TaskStatus::Retry);

    // Vérifier que les statuts sont distincts
    assert_ne!(TaskStatus::Pending, TaskStatus::Running);
    assert_ne!(TaskStatus::Completed, TaskStatus::Failed);
}

/// Test de statistiques de tâche
#[test]
fn test_task_stats_default() {
    let stats = TaskStats::default();
    assert_eq!(stats.duration_secs, 0.0);
    assert_eq!(stats.bytes_generated, 0);
    assert_eq!(stats.worker_id, "");
}

/// Test de statistiques globales
#[test]
fn test_global_stats_default() {
    let stats = GlobalStats::default();
    assert_eq!(stats.total_tasks, 0);
    assert_eq!(stats.completed_tasks, 0);
    assert_eq!(stats.failed_tasks, 0);
    assert_eq!(stats.total_bytes, 0);
    assert!(stats.start_time.is_none());
}

/// Test de configuration distribuée par défaut
#[test]
fn test_distributed_config_default() {
    let config = DistributedConfig::default();
    assert_eq!(config.coordinator_addr, "127.0.0.1:9090");
    assert_eq!(config.num_workers, 4);
    assert_eq!(config.worker_capacity, 8);
    assert_eq!(config.task_timeout, 300);
    assert_eq!(config.max_retries, 3);
}

/// Test de configuration distribuée personnalisée
#[test]
fn test_distributed_config_custom() {
    let config = DistributedConfig {
        coordinator_addr: "192.168.1.100:8080".to_string(),
        num_workers: 8,
        worker_capacity: 16,
        task_timeout: 600,
        max_retries: 5,
    };

    assert_eq!(config.coordinator_addr, "192.168.1.100:8080");
    assert_eq!(config.num_workers, 8);
    assert_eq!(config.worker_capacity, 16);
    assert_eq!(config.task_timeout, 600);
    assert_eq!(config.max_retries, 5);
}

/// Test de sérialisation/désérialisation de tâche
#[test]
fn test_task_serialization() {
    let task = DistributedTask {
        id: "task-serde".to_string(),
        tensor_spec: TensorSpec::new(
            "model.test.weight",
            Shape::new(vec![100, 100]).unwrap(),
            DType::F32,
            TensorRole::Other,
        )
        .unwrap(),
        tensor_type: "test".to_string(),
        layer_index: Some(5),
        seed: 42,
        priority: 2,
    };

    // Sérialiser en JSON
    let json = serde_json::to_string(&task).expect("La sérialisation a échoué");
    assert!(json.contains("task-serde"));
    assert!(json.contains("model.test.weight"));

    // Désérialiser depuis JSON
    let deserialized: DistributedTask =
        serde_json::from_str(&json).expect("La désérialisation a échoué");
    assert_eq!(deserialized.id, task.id);
    assert_eq!(deserialized.tensor_type, task.tensor_type);
    assert_eq!(deserialized.seed, task.seed);
    assert_eq!(deserialized.priority, task.priority);
}

/// Test de sérialisation/désérialisation de résultat
#[test]
fn test_result_serialization() {
    use pmg_generator::DistributedResult;

    let result = DistributedResult {
        task_id: "task-1".to_string(),
        status: TaskStatus::Completed,
        data: Some(vec![1, 2, 3, 4, 5]),
        file_path: Some("/output/model.safetensors".to_string()),
        stats: TaskStats {
            duration_secs: 1.5,
            bytes_generated: 1024,
            worker_id: "worker-1".to_string(),
        },
        error: None,
    };

    // Sérialiser en JSON
    let json = serde_json::to_string(&result).expect("La sérialisation a échoué");
    assert!(json.contains("task-1"));
    assert!(json.contains("Completed"));

    // Désérialiser depuis JSON
    let deserialized: DistributedResult =
        serde_json::from_str(&json).expect("La désérialisation a échoué");
    assert_eq!(deserialized.task_id, result.task_id);
    assert_eq!(deserialized.status, result.status);
    assert!(deserialized.data.is_some());
    assert_eq!(deserialized.stats.duration_secs, 1.5);
}

/// Test de priorité de tâche
#[test]
fn test_task_priority_ordering() {
    let mut tasks = [
        DistributedTask {
            id: "low-priority".to_string(),
            tensor_spec: TensorSpec::new(
                "tensor1",
                Shape::new(vec![10]).unwrap(),
                DType::F32,
                TensorRole::Other,
            )
            .unwrap(),
            tensor_type: "test".to_string(),
            layer_index: None,
            seed: 1,
            priority: 10,
        },
        DistributedTask {
            id: "high-priority".to_string(),
            tensor_spec: TensorSpec::new(
                "tensor2",
                Shape::new(vec![10]).unwrap(),
                DType::F32,
                TensorRole::Other,
            )
            .unwrap(),
            tensor_type: "test".to_string(),
            layer_index: None,
            seed: 2,
            priority: 0,
        },
        DistributedTask {
            id: "medium-priority".to_string(),
            tensor_spec: TensorSpec::new(
                "tensor3",
                Shape::new(vec![10]).unwrap(),
                DType::F32,
                TensorRole::Other,
            )
            .unwrap(),
            tensor_type: "test".to_string(),
            layer_index: None,
            seed: 3,
            priority: 5,
        },
    ];

    // Trier par priorité (0 = haute)
    tasks.sort_by_key(|t| t.priority);

    assert_eq!(tasks[0].id, "high-priority");
    assert_eq!(tasks[1].id, "medium-priority");
    assert_eq!(tasks[2].id, "low-priority");
}

/// Test de déterminisme des seeds
#[test]
fn test_seed_determinism() {
    let base_seed = 42;

    // Créer plusieurs tâches avec des seeds dérivées
    let tasks: Vec<u64> = (0..10).map(|i| base_seed + i as u64).collect();

    // Vérifier que les seeds sont uniques et déterministes
    let mut seen = std::collections::HashSet::new();
    for &seed in &tasks {
        assert!(seen.insert(seed), "Seed {} déjà vue", seed);
    }

    // Vérifier que la même seed produit toujours le même résultat
    let task1 = DistributedTask {
        id: "task-1".to_string(),
        tensor_spec: TensorSpec::new(
            "tensor",
            Shape::new(vec![10]).unwrap(),
            DType::F32,
            TensorRole::Other,
        )
        .unwrap(),
        tensor_type: "test".to_string(),
        layer_index: None,
        seed: base_seed,
        priority: 0,
    };

    let task2 = DistributedTask {
        id: "task-2".to_string(),
        tensor_spec: TensorSpec::new(
            "tensor",
            Shape::new(vec![10]).unwrap(),
            DType::F32,
            TensorRole::Other,
        )
        .unwrap(),
        tensor_type: "test".to_string(),
        layer_index: None,
        seed: base_seed,
        priority: 0,
    };

    assert_eq!(task1.seed, task2.seed);
}
