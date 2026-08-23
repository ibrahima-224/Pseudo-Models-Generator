//! Métriques de performance GPU/CPU
//!
//! Ce module fournit des métriques pour mesurer les performances
//! de l'accélération GPU par rapport au CPU.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Métriques de performance
#[derive(Debug, Default)]
pub struct PerformanceMetrics {
    /// Nombre d'opérations exécutées
    operations_count: AtomicU64,
    /// Temps total d'exécution
    total_duration: AtomicU64,
    /// Nombre d'opérations GPU
    gpu_operations: AtomicU64,
    /// Nombre d'opérations CPU
    cpu_operations: AtomicU64,
    /// Mémoire GPU utilisée (en octets)
    gpu_memory_used: AtomicU64,
}

impl PerformanceMetrics {
    /// Crée de nouvelles métriques
    pub fn new() -> Self {
        Self::default()
    }

    /// Enregistre une opération
    pub fn record_operation(&self, duration: Duration, is_gpu: bool) {
        self.operations_count.fetch_add(1, Ordering::SeqCst);
        self.total_duration
            .fetch_add(duration.as_nanos() as u64, Ordering::SeqCst);

        if is_gpu {
            self.gpu_operations.fetch_add(1, Ordering::SeqCst);
        } else {
            self.cpu_operations.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Met à jour la mémoire GPU utilisée
    pub fn update_gpu_memory(&self, bytes: u64) {
        self.gpu_memory_used.store(bytes, Ordering::SeqCst);
    }

    /// Retourne le nombre total d'opérations
    pub fn operations_count(&self) -> u64 {
        self.operations_count.load(Ordering::SeqCst)
    }

    /// Retourne le temps moyen par opération
    pub fn average_duration(&self) -> Duration {
        let total_nanos = self.total_duration.load(Ordering::SeqCst);
        let count = self.operations_count.load(Ordering::SeqCst);

        // Utilisation de checked_div pour éviter la division par zéro de manière idiomatique
        total_nanos
            .checked_div(count)
            .map(Duration::from_nanos)
            .unwrap_or(Duration::ZERO)
    }

    /// Retourne le ratio GPU/CPU
    pub fn gpu_cpu_ratio(&self) -> f64 {
        let gpu = self.gpu_operations.load(Ordering::SeqCst) as f64;
        let cpu = self.cpu_operations.load(Ordering::SeqCst) as f64;

        if cpu == 0.0 {
            if gpu == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            gpu / cpu
        }
    }

    /// Retourne la mémoire GPU utilisée
    pub fn gpu_memory_used(&self) -> u64 {
        self.gpu_memory_used.load(Ordering::SeqCst)
    }

    /// Génère un rapport de performance
    pub fn report(&self) -> PerformanceReport {
        PerformanceReport {
            operations_count: self.operations_count(),
            average_duration: self.average_duration(),
            gpu_operations: self.gpu_operations.load(Ordering::SeqCst),
            cpu_operations: self.cpu_operations.load(Ordering::SeqCst),
            gpu_cpu_ratio: self.gpu_cpu_ratio(),
            gpu_memory_used: self.gpu_memory_used(),
        }
    }
}

/// Rapport de performance
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub operations_count: u64,
    pub average_duration: Duration,
    pub gpu_operations: u64,
    pub cpu_operations: u64,
    pub gpu_cpu_ratio: f64,
    pub gpu_memory_used: u64,
}

impl std::fmt::Display for PerformanceReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Rapport de Performance ===")?;
        writeln!(f, "Opérations totales: {}", self.operations_count)?;
        writeln!(f, "Durée moyenne: {:?}", self.average_duration)?;
        writeln!(f, "Opérations GPU: {}", self.gpu_operations)?;
        writeln!(f, "Opérations CPU: {}", self.cpu_operations)?;
        writeln!(f, "Ratio GPU/CPU: {:.2}", self.gpu_cpu_ratio)?;
        writeln!(f, "Mémoire GPU: {} octets", self.gpu_memory_used)?;
        Ok(())
    }
}

/// Timer pour mesurer la durée des opérations
pub struct OperationTimer {
    start: std::time::Instant,
    is_gpu: bool,
}

impl OperationTimer {
    /// Démarre un timer pour une opération GPU
    pub fn gpu() -> Self {
        Self {
            start: std::time::Instant::now(),
            is_gpu: true,
        }
    }

    /// Démarre un timer pour une opération CPU
    pub fn cpu() -> Self {
        Self {
            start: std::time::Instant::now(),
            is_gpu: false,
        }
    }

    /// Arrête le timer et retourne la durée
    pub fn stop(self) -> (Duration, bool) {
        (self.start.elapsed(), self.is_gpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics() {
        let metrics = PerformanceMetrics::new();

        // Enregistrer des opérations
        metrics.record_operation(Duration::from_millis(10), true);
        metrics.record_operation(Duration::from_millis(20), false);
        metrics.record_operation(Duration::from_millis(15), true);

        assert_eq!(metrics.operations_count(), 3);
        assert!(metrics.average_duration() > Duration::ZERO);
        assert!(metrics.gpu_cpu_ratio() > 0.0);
    }

    #[test]
    fn test_operation_timer() {
        let timer = OperationTimer::gpu();
        std::thread::sleep(Duration::from_millis(10));
        let (duration, is_gpu) = timer.stop();

        assert!(duration >= Duration::from_millis(10));
        assert!(is_gpu);
    }
}
