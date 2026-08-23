//! Tests de performance pour le module GPU
//!
//! Ces tests mesurent les performances des opérations GPU
//! et comparent les modes GPU et CPU.

use super::*;
use crate::kernel::{GpuKernel, KernelConfig, NORMAL_GENERATION_KERNEL};
use std::time::{Duration, Instant};

/// Structure pour les résultats de benchmark
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Nom du test
    pub name: String,
    /// Durée moyenne
    pub avg_duration: Duration,
    /// Nombre d'itérations
    pub iterations: usize,
    /// Débit (opérations par seconde)
    pub throughput: f64,
    /// Mode utilisé (GPU ou CPU)
    pub mode: String,
}

impl BenchmarkResult {
    /// Affiche les résultats
    pub fn display(&self) {
        println!(
            "{}: {:.3}ms ({} itérations, {:.2} ops/s, mode: {})",
            self.name,
            self.avg_duration.as_secs_f64() * 1000.0,
            self.iterations,
            self.throughput,
            self.mode
        );
    }
}

/// Benchmark d'allocation mémoire GPU
pub fn benchmark_allocation(iterations: usize) -> BenchmarkResult {
    let device = GpuDevice::new(0).unwrap();
    let mut durations = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        let _pointer = device.allocate(1024 * 1024).unwrap(); // 1 MB
        durations.push(start.elapsed());
    }

    let avg_duration = durations.iter().sum::<Duration>() / durations.len() as u32;
    let throughput = iterations as f64 / avg_duration.as_secs_f64();

    BenchmarkResult {
        name: "Allocation GPU (1 MB)".to_string(),
        avg_duration,
        iterations,
        throughput,
        mode: if cfg!(feature = "gpu") {
            "GPU"
        } else {
            "CPU (simulé)"
        }
        .to_string(),
    }
}

/// Benchmark de transfert mémoire
pub fn benchmark_transfer(iterations: usize, size: usize) -> BenchmarkResult {
    let device = GpuDevice::new(0).unwrap();
    let data = vec![0u8; size];
    let mut durations = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let pointer = device.allocate(size).unwrap();
        let start = Instant::now();
        device.memcpy_to_device(&pointer, &data).unwrap();
        durations.push(start.elapsed());
    }

    let avg_duration = durations.iter().sum::<Duration>() / durations.len() as u32;
    let throughput = iterations as f64 / avg_duration.as_secs_f64();

    BenchmarkResult {
        name: format!("Transfert GPU ({} KB)", size / 1024),
        avg_duration,
        iterations,
        throughput,
        mode: if cfg!(feature = "gpu") {
            "GPU"
        } else {
            "CPU (simulé)"
        }
        .to_string(),
    }
}

/// Benchmark de kernel simple
pub fn benchmark_kernel_execution(iterations: usize) -> BenchmarkResult {
    let device = GpuDevice::new(0).unwrap();
    let kernel = GpuKernel::new("test_kernel", "test_ptx").unwrap();
    let config = KernelConfig::default();
    let mut durations = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        kernel.launch(&device, config.clone(), &[]).unwrap();
        durations.push(start.elapsed());
    }

    let avg_duration = durations.iter().sum::<Duration>() / durations.len() as u32;
    let throughput = iterations as f64 / avg_duration.as_secs_f64();

    BenchmarkResult {
        name: "Exécution kernel".to_string(),
        avg_duration,
        iterations,
        throughput,
        mode: if cfg!(feature = "gpu") {
            "GPU"
        } else {
            "CPU (simulé)"
        }
        .to_string(),
    }
}

/// Benchmark de génération normale
pub fn benchmark_normal_generation(iterations: usize, num_elements: usize) -> BenchmarkResult {
    let device = GpuDevice::new(0).unwrap();
    let kernel = GpuKernel::new("normal_generation_kernel", NORMAL_GENERATION_KERNEL).unwrap();
    let config = KernelConfig::auto_grid(256);

    let mut durations = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        // Allouer les buffers
        let output = device.allocate(num_elements * 4).unwrap(); // f32
        let seeds = device.allocate(num_elements * 8).unwrap(); // u64

        let start = Instant::now();
        kernel
            .launch(
                &device,
                config.clone(),
                &[
                    &output as &dyn crate::kernel::ToGpuArg,
                    &seeds as &dyn crate::kernel::ToGpuArg,
                    &(num_elements as u32) as &dyn crate::kernel::ToGpuArg,
                ],
            )
            .unwrap();
        durations.push(start.elapsed());
    }

    let avg_duration = durations.iter().sum::<Duration>() / durations.len() as u32;
    let throughput = iterations as f64 / avg_duration.as_secs_f64();

    BenchmarkResult {
        name: format!("Génération normale ({} éléments)", num_elements),
        avg_duration,
        iterations,
        throughput,
        mode: if cfg!(feature = "gpu") {
            "GPU"
        } else {
            "CPU (simulé)"
        }
        .to_string(),
    }
}

/// Comparaison CPU vs GPU pour la génération
pub fn compare_cpu_gpu_performance() {
    println!("\n=== Comparaison CPU vs GPU ===\n");

    // Benchmark CPU (simulation)
    let cpu_start = Instant::now();
    let iterations = 1000;
    let num_elements = 10000;

    for _ in 0..iterations {
        // Simulation de génération CPU
        let _: Vec<f32> = (0..num_elements)
            .map(|i| (i as f32 * 0.001).sin())
            .collect();
    }
    let cpu_duration = cpu_start.elapsed();

    // Benchmark GPU (simulation)
    let gpu_start = Instant::now();
    for _ in 0..iterations {
        let device = GpuDevice::new(0).unwrap();
        let kernel = GpuKernel::new("normal_generation_kernel", NORMAL_GENERATION_KERNEL).unwrap();
        let config = KernelConfig::auto_grid(256);
        let output = device.allocate(num_elements * 4).unwrap();
        let seeds = device.allocate(num_elements * 8).unwrap();
        kernel
            .launch(
                &device,
                config,
                &[
                    &output as &dyn crate::kernel::ToGpuArg,
                    &seeds as &dyn crate::kernel::ToGpuArg,
                    &(num_elements as u32) as &dyn crate::kernel::ToGpuArg,
                ],
            )
            .unwrap();
    }
    let gpu_duration = gpu_start.elapsed();

    println!(
        "CPU: {:.3}ms pour {} itérations",
        cpu_duration.as_secs_f64() * 1000.0,
        iterations
    );
    println!(
        "GPU (simulé): {:.3}ms pour {} itérations",
        gpu_duration.as_secs_f64() * 1000.0,
        iterations
    );

    let speedup = cpu_duration.as_secs_f64() / gpu_duration.as_secs_f64();
    println!("Accélération GPU: {:.2}x", speedup);

    if speedup > 2.0 {
        println!("✅ Objectif d'accélération atteint (>2x)");
    } else {
        println!("⚠️  Accélération insuffisante (<2x)");
    }
}

/// Exécute tous les benchmarks
pub fn run_all_benchmarks() {
    println!("=== Benchmarks GPU PMG ===\n");

    let results = vec![
        benchmark_allocation(100),
        benchmark_transfer(100, 1024 * 1024),      // 1 MB
        benchmark_transfer(100, 10 * 1024 * 1024), // 10 MB
        benchmark_kernel_execution(1000),
        benchmark_normal_generation(100, 10000),
        benchmark_normal_generation(100, 100000),
    ];

    for result in &results {
        result.display();
    }

    compare_cpu_gpu_performance();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_allocation() {
        let result = benchmark_allocation(10);
        assert!(result.avg_duration > Duration::from_nanos(0));
        assert!(result.throughput > 0.0);
    }

    #[test]
    fn test_benchmark_transfer() {
        let result = benchmark_transfer(10, 1024);
        assert!(result.avg_duration > Duration::from_nanos(0));
        assert!(result.throughput > 0.0);
    }

    #[test]
    fn test_benchmark_kernel_execution() {
        let result = benchmark_kernel_execution(10);
        assert!(result.avg_duration > Duration::from_nanos(0));
        assert!(result.throughput > 0.0);
    }

    #[test]
    fn test_compare_cpu_gpu_performance() {
        // Ce test vérifie que la comparaison ne panic pas
        compare_cpu_gpu_performance();
    }

    #[test]
    fn test_run_all_benchmarks() {
        // Ce test exécute tous les benchmarks
        // Note: il peut prendre du temps
        run_all_benchmarks();
    }
}
