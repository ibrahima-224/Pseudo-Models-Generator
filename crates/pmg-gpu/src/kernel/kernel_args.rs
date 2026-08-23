//! Interfaces d'argumentation pour les kernels GPU
//!
//! Ce module définit le trait `ToGpuArg` qui permet de convertir des types Rust
//! en arguments interprétables par les kernels CUDA/PTX. Il fournit également
//! des implémentations pour les types courants.

/// Trait pour les arguments de kernel
///
/// Permet de passer des arguments typés aux kernels GPU.
pub trait ToGpuArg {
    /// Convertit l'argument en pointeur brut pour le kernel
    fn to_gpu_ptr(&self) -> *mut std::ffi::c_void;

    /// Retourne la taille des données en octets
    fn size(&self) -> usize;

    /// Retourne le nom de l'argument (pour le débogage)
    fn name(&self) -> &str {
        "unnamed"
    }
}

/// Implémentation pour les slices
impl<T> ToGpuArg for &[T] {
    fn to_gpu_ptr(&self) -> *mut std::ffi::c_void {
        self.as_ptr() as *mut std::ffi::c_void
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(*self)
    }
}

/// Implémentation pour les slices mutables
impl<T> ToGpuArg for &mut [T] {
    fn to_gpu_ptr(&self) -> *mut std::ffi::c_void {
        // CORRECTION : Retourner un pointeur mutable pour le GPU
        // Le paramètre self est de type &&mut [T], donc nous utilisons as_ptr()
        // puis cast en *mut pour obtenir le pointeur mutable requis par les kernels GPU
        // C'est sûr car le slice est réellement mutable (il vient d'un &mut [T])
        self.as_ptr() as *mut std::ffi::c_void
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(*self)
    }
}

/// Implémentation pour les pointeurs bruts
impl<T> ToGpuArg for *mut T {
    fn to_gpu_ptr(&self) -> *mut std::ffi::c_void {
        *self as *mut std::ffi::c_void
    }

    fn size(&self) -> usize {
        std::mem::size_of::<T>()
    }
}

/// Implémentation pour les GpuPointer
impl ToGpuArg for crate::device::GpuPointer {
    fn to_gpu_ptr(&self) -> *mut std::ffi::c_void {
        #[cfg(feature = "gpu")]
        {
            self.raw_ptr() as *mut std::ffi::c_void
        }
        #[cfg(not(feature = "gpu"))]
        {
            std::ptr::null_mut()
        }
    }

    fn size(&self) -> usize {
        self.size
    }
}

/// Implémentation pour u32
impl ToGpuArg for u32 {
    fn to_gpu_ptr(&self) -> *mut std::ffi::c_void {
        self as *const u32 as *mut std::ffi::c_void
    }

    fn size(&self) -> usize {
        std::mem::size_of::<u32>()
    }
}
