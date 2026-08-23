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

//! # Pool Typé et Buffer avec Remise Automatique
//!
//! Ce module fournit un pool typé pour les types spécifiques (u8, f64)
//! et un wrapper `PooledBuffer` qui retourne automatiquement le buffer
//! au pool lors de sa destruction (via `Drop`).

use std::ops::{Deref, DerefMut};

use super::buffer_pool::UnifiedBufferPool;

/// Pool typé pour les buffers d'un type spécifique.
///
/// Ce pool encapsule un `UnifiedBufferPool` et fournit des méthodes
/// typées pour acquérir et libérer des buffers d'un type donné.
pub struct TypedPool<T> {
    /// Pool de buffers sous-jacent partagé.
    pool: UnifiedBufferPool,

    /// Taille en octets d'un élément du type `T`.
    _element_size: usize,

    /// Marqueur de phantom pour le type `T`.
    _marker: std::marker::PhantomData<T>,
}

/// Buffer acquis depuis un pool, avec remise automatique au pool when dropped.
///
/// Ce wrapper garantit que le buffer est toujours retourné au pool,
/// même en cas de panic ou de sortie précoce de la portée.
pub struct PooledBuffer<T> {
    /// Le buffer interne.
    buffer: Option<Vec<T>>,

    /// Référence vers le pool pour la remise.
    pool: UnifiedBufferPool,
}

// =============================================================================
// Implémentation de TypedPool<u8>
// =============================================================================

impl TypedPool<u8> {
    /// Crée un pool typé pour des buffers `u8`.
    ///
    /// # Paramètres
    /// - `pool` : pool de buffers unifié sous-jacent
    pub fn new_u8(pool: UnifiedBufferPool) -> Self {
        Self {
            pool,
            _element_size: std::mem::size_of::<u8>(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Acquiert un buffer `u8` de la taille minimale spécifiée.
    ///
    /// # Paramètres
    /// - `min_size` : taille minimale requise (en octets)
    ///
    /// # Retourne
    /// Un `PooledBuffer<u8>` qui sera automatiquement retourné au pool.
    pub fn acquire(&self, min_size: usize) -> PooledBuffer<u8> {
        let buffer = self.pool.acquire_u8(min_size);
        PooledBuffer {
            buffer: Some(buffer),
            pool: self.pool.clone(),
        }
    }
}

// =============================================================================
// Implémentation de TypedPool<f64>
// =============================================================================

impl TypedPool<f64> {
    /// Crée un pool typé pour des buffers `f64`.
    ///
    /// # Paramètres
    /// - `pool` : pool de buffers unifié sous-jacent
    pub fn new_f64(pool: UnifiedBufferPool) -> Self {
        Self {
            pool,
            _element_size: std::mem::size_of::<f64>(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Acquiert un buffer `f64` de la taille minimale spécifiée.
    ///
    /// # Paramètres
    /// - `min_len` : nombre minimal d'éléments `f64` requis
    ///
    /// # Retourne
    /// Un `PooledBuffer<f64>` qui sera automatiquement retourné au pool.
    pub fn acquire(&self, min_len: usize) -> PooledBuffer<f64> {
        let buffer = self.pool.acquire_f64(min_len);
        PooledBuffer {
            buffer: Some(buffer),
            pool: self.pool.clone(),
        }
    }
}

// =============================================================================
// Implémentation générique de PooledBuffer<T>
// =============================================================================

impl<T: Clone> PooledBuffer<T> {
    /// Crée un nouveau PooledBuffer à partir d'un buffer et d'un pool.
    ///
    /// # Paramètres
    /// - `buffer` : le buffer à wrapper
    /// - `pool` : le pool auquel retourner le buffer when dropped
    ///
    /// # Retourne
    /// Un nouveau PooledBuffer.
    pub fn new(buffer: Vec<T>, pool: UnifiedBufferPool) -> Self {
        Self {
            buffer: Some(buffer),
            pool,
        }
    }

    /// Retourne la capacité du buffer sous-jacent.
    ///
    /// # Retourne
    /// La capacité en éléments du type `T`.
    pub fn capacity(&self) -> usize {
        self.buffer.as_ref().map(|b| b.capacity()).unwrap_or(0)
    }

    /// Retourne une référence immuable vers le buffer interne.
    ///
    /// # Retourne
    /// Une référence `&Vec<T>` vers le buffer.
    pub fn inner(&self) -> &Vec<T> {
        self.buffer
            .as_ref()
            .expect("PooledBuffer vide après extraction")
    }

    /// Ajoute des éléments à la fin du buffer.
    ///
    /// # Paramètres
    /// - `slice` : les éléments à ajouter
    pub fn extend_from_slice(&mut self, slice: &[T]) {
        if let Some(ref mut buf) = self.buffer {
            buf.extend_from_slice(slice);
        }
    }

    /// Extrait le buffer du wrapper sans le retourner au pool.
    ///
    /// # Attention
    /// Après cet appel, le buffer ne sera plus automatiquement retourné
    /// au pool. L'appelant doit gérer manuellement la mémoire.
    ///
    /// # Retourne
    /// Le buffer `Vec<T>` extrait.
    pub fn into_inner(mut self) -> Vec<T> {
        self.buffer
            .take()
            .expect("PooledBuffer vide après extraction")
    }
}

// =============================================================================
// Implémentation spécifique pour PooledBuffer<u8>
// =============================================================================

impl PooledBuffer<u8> {
    /// Crée un nouveau PooledBuffer<u8> à partir d'un buffer et d'un pool.
    ///
    /// # Paramètres
    /// - `buffer` : le buffer u8 à wrapper
    /// - `pool` : le pool auquel retourner le buffer when dropped
    ///
    /// # Retourne
    /// Un nouveau PooledBuffer<u8>.
    pub fn new_u8(buffer: Vec<u8>, pool: UnifiedBufferPool) -> Self {
        Self {
            buffer: Some(buffer),
            pool,
        }
    }
}

// =============================================================================
// Implémentation spécifique pour PooledBuffer<f64>
// =============================================================================

impl PooledBuffer<f64> {
    /// Crée un nouveau PooledBuffer<f64> à partir d'un buffer et d'un pool.
    ///
    /// # Paramètres
    /// - `buffer` : le buffer f64 à wrapper
    /// - `pool` : le pool auquel retourner le buffer when dropped
    ///
    /// # Retourne
    /// Un nouveau PooledBuffer<f64>.
    pub fn new_f64(buffer: Vec<f64>, pool: UnifiedBufferPool) -> Self {
        Self {
            buffer: Some(buffer),
            pool,
        }
    }
}

// =============================================================================
// Implémentation de Drop pour PooledBuffer<T>
// =============================================================================

impl<T> Drop for PooledBuffer<T> {
    /// Retourne automatiquement le buffer au pool when dropped.
    ///
    /// Pour `u8`, le buffer est directement retourné.
    /// Pour `f64`, le buffer est converti en `u8` avant retour.
    fn drop(&mut self) {
        // Ne fait rien si le buffer a déjà été extrait
        if self.buffer.is_none() {
            return;
        }

        // Retourne le buffer au pool selon le type
        // Utilise un trick avec la taille pour distinguer les types
        if std::mem::size_of::<T>() == std::mem::size_of::<u8>()
            && std::mem::align_of::<T>() == std::mem::align_of::<u8>()
        {
            // SAFETY : T est vérifié être u8 par les conditions size_of et align_of.
            // Les taille et alignement sont identiques, donc la conversion est sûre.
            // Préconditions :
            // 1. size_of::<T>() == size_of::<u8>() (taille identique)
            // 2. align_of::<T>() == align_of::<u8>() (alignement identique)
            // 3. buf contient des éléments T valides
            // 4. buf n'est pas encore libéré (on utilise take() pour extraire le buffer)
            debug_assert_eq!(
                std::mem::size_of::<T>(),
                std::mem::size_of::<u8>(),
                "La taille de T doit être identique à celle de u8"
            );
            debug_assert_eq!(
                std::mem::align_of::<T>(),
                std::mem::align_of::<u8>(),
                "L'alignement de T doit être identique à celui de u8"
            );
            let mut buf = self.buffer.take().unwrap();
            let ptr = buf.as_mut_ptr() as *mut u8;
            let len = buf.len();
            let cap = buf.capacity();
            debug_assert!(len <= cap, "La longueur ne doit pas dépasser la capacité");
            debug_assert!(
                ptr as usize % std::mem::align_of::<u8>() == 0,
                "Le pointeur doit être aligné pour u8"
            );

            std::mem::forget(buf);
            let buffer: Vec<u8> = unsafe { Vec::from_raw_parts(ptr, len, cap) };
            self.pool.release_u8(buffer);
        } else if std::mem::size_of::<T>() == std::mem::size_of::<f64>()
            && std::mem::align_of::<T>() == std::mem::align_of::<f64>()
        {
            // SAFETY : T est vérifié être f64 par les conditions size_of et align_of.
            // Les taille et alignement sont identiques, donc la conversion est sûre.
            // Préconditions :
            // 1. size_of::<T>() == size_of::<f64>() (taille identique)
            // 2. align_of::<T>() == align_of::<f64>() (alignement identique)
            // 3. buf contient des éléments T valides
            // 4. buf n'est pas encore libéré (on utilise take() pour extraire le buffer)
            debug_assert_eq!(
                std::mem::size_of::<T>(),
                std::mem::size_of::<f64>(),
                "La taille de T doit être identique à celle de f64"
            );
            debug_assert_eq!(
                std::mem::align_of::<T>(),
                std::mem::align_of::<f64>(),
                "L'alignement de T doit être identique à celui de f64"
            );
            let mut buf = self.buffer.take().unwrap();
            let ptr = buf.as_mut_ptr() as *mut f64;
            let len = buf.len();
            let cap = buf.capacity();
            debug_assert!(len <= cap, "La longueur ne doit pas dépasser la capacité");
            debug_assert!(
                ptr as usize % std::mem::align_of::<f64>() == 0,
                "Le pointeur doit être aligné pour f64"
            );

            std::mem::forget(buf);
            let buffer: Vec<f64> = unsafe { Vec::from_raw_parts(ptr, len, cap) };
            self.pool.release_f64(buffer);
        } else {
            // Pour les types inconnus, on ne fait rien (le buffer sera libéré par Drop de Vec<T>)
            let _ = self.buffer.take();
        }
    }
}

// =============================================================================
// Implémentation de Deref et DerefMut
// =============================================================================

impl<T> Deref for PooledBuffer<T> {
    type Target = [T];

    /// Déréférence vers la tranche sous-jacente.
    fn deref(&self) -> &[T] {
        match &self.buffer {
            Some(buf) => buf.as_slice(),
            None => &[],
        }
    }
}

impl<T> DerefMut for PooledBuffer<T> {
    /// Déréférence mutable vers la tranche sous-jacente.
    fn deref_mut(&mut self) -> &mut [T] {
        match &mut self.buffer {
            Some(buf) => buf.as_mut_slice(),
            None => &mut [],
        }
    }
}

// =============================================================================
// Implémentation de AsRef et AsMut
// =============================================================================

impl<T> AsRef<[T]> for PooledBuffer<T> {
    /// Convertit en référence vers la tranche sous-jacente.
    fn as_ref(&self) -> &[T] {
        self.deref()
    }
}

impl<T> AsMut<[T]> for PooledBuffer<T> {
    /// Convertit en référence mutable vers la tranche sous-jacente.
    fn as_mut(&mut self) -> &mut [T] {
        self.deref_mut()
    }
}
