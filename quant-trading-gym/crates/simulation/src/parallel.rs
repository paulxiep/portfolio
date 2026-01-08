//! Parallel execution utilities for the simulation.
//!
//! This module provides declarative helpers that abstract over parallel vs sequential
//! execution based on the `parallel` feature flag. The `cfg` logic lives here in ONE
//! place, keeping call sites clean.
//!
//! # Design
//!
//! Each helper takes a closure and applies it over a collection. When `parallel` is
//! enabled, uses rayon's parallel iterators; otherwise uses standard iterators.
//!
//! # Runtime Override (V3.7)
//!
//! All functions accept an optional `force_sequential` parameter. When `true`, execution
//! is sequential even if the `parallel` feature is enabled. This allows runtime profiling
//! and testing of parallel vs sequential performance.
//!
//! # Example
//!
//! ```ignore
//! use crate::parallel;
//!
//! // Instead of:
//! // #[cfg(feature = "parallel")]
//! // let results: Vec<_> = items.par_iter().map(|x| process(x)).collect();
//! // #[cfg(not(feature = "parallel"))]
//! // let results: Vec<_> = items.iter().map(|x| process(x)).collect();
//!
//! // Just write:
//! let results = parallel::map_slice(&items, |x| process(x), false);
//!
//! // Or force sequential for profiling:
//! let results = parallel::map_slice(&items, |x| process(x), true);
//! ```

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// =============================================================================
// Slice Operations
// =============================================================================

/// Map a function over a slice, potentially in parallel.
///
/// Returns a Vec of results in the same order as input (parallel preserves order).
///
/// # Parameters
/// - `force_sequential`: When true, forces sequential execution even if parallel feature is enabled
#[inline]
pub fn map_slice<T, F, R>(slice: &[T], f: F, force_sequential: bool) -> Vec<R>
where
    T: Sync,
    F: Fn(&T) -> R + Sync + Send,
    R: Send,
{
    #[cfg(feature = "parallel")]
    {
        if force_sequential {
            slice.iter().map(f).collect()
        } else {
            slice.par_iter().map(f).collect()
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        let _ = force_sequential; // Suppress unused warning
        slice.iter().map(f).collect()
    }
}

/// Filter-map over a slice, potentially in parallel.
///
/// Applies `f` to each element, collecting `Some` results and discarding `None`.
///
/// # Parameters
/// - `force_sequential`: When true, forces sequential execution even if parallel feature is enabled
#[inline]
pub fn filter_map_slice<T, F, R>(slice: &[T], f: F, force_sequential: bool) -> Vec<R>
where
    T: Sync,
    F: Fn(&T) -> Option<R> + Sync + Send,
    R: Send,
{
    #[cfg(feature = "parallel")]
    {
        if force_sequential {
            slice.iter().filter_map(f).collect()
        } else {
            slice.par_iter().filter_map(f).collect()
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        let _ = force_sequential;
        slice.iter().filter_map(f).collect()
    }
}

/// Execute a side-effectful closure for each element, potentially in parallel.
///
/// Note: The closure must be safe to call concurrently (e.g., only mutating
/// thread-local state or using interior mutability with proper synchronization).
///
/// # Parameters
/// - `force_sequential`: When true, forces sequential execution even if parallel feature is enabled
#[inline]
pub fn for_each_slice<T, F>(slice: &[T], f: F, force_sequential: bool)
where
    T: Sync,
    F: Fn(&T) + Sync + Send,
{
    #[cfg(feature = "parallel")]
    {
        if force_sequential {
            slice.iter().for_each(f);
        } else {
            slice.par_iter().for_each(f);
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        let _ = force_sequential;
        slice.iter().for_each(f);
    }
}

// =============================================================================
// Index-based Operations (for Mutex-wrapped collections)
// =============================================================================

/// Map over indices, potentially in parallel.
///
/// Useful when you have a `Vec<Mutex<T>>` and need to access by index.
///
/// # Parameters
/// - `force_sequential`: When true, forces sequential execution even if parallel feature is enabled
#[inline]
pub fn map_indices<F, R>(indices: &[usize], f: F, force_sequential: bool) -> Vec<R>
where
    F: Fn(usize) -> R + Sync + Send,
    R: Send,
{
    #[cfg(feature = "parallel")]
    {
        if force_sequential {
            indices.iter().map(|&i| f(i)).collect()
        } else {
            indices.par_iter().map(|&i| f(i)).collect()
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        let _ = force_sequential;
        indices.iter().map(|&i| f(i)).collect()
    }
}

/// Filter-map over indices, potentially in parallel.
///
/// # Parameters
/// - `force_sequential`: When true, forces sequential execution even if parallel feature is enabled
#[inline]
pub fn filter_map_indices<F, R>(indices: &[usize], f: F, force_sequential: bool) -> Vec<R>
where
    F: Fn(usize) -> Option<R> + Sync + Send,
    R: Send,
{
    #[cfg(feature = "parallel")]
    {
        if force_sequential {
            indices.iter().filter_map(|&i| f(i)).collect()
        } else {
            indices.par_iter().filter_map(|&i| f(i)).collect()
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        let _ = force_sequential;
        indices.iter().filter_map(|&i| f(i)).collect()
    }
}

// =============================================================================
// Vec Operations (owned iteration)
// =============================================================================

/// Map over a Vec, consuming it, potentially in parallel.
///
/// # Parameters
/// - `force_sequential`: When true, forces sequential execution even if parallel feature is enabled
#[inline]
pub fn map_vec<T, F, R>(vec: Vec<T>, f: F, force_sequential: bool) -> Vec<R>
where
    T: Send,
    F: Fn(T) -> R + Sync + Send,
    R: Send,
{
    #[cfg(feature = "parallel")]
    {
        if force_sequential {
            vec.into_iter().map(f).collect()
        } else {
            vec.into_par_iter().map(f).collect()
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        let _ = force_sequential;
        vec.into_iter().map(f).collect()
    }
}

/// Filter-map over a Vec, consuming it, potentially in parallel.
///
/// # Parameters
/// - `force_sequential`: When true, forces sequential execution even if parallel feature is enabled
#[inline]
pub fn filter_map_vec<T, F, R>(vec: Vec<T>, f: F, force_sequential: bool) -> Vec<R>
where
    T: Send,
    F: Fn(T) -> Option<R> + Sync + Send,
    R: Send,
{
    #[cfg(feature = "parallel")]
    {
        if force_sequential {
            vec.into_iter().filter_map(f).collect()
        } else {
            vec.into_par_iter().filter_map(f).collect()
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        let _ = force_sequential;
        vec.into_iter().filter_map(f).collect()
    }
}

// =============================================================================
// HashMap Operations
// =============================================================================

use std::collections::HashMap;
use std::hash::Hash;

/// Map a slice to a HashMap, potentially in parallel.
///
/// # Parameters
/// - `force_sequential`: When true, forces sequential execution even if parallel feature is enabled
#[inline]
pub fn map_to_hashmap<T, F, K, V>(slice: &[T], f: F, force_sequential: bool) -> HashMap<K, V>
where
    T: Sync,
    F: Fn(&T) -> (K, V) + Sync + Send,
    K: Eq + Hash + Send,
    V: Send,
{
    #[cfg(feature = "parallel")]
    {
        if force_sequential {
            slice.iter().map(f).collect()
        } else {
            slice.par_iter().map(f).collect()
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        let _ = force_sequential;
        slice.iter().map(f).collect()
    }
}

// =============================================================================
// Specialized Mutex Operations
// =============================================================================

use parking_lot::Mutex;

/// Map over a slice of Mutex-wrapped items, locking each.
///
/// This is the common pattern for our agent collection.
///
/// # Parameters
/// - `force_sequential`: When true, forces sequential execution even if parallel feature is enabled
#[inline]
pub fn map_mutex_slice<T, F, R>(slice: &[Mutex<T>], f: F, force_sequential: bool) -> Vec<R>
where
    T: Send,
    F: Fn(&mut T) -> R + Sync + Send,
    R: Send,
{
    #[cfg(feature = "parallel")]
    {
        if force_sequential {
            slice.iter().map(|m| f(&mut *m.lock())).collect()
        } else {
            slice.par_iter().map(|m| f(&mut *m.lock())).collect()
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        let _ = force_sequential;
        slice.iter().map(|m| f(&mut *m.lock())).collect()
    }
}

/// Map over a slice of Mutex-wrapped items with immutable access.
///
/// # Parameters
/// - `force_sequential`: When true, forces sequential execution even if parallel feature is enabled
#[inline]
pub fn map_mutex_slice_ref<T, F, R>(slice: &[Mutex<T>], f: F, force_sequential: bool) -> Vec<R>
where
    T: Send,
    F: Fn(&T) -> R + Sync + Send,
    R: Send,
{
    #[cfg(feature = "parallel")]
    {
        if force_sequential {
            slice.iter().map(|m| f(&*m.lock())).collect()
        } else {
            slice.par_iter().map(|m| f(&*m.lock())).collect()
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        let _ = force_sequential;
        slice.iter().map(|m| f(&*m.lock())).collect()
    }
}
