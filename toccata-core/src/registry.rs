// Copyright (c) 2026 Cameron Bytheway. SPDX-License-Identifier: MIT

//! App-configurable sub-heap registry.
//!
//! Components **register** a named sub-heap declaratively via
//! the [`register_subheap!`] macro, without knowing its budget. At startup the
//! top-level application calls [`configure`] with a budget map (typically from
//! its own config file), which validates the total against the reservation and
//! builds each sub-heap. Components then fetch their live handle with
//! [`subheap`].
//!
//! Registration is **static** (a `linkme` distributed slice collected at link
//! time) — no dynamic registration, for deterministic startup. A registered but
//! unconfigured sub-heap fails loudly at `configure` time.

use crate::subheap::{OnExhaust, SubHeap, SubHeapBuilder};
use linkme::distributed_slice;
use std::sync::OnceLock;

/// One component's static declaration of a sub-heap it needs. Collected at link
/// time into [`REGISTRATIONS`].
pub struct SubHeapRegistration {
    /// Unique name; the app's budget map is keyed on this.
    pub name: &'static str,
    /// Default exhaustion policy (the app can't currently override this).
    pub on_exhaust: OnExhaust,
    /// Per-(CPU,class) stack capacity hint.
    pub cap_per_class: u32,
}

/// The link-time-collected set of all declared sub-heaps. Components append via
/// [`register_subheap!`]; the registry reads it in [`configure`].
#[distributed_slice]
pub static REGISTRATIONS: [SubHeapRegistration] = [..];

/// Declare a sub-heap from any crate. Expands to a `linkme` slice element.
///
/// ```ignore
/// toccata_core::register_subheap!(PACKET, "packet");
/// toccata_core::register_subheap!(LINKS, "links", on_exhaust = OnExhaust::None, cap_per_class = 512);
/// ```
#[macro_export]
macro_rules! register_subheap {
    ($ident:ident, $name:expr $(, on_exhaust = $oe:expr)? $(, cap_per_class = $cap:expr)? $(,)?) => {
        #[$crate::linkme::distributed_slice($crate::registry::REGISTRATIONS)]
        #[linkme(crate = $crate::linkme)]
        static $ident: $crate::registry::SubHeapRegistration = $crate::registry::SubHeapRegistration {
            name: $name,
            on_exhaust: { let oe = $crate::OnExhaust::None; $(let oe = $oe;)? oe },
            cap_per_class: { let cap = 256u32; $(let cap = $cap;)? cap },
        };
    };
}

/// The configured sub-heaps, set once by [`configure`]. A small system-backed
/// vector (a handful of entries; linear scan beats hashing and — crucially —
/// allocates via the **system** allocator, never recursing through toccata's
/// global allocator).
static CONFIGURED: OnceLock<crate::SysVec<(&'static str, &'static SubHeap)>> = OnceLock::new();

/// App-supplied budget configuration: a total reservation cap plus a per-named-
/// sub-heap byte budget. Built by the application's config function (which may
/// read env/files freely — its allocations go to the **system** allocator).
/// Backed by a [`crate::SysVec`] so constructing it never recurses through
/// toccata's global allocator.
pub struct Budgets {
    total: usize,
    subs: crate::SysVec<(&'static str, usize)>,
}

impl Budgets {
    /// Start a budget config with a `total_bytes` cap (the max toccata will
    /// reserve+lock across all sub-heaps).
    pub fn new(total_bytes: usize) -> Self {
        Self {
            total: total_bytes,
            subs: allocator_api2::vec::Vec::new_in(crate::Sys),
        }
    }

    /// Assign `bytes` to the sub-heap registered under `name`. Chainable.
    pub fn sub(mut self, name: &'static str, bytes: usize) -> Self {
        self.subs.push((name, bytes));
        self
    }

    #[inline]
    pub fn total(&self) -> usize {
        self.total
    }

    #[inline]
    fn lookup(&self, name: &str) -> Option<usize> {
        self.subs.iter().find(|(n, _)| *n == name).map(|(_, b)| *b)
    }
}

/// Errors from [`configure`].
#[derive(Debug)]
pub enum ConfigureError {
    /// Already configured (idempotent guard).
    AlreadyConfigured,
    /// A registered sub-heap had no budget entry in the app's map.
    Unconfigured(&'static str),
    /// The sum of budgets exceeds `total_bytes`.
    OverBudget { requested: usize, total: usize },
    /// A sub-heap reservation failed.
    Reserve(&'static str, crate::ReserveError),
}

impl std::fmt::Display for ConfigureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigureError::AlreadyConfigured => write!(f, "toccata registry already configured"),
            ConfigureError::Unconfigured(n) => {
                write!(
                    f,
                    "registered sub-heap '{n}' has no budget in the config map"
                )
            }
            ConfigureError::OverBudget { requested, total } => write!(
                f,
                "sub-heap budgets sum to {requested} bytes, exceeding total_bytes {total}"
            ),
            ConfigureError::Reserve(n, e) => write!(f, "sub-heap '{n}' reservation failed: {e}"),
        }
    }
}

impl std::error::Error for ConfigureError {}

/// Configure all registered sub-heaps from an app-supplied budget map (name ->
/// bytes). Validates that every registration has a budget and that the sum fits
/// `total_bytes`, then builds each sub-heap. Call once at startup.
///
/// Each sub-heap currently gets its own standalone reservation; a future change
/// can carve them from one shared reservation. `total_bytes` bounds the sum.
/// `budgets` is a slice of `(name, bytes)` provided by the caller (no allocation
/// by toccata). Every registration must have a matching entry; the sum must fit
/// `total_bytes`.
pub fn configure(budgets: &Budgets) -> Result<(), ConfigureError> {
    if CONFIGURED.get().is_some() {
        return Err(ConfigureError::AlreadyConfigured);
    }

    // Validate: every registration has a budget; the sum fits.
    let mut sum = 0usize;
    for reg in REGISTRATIONS {
        let b = budgets
            .lookup(reg.name)
            .ok_or(ConfigureError::Unconfigured(reg.name))?;
        sum = sum.saturating_add(b);
    }
    if sum > budgets.total {
        return Err(ConfigureError::OverBudget {
            requested: sum,
            total: budgets.total,
        });
    }

    // Build each sub-heap; store handles in a SYSTEM-backed vec, and leak each
    // SubHeap via a SYSTEM-backed box — no allocation here touches toccata.
    let mut map: crate::SysVec<(&'static str, &'static SubHeap)> =
        allocator_api2::vec::Vec::new_in(crate::Sys);
    for reg in REGISTRATIONS {
        let bytes = budgets.lookup(reg.name).unwrap();
        let sh = SubHeapBuilder::new(reg.name, bytes)
            .on_exhaust(reg.on_exhaust)
            .cap_per_class(reg.cap_per_class)
            .build_standalone()
            .map_err(|e| ConfigureError::Reserve(reg.name, e))?;
        let leaked: &'static SubHeap =
            allocator_api2::boxed::Box::leak(allocator_api2::boxed::Box::new_in(sh, crate::Sys));
        map.push((reg.name, leaked));
    }

    CONFIGURED
        .set(map)
        .map_err(|_| ConfigureError::AlreadyConfigured)?;
    Ok(())
}

/// Fetch a configured sub-heap by name. `None` before [`configure`] or for an
/// unregistered name.
pub fn subheap(name: &str) -> Option<&'static SubHeap> {
    CONFIGURED
        .get()?
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, sh)| *sh)
}

/// Iterate all configured sub-heaps (for the metrics layer).
pub fn all() -> impl Iterator<Item = (&'static str, &'static SubHeap)> {
    CONFIGURED
        .get()
        .into_iter()
        .flat_map(|m| m.iter().map(|(k, v)| (*k, *v)))
}

/// Whether the registry has been configured.
pub fn is_configured() -> bool {
    CONFIGURED.get().is_some()
}
