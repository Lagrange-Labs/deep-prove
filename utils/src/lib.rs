use std::{
    fmt::Display,
    time::{Duration, Instant},
};

use bytesize::ByteSize;
use memory_stats::{MemoryStats, memory_stats};
use thousands::Separable;

#[derive(Default)]
struct AllocatorMetrics {
    /// Total number of bytes allocated so far.
    allocated: usize,

    /// Total number of bytes deallocated so far.
    deallocated: usize,

    /// Number of alloc calls.
    alloc_calls: usize,

    /// Peak memory usage in bytes.
    ///
    /// Note: The peak memory usage can be reset, this is used to measure
    /// the memore usage in a span of time.
    peak: usize,
}

/// Generates memory flame graphs for the period of time this object is alive.
///
/// Data collection starts when the struct is created, and it finishes when the
/// struct is dropped.
///
/// Note: Flame graph generation takes a long time, the execution of the drop will
/// take a long time.
/// Note: To generate flamegraphs an additional environment variable called `FLAMEGRAPH`
/// must be present with a non-empty string, the variable's contents will be used
/// as the generate file prefix.
///
/// # Panics
///
/// Only one of this structures may exist at any time.
pub struct MemoryFlameGraph {}

impl MemoryFlameGraph {
    /// Starts memory flame graph collection
    ///
    /// # Panics
    ///
    /// If there is already a flame graph being collected.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        track::flame_graph_enable();
        Self {}
    }
}

impl Drop for MemoryFlameGraph {
    fn drop(&mut self) {
        track::flame_graph();
    }
}

#[cfg(feature = "mem-track")]
mod track {
    use std::{
        alloc::System,
        env,
        fs::OpenOptions,
        io::BufWriter,
        sync::atomic::{AtomicBool, Ordering},
        thread,
    };

    use mem_track::{
        flame::{FlameAlloc, format_flame_graph},
        peak::global::GlobalPeakTracker,
    };

    use crate::AllocatorMetrics;

    #[global_allocator]
    static ALLOCATOR: GlobalPeakTracker<FlameAlloc<System>> =
        GlobalPeakTracker::init(FlameAlloc::init(System));

    static IS_FLAME_GRAPH_ENABLED: AtomicBool = AtomicBool::new(false);

    /// Collects memory metrics from the custom global allocator.
    ///
    /// NOTE: This will also reset the memory allocator peak to current usage.
    pub(crate) fn allocator_metrics() -> Option<AllocatorMetrics> {
        let metrics = AllocatorMetrics {
            peak: ALLOCATOR.peak(),
            allocated: ALLOCATOR.allocated(),
            deallocated: ALLOCATOR.deallocated(),
            alloc_calls: ALLOCATOR.alloc_calls(),
        };
        ALLOCATOR.reset_peak();

        Some(metrics)
    }

    /// Enables the flame graph and clean any data.
    pub(crate) fn flame_graph_enable() {
        assert_eq!(
            Ok(false),
            IS_FLAME_GRAPH_ENABLED.compare_exchange(
                false,
                true,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ),
            "Can not have two flame graphs being collected at the same time",
        );
        let _graph = ALLOCATOR.inner().global_flame_graph();
        ALLOCATOR.inner().enable();
    }

    /// Save the flamegraph data to disk.
    ///
    /// NOTE: To enable flame graph generation a environment variable called `FLAMEGRAPH` is required.
    /// The contents of this variable defines the file's name prefix.
    ///
    /// # Panics
    ///
    /// If `flame_graph_enable` wasn't called.
    pub(crate) fn flame_graph() {
        if let Ok(file_prefix) = env::var("FLAMEGRAPH") {
            let graph = ALLOCATOR.inner().disable();
            assert_eq!(
                Ok(true),
                IS_FLAME_GRAPH_ENABLED.compare_exchange(
                    true,
                    false,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ),
                "Must have called flame_graph_enable",
            );
            let iterator = graph.iter();

            for i in 0..128 {
                if let Ok(file) = OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(format!("{file_prefix}_bytes_{i}.flame"))
                {
                    // Formatting the flame graph is an expensive operation, do it in parallel.
                    thread::scope(|s| {
                        let iterator = iterator.clone();
                        s.spawn(move || {
                            let mut file = BufWriter::new(file);
                            let _ = format_flame_graph(&mut file, iterator, |v| v.bytes_allocated);
                            if let Ok(file) = file.into_inner() {
                                let _ = file.sync_all();
                            }
                        });

                        if let Ok(file) = OpenOptions::new()
                            .write(true)
                            .create_new(true)
                            .open(format!("{file_prefix}_calls_{i}.flame"))
                        {
                            let mut file = BufWriter::new(file);
                            let _ = format_flame_graph(&mut file, graph.iter(), |v| v.alloc_calls);
                            if let Ok(file) = file.into_inner() {
                                let _ = file.sync_all();
                            }
                        }
                    });

                    break;
                }
            }
        }
    }
}

#[cfg(not(feature = "mem-track"))]
mod track {
    use crate::AllocatorMetrics;

    /// Collects memory metrics from the allocator.
    ///
    /// NOTE: This will also reset the memory allocator peak to current usage.
    pub(crate) fn allocator_metrics() -> Option<AllocatorMetrics> {
        None
    }

    pub(crate) fn flame_graph_enable() {
        // empty
    }

    pub(crate) fn flame_graph() {
        // empty
    }
}

#[must_use]
pub struct Metrics {
    /// When the measurement happened.
    when: Instant,

    /// Allocator metrics, if available, contains the exact number of bytes
    /// allocated/deallocated/inuse and peak memory usage as reported by the memory
    /// allocator.
    allocator_metrics: Option<AllocatorMetrics>,

    /// The memory stats, if available, constains the memory usage of the program as
    /// reported by the OS.
    ///
    /// This includes memory mapped files, stack space, and memory requested by the memory
    /// allocator.
    memory_stats: Option<MemoryStats>,
}

impl Metrics {
    /// Start measuring a time span.
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            when: Instant::now(),
            allocator_metrics: track::allocator_metrics(),
            memory_stats: memory_stats(),
        }
    }

    /// End the measuring span and collect metrics.
    pub fn to_span(self) -> MetricsSpan {
        let new = Self::new();
        MetricsSpan::from_measurements(self, new)
    }
}

pub struct MetricsSpan {
    pub physical_mem: Option<usize>,
    pub virtual_mem: Option<usize>,
    pub physical_mem_diff: Option<isize>,
    pub virtual_mem_diff: Option<isize>,
    pub allocated: Option<usize>,
    pub deallocated: Option<usize>,
    pub alloc_calls: Option<usize>,
    pub peak: Option<usize>,
    pub in_use: Option<usize>,
    pub elapsed: Duration,
}

impl MetricsSpan {
    fn from_measurements(old: Metrics, new: Metrics) -> Self {
        let allocated = new.allocator_metrics.as_ref().map(|v| v.allocated);
        let deallocated = new.allocator_metrics.as_ref().map(|v| v.deallocated);
        let peak = new.allocator_metrics.as_ref().map(|v| v.peak);
        let in_use = new
            .allocator_metrics
            .as_ref()
            .map(|v| v.allocated.saturating_sub(v.deallocated));
        let alloc_calls = new.allocator_metrics.as_ref().map(|v| v.alloc_calls);

        let elapsed = new.when.duration_since(old.when);

        match (old.memory_stats, new.memory_stats) {
            (Some(old_memory_stats), Some(new_memory_stats)) => {
                let physical_mem_diff = Some(
                    isize::try_from(new_memory_stats.physical_mem)
                        .expect("diff must fit in an isize")
                        - isize::try_from(old_memory_stats.physical_mem)
                            .expect("diff must fit in an isize"),
                );
                let virtual_mem_diff = Some(
                    isize::try_from(new_memory_stats.virtual_mem)
                        .expect("diff must fit in an isize")
                        - isize::try_from(old_memory_stats.virtual_mem)
                            .expect("diff must fit in an isize"),
                );
                Self {
                    physical_mem: Some(new_memory_stats.physical_mem),
                    virtual_mem: Some(new_memory_stats.virtual_mem),
                    physical_mem_diff,
                    virtual_mem_diff,
                    allocated,
                    deallocated,
                    alloc_calls,
                    peak,
                    in_use,
                    elapsed,
                }
            }
            (None, Some(new_memory_stats)) => Self {
                physical_mem: Some(new_memory_stats.physical_mem),
                virtual_mem: Some(new_memory_stats.virtual_mem),
                physical_mem_diff: None,
                virtual_mem_diff: None,
                allocated,
                deallocated,
                alloc_calls,
                peak,
                in_use,
                elapsed,
            },
            (None, None) | (Some(_), None) => Self {
                physical_mem: None,
                virtual_mem: None,
                physical_mem_diff: None,
                virtual_mem_diff: None,
                allocated,
                deallocated,
                alloc_calls,
                peak,
                in_use,
                elapsed,
            },
        }
    }
}

impl Display for MetricsSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "elapsed={:?}", self.elapsed)?;

        write!(f, " peak=")?;
        if let Some(peak) = self.peak {
            format_bytes_usize(f, peak)?;
        };

        write!(f, " in_use=")?;
        if let Some(in_use) = self.in_use {
            format_bytes_usize(f, in_use)?;
        };

        if f.alternate() {
            write!(f, " physical_mem=")?;
            if let Some(physical_mem) = self.physical_mem {
                format_bytes_usize(f, physical_mem)?;
            };

            write!(f, " virtual_mem=")?;
            if let Some(virtual_mem) = self.virtual_mem {
                format_bytes_usize(f, virtual_mem)?;
            };

            write!(f, " physical_mem_diff=")?;
            if let Some(physical_mem_diff) = self.physical_mem_diff {
                format_bytes_isize(f, physical_mem_diff)?;
            };

            write!(f, " virtual_mem_diff=")?;
            if let Some(virtual_mem_diff) = self.virtual_mem_diff {
                format_bytes_isize(f, virtual_mem_diff)?;
            };

            write!(f, " allocated=")?;
            if let Some(allocated) = self.allocated {
                format_bytes_usize(f, allocated)?;
            };

            write!(f, " deallocated=")?;
            if let Some(deallocated) = self.deallocated {
                format_bytes_usize(f, deallocated)?;
            };

            write!(f, " alloc_calls=")?;
            if let Some(alloc_calls) = self.alloc_calls {
                f.write_str(&alloc_calls.separate_with_commas())?;
            };
        }

        Ok(())
    }
}

fn format_bytes_usize(f: &mut std::fmt::Formatter<'_>, v: usize) -> std::fmt::Result {
    write!(
        f,
        "{}",
        ByteSize::b(v.try_into().expect("Should fit in a u64")).display()
    )
}

fn format_bytes_isize(f: &mut std::fmt::Formatter<'_>, v: isize) -> std::fmt::Result {
    let prefix = if v.is_negative() { "-" } else { "" };
    let formatter = ByteSize::b(v.abs().try_into().expect("Should fit in a u64")).display();
    write!(f, "{prefix}{formatter}")
}
