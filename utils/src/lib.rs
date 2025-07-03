use std::{
    borrow::Cow,
    time::{Duration, Instant},
};

use bytesize::ByteSize;
use memory_stats::{MemoryStats, memory_stats};
use thousands::Separable;
use tracing::info;

#[derive(Default)]
struct AllocatorMetrics {
    allocated: usize,
    deallocated: usize,
    num_alloc_calls: usize,
    peak: usize,
    in_use: usize,
}

impl AllocatorMetrics {
    #[cfg(feature = "mem-track")]
    fn with_peak_and_in_use(peak: usize, in_use: usize) -> Self {
        AllocatorMetrics {
            peak,
            in_use,
            ..Default::default()
        }
    }
}

#[cfg(feature = "mem-track")]
mod track {
    use std::alloc::System;

    use mem_track::peak::{global::GlobalPeakTracker, thread::BytesInUseTracker};

    use crate::AllocatorMetrics;

    #[global_allocator]
    static ALLOCATOR: BytesInUseTracker<GlobalPeakTracker<System>> =
        BytesInUseTracker::init(GlobalPeakTracker::init(System));

    pub(crate) fn allocator_metrics() -> Option<AllocatorMetrics> {
        let global = ALLOCATOR.inner();
        let metrics = AllocatorMetrics::with_peak_and_in_use(global.peak(), global.in_use());
        global.reset();

        let details = ALLOCATOR.reset_all_threads();
        let metrics = details.into_iter().fold(metrics, |mut state, data| {
            state.allocated += data.allocated;
            state.deallocated += data.deallocated;
            state.num_alloc_calls += data.num_alloc_calls;
            state
        });

        Some(metrics)
    }
}

#[cfg(not(feature = "mem-track"))]
mod track {
    use crate::AllocatorMetrics;

    pub(crate) fn allocator_metrics() -> Option<AllocatorMetrics> {
        None
    }
}

#[macro_export]
macro_rules! info_metrics {
    ($open: expr, $close: expr $(,)?) => {
        let _guard = $crate::MeasureStage::new($open.into(), $close.into()).guard();
    };
}

pub struct MemoryMetrics {
    pub physical_mem: Option<usize>,
    pub virtual_mem: Option<usize>,
    pub physical_mem_diff: Option<isize>,
    pub virtual_mem_diff: Option<isize>,
    pub allocated: Option<usize>,
    pub deallocated: Option<usize>,
    pub num_alloc_calls: Option<usize>,
    pub peak: Option<usize>,
    pub in_use: Option<usize>,
}

impl MemoryMetrics {
    fn from_measurements(
        old_memory_stats: Option<MemoryStats>,
        new_memory_stats: Option<MemoryStats>,
        allocator_metrics: Option<AllocatorMetrics>,
    ) -> Self {
        let allocated = allocator_metrics.as_ref().map(|v| v.allocated);
        let deallocated = allocator_metrics.as_ref().map(|v| v.deallocated);
        let peak = allocator_metrics.as_ref().map(|v| v.peak);
        let in_use = allocator_metrics.as_ref().map(|v| v.in_use);
        let num_alloc_calls = allocator_metrics.as_ref().map(|v| v.num_alloc_calls);

        match (old_memory_stats, new_memory_stats) {
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
                    num_alloc_calls,
                    peak,
                    in_use,
                }
            }
            (None, Some(new_memory_stats)) => Self {
                physical_mem: Some(new_memory_stats.physical_mem),
                virtual_mem: Some(new_memory_stats.virtual_mem),
                physical_mem_diff: None,
                virtual_mem_diff: None,
                allocated,
                deallocated,
                num_alloc_calls,
                peak,
                in_use,
            },
            (None, None) | (Some(_), None) => Self {
                physical_mem: None,
                virtual_mem: None,
                physical_mem_diff: None,
                virtual_mem_diff: None,
                allocated,
                deallocated,
                num_alloc_calls,
                peak,
                in_use,
            },
        }
    }
}

pub struct Metrics {
    pub elapsed: Duration,
    pub memory: MemoryMetrics,
}

#[must_use]
pub struct MeasureStage<'a> {
    close: Cow<'a, str>,
    start: Instant,
    memory_stats: Option<MemoryStats>,
}

pub struct Guard<'a> {
    measure: MeasureStage<'a>,
}

impl<'a> Drop for Guard<'a> {
    fn drop(&mut self) {
        let _ = self.measure.metrics();
    }
}

fn format_bytes_usize(v: Option<usize>) -> Option<String> {
    v.map(|v| {
        let formatter = ByteSize::b(v.try_into().expect("Should fit in a u64")).display();
        format!("{}", formatter)
    })
}

fn format_bytes_isize(v: Option<isize>) -> Option<String> {
    v.map(|v| {
        let prefix = if v.is_negative() { "-" } else { "" };
        let formatter = ByteSize::b(v.abs().try_into().expect("Should fit in a u64")).display();
        format!("{}{}", prefix, formatter)
    })
}

impl<'a> MeasureStage<'a> {
    /// Start measuring a time span.
    pub fn new(start: Cow<'_, str>, close: Cow<'a, str>) -> Self {
        // Reset the metrics and report the current values. This is done so the
        // span will report the memory accumulated in its lifetime
        //
        // NOTE: Because of this nesting spans is currently not supported. That should
        // be possible to add with a stack of measurements.
        let allocator_metrics = track::allocator_metrics();
        let memory_stats = memory_stats();
        let memory = MemoryMetrics::from_measurements(None, memory_stats, allocator_metrics);

        info!(
            physical_mem = format_bytes_usize(memory.physical_mem),
            virtual_mem = format_bytes_usize(memory.virtual_mem),
            allocated = format_bytes_usize(memory.allocated),
            deallocated = format_bytes_usize(memory.deallocated),
            num_alloc_calls = memory.num_alloc_calls.map(|v| v.separate_with_commas()),
            peak = format_bytes_usize(memory.peak),
            in_use = format_bytes_usize(memory.in_use),
            "{start}"
        );

        Self {
            close,
            start: Instant::now(),
            memory_stats,
        }
    }

    /// End the measuring span and collect metrics.
    pub fn into_metrics(self) -> Metrics {
        self.metrics()
    }

    /// Consumes the metric and returns a guard.
    ///
    /// The guard will produce a log line with the metrics when dropped.
    pub fn guard(self) -> Guard<'a> {
        Guard { measure: self }
    }

    /// End the measuring span and collect metrics.
    pub fn metrics(&self) -> Metrics {
        let elapsed = self.start.elapsed();
        let memory = MemoryMetrics::from_measurements(
            self.memory_stats,
            memory_stats(),
            track::allocator_metrics(),
        );

        info!(
            physical_mem = format_bytes_usize(memory.physical_mem),
            virtual_mem = format_bytes_usize(memory.virtual_mem),
            physical_mem_diff = format_bytes_isize(memory.physical_mem_diff),
            virtual_mem_diff = format_bytes_isize(memory.virtual_mem_diff),
            allocated = format_bytes_usize(memory.allocated),
            deallocated = format_bytes_usize(memory.deallocated),
            num_alloc_calls = memory.num_alloc_calls.map(|v|v.separate_with_commas()),
            peak = format_bytes_usize(memory.peak),
            in_use = format_bytes_usize(memory.in_use),
            elapsed = ?elapsed,
            "{}",
            self.close
        );

        Metrics { elapsed, memory }
    }
}

#[cfg(test)]
mod tests {
    use crate::MeasureStage;

    #[test]
    fn test_measure() {
        let m = MeasureStage::new("start".into(), "end".into());
        m.metrics();
    }
}
