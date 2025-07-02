use std::{
    borrow::Cow,
    time::{Duration, Instant},
};

use memory_stats::{MemoryStats, memory_stats};
use tracing::info;

#[macro_export]

macro_rules! info_metrics {
    ($open: expr, $close: expr $(,)?) => {
        let _guard = utils::MeasureStage::new($open.into(), $close.into()).guard();
    };
}

pub struct MemoryMetrics {
    pub physical_mem: Option<usize>,
    pub virtual_mem: Option<usize>,
    pub physical_mem_diff: Option<usize>,
    pub virtual_mem_diff: Option<usize>,
}

impl MemoryMetrics {
    fn from_measurements(
        old_memory_stats: Option<MemoryStats>,
        new_memory_stats: Option<MemoryStats>,
    ) -> Self {
        match (old_memory_stats, new_memory_stats) {
            (Some(old_memory_stats), Some(new_memory_stats)) => Self {
                physical_mem: Some(new_memory_stats.physical_mem),
                virtual_mem: Some(new_memory_stats.virtual_mem),
                physical_mem_diff: Some(
                    new_memory_stats.physical_mem - old_memory_stats.physical_mem,
                ),
                virtual_mem_diff: Some(new_memory_stats.virtual_mem - old_memory_stats.virtual_mem),
            },
            (None, Some(new_memory_stats)) => Self {
                physical_mem: Some(new_memory_stats.physical_mem),
                virtual_mem: Some(new_memory_stats.virtual_mem),
                physical_mem_diff: None,
                virtual_mem_diff: None,
            },
            (None, None) | (Some(_), None) => Self {
                physical_mem: None,
                virtual_mem: None,
                physical_mem_diff: None,
                virtual_mem_diff: None,
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

impl<'a> MeasureStage<'a> {
    /// Start measuring a time span.
    pub fn new(start: Cow<'_, str>, close: Cow<'a, str>) -> Self {
        let memory_stats = memory_stats();
        let memory = MemoryMetrics::from_measurements(None, memory_stats);

        info!(
            physical_mem = memory.physical_mem,
            virtual_mem = memory.virtual_mem,
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
        let memory = MemoryMetrics::from_measurements(self.memory_stats, memory_stats());

        info!(
            physical_mem = memory.physical_mem,
            virtual_mem = memory.virtual_mem,
            physical_mem_diff = memory.physical_mem_diff,
            virtual_mem_diff = memory.virtual_mem_diff,
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
