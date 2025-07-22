use std::{
    borrow::Cow,
    collections::BTreeMap,
    fmt::Display,
    fs::{File, OpenOptions},
    io::{self, Write},
    ops::DerefMut,
    path::Path,
    sync::Mutex,
    time::{Duration, Instant},
};

use anyhow::Context;
use bytesize::ByteSize;
use csv::{Writer, WriterBuilder};
use memory_stats::{MemoryStats, memory_stats};
use thiserror::Error;
use thousands::Separable;
use tracing::warn;

type CsvStreamer = StreamingRecorder<Writer<File>>;
static CSV_RECORDER: Mutex<Option<CsvStreamer>> = Mutex::new(None);

pub fn init_csv_recorder<I, T, P>(additional_columns: I, file: P) -> anyhow::Result<()>
where
    I: IntoIterator<Item = T>,
    T: AsRef<str>,
    P: AsRef<Path>,
{
    let mut lock = CSV_RECORDER.lock().expect("Should not be poison");
    if lock.is_some() {
        warn!("CSV_RECORDER is already initialised");
        return Ok(());
    }

    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&file)
        .context(format!("opening file {:?}", file.as_ref()))?;

    *lock = Some(StreamingRecorder::csv_streaming(additional_columns, file)?);
    Ok(())
}

/// Stream the metrics out.
pub fn stream_metrics(name: impl AsRef<str>, metrics: &MetricsSpan) {
    let mut lock = CSV_RECORDER.lock().expect("Should not be poison");
    match lock.deref_mut() {
        Some(stream) => {
            match stream.stream_metrics(name, metrics) {
                Ok(_) => {}
                Err(err) => {
                    warn!("Error while streaming via CSV_RECORDER. err: {err:?}");
                }
            };
        }
        None => warn!("CSV_RECORDER is not initialised"),
    }
}

pub fn stream_data<K, V, I>(name: impl AsRef<str>, metrics: &MetricsSpan, data: I)
where
    K: AsRef<str>,
    V: ToString,
    I: IntoIterator<Item = (K, V)>,
{
    let mut lock = CSV_RECORDER.lock().expect("Should not be poison");
    match lock.deref_mut() {
        Some(stream) => {
            match stream.stream_data(name, metrics, data) {
                Ok(_) => {}
                Err(err) => {
                    warn!("Error while streaming via CSV_RECORDER. err: {err:?}");
                }
            };
        }
        None => warn!("CSV_RECORDER is not initialised"),
    }
}

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
        io::{BufWriter, Write},
        sync::atomic::{AtomicBool, Ordering},
        thread,
    };

    use flate2::{Compression, write::GzEncoder};
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
                    .open(format!("{file_prefix}_bytes_{i}.flame.gz"))
                {
                    // Formatting the flame graph is an expensive operation, do it in parallel.
                    thread::scope(|s| {
                        let iterator = iterator.clone();
                        s.spawn(move || {
                            let mut file =
                                BufWriter::new(GzEncoder::new(file, Compression::default()));
                            let _ = format_flame_graph(&mut file, iterator, |v| v.bytes_allocated);
                            let _ = file.flush();
                        });

                        if let Ok(file) = OpenOptions::new()
                            .write(true)
                            .create_new(true)
                            .open(format!("{file_prefix}_calls_{i}.flame.gz"))
                        {
                            let mut file =
                                BufWriter::new(GzEncoder::new(file, Compression::default()));
                            let _ = format_flame_graph(&mut file, graph.iter(), |v| v.alloc_calls);
                            let _ = file.flush();
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

/// Collects profilling data.
pub struct StreamingRecorder<W> {
    /// The list of allowed column names.
    ///
    /// The data is streamed out, the column names must be known a prior.
    column_names: Vec<String>,

    /// The data is streamed to this writer.
    writer: W,
}

#[derive(Debug, Error)]
pub enum RecorderError<E> {
    /// Record names must be unique, found a duplicate.
    #[error("Found duplicate column name: {0}")]
    DuplicatedName(String),

    /// User tried to write to a unknown column name.
    #[error("Unknown column during metrics streaming: {0}")]
    UnkownName(String),

    /// Error while writing the data..
    #[error("Failed to write data, err: {0}")]
    WriteErr(E),

    /// Error while streaming the data.
    #[error("Io error, err: {0}")]
    IoErr(io::Error),
}

fn format_field<T: ToString>(v: Option<T>) -> Cow<'static, str> {
    v.map(|v| v.to_string().into()).unwrap_or("".into())
}

impl<W: Write> StreamingRecorder<Writer<W>> {
    const FIXED_COLUMNS: usize = 11;

    /// Creates a new [StreamingRecorder] with the configured `additional_column` streaming data
    /// csv data to the given [Write] sink.
    pub fn csv_streaming<I, T>(
        additional_columns: I,
        write: W,
    ) -> Result<Self, RecorderError<csv::Error>>
    where
        I: IntoIterator<Item = T>,
        T: AsRef<str>,
    {
        // The column_names determines the order of the output, try to allocate the columns in
        // a human friendly way.
        //
        // NOTE: Keep in sync with the order on the other methods
        let mut column_names = vec!["name".to_string(), "elapsed".to_string()];
        column_names.extend(
            additional_columns
                .into_iter()
                .map(|s| s.as_ref().to_string()),
        );
        let additional_columns_count = column_names.len() - 2;
        column_names.extend([
            "in_use".to_string(),
            "peak".to_string(),
            "physical_mem".to_string(),
            "virtual_mem".to_string(),
            "physical_mem_diff".to_string(),
            "virtual_mem_diff".to_string(),
            "allocated".to_string(),
            "deallocated".to_string(),
            "alloc_calls".to_string(),
        ]);

        debug_assert!(Self::FIXED_COLUMNS + additional_columns_count == column_names.len());

        let counts = column_names
            .iter()
            .fold(BTreeMap::<&String, u32>::new(), |mut s, v| {
                *s.entry(v).or_default() += 1;
                s
            });
        if let Some(duplicated) = counts.iter().find(|(_key, value)| **value > 1) {
            return Err(RecorderError::DuplicatedName(duplicated.0.to_string()));
        }

        let mut writer: Writer<W> = WriterBuilder::new().flexible(false).from_writer(write);

        // write the header out
        writer
            .write_record(&column_names)
            .map_err(RecorderError::WriteErr)?;

        Ok(Self {
            column_names,
            writer,
        })
    }

    /// Stream the metrics out.
    pub fn stream_metrics(
        &mut self,
        name: impl AsRef<str>,
        metrics: &MetricsSpan,
    ) -> Result<(), RecorderError<csv::Error>> {
        // NOTE: Keep in sync with order in the constructor
        let mut record: Vec<Cow<'_, str>> = Vec::with_capacity(self.column_names.len());

        record.push(name.as_ref().into());
        record.push(metrics.elapsed.as_millis().to_string().into());

        let user_columns = self.column_names.len() - Self::FIXED_COLUMNS;
        for _i in 0..user_columns {
            record.push("".into());
        }

        record.push(format_field(metrics.physical_mem));
        record.push(format_field(metrics.virtual_mem));
        record.push(format_field(metrics.physical_mem_diff));
        record.push(format_field(metrics.virtual_mem_diff));
        record.push(format_field(metrics.allocated));
        record.push(format_field(metrics.deallocated));
        record.push(format_field(metrics.alloc_calls));
        record.push(format_field(metrics.peak));
        record.push(format_field(metrics.in_use));

        self.writer
            .write_record(record.iter().map(|v| v.as_bytes()))
            .map_err(RecorderError::WriteErr)?;
        self.writer.flush().map_err(RecorderError::IoErr)?;

        Ok(())
    }

    /// Stream data out.
    pub fn stream_data<K, V, I>(
        &mut self,
        name: impl AsRef<str>,
        metrics: &MetricsSpan,
        data: I,
    ) -> Result<(), RecorderError<csv::Error>>
    where
        K: AsRef<str>,
        V: ToString,
        I: IntoIterator<Item = (K, V)>,
    {
        // NOTE: Keep in sync with order in the constructor
        let mut record: Vec<Cow<'_, str>> = Vec::with_capacity(self.column_names.len());
        record.push(name.as_ref().into());
        record.push(metrics.elapsed.as_millis().to_string().into());

        let new_length = self.column_names.len() - Self::FIXED_COLUMNS + 2;
        record.resize_with(new_length, || "".into());

        for (name, value) in data.into_iter() {
            let pos = self
                .column_names
                .iter()
                .position(|column_name| name.as_ref() == column_name)
                .ok_or_else(|| RecorderError::UnkownName(name.as_ref().to_string()))?;

            let storage = record
                .get_mut(pos)
                .expect("Position returned form a valid iterator above");
            *storage = value.to_string().into();
        }

        record.push(format_field(metrics.physical_mem));
        record.push(format_field(metrics.virtual_mem));
        record.push(format_field(metrics.physical_mem_diff));
        record.push(format_field(metrics.virtual_mem_diff));
        record.push(format_field(metrics.allocated));
        record.push(format_field(metrics.deallocated));
        record.push(format_field(metrics.alloc_calls));
        record.push(format_field(metrics.peak));
        record.push(format_field(metrics.in_use));

        self.writer
            .write_record(record.iter().map(|v| v.as_bytes()))
            .map_err(RecorderError::WriteErr)?;
        self.writer.flush().map_err(RecorderError::IoErr)?;

        Ok(())
    }
}
