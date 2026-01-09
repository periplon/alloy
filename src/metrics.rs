//! Prometheus-compatible metrics for Alloy MCP server.
//!
//! This module provides observability metrics for monitoring
//! the Alloy indexing and search operations.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Global metrics instance.
static METRICS: std::sync::OnceLock<Arc<Metrics>> = std::sync::OnceLock::new();

/// Get or initialize the global metrics instance.
pub fn get_metrics() -> Arc<Metrics> {
    METRICS.get_or_init(|| Arc::new(Metrics::new())).clone()
}

/// Reset the global metrics (mainly for testing).
#[cfg(test)]
pub fn reset_metrics() {
    if let Some(m) = METRICS.get() {
        m.reset();
    }
}

/// Counter metric that can only increase.
#[derive(Debug, Default)]
pub struct Counter {
    value: AtomicU64,
}

impl Counter {
    /// Create a new counter.
    pub fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }

    /// Increment the counter by 1.
    pub fn inc(&self) {
        self.add(1);
    }

    /// Add a value to the counter.
    pub fn add(&self, n: u64) {
        self.value.fetch_add(n, Ordering::Relaxed);
    }

    /// Get the current value.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Reset the counter to 0.
    pub fn reset(&self) {
        self.value.store(0, Ordering::Relaxed);
    }
}

/// Gauge metric that can go up and down.
#[derive(Debug, Default)]
pub struct Gauge {
    value: AtomicU64,
}

impl Gauge {
    /// Create a new gauge.
    pub fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }

    /// Set the gauge value.
    pub fn set(&self, value: u64) {
        self.value.store(value, Ordering::Relaxed);
    }

    /// Increment the gauge by 1.
    pub fn inc(&self) {
        self.add(1);
    }

    /// Decrement the gauge by 1.
    pub fn dec(&self) {
        self.sub(1);
    }

    /// Add a value to the gauge.
    pub fn add(&self, n: u64) {
        self.value.fetch_add(n, Ordering::Relaxed);
    }

    /// Subtract a value from the gauge.
    pub fn sub(&self, n: u64) {
        self.value.fetch_sub(n, Ordering::Relaxed);
    }

    /// Get the current value.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Reset the gauge to 0.
    pub fn reset(&self) {
        self.value.store(0, Ordering::Relaxed);
    }
}

/// Float gauge for values that need floating point precision.
#[derive(Debug, Default)]
pub struct FloatGauge {
    /// Using an atomic u64 to store the bits of an f64.
    bits: AtomicU64,
}

impl FloatGauge {
    /// Create a new float gauge.
    pub fn new() -> Self {
        Self {
            bits: AtomicU64::new(0.0f64.to_bits()),
        }
    }

    /// Set the gauge value.
    pub fn set(&self, value: f64) {
        self.bits.store(value.to_bits(), Ordering::Relaxed);
    }

    /// Get the current value.
    pub fn get(&self) -> f64 {
        f64::from_bits(self.bits.load(Ordering::Relaxed))
    }

    /// Reset to 0.
    pub fn reset(&self) {
        self.set(0.0);
    }
}

/// Histogram for tracking value distributions.
#[derive(Debug)]
pub struct Histogram {
    /// Bucket boundaries.
    buckets: Vec<f64>,
    /// Count of observations in each bucket.
    bucket_counts: Vec<AtomicU64>,
    /// Sum of all observed values.
    sum: AtomicU64,
    /// Count of all observations.
    count: AtomicU64,
}

impl Histogram {
    /// Create a new histogram with default buckets for latency tracking.
    pub fn new_latency() -> Self {
        // Default latency buckets in milliseconds
        Self::new(vec![
            1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0,
        ])
    }

    /// Create a new histogram with custom buckets.
    pub fn new(buckets: Vec<f64>) -> Self {
        let bucket_counts = buckets.iter().map(|_| AtomicU64::new(0)).collect();
        Self {
            buckets,
            bucket_counts,
            sum: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// Observe a value.
    pub fn observe(&self, value: f64) {
        // Update sum and count
        let current_sum = f64::from_bits(self.sum.load(Ordering::Relaxed));
        self.sum
            .store((current_sum + value).to_bits(), Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);

        // Update bucket counts
        for (i, bucket) in self.buckets.iter().enumerate() {
            if value <= *bucket {
                self.bucket_counts[i].fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Observe a duration in milliseconds.
    pub fn observe_duration(&self, duration: Duration) {
        self.observe(duration.as_secs_f64() * 1000.0);
    }

    /// Start a timer that records duration on drop.
    pub fn start_timer(&self) -> HistogramTimer<'_> {
        HistogramTimer {
            histogram: self,
            start: Instant::now(),
        }
    }

    /// Get the sum of all observations.
    pub fn sum(&self) -> f64 {
        f64::from_bits(self.sum.load(Ordering::Relaxed))
    }

    /// Get the count of all observations.
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get bucket counts with their boundaries.
    pub fn buckets(&self) -> Vec<(f64, u64)> {
        self.buckets
            .iter()
            .zip(self.bucket_counts.iter())
            .map(|(b, c)| (*b, c.load(Ordering::Relaxed)))
            .collect()
    }

    /// Reset the histogram.
    pub fn reset(&self) {
        self.sum.store(0.0f64.to_bits(), Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
        for c in &self.bucket_counts {
            c.store(0, Ordering::Relaxed);
        }
    }
}

/// Timer that records duration to a histogram when dropped.
pub struct HistogramTimer<'a> {
    histogram: &'a Histogram,
    start: Instant,
}

impl<'a> Drop for HistogramTimer<'a> {
    fn drop(&mut self) {
        self.histogram.observe_duration(self.start.elapsed());
    }
}

impl<'a> HistogramTimer<'a> {
    /// Get the elapsed time without stopping the timer.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Stop the timer and return the elapsed duration.
    pub fn stop(self) -> Duration {
        self.start.elapsed()
        // Duration is recorded in drop
    }
}

/// All metrics for the Alloy server.
#[derive(Debug)]
pub struct Metrics {
    // =========================================================================
    // Counters
    // =========================================================================
    /// Total number of documents indexed.
    pub documents_indexed_total: Counter,
    /// Total number of search queries executed.
    pub search_queries_total: Counter,
    /// Total number of indexing errors.
    pub indexing_errors_total: Counter,
    /// Total number of search errors.
    pub search_errors_total: Counter,
    /// Total number of chunks created.
    pub chunks_created_total: Counter,
    /// Total number of embeddings generated.
    pub embeddings_generated_total: Counter,
    /// Total number of duplicates detected.
    pub duplicates_detected_total: Counter,
    /// Total number of cache hits.
    pub cache_hits_total: Counter,
    /// Total number of cache misses.
    pub cache_misses_total: Counter,

    // =========================================================================
    // Gauges
    // =========================================================================
    /// Current number of indexed documents.
    pub documents_count: Gauge,
    /// Current number of chunks.
    pub chunks_count: Gauge,
    /// Index storage size in bytes.
    pub index_size_bytes: Gauge,
    /// Number of items in embedding queue.
    pub embedding_queue_size: Gauge,
    /// Number of active sources.
    pub sources_count: Gauge,
    /// Number of active file watchers.
    pub watchers_count: Gauge,
    /// Uptime in seconds.
    pub uptime_seconds: Gauge,

    // =========================================================================
    // Histograms
    // =========================================================================
    /// Search query duration in milliseconds.
    pub search_duration_ms: Histogram,
    /// Indexing duration per document in milliseconds.
    pub indexing_duration_ms: Histogram,
    /// Embedding generation duration in milliseconds.
    pub embedding_duration_ms: Histogram,
    /// Reranking duration in milliseconds.
    pub reranking_duration_ms: Histogram,
    /// Query expansion duration in milliseconds.
    pub query_expansion_duration_ms: Histogram,

    /// Server start time.
    start_time: RwLock<Instant>,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    /// Create a new metrics instance.
    pub fn new() -> Self {
        Self {
            // Counters
            documents_indexed_total: Counter::new(),
            search_queries_total: Counter::new(),
            indexing_errors_total: Counter::new(),
            search_errors_total: Counter::new(),
            chunks_created_total: Counter::new(),
            embeddings_generated_total: Counter::new(),
            duplicates_detected_total: Counter::new(),
            cache_hits_total: Counter::new(),
            cache_misses_total: Counter::new(),

            // Gauges
            documents_count: Gauge::new(),
            chunks_count: Gauge::new(),
            index_size_bytes: Gauge::new(),
            embedding_queue_size: Gauge::new(),
            sources_count: Gauge::new(),
            watchers_count: Gauge::new(),
            uptime_seconds: Gauge::new(),

            // Histograms
            search_duration_ms: Histogram::new_latency(),
            indexing_duration_ms: Histogram::new_latency(),
            embedding_duration_ms: Histogram::new_latency(),
            reranking_duration_ms: Histogram::new_latency(),
            query_expansion_duration_ms: Histogram::new_latency(),

            start_time: RwLock::new(Instant::now()),
        }
    }

    /// Reset all metrics.
    pub fn reset(&self) {
        // Counters
        self.documents_indexed_total.reset();
        self.search_queries_total.reset();
        self.indexing_errors_total.reset();
        self.search_errors_total.reset();
        self.chunks_created_total.reset();
        self.embeddings_generated_total.reset();
        self.duplicates_detected_total.reset();
        self.cache_hits_total.reset();
        self.cache_misses_total.reset();

        // Gauges
        self.documents_count.reset();
        self.chunks_count.reset();
        self.index_size_bytes.reset();
        self.embedding_queue_size.reset();
        self.sources_count.reset();
        self.watchers_count.reset();
        self.uptime_seconds.reset();

        // Histograms
        self.search_duration_ms.reset();
        self.indexing_duration_ms.reset();
        self.embedding_duration_ms.reset();
        self.reranking_duration_ms.reset();
        self.query_expansion_duration_ms.reset();

        *self.start_time.write() = Instant::now();
    }

    /// Update the uptime gauge.
    pub fn update_uptime(&self) {
        let uptime = self.start_time.read().elapsed();
        self.uptime_seconds.set(uptime.as_secs());
    }

    /// Export metrics in Prometheus text format.
    pub fn export_prometheus(&self) -> String {
        self.update_uptime();
        let mut output = String::new();

        // Helper to add metric
        macro_rules! add_counter {
            ($name:expr, $help:expr, $metric:expr) => {
                output.push_str(&format!("# HELP {} {}\n", $name, $help));
                output.push_str(&format!("# TYPE {} counter\n", $name));
                output.push_str(&format!("{} {}\n", $name, $metric.get()));
            };
        }

        macro_rules! add_gauge {
            ($name:expr, $help:expr, $metric:expr) => {
                output.push_str(&format!("# HELP {} {}\n", $name, $help));
                output.push_str(&format!("# TYPE {} gauge\n", $name));
                output.push_str(&format!("{} {}\n", $name, $metric.get()));
            };
        }

        macro_rules! add_histogram {
            ($name:expr, $help:expr, $metric:expr) => {
                output.push_str(&format!("# HELP {} {}\n", $name, $help));
                output.push_str(&format!("# TYPE {} histogram\n", $name));
                let mut cumulative = 0u64;
                for (bucket, count) in $metric.buckets() {
                    cumulative += count;
                    output.push_str(&format!(
                        "{}_bucket{{le=\"{}\"}} {}\n",
                        $name, bucket, cumulative
                    ));
                }
                output.push_str(&format!(
                    "{}_bucket{{le=\"+Inf\"}} {}\n",
                    $name,
                    $metric.count()
                ));
                output.push_str(&format!("{}_sum {}\n", $name, $metric.sum()));
                output.push_str(&format!("{}_count {}\n", $name, $metric.count()));
            };
        }

        // Counters
        add_counter!(
            "alloy_documents_indexed_total",
            "Total number of documents indexed",
            self.documents_indexed_total
        );
        add_counter!(
            "alloy_search_queries_total",
            "Total number of search queries executed",
            self.search_queries_total
        );
        add_counter!(
            "alloy_indexing_errors_total",
            "Total number of indexing errors",
            self.indexing_errors_total
        );
        add_counter!(
            "alloy_search_errors_total",
            "Total number of search errors",
            self.search_errors_total
        );
        add_counter!(
            "alloy_chunks_created_total",
            "Total number of chunks created",
            self.chunks_created_total
        );
        add_counter!(
            "alloy_embeddings_generated_total",
            "Total number of embeddings generated",
            self.embeddings_generated_total
        );
        add_counter!(
            "alloy_duplicates_detected_total",
            "Total number of duplicate documents detected",
            self.duplicates_detected_total
        );
        add_counter!(
            "alloy_cache_hits_total",
            "Total number of cache hits",
            self.cache_hits_total
        );
        add_counter!(
            "alloy_cache_misses_total",
            "Total number of cache misses",
            self.cache_misses_total
        );

        // Gauges
        add_gauge!(
            "alloy_documents_count",
            "Current number of indexed documents",
            self.documents_count
        );
        add_gauge!(
            "alloy_chunks_count",
            "Current number of chunks in index",
            self.chunks_count
        );
        add_gauge!(
            "alloy_index_size_bytes",
            "Index storage size in bytes",
            self.index_size_bytes
        );
        add_gauge!(
            "alloy_embedding_queue_size",
            "Number of items in embedding queue",
            self.embedding_queue_size
        );
        add_gauge!(
            "alloy_sources_count",
            "Number of active indexed sources",
            self.sources_count
        );
        add_gauge!(
            "alloy_watchers_count",
            "Number of active file watchers",
            self.watchers_count
        );
        add_gauge!(
            "alloy_uptime_seconds",
            "Server uptime in seconds",
            self.uptime_seconds
        );

        // Histograms
        add_histogram!(
            "alloy_search_duration_ms",
            "Search query duration in milliseconds",
            self.search_duration_ms
        );
        add_histogram!(
            "alloy_indexing_duration_ms",
            "Document indexing duration in milliseconds",
            self.indexing_duration_ms
        );
        add_histogram!(
            "alloy_embedding_duration_ms",
            "Embedding generation duration in milliseconds",
            self.embedding_duration_ms
        );
        add_histogram!(
            "alloy_reranking_duration_ms",
            "Reranking duration in milliseconds",
            self.reranking_duration_ms
        );
        add_histogram!(
            "alloy_query_expansion_duration_ms",
            "Query expansion duration in milliseconds",
            self.query_expansion_duration_ms
        );

        output
    }

    /// Export metrics as JSON.
    pub fn export_json(&self) -> MetricsSnapshot {
        self.update_uptime();
        MetricsSnapshot {
            counters: MetricsCounters {
                documents_indexed_total: self.documents_indexed_total.get(),
                search_queries_total: self.search_queries_total.get(),
                indexing_errors_total: self.indexing_errors_total.get(),
                search_errors_total: self.search_errors_total.get(),
                chunks_created_total: self.chunks_created_total.get(),
                embeddings_generated_total: self.embeddings_generated_total.get(),
                duplicates_detected_total: self.duplicates_detected_total.get(),
                cache_hits_total: self.cache_hits_total.get(),
                cache_misses_total: self.cache_misses_total.get(),
            },
            gauges: MetricsGauges {
                documents_count: self.documents_count.get(),
                chunks_count: self.chunks_count.get(),
                index_size_bytes: self.index_size_bytes.get(),
                embedding_queue_size: self.embedding_queue_size.get(),
                sources_count: self.sources_count.get(),
                watchers_count: self.watchers_count.get(),
                uptime_seconds: self.uptime_seconds.get(),
            },
            histograms: MetricsHistograms {
                search_duration_ms: HistogramSnapshot::from(&self.search_duration_ms),
                indexing_duration_ms: HistogramSnapshot::from(&self.indexing_duration_ms),
                embedding_duration_ms: HistogramSnapshot::from(&self.embedding_duration_ms),
                reranking_duration_ms: HistogramSnapshot::from(&self.reranking_duration_ms),
                query_expansion_duration_ms: HistogramSnapshot::from(
                    &self.query_expansion_duration_ms,
                ),
            },
        }
    }
}

/// Snapshot of all metrics for serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub counters: MetricsCounters,
    pub gauges: MetricsGauges,
    pub histograms: MetricsHistograms,
}

/// Counter metrics snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsCounters {
    pub documents_indexed_total: u64,
    pub search_queries_total: u64,
    pub indexing_errors_total: u64,
    pub search_errors_total: u64,
    pub chunks_created_total: u64,
    pub embeddings_generated_total: u64,
    pub duplicates_detected_total: u64,
    pub cache_hits_total: u64,
    pub cache_misses_total: u64,
}

/// Gauge metrics snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsGauges {
    pub documents_count: u64,
    pub chunks_count: u64,
    pub index_size_bytes: u64,
    pub embedding_queue_size: u64,
    pub sources_count: u64,
    pub watchers_count: u64,
    pub uptime_seconds: u64,
}

/// Histogram metrics snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsHistograms {
    pub search_duration_ms: HistogramSnapshot,
    pub indexing_duration_ms: HistogramSnapshot,
    pub embedding_duration_ms: HistogramSnapshot,
    pub reranking_duration_ms: HistogramSnapshot,
    pub query_expansion_duration_ms: HistogramSnapshot,
}

/// Snapshot of a histogram for serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramSnapshot {
    pub count: u64,
    pub sum: f64,
    pub buckets: Vec<(f64, u64)>,
    pub mean: Option<f64>,
}

impl From<&Histogram> for HistogramSnapshot {
    fn from(h: &Histogram) -> Self {
        let count = h.count();
        let sum = h.sum();
        let mean = if count > 0 {
            Some(sum / count as f64)
        } else {
            None
        };
        Self {
            count,
            sum,
            buckets: h.buckets(),
            mean,
        }
    }
}

/// Health status for the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: HealthState,
    pub version: String,
    pub uptime_seconds: u64,
    pub checks: Vec<HealthCheck>,
}

/// Health state enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HealthState {
    Healthy,
    Degraded,
    Unhealthy,
}

impl HealthState {
    /// Convert to HTTP status code.
    pub fn to_status_code(self) -> u16 {
        match self {
            HealthState::Healthy => 200,
            HealthState::Degraded => 200, // Still operational
            HealthState::Unhealthy => 503,
        }
    }
}

/// Individual health check result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub name: String,
    pub status: HealthState,
    pub message: Option<String>,
    pub duration_ms: Option<u64>,
}

impl HealthCheck {
    /// Create a healthy check.
    pub fn healthy(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: HealthState::Healthy,
            message: None,
            duration_ms: None,
        }
    }

    /// Create a healthy check with duration.
    pub fn healthy_with_duration(name: impl Into<String>, duration_ms: u64) -> Self {
        Self {
            name: name.into(),
            status: HealthState::Healthy,
            message: None,
            duration_ms: Some(duration_ms),
        }
    }

    /// Create a degraded check.
    pub fn degraded(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: HealthState::Degraded,
            message: Some(message.into()),
            duration_ms: None,
        }
    }

    /// Create an unhealthy check.
    pub fn unhealthy(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: HealthState::Unhealthy,
            message: Some(message.into()),
            duration_ms: None,
        }
    }
}

/// Readiness status for the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadinessStatus {
    pub ready: bool,
    pub message: Option<String>,
}

impl ReadinessStatus {
    /// Create a ready status.
    pub fn ready() -> Self {
        Self {
            ready: true,
            message: None,
        }
    }

    /// Create a not-ready status.
    pub fn not_ready(message: impl Into<String>) -> Self {
        Self {
            ready: false,
            message: Some(message.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter() {
        let counter = Counter::new();
        assert_eq!(counter.get(), 0);
        counter.inc();
        assert_eq!(counter.get(), 1);
        counter.add(5);
        assert_eq!(counter.get(), 6);
    }

    #[test]
    fn test_gauge() {
        let gauge = Gauge::new();
        assert_eq!(gauge.get(), 0);
        gauge.set(10);
        assert_eq!(gauge.get(), 10);
        gauge.inc();
        assert_eq!(gauge.get(), 11);
        gauge.dec();
        assert_eq!(gauge.get(), 10);
    }

    #[test]
    fn test_histogram() {
        let hist = Histogram::new_latency();
        hist.observe(5.0);
        hist.observe(25.0);
        hist.observe(100.0);

        assert_eq!(hist.count(), 3);
        assert!((hist.sum() - 130.0).abs() < 0.001);
    }

    #[test]
    fn test_histogram_timer() {
        let hist = Histogram::new_latency();
        {
            let _timer = hist.start_timer();
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(hist.count() > 0);
        assert!(hist.sum() >= 10.0);
    }

    #[test]
    fn test_prometheus_export() {
        let metrics = Metrics::new();
        metrics.documents_indexed_total.add(100);
        metrics.search_queries_total.add(50);
        metrics.documents_count.set(100);

        let output = metrics.export_prometheus();
        assert!(output.contains("alloy_documents_indexed_total 100"));
        assert!(output.contains("alloy_search_queries_total 50"));
        assert!(output.contains("alloy_documents_count 100"));
    }

    #[test]
    fn test_json_export() {
        let metrics = Metrics::new();
        metrics.documents_indexed_total.add(100);

        let snapshot = metrics.export_json();
        assert_eq!(snapshot.counters.documents_indexed_total, 100);
    }

    #[test]
    fn test_global_metrics() {
        let metrics = get_metrics();
        metrics.documents_indexed_total.inc();
        assert!(metrics.documents_indexed_total.get() >= 1);
    }
}
