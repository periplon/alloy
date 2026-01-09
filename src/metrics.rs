//! Prometheus-compatible metrics for Alloy MCP server.
//!
//! This module provides observability metrics for monitoring
//! the Alloy indexing and search operations using the prometheus crate.

use prometheus::{self, Histogram, HistogramOpts, IntCounter, IntGauge, Registry};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;

/// Global metrics instance.
static METRICS: std::sync::OnceLock<Arc<Metrics>> = std::sync::OnceLock::new();

/// Get or initialize the global metrics instance.
pub fn get_metrics() -> Arc<Metrics> {
    METRICS.get_or_init(|| Arc::new(Metrics::new())).clone()
}

/// Default histogram buckets for latency tracking (in seconds).
/// Covers from 1ms to 10s with reasonable granularity.
fn default_latency_buckets() -> Vec<f64> {
    vec![
        0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
    ]
}

/// All metrics for the Alloy server.
pub struct Metrics {
    /// Prometheus registry for all metrics.
    pub registry: Registry,

    // =========================================================================
    // Counters
    // =========================================================================
    /// Total number of documents indexed.
    pub documents_indexed_total: IntCounter,
    /// Total number of search queries executed.
    pub search_queries_total: IntCounter,
    /// Total number of indexing errors.
    pub indexing_errors_total: IntCounter,
    /// Total number of search errors.
    pub search_errors_total: IntCounter,
    /// Total number of chunks created.
    pub chunks_created_total: IntCounter,
    /// Total number of embeddings generated.
    pub embeddings_generated_total: IntCounter,
    /// Total number of duplicates detected.
    pub duplicates_detected_total: IntCounter,
    /// Total number of cache hits.
    pub cache_hits_total: IntCounter,
    /// Total number of cache misses.
    pub cache_misses_total: IntCounter,

    // =========================================================================
    // Gauges
    // =========================================================================
    /// Current number of indexed documents.
    pub documents_count: IntGauge,
    /// Current number of chunks.
    pub chunks_count: IntGauge,
    /// Index storage size in bytes.
    pub index_size_bytes: IntGauge,
    /// Number of items in embedding queue.
    pub embedding_queue_size: IntGauge,
    /// Number of active sources.
    pub sources_count: IntGauge,
    /// Number of active file watchers.
    pub watchers_count: IntGauge,
    /// Uptime in seconds.
    pub uptime_seconds: IntGauge,

    // =========================================================================
    // Histograms (durations in seconds)
    // =========================================================================
    /// Search query duration in seconds.
    pub search_duration_seconds: Histogram,
    /// Indexing duration per document in seconds.
    pub indexing_duration_seconds: Histogram,
    /// Embedding generation duration in seconds.
    pub embedding_duration_seconds: Histogram,
    /// Reranking duration in seconds.
    pub reranking_duration_seconds: Histogram,
    /// Query expansion duration in seconds.
    pub query_expansion_duration_seconds: Histogram,

    /// Server start time.
    start_time: RwLock<Instant>,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    /// Create a new metrics instance with all metrics registered.
    pub fn new() -> Self {
        let registry = Registry::new();

        // Counters
        let documents_indexed_total = IntCounter::new(
            "alloy_documents_indexed_total",
            "Total number of documents indexed",
        )
        .expect("failed to create counter");

        let search_queries_total = IntCounter::new(
            "alloy_search_queries_total",
            "Total number of search queries executed",
        )
        .expect("failed to create counter");

        let indexing_errors_total = IntCounter::new(
            "alloy_indexing_errors_total",
            "Total number of indexing errors",
        )
        .expect("failed to create counter");

        let search_errors_total =
            IntCounter::new("alloy_search_errors_total", "Total number of search errors")
                .expect("failed to create counter");

        let chunks_created_total = IntCounter::new(
            "alloy_chunks_created_total",
            "Total number of chunks created",
        )
        .expect("failed to create counter");

        let embeddings_generated_total = IntCounter::new(
            "alloy_embeddings_generated_total",
            "Total number of embeddings generated",
        )
        .expect("failed to create counter");

        let duplicates_detected_total = IntCounter::new(
            "alloy_duplicates_detected_total",
            "Total number of duplicate documents detected",
        )
        .expect("failed to create counter");

        let cache_hits_total =
            IntCounter::new("alloy_cache_hits_total", "Total number of cache hits")
                .expect("failed to create counter");

        let cache_misses_total =
            IntCounter::new("alloy_cache_misses_total", "Total number of cache misses")
                .expect("failed to create counter");

        // Gauges
        let documents_count = IntGauge::new(
            "alloy_documents_count",
            "Current number of indexed documents",
        )
        .expect("failed to create gauge");

        let chunks_count = IntGauge::new("alloy_chunks_count", "Current number of chunks in index")
            .expect("failed to create gauge");

        let index_size_bytes =
            IntGauge::new("alloy_index_size_bytes", "Index storage size in bytes")
                .expect("failed to create gauge");

        let embedding_queue_size = IntGauge::new(
            "alloy_embedding_queue_size",
            "Number of items in embedding queue",
        )
        .expect("failed to create gauge");

        let sources_count =
            IntGauge::new("alloy_sources_count", "Number of active indexed sources")
                .expect("failed to create gauge");

        let watchers_count =
            IntGauge::new("alloy_watchers_count", "Number of active file watchers")
                .expect("failed to create gauge");

        let uptime_seconds = IntGauge::new("alloy_uptime_seconds", "Server uptime in seconds")
            .expect("failed to create gauge");

        // Histograms with latency buckets (in seconds)
        let search_duration_seconds = Histogram::with_opts(
            HistogramOpts::new(
                "alloy_search_duration_seconds",
                "Search query duration in seconds",
            )
            .buckets(default_latency_buckets()),
        )
        .expect("failed to create histogram");

        let indexing_duration_seconds = Histogram::with_opts(
            HistogramOpts::new(
                "alloy_indexing_duration_seconds",
                "Document indexing duration in seconds",
            )
            .buckets(default_latency_buckets()),
        )
        .expect("failed to create histogram");

        let embedding_duration_seconds = Histogram::with_opts(
            HistogramOpts::new(
                "alloy_embedding_duration_seconds",
                "Embedding generation duration in seconds",
            )
            .buckets(default_latency_buckets()),
        )
        .expect("failed to create histogram");

        let reranking_duration_seconds = Histogram::with_opts(
            HistogramOpts::new(
                "alloy_reranking_duration_seconds",
                "Reranking duration in seconds",
            )
            .buckets(default_latency_buckets()),
        )
        .expect("failed to create histogram");

        let query_expansion_duration_seconds = Histogram::with_opts(
            HistogramOpts::new(
                "alloy_query_expansion_duration_seconds",
                "Query expansion duration in seconds",
            )
            .buckets(default_latency_buckets()),
        )
        .expect("failed to create histogram");

        // Register all metrics
        registry
            .register(Box::new(documents_indexed_total.clone()))
            .expect("failed to register metric");
        registry
            .register(Box::new(search_queries_total.clone()))
            .expect("failed to register metric");
        registry
            .register(Box::new(indexing_errors_total.clone()))
            .expect("failed to register metric");
        registry
            .register(Box::new(search_errors_total.clone()))
            .expect("failed to register metric");
        registry
            .register(Box::new(chunks_created_total.clone()))
            .expect("failed to register metric");
        registry
            .register(Box::new(embeddings_generated_total.clone()))
            .expect("failed to register metric");
        registry
            .register(Box::new(duplicates_detected_total.clone()))
            .expect("failed to register metric");
        registry
            .register(Box::new(cache_hits_total.clone()))
            .expect("failed to register metric");
        registry
            .register(Box::new(cache_misses_total.clone()))
            .expect("failed to register metric");
        registry
            .register(Box::new(documents_count.clone()))
            .expect("failed to register metric");
        registry
            .register(Box::new(chunks_count.clone()))
            .expect("failed to register metric");
        registry
            .register(Box::new(index_size_bytes.clone()))
            .expect("failed to register metric");
        registry
            .register(Box::new(embedding_queue_size.clone()))
            .expect("failed to register metric");
        registry
            .register(Box::new(sources_count.clone()))
            .expect("failed to register metric");
        registry
            .register(Box::new(watchers_count.clone()))
            .expect("failed to register metric");
        registry
            .register(Box::new(uptime_seconds.clone()))
            .expect("failed to register metric");
        registry
            .register(Box::new(search_duration_seconds.clone()))
            .expect("failed to register metric");
        registry
            .register(Box::new(indexing_duration_seconds.clone()))
            .expect("failed to register metric");
        registry
            .register(Box::new(embedding_duration_seconds.clone()))
            .expect("failed to register metric");
        registry
            .register(Box::new(reranking_duration_seconds.clone()))
            .expect("failed to register metric");
        registry
            .register(Box::new(query_expansion_duration_seconds.clone()))
            .expect("failed to register metric");

        Self {
            registry,
            // Counters
            documents_indexed_total,
            search_queries_total,
            indexing_errors_total,
            search_errors_total,
            chunks_created_total,
            embeddings_generated_total,
            duplicates_detected_total,
            cache_hits_total,
            cache_misses_total,
            // Gauges
            documents_count,
            chunks_count,
            index_size_bytes,
            embedding_queue_size,
            sources_count,
            watchers_count,
            uptime_seconds,
            // Histograms
            search_duration_seconds,
            indexing_duration_seconds,
            embedding_duration_seconds,
            reranking_duration_seconds,
            query_expansion_duration_seconds,
            // Internal state
            start_time: RwLock::new(Instant::now()),
        }
    }

    /// Update the uptime gauge.
    pub fn update_uptime(&self) {
        let uptime = self.start_time.read().elapsed();
        self.uptime_seconds.set(uptime.as_secs() as i64);
    }

    /// Export metrics in Prometheus text format.
    pub fn export_prometheus(&self) -> String {
        use prometheus::Encoder;
        self.update_uptime();

        let encoder = prometheus::TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
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
                search_duration_seconds: HistogramSnapshot::from_prometheus(
                    &self.search_duration_seconds,
                ),
                indexing_duration_seconds: HistogramSnapshot::from_prometheus(
                    &self.indexing_duration_seconds,
                ),
                embedding_duration_seconds: HistogramSnapshot::from_prometheus(
                    &self.embedding_duration_seconds,
                ),
                reranking_duration_seconds: HistogramSnapshot::from_prometheus(
                    &self.reranking_duration_seconds,
                ),
                query_expansion_duration_seconds: HistogramSnapshot::from_prometheus(
                    &self.query_expansion_duration_seconds,
                ),
            },
        }
    }

    /// Start a timer that records duration to a histogram when dropped.
    /// Returns a guard that will observe the duration in seconds.
    pub fn start_timer(histogram: &Histogram) -> HistogramTimer {
        HistogramTimer {
            histogram: histogram.clone(),
            start: Instant::now(),
        }
    }
}

/// Timer that records duration to a histogram when dropped.
pub struct HistogramTimer {
    histogram: Histogram,
    start: Instant,
}

impl Drop for HistogramTimer {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        self.histogram.observe(duration.as_secs_f64());
    }
}

impl HistogramTimer {
    /// Get the elapsed time without stopping the timer.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Stop the timer and return the elapsed duration.
    /// The duration is recorded in the histogram on drop.
    pub fn stop(self) -> Duration {
        self.start.elapsed()
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
    pub documents_count: i64,
    pub chunks_count: i64,
    pub index_size_bytes: i64,
    pub embedding_queue_size: i64,
    pub sources_count: i64,
    pub watchers_count: i64,
    pub uptime_seconds: i64,
}

/// Histogram metrics snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsHistograms {
    pub search_duration_seconds: HistogramSnapshot,
    pub indexing_duration_seconds: HistogramSnapshot,
    pub embedding_duration_seconds: HistogramSnapshot,
    pub reranking_duration_seconds: HistogramSnapshot,
    pub query_expansion_duration_seconds: HistogramSnapshot,
}

/// Snapshot of a histogram for serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramSnapshot {
    pub count: u64,
    pub sum: f64,
    pub mean: Option<f64>,
}

impl HistogramSnapshot {
    /// Create a snapshot from a prometheus histogram.
    pub fn from_prometheus(h: &Histogram) -> Self {
        let sample_count = h.get_sample_count();
        let sample_sum = h.get_sample_sum();
        let mean = if sample_count > 0 {
            Some(sample_sum / sample_count as f64)
        } else {
            None
        };
        Self {
            count: sample_count,
            sum: sample_sum,
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
        let counter = IntCounter::new("test_counter", "test").unwrap();
        assert_eq!(counter.get(), 0);
        counter.inc();
        assert_eq!(counter.get(), 1);
        counter.inc_by(5);
        assert_eq!(counter.get(), 6);
    }

    #[test]
    fn test_gauge() {
        let gauge = IntGauge::new("test_gauge", "test").unwrap();
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
        let hist = Histogram::with_opts(
            HistogramOpts::new("test_histogram", "test").buckets(default_latency_buckets()),
        )
        .unwrap();
        hist.observe(0.005); // 5ms
        hist.observe(0.025); // 25ms
        hist.observe(0.1); // 100ms

        assert_eq!(hist.get_sample_count(), 3);
        assert!((hist.get_sample_sum() - 0.13).abs() < 0.001);
    }

    #[test]
    fn test_histogram_timer() {
        let hist = Histogram::with_opts(
            HistogramOpts::new("test_timer_histogram", "test").buckets(default_latency_buckets()),
        )
        .unwrap();
        {
            let _timer = Metrics::start_timer(&hist);
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(hist.get_sample_count() > 0);
        assert!(hist.get_sample_sum() >= 0.01); // At least 10ms = 0.01s
    }

    #[test]
    fn test_prometheus_export() {
        let metrics = Metrics::new();
        metrics.documents_indexed_total.inc_by(100);
        metrics.search_queries_total.inc_by(50);
        metrics.documents_count.set(100);

        let output = metrics.export_prometheus();
        assert!(output.contains("alloy_documents_indexed_total 100"));
        assert!(output.contains("alloy_search_queries_total 50"));
        assert!(output.contains("alloy_documents_count 100"));

        // Verify we use seconds not ms for all histograms
        assert!(output.contains("alloy_search_duration_seconds"));
        assert!(output.contains("alloy_indexing_duration_seconds"));
        assert!(output.contains("alloy_embedding_duration_seconds"));
        assert!(output.contains("alloy_reranking_duration_seconds"));
        assert!(output.contains("alloy_query_expansion_duration_seconds"));

        // Verify NO _ms suffixes exist
        assert!(!output.contains("alloy_search_duration_ms"));
        assert!(!output.contains("alloy_indexing_duration_ms"));
        assert!(!output.contains("alloy_embedding_duration_ms"));
        assert!(!output.contains("alloy_reranking_duration_ms"));
        assert!(!output.contains("alloy_query_expansion_duration_ms"));
    }

    #[test]
    fn test_json_export() {
        let metrics = Metrics::new();
        metrics.documents_indexed_total.inc_by(100);

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
