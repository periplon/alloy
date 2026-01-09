//! Static file serving for the web UI.
//!
//! The web UI is embedded directly in the binary for easy deployment.

use axum::{
    body::Body,
    http::{header, Response, StatusCode},
    response::IntoResponse,
};

/// CSS styles for the web UI.
pub const CSS: &str = r#"
:root {
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-tertiary: #334155;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --accent: #3b82f6;
    --accent-hover: #2563eb;
    --success: #22c55e;
    --warning: #eab308;
    --error: #ef4444;
    --border: #475569;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

header {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    padding: 15px 0;
    margin-bottom: 30px;
}

header .container {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--accent);
    text-decoration: none;
}

.logo span {
    color: var(--text-primary);
}

nav a {
    color: var(--text-secondary);
    text-decoration: none;
    margin-left: 30px;
    transition: color 0.2s;
}

nav a:hover, nav a.active {
    color: var(--accent);
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}

.stat-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
}

.stat-card h3 {
    font-size: 0.875rem;
    color: var(--text-secondary);
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.stat-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
}

.stat-card .value.success { color: var(--success); }
.stat-card .value.warning { color: var(--warning); }
.stat-card .value.error { color: var(--error); }

.search-box {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 25px;
    margin-bottom: 30px;
}

.search-form {
    display: flex;
    gap: 15px;
}

.search-input {
    flex: 1;
    padding: 12px 16px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 1rem;
}

.search-input:focus {
    outline: none;
    border-color: var(--accent);
}

.search-button {
    padding: 12px 24px;
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 6px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
}

.search-button:hover {
    background: var(--accent-hover);
}

.search-options {
    display: flex;
    gap: 20px;
    margin-top: 15px;
    flex-wrap: wrap;
}

.search-options label {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-secondary);
    font-size: 0.875rem;
}

.search-options input[type="number"],
.search-options select {
    padding: 6px 10px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text-primary);
    width: 100px;
}

.search-options input[type="checkbox"] {
    width: 16px;
    height: 16px;
    accent-color: var(--accent);
}

.results {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
}

.results-header {
    padding: 15px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.results-header h2 {
    font-size: 1rem;
    font-weight: 600;
}

.results-meta {
    color: var(--text-secondary);
    font-size: 0.875rem;
}

.result-item {
    padding: 20px;
    border-bottom: 1px solid var(--border);
}

.result-item:last-child {
    border-bottom: none;
}

.result-item:hover {
    background: var(--bg-tertiary);
}

.result-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 10px;
}

.result-title {
    font-weight: 600;
    color: var(--accent);
    word-break: break-all;
}

.result-score {
    background: var(--bg-tertiary);
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--success);
}

.result-content {
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.7;
    margin-bottom: 10px;
}

.result-meta {
    display: flex;
    gap: 20px;
    font-size: 0.75rem;
    color: var(--text-secondary);
}

.sources-list {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
}

.source-item {
    padding: 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.source-item:last-child {
    border-bottom: none;
}

.source-info h3 {
    font-weight: 600;
    margin-bottom: 5px;
}

.source-info p {
    color: var(--text-secondary);
    font-size: 0.875rem;
}

.source-stats {
    display: flex;
    gap: 20px;
    align-items: center;
}

.source-stats .count {
    text-align: center;
}

.source-stats .count .value {
    font-size: 1.25rem;
    font-weight: 700;
}

.source-stats .count .label {
    font-size: 0.75rem;
    color: var(--text-secondary);
}

.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
}

.badge.watching {
    background: rgba(34, 197, 94, 0.2);
    color: var(--success);
}

.badge.static {
    background: rgba(148, 163, 184, 0.2);
    color: var(--text-secondary);
}

.btn {
    padding: 8px 16px;
    border: none;
    border-radius: 6px;
    font-size: 0.875rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
}

.btn-danger {
    background: rgba(239, 68, 68, 0.2);
    color: var(--error);
}

.btn-danger:hover {
    background: rgba(239, 68, 68, 0.3);
}

.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: var(--text-secondary);
}

.empty-state svg {
    width: 64px;
    height: 64px;
    margin-bottom: 20px;
    opacity: 0.5;
}

.loading {
    display: flex;
    justify-content: center;
    padding: 40px;
}

.spinner {
    width: 40px;
    height: 40px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

.health-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
}

.health-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
}

.health-dot.healthy { background: var(--success); }
.health-dot.degraded { background: var(--warning); }
.health-dot.unhealthy { background: var(--error); }

.page-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 30px;
}

.page-header h1 {
    font-size: 1.5rem;
}

.section {
    margin-bottom: 30px;
}

.section h2 {
    font-size: 1.125rem;
    margin-bottom: 15px;
    color: var(--text-secondary);
}

@media (max-width: 768px) {
    .search-form {
        flex-direction: column;
    }

    .stats-grid {
        grid-template-columns: 1fr 1fr;
    }

    .source-item {
        flex-direction: column;
        align-items: flex-start;
        gap: 15px;
    }
}
"#;

/// JavaScript for the web UI.
pub const JS: &str = r#"
// Alloy Web UI JavaScript

const API_BASE = '/api/v1';

// State
let searchResults = [];
let sources = [];
let stats = {};

// DOM Elements
const elements = {};

// Initialize on load
document.addEventListener('DOMContentLoaded', init);

async function init() {
    // Cache DOM elements
    elements.searchInput = document.getElementById('search-input');
    elements.searchForm = document.getElementById('search-form');
    elements.resultsContainer = document.getElementById('results-container');
    elements.sourcesContainer = document.getElementById('sources-container');
    elements.statsContainer = document.getElementById('stats-container');

    // Set up event listeners
    if (elements.searchForm) {
        elements.searchForm.addEventListener('submit', handleSearch);
    }

    // Load initial data
    await Promise.all([
        loadStats(),
        loadSources()
    ]);

    // Refresh stats periodically
    setInterval(loadStats, 30000);
}

// API Functions
async function api(endpoint, options = {}) {
    try {
        const response = await fetch(API_BASE + endpoint, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'API request failed');
        }

        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// Load Functions
async function loadStats() {
    try {
        stats = await api('/stats');
        renderStats();
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

async function loadSources() {
    try {
        const result = await api('/sources');
        sources = result.sources || [];
        renderSources();
    } catch (error) {
        console.error('Failed to load sources:', error);
    }
}

// Search
async function handleSearch(event) {
    event.preventDefault();

    const query = elements.searchInput?.value?.trim();
    if (!query) return;

    const limit = document.getElementById('limit')?.value || 10;
    const vectorWeight = document.getElementById('vector-weight')?.value || 0.5;
    const expand = document.getElementById('expand')?.checked || false;
    const rerank = document.getElementById('rerank')?.checked || false;

    showLoading('results-container');

    try {
        const params = new URLSearchParams({
            q: query,
            limit,
            vector_weight: vectorWeight,
            expand,
            rerank
        });

        const result = await api('/search?' + params.toString());
        searchResults = result.results || [];
        renderResults(result);
    } catch (error) {
        showError('results-container', 'Search failed: ' + error.message);
    }
}

// Render Functions
function renderStats() {
    if (!elements.statsContainer) return;

    elements.statsContainer.innerHTML = `
        <div class="stat-card">
            <h3>Sources</h3>
            <div class="value">${stats.source_count || 0}</div>
        </div>
        <div class="stat-card">
            <h3>Documents</h3>
            <div class="value">${formatNumber(stats.document_count || 0)}</div>
        </div>
        <div class="stat-card">
            <h3>Chunks</h3>
            <div class="value">${formatNumber(stats.chunk_count || 0)}</div>
        </div>
        <div class="stat-card">
            <h3>Storage</h3>
            <div class="value">${formatBytes(stats.storage_bytes || 0)}</div>
        </div>
    `;
}

function renderSources() {
    if (!elements.sourcesContainer) return;

    if (sources.length === 0) {
        elements.sourcesContainer.innerHTML = `
            <div class="empty-state">
                <p>No sources indexed yet.</p>
                <p>Use the CLI or API to index a directory.</p>
            </div>
        `;
        return;
    }

    elements.sourcesContainer.innerHTML = sources.map(source => `
        <div class="source-item" data-id="${source.id}">
            <div class="source-info">
                <h3>${escapeHtml(source.path)}</h3>
                <p>${source.source_type} • Last scan: ${formatDate(source.last_scan)}</p>
            </div>
            <div class="source-stats">
                <div class="count">
                    <div class="value">${source.document_count}</div>
                    <div class="label">Documents</div>
                </div>
                <span class="badge ${source.watching ? 'watching' : 'static'}">
                    ${source.watching ? 'Watching' : 'Static'}
                </span>
                <button class="btn btn-danger" onclick="removeSource('${source.id}')">
                    Remove
                </button>
            </div>
        </div>
    `).join('');
}

function renderResults(result) {
    if (!elements.resultsContainer) return;

    const results = result.results || [];

    if (results.length === 0) {
        elements.resultsContainer.innerHTML = `
            <div class="results-header">
                <h2>Search Results</h2>
                <span class="results-meta">No results found</span>
            </div>
            <div class="empty-state">
                <p>No documents matched your search.</p>
            </div>
        `;
        return;
    }

    elements.resultsContainer.innerHTML = `
        <div class="results-header">
            <h2>Search Results</h2>
            <span class="results-meta">${results.length} results in ${result.took_ms}ms</span>
        </div>
        ${results.map((r, i) => `
            <div class="result-item">
                <div class="result-header">
                    <span class="result-title">${escapeHtml(r.path || r.document_id)}</span>
                    <span class="result-score">${(r.score * 100).toFixed(1)}%</span>
                </div>
                <div class="result-content">${escapeHtml(truncate(r.content, 300))}</div>
                <div class="result-meta">
                    <span>Document: ${escapeHtml(r.document_id)}</span>
                    <span>Chunk: ${escapeHtml(r.chunk_id)}</span>
                </div>
            </div>
        `).join('')}
    `;
}

// Actions
async function removeSource(sourceId) {
    if (!confirm('Are you sure you want to remove this source? All indexed documents will be deleted.')) {
        return;
    }

    try {
        await api('/sources/' + sourceId, { method: 'DELETE' });
        await loadSources();
        await loadStats();
    } catch (error) {
        alert('Failed to remove source: ' + error.message);
    }
}

// Utility Functions
function showLoading(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    }
}

function showError(containerId, message) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `
            <div class="empty-state">
                <p class="error">${escapeHtml(message)}</p>
            </div>
        `;
    }
}

function formatNumber(num) {
    return new Intl.NumberFormat().format(num);
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(dateStr) {
    if (!dateStr) return 'Never';
    return new Date(dateStr).toLocaleString();
}

function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function truncate(str, maxLen) {
    if (!str || str.length <= maxLen) return str;
    return str.substring(0, maxLen) + '...';
}
"#;

/// HTML template for the web UI.
pub const INDEX_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Alloy - Document Indexing Dashboard</title>
    <link rel="stylesheet" href="/ui/style.css">
</head>
<body>
    <header>
        <div class="container">
            <a href="/ui" class="logo">Alloy<span>Index</span></a>
            <nav>
                <a href="/ui" class="active">Dashboard</a>
                <a href="/ui/sources">Sources</a>
                <a href="/ui/search">Search</a>
                <a href="/health" target="_blank">Health</a>
                <a href="/api" target="_blank">API</a>
            </nav>
        </div>
    </header>

    <main class="container">
        <div class="page-header">
            <h1>Dashboard</h1>
            <div class="health-status">
                <span class="health-dot healthy"></span>
                <span>Healthy</span>
            </div>
        </div>

        <div class="stats-grid" id="stats-container">
            <div class="loading"><div class="spinner"></div></div>
        </div>

        <div class="section">
            <h2>Search</h2>
            <div class="search-box">
                <form id="search-form" class="search-form">
                    <input type="text" id="search-input" class="search-input"
                           placeholder="Search indexed documents..." autocomplete="off">
                    <button type="submit" class="search-button">Search</button>
                </form>
                <div class="search-options">
                    <label>
                        Limit:
                        <input type="number" id="limit" value="10" min="1" max="100">
                    </label>
                    <label>
                        Vector Weight:
                        <input type="number" id="vector-weight" value="0.5" min="0" max="1" step="0.1">
                    </label>
                    <label>
                        <input type="checkbox" id="expand">
                        Query Expansion
                    </label>
                    <label>
                        <input type="checkbox" id="rerank">
                        Reranking
                    </label>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Search Results</h2>
            <div class="results" id="results-container">
                <div class="empty-state">
                    <p>Enter a search query to find documents.</p>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Indexed Sources</h2>
            <div class="sources-list" id="sources-container">
                <div class="loading"><div class="spinner"></div></div>
            </div>
        </div>
    </main>

    <script src="/ui/app.js"></script>
</body>
</html>
"#;

/// Serve the main HTML page.
pub async fn serve_index() -> impl IntoResponse {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
        .body(Body::from(INDEX_HTML))
        .unwrap()
}

/// Serve CSS styles.
pub async fn serve_css() -> impl IntoResponse {
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/css; charset=utf-8")
        .header(header::CACHE_CONTROL, "public, max-age=3600")
        .body(Body::from(CSS))
        .unwrap()
}

/// Serve JavaScript.
pub async fn serve_js() -> impl IntoResponse {
    Response::builder()
        .status(StatusCode::OK)
        .header(
            header::CONTENT_TYPE,
            "application/javascript; charset=utf-8",
        )
        .header(header::CACHE_CONTROL, "public, max-age=3600")
        .body(Body::from(JS))
        .unwrap()
}
