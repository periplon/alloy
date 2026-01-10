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

.badge.indexing {
    background: rgba(59, 130, 246, 0.2);
    color: var(--primary);
    animation: pulse 1.5s ease-in-out infinite;
}

.badge.failed {
    background: rgba(239, 68, 68, 0.2);
    color: var(--error);
    cursor: help;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.source-item.indexing {
    border-left: 3px solid var(--primary);
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

/* Button styles */
.btn-primary {
    background: var(--accent);
    color: white;
}

.btn-primary:hover {
    background: var(--accent-hover);
}

.btn-secondary {
    background: var(--bg-tertiary);
    color: var(--text-primary);
}

.btn-secondary:hover {
    background: var(--border);
}

.btn-link {
    background: transparent;
    color: var(--accent);
    text-decoration: none;
    padding: 8px 12px;
}

.btn-link:hover {
    text-decoration: underline;
}

.btn-close {
    background: transparent;
    border: none;
    color: var(--text-secondary);
    font-size: 1.5rem;
    cursor: pointer;
    padding: 0 8px;
}

.btn-close:hover {
    color: var(--text-primary);
}

/* Section header with action */
.section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
}

.section-header h2 {
    margin-bottom: 0;
}

/* Modal styles */
.modal {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
}

.modal-content {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 12px;
    width: 90%;
    max-width: 500px;
    max-height: 90vh;
    overflow: auto;
}

.modal-content.modal-large {
    max-width: 800px;
}

.modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px;
    border-bottom: 1px solid var(--border);
}

.modal-header h2 {
    font-size: 1.125rem;
    margin: 0;
}

/* Form styles */
.form-group {
    padding: 0 20px;
    margin: 20px 0;
}

.form-group label {
    display: block;
    margin-bottom: 8px;
    font-size: 0.875rem;
    color: var(--text-secondary);
}

.form-input, .form-select {
    width: 100%;
    padding: 10px 14px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 0.9rem;
}

.form-input:focus, .form-select:focus {
    outline: none;
    border-color: var(--accent);
}

.form-group input[type="checkbox"] {
    width: 16px;
    height: 16px;
    accent-color: var(--accent);
    margin-right: 8px;
}

.form-group label:has(input[type="checkbox"]) {
    display: flex;
    align-items: center;
    color: var(--text-primary);
}

.form-actions {
    display: flex;
    justify-content: flex-end;
    gap: 12px;
    padding: 20px;
    border-top: 1px solid var(--border);
}

/* Filter bar */
.filter-bar {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    align-items: center;
}

.filter-bar label {
    display: flex;
    align-items: center;
    gap: 10px;
    color: var(--text-secondary);
}

.filter-bar select {
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text-primary);
    min-width: 250px;
}

/* Documents grid */
.documents-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
}

.document-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    display: flex;
    gap: 16px;
    cursor: pointer;
    transition: all 0.2s;
}

.document-card:hover {
    background: var(--bg-tertiary);
    border-color: var(--accent);
}

.document-icon {
    font-size: 2rem;
    flex-shrink: 0;
}

.document-info h3 {
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 4px;
    word-break: break-word;
}

.document-path {
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-bottom: 8px;
    word-break: break-all;
}

.document-chunks {
    font-size: 0.75rem;
    background: var(--bg-tertiary);
    padding: 2px 8px;
    border-radius: 4px;
    color: var(--text-secondary);
}

/* Document detail */
.document-detail {
    padding: 20px;
}

.detail-section {
    margin-bottom: 24px;
}

.detail-section:last-child {
    margin-bottom: 0;
}

.detail-section h3 {
    font-size: 0.875rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 12px;
}

.detail-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
}

.detail-item {
    background: var(--bg-tertiary);
    padding: 12px;
    border-radius: 6px;
}

.detail-item label {
    display: block;
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-bottom: 4px;
}

.detail-item span {
    font-size: 0.9rem;
    word-break: break-all;
}

.document-content {
    background: var(--bg-primary);
    padding: 16px;
    border-radius: 6px;
    font-family: 'SF Mono', Monaco, monospace;
    font-size: 0.85rem;
    line-height: 1.6;
    overflow-x: auto;
    max-height: 400px;
    overflow-y: auto;
    white-space: pre-wrap;
}

/* Settings page */
.settings-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
    gap: 20px;
}

.settings-section {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 20px;
}

.settings-section h2 {
    font-size: 1rem;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
}

.settings-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.setting-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
}

.setting-item label {
    color: var(--text-secondary);
    font-size: 0.875rem;
}

.setting-item span, .setting-item a {
    font-weight: 500;
}

.setting-item .link {
    color: var(--accent);
    text-decoration: none;
}

.setting-item .link:hover {
    text-decoration: underline;
}

.settings-actions {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
}

/* Stat card small value variant */
.stat-card .value.small {
    font-size: 1.25rem;
}

/* Clickable result items */
.result-item {
    cursor: pointer;
}

/* Error text */
.error {
    color: var(--error);
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

    .settings-grid {
        grid-template-columns: 1fr;
    }

    .documents-grid {
        grid-template-columns: 1fr;
    }

    .detail-grid {
        grid-template-columns: 1fr;
    }

    .modal-content {
        width: 95%;
        margin: 10px;
    }
}
"#;

/// JavaScript for the web UI.
pub const JS: &str = r#"
// Alloy Web UI JavaScript - Single Page Application

const API_BASE = '/api/v1';

// State
let state = {
    stats: {},
    sources: [],
    documents: [],
    searchResults: [],
    health: {},
    config: {},
    currentPage: 'dashboard',
    selectedSource: null,
};

// DOM Elements
const mainContent = () => document.getElementById('main-content');

// Initialize on load
document.addEventListener('DOMContentLoaded', init);

async function init() {
    // Set up navigation
    setupNavigation();

    // Parse current URL and navigate to the correct page
    const path = window.location.pathname.replace('/ui', '').replace('/', '') || 'dashboard';
    navigateTo(path, false);

    // Handle browser back/forward
    window.addEventListener('popstate', (e) => {
        const path = window.location.pathname.replace('/ui', '').replace('/', '') || 'dashboard';
        navigateTo(path, false);
    });

    // Refresh stats periodically
    setInterval(() => {
        if (state.currentPage === 'dashboard') loadStats();
    }, 30000);
}

function setupNavigation() {
    document.querySelectorAll('nav a, .logo').forEach(link => {
        link.addEventListener('click', (e) => {
            if (link.dataset.page) {
                e.preventDefault();
                navigateTo(link.dataset.page);
            }
        });
    });
}

function navigateTo(page, pushState = true) {
    state.currentPage = page;

    // Update URL
    const url = page === 'dashboard' ? '/ui' : `/ui/${page}`;
    if (pushState) {
        history.pushState({ page }, '', url);
    }

    // Update active nav
    document.querySelectorAll('nav a').forEach(a => {
        a.classList.toggle('active', a.dataset.page === page);
    });

    // Render page
    renderPage(page);
}

async function renderPage(page) {
    const container = mainContent();
    if (!container) return;

    container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

    switch (page) {
        case 'dashboard':
            await renderDashboard();
            break;
        case 'search':
            await renderSearchPage();
            break;
        case 'sources':
            await renderSourcesPage();
            break;
        case 'documents':
            await renderDocumentsPage();
            break;
        case 'settings':
            await renderSettingsPage();
            break;
        default:
            container.innerHTML = '<div class="empty-state"><p>Page not found</p></div>';
    }
}

// ============================================================================
// API Functions
// ============================================================================

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
            const error = await response.json().catch(() => ({}));
            throw new Error(error.error || `HTTP ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

async function loadStats() {
    try {
        state.stats = await api('/stats');
        return state.stats;
    } catch (error) {
        console.error('Failed to load stats:', error);
        state.stats = { error: error.message };
        return state.stats;
    }
}

async function loadSources() {
    try {
        const result = await api('/sources');
        state.sources = result.sources || [];
        return state.sources;
    } catch (error) {
        console.error('Failed to load sources:', error);
        state.sources = [];
        return [];
    }
}

async function loadHealth() {
    try {
        const response = await fetch('/health');
        state.health = await response.json();
        return state.health;
    } catch (error) {
        console.error('Failed to load health:', error);
        state.health = { status: 'unknown', error: error.message };
        return state.health;
    }
}

// ============================================================================
// Dashboard Page
// ============================================================================

async function renderDashboard() {
    const container = mainContent();

    // Load data in parallel
    const [stats, sources, health] = await Promise.all([
        loadStats(),
        loadSources(),
        loadHealth()
    ]);

    const healthClass = health.status === 'healthy' ? 'healthy' :
                        health.status === 'degraded' ? 'degraded' : 'unhealthy';

    container.innerHTML = `
        <div class="page-header">
            <h1>Dashboard</h1>
            <div class="health-status">
                <span class="health-dot ${healthClass}"></span>
                <span>${capitalize(health.status || 'unknown')}</span>
            </div>
        </div>

        <div class="stats-grid">
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
            <div class="stat-card">
                <h3>Uptime</h3>
                <div class="value">${formatDuration(stats.uptime_secs || 0)}</div>
            </div>
            <div class="stat-card">
                <h3>Backend</h3>
                <div class="value small">${stats.storage_backend || 'N/A'}</div>
            </div>
        </div>

        <div class="section">
            <h2>Quick Search</h2>
            <div class="search-box">
                <form id="quick-search-form" class="search-form">
                    <input type="text" id="quick-search-input" class="search-input"
                           placeholder="Search indexed documents..." autocomplete="off">
                    <button type="submit" class="search-button">Search</button>
                </form>
            </div>
        </div>

        <div class="section">
            <div class="section-header">
                <h2>Recent Sources</h2>
                <a href="/ui/sources" class="btn btn-link" data-page="sources">View All</a>
            </div>
            <div class="sources-list" id="recent-sources">
                ${renderSourcesList(sources.slice(0, 5), true)}
            </div>
        </div>
    `;

    // Setup quick search
    document.getElementById('quick-search-form')?.addEventListener('submit', (e) => {
        e.preventDefault();
        const query = document.getElementById('quick-search-input')?.value?.trim();
        if (query) {
            state.searchQuery = query;
            navigateTo('search');
        }
    });

    // Setup view all link
    document.querySelector('[data-page="sources"]')?.addEventListener('click', (e) => {
        e.preventDefault();
        navigateTo('sources');
    });
}

// ============================================================================
// Search Page
// ============================================================================

async function renderSearchPage() {
    const container = mainContent();

    await loadSources();

    container.innerHTML = `
        <div class="page-header">
            <h1>Search Documents</h1>
        </div>

        <div class="search-box">
            <form id="search-form" class="search-form">
                <input type="text" id="search-input" class="search-input"
                       placeholder="Enter your search query..." autocomplete="off"
                       value="${escapeHtml(state.searchQuery || '')}">
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
                    Source:
                    <select id="source-filter">
                        <option value="">All Sources</option>
                        ${state.sources.map(s => `
                            <option value="${escapeHtml(s.id)}">${escapeHtml(s.path)}</option>
                        `).join('')}
                    </select>
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

        <div class="results" id="results-container">
            <div class="empty-state">
                <p>Enter a search query to find documents.</p>
            </div>
        </div>
    `;

    // Setup search form
    document.getElementById('search-form')?.addEventListener('submit', handleSearch);

    // Auto-search if we have a query
    if (state.searchQuery) {
        document.getElementById('search-input').value = state.searchQuery;
        handleSearch(new Event('submit'));
    }
}

async function handleSearch(event) {
    event.preventDefault();

    const query = document.getElementById('search-input')?.value?.trim();
    if (!query) return;

    state.searchQuery = query;
    const limit = document.getElementById('limit')?.value || 10;
    const vectorWeight = document.getElementById('vector-weight')?.value || 0.5;
    const sourceId = document.getElementById('source-filter')?.value || '';
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
        if (sourceId) params.set('source_id', sourceId);

        const result = await api('/search?' + params.toString());
        state.searchResults = result.results || [];
        renderSearchResults(result);
    } catch (error) {
        showError('results-container', 'Search failed: ' + error.message);
    }
}

function renderSearchResults(result) {
    const container = document.getElementById('results-container');
    if (!container) return;

    const results = result.results || [];

    if (results.length === 0) {
        container.innerHTML = `
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

    container.innerHTML = `
        <div class="results-header">
            <h2>Search Results</h2>
            <span class="results-meta">${results.length} results in ${result.took_ms}ms${
                result.query_expanded ? ' (expanded)' : ''
            }${result.reranked ? ' (reranked)' : ''}</span>
        </div>
        ${results.map((r, i) => `
            <div class="result-item" onclick="viewDocument('${escapeHtml(r.document_id)}')">
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

// ============================================================================
// Sources Page
// ============================================================================

async function renderSourcesPage() {
    const container = mainContent();

    await loadSources();

    container.innerHTML = `
        <div class="page-header">
            <h1>Indexed Sources</h1>
            <button class="btn btn-primary" id="add-source-btn">Add Source</button>
        </div>

        <div class="section">
            <div class="sources-list" id="sources-container">
                ${renderSourcesList(state.sources, false)}
            </div>
        </div>

        <!-- Add Source Modal -->
        <div class="modal" id="add-source-modal" style="display: none;">
            <div class="modal-content">
                <div class="modal-header">
                    <h2>Add Source</h2>
                    <button class="btn-close" id="close-modal">&times;</button>
                </div>
                <form id="add-source-form">
                    <div class="form-group">
                        <label for="source-path">Path or S3 URI</label>
                        <input type="text" id="source-path" class="form-input"
                               placeholder="/path/to/directory or s3://bucket/prefix" required>
                    </div>
                    <div class="form-group">
                        <label for="source-pattern">File Pattern (optional)</label>
                        <input type="text" id="source-pattern" class="form-input"
                               placeholder="*.md, **/*.rs">
                    </div>
                    <div class="form-group">
                        <label>
                            <input type="checkbox" id="source-watch">
                            Watch for changes
                        </label>
                    </div>
                    <div class="form-actions">
                        <button type="button" class="btn btn-secondary" id="cancel-add-source">Cancel</button>
                        <button type="submit" class="btn btn-primary">Index</button>
                    </div>
                </form>
            </div>
        </div>
    `;

    // Setup modal
    const modal = document.getElementById('add-source-modal');
    document.getElementById('add-source-btn')?.addEventListener('click', () => {
        modal.style.display = 'flex';
    });
    document.getElementById('close-modal')?.addEventListener('click', () => {
        modal.style.display = 'none';
    });
    document.getElementById('cancel-add-source')?.addEventListener('click', () => {
        modal.style.display = 'none';
    });

    // Setup form
    document.getElementById('add-source-form')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        await addSource();
    });

    // Auto-refresh if any sources are indexing
    startSourcesAutoRefresh();
}

// Auto-refresh interval for sources page
let sourcesRefreshInterval = null;

function startSourcesAutoRefresh() {
    // Clear any existing interval
    if (sourcesRefreshInterval) {
        clearInterval(sourcesRefreshInterval);
        sourcesRefreshInterval = null;
    }

    // Check if any sources are indexing
    const hasIndexing = state.sources.some(s => s.status === 'indexing');
    if (!hasIndexing) return;

    // Start auto-refresh every 2 seconds
    sourcesRefreshInterval = setInterval(async () => {
        if (state.currentPage !== 'sources') {
            clearInterval(sourcesRefreshInterval);
            sourcesRefreshInterval = null;
            return;
        }

        await loadSources();
        const container = document.getElementById('sources-container');
        if (container) {
            container.innerHTML = renderSourcesList(state.sources, false);
        }

        // Stop if no more indexing sources
        const stillIndexing = state.sources.some(s => s.status === 'indexing');
        if (!stillIndexing) {
            clearInterval(sourcesRefreshInterval);
            sourcesRefreshInterval = null;
        }
    }, 2000);
}

function renderSourcesList(sources, compact = false) {
    if (sources.length === 0) {
        return `
            <div class="empty-state">
                <p>No sources indexed yet.</p>
                <p>Use the "Add Source" button or CLI to index a directory.</p>
            </div>
        `;
    }

    return sources.map(source => {
        const status = source.status || 'ready';
        const statusBadge = status === 'indexing'
            ? '<span class="badge indexing">Indexing...</span>'
            : status === 'failed'
            ? `<span class="badge failed" title="${escapeHtml(source.error || 'Unknown error')}">Failed</span>`
            : '';

        return `
            <div class="source-item ${status === 'indexing' ? 'indexing' : ''}" data-id="${source.id}">
                <div class="source-info">
                    <h3>${escapeHtml(source.path)}</h3>
                    <p>${source.source_type} • Last scan: ${formatDate(source.last_scan)}</p>
                </div>
                <div class="source-stats">
                    <div class="count">
                        <div class="value">${source.document_count}</div>
                        <div class="label">Documents</div>
                    </div>
                    ${statusBadge}
                    <span class="badge ${source.watching ? 'watching' : 'static'}">
                        ${source.watching ? 'Watching' : 'Static'}
                    </span>
                    ${compact ? '' : `
                        <button class="btn btn-secondary" onclick="refreshSource('${source.id}')" ${status === 'indexing' ? 'disabled' : ''}>
                            Refresh
                        </button>
                        <button class="btn btn-danger" onclick="removeSource('${source.id}')">
                            Remove
                        </button>
                    `}
                </div>
            </div>
        `;
    }).join('');
}

async function addSource() {
    const path = document.getElementById('source-path')?.value?.trim();
    const pattern = document.getElementById('source-pattern')?.value?.trim();
    const watch = document.getElementById('source-watch')?.checked;

    if (!path) return;

    try {
        const body = { path, watch };
        if (pattern) body.pattern = pattern;

        await api('/index', {
            method: 'POST',
            body: JSON.stringify(body)
        });

        document.getElementById('add-source-modal').style.display = 'none';
        await renderSourcesPage();
    } catch (error) {
        alert('Failed to add source: ' + error.message);
    }
}

async function removeSource(sourceId) {
    if (!confirm('Are you sure you want to remove this source? All indexed documents will be deleted.')) {
        return;
    }

    try {
        await api('/sources/' + sourceId, { method: 'DELETE' });
        await loadSources();
        await loadStats();

        // Re-render current page
        if (state.currentPage === 'sources') {
            await renderSourcesPage();
        } else if (state.currentPage === 'dashboard') {
            await renderDashboard();
        }
    } catch (error) {
        alert('Failed to remove source: ' + error.message);
    }
}

async function refreshSource(sourceId) {
    try {
        const source = state.sources.find(s => s.id === sourceId);
        if (source) {
            await api('/index', {
                method: 'POST',
                body: JSON.stringify({ path: source.path })
            });
            await renderSourcesPage();
        }
    } catch (error) {
        alert('Failed to refresh source: ' + error.message);
    }
}

// ============================================================================
// Documents Page
// ============================================================================

async function renderDocumentsPage() {
    const container = mainContent();

    await loadSources();

    container.innerHTML = `
        <div class="page-header">
            <h1>Browse Documents</h1>
        </div>

        <div class="section">
            <div class="filter-bar">
                <label>
                    Source:
                    <select id="doc-source-filter" class="form-select">
                        <option value="">All Sources</option>
                        ${state.sources.map(s => `
                            <option value="${escapeHtml(s.id)}"
                                ${state.selectedSource === s.id ? 'selected' : ''}>
                                ${escapeHtml(s.path)} (${s.document_count} docs)
                            </option>
                        `).join('')}
                    </select>
                </label>
            </div>
        </div>

        <div class="section">
            <div class="documents-list" id="documents-container">
                ${state.selectedSource
                    ? '<div class="loading"><div class="spinner"></div></div>'
                    : '<div class="empty-state"><p>Select a source to browse documents.</p></div>'
                }
            </div>
        </div>

        <!-- Document Detail Modal -->
        <div class="modal" id="document-modal" style="display: none;">
            <div class="modal-content modal-large">
                <div class="modal-header">
                    <h2 id="document-title">Document Details</h2>
                    <button class="btn-close" id="close-doc-modal">&times;</button>
                </div>
                <div id="document-detail-content">
                    <div class="loading"><div class="spinner"></div></div>
                </div>
            </div>
        </div>
    `;

    // Setup source filter
    document.getElementById('doc-source-filter')?.addEventListener('change', async (e) => {
        state.selectedSource = e.target.value;
        if (state.selectedSource) {
            await loadDocumentsForSource(state.selectedSource);
        } else {
            document.getElementById('documents-container').innerHTML =
                '<div class="empty-state"><p>Select a source to browse documents.</p></div>';
        }
    });

    // Setup modal
    document.getElementById('close-doc-modal')?.addEventListener('click', () => {
        document.getElementById('document-modal').style.display = 'none';
    });

    // Load documents if source is selected
    if (state.selectedSource) {
        await loadDocumentsForSource(state.selectedSource);
    }
}

async function loadDocumentsForSource(sourceId) {
    const container = document.getElementById('documents-container');
    container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

    try {
        // Search for all documents from this source
        const result = await api(`/search?q=*&source_id=${encodeURIComponent(sourceId)}&limit=100`);
        const docs = result.results || [];

        // Group by document
        const docMap = new Map();
        for (const chunk of docs) {
            if (!docMap.has(chunk.document_id)) {
                docMap.set(chunk.document_id, {
                    document_id: chunk.document_id,
                    path: chunk.path,
                    chunks: []
                });
            }
            docMap.get(chunk.document_id).chunks.push(chunk);
        }

        state.documents = Array.from(docMap.values());
        renderDocumentsList();
    } catch (error) {
        container.innerHTML = `<div class="empty-state"><p>Failed to load documents: ${escapeHtml(error.message)}</p></div>`;
    }
}

function renderDocumentsList() {
    const container = document.getElementById('documents-container');

    if (state.documents.length === 0) {
        container.innerHTML = '<div class="empty-state"><p>No documents found in this source.</p></div>';
        return;
    }

    container.innerHTML = `
        <div class="documents-grid">
            ${state.documents.map(doc => `
                <div class="document-card" onclick="viewDocument('${escapeHtml(doc.document_id)}')">
                    <div class="document-icon">${getFileIcon(doc.path)}</div>
                    <div class="document-info">
                        <h3>${escapeHtml(getFileName(doc.path))}</h3>
                        <p class="document-path">${escapeHtml(doc.path)}</p>
                        <span class="document-chunks">${doc.chunks.length} chunks</span>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

async function viewDocument(documentId) {
    const modal = document.getElementById('document-modal');
    const content = document.getElementById('document-detail-content');
    const title = document.getElementById('document-title');

    modal.style.display = 'flex';
    content.innerHTML = '<div class="loading"><div class="spinner"></div></div>';

    try {
        const doc = await api(`/documents/${encodeURIComponent(documentId)}`);

        title.textContent = getFileName(doc.path);

        content.innerHTML = `
            <div class="document-detail">
                <div class="detail-section">
                    <h3>Metadata</h3>
                    <div class="detail-grid">
                        <div class="detail-item">
                            <label>Path</label>
                            <span>${escapeHtml(doc.path)}</span>
                        </div>
                        <div class="detail-item">
                            <label>MIME Type</label>
                            <span>${escapeHtml(doc.mime_type)}</span>
                        </div>
                        <div class="detail-item">
                            <label>Size</label>
                            <span>${formatBytes(doc.size_bytes)}</span>
                        </div>
                        <div class="detail-item">
                            <label>Indexed</label>
                            <span>${formatDate(doc.indexed_at)}</span>
                        </div>
                        <div class="detail-item">
                            <label>Modified</label>
                            <span>${formatDate(doc.modified_at)}</span>
                        </div>
                        <div class="detail-item">
                            <label>Chunks</label>
                            <span>${doc.chunk_count}</span>
                        </div>
                    </div>
                </div>
                ${doc.content ? `
                    <div class="detail-section">
                        <h3>Content</h3>
                        <pre class="document-content">${escapeHtml(doc.content)}</pre>
                    </div>
                ` : ''}
            </div>
        `;
    } catch (error) {
        content.innerHTML = `<div class="empty-state"><p>Failed to load document: ${escapeHtml(error.message)}</p></div>`;
    }
}

// ============================================================================
// Settings Page
// ============================================================================

async function renderSettingsPage() {
    const container = mainContent();

    const [stats, health] = await Promise.all([
        loadStats(),
        loadHealth()
    ]);

    container.innerHTML = `
        <div class="page-header">
            <h1>Settings & Configuration</h1>
        </div>

        <div class="settings-grid">
            <div class="settings-section">
                <h2>Server Information</h2>
                <div class="settings-list">
                    <div class="setting-item">
                        <label>Version</label>
                        <span>${health.version || 'Unknown'}</span>
                    </div>
                    <div class="setting-item">
                        <label>Status</label>
                        <span class="badge ${health.status === 'healthy' ? 'watching' : 'static'}">
                            ${capitalize(health.status || 'unknown')}
                        </span>
                    </div>
                    <div class="setting-item">
                        <label>Uptime</label>
                        <span>${formatDuration(health.uptime_seconds || 0)}</span>
                    </div>
                </div>
            </div>

            <div class="settings-section">
                <h2>Storage</h2>
                <div class="settings-list">
                    <div class="setting-item">
                        <label>Backend</label>
                        <span>${stats.storage_backend || 'N/A'}</span>
                    </div>
                    <div class="setting-item">
                        <label>Embedding Provider</label>
                        <span>${stats.embedding_provider || 'N/A'}</span>
                    </div>
                    <div class="setting-item">
                        <label>Embedding Dimension</label>
                        <span>${stats.embedding_dimension || 'N/A'}</span>
                    </div>
                    <div class="setting-item">
                        <label>Storage Used</label>
                        <span>${formatBytes(stats.storage_bytes || 0)}</span>
                    </div>
                </div>
            </div>

            <div class="settings-section">
                <h2>Index Statistics</h2>
                <div class="settings-list">
                    <div class="setting-item">
                        <label>Total Sources</label>
                        <span>${stats.source_count || 0}</span>
                    </div>
                    <div class="setting-item">
                        <label>Total Documents</label>
                        <span>${formatNumber(stats.document_count || 0)}</span>
                    </div>
                    <div class="setting-item">
                        <label>Total Chunks</label>
                        <span>${formatNumber(stats.chunk_count || 0)}</span>
                    </div>
                </div>
            </div>

            <div class="settings-section">
                <h2>Health Checks</h2>
                <div class="settings-list">
                    ${(health.checks || []).map(check => `
                        <div class="setting-item">
                            <label>${capitalize(check.component)}</label>
                            <span class="badge ${check.status === 'healthy' ? 'watching' : 'static'}">
                                ${capitalize(check.status)}
                            </span>
                        </div>
                    `).join('')}
                </div>
            </div>

            <div class="settings-section">
                <h2>API Endpoints</h2>
                <div class="settings-list">
                    <div class="setting-item">
                        <label>REST API</label>
                        <a href="/api" target="_blank" class="link">/api/v1</a>
                    </div>
                    <div class="setting-item">
                        <label>Health</label>
                        <a href="/health" target="_blank" class="link">/health</a>
                    </div>
                    <div class="setting-item">
                        <label>Metrics</label>
                        <a href="/metrics" target="_blank" class="link">/metrics</a>
                    </div>
                </div>
            </div>

            <div class="settings-section">
                <h2>Actions</h2>
                <div class="settings-actions">
                    <button class="btn btn-secondary" onclick="clearCache()">Clear Cache</button>
                    <button class="btn btn-secondary" onclick="createBackup()">Create Backup</button>
                    <a href="/api" target="_blank" class="btn btn-secondary">View API Docs</a>
                </div>
            </div>
        </div>
    `;
}

async function clearCache() {
    try {
        // Note: This would need a specific endpoint - for now show a message
        alert('Cache clearing is available via the MCP tool: clear_cache');
    } catch (error) {
        alert('Failed to clear cache: ' + error.message);
    }
}

async function createBackup() {
    try {
        // Note: This would need a specific endpoint - for now show a message
        alert('Backup creation is available via the MCP tool: create_backup\n\nOr use the CLI: alloy backup');
    } catch (error) {
        alert('Failed to create backup: ' + error.message);
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

function showLoading(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = '<div class="loading"><div class="spinner"></div></div>';
    }
}

function showError(containerId, message) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = `<div class="empty-state"><p class="error">${escapeHtml(message)}</p></div>`;
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

function formatDuration(seconds) {
    if (!seconds) return '0s';
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;

    if (days > 0) return `${days}d ${hours}h`;
    if (hours > 0) return `${hours}h ${mins}m`;
    if (mins > 0) return `${mins}m ${secs}s`;
    return `${secs}s`;
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

function capitalize(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
}

function getFileName(path) {
    if (!path) return 'Unknown';
    return path.split('/').pop() || path;
}

function getFileIcon(path) {
    if (!path) return '📄';
    const ext = path.split('.').pop()?.toLowerCase() || '';
    const icons = {
        md: '📝', txt: '📄', pdf: '📕', doc: '📘', docx: '📘',
        rs: '🦀', py: '🐍', js: '📜', ts: '📜', jsx: '⚛️', tsx: '⚛️',
        json: '📋', yaml: '📋', yml: '📋', toml: '📋',
        html: '🌐', css: '🎨', svg: '🖼️', png: '🖼️', jpg: '🖼️', jpeg: '🖼️',
        go: '🔵', java: '☕', cpp: '⚙️', c: '⚙️', h: '⚙️',
    };
    return icons[ext] || '📄';
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
            <a href="/ui" class="logo" data-page="dashboard">Alloy<span>Index</span></a>
            <nav>
                <a href="/ui" data-page="dashboard">Dashboard</a>
                <a href="/ui/search" data-page="search">Search</a>
                <a href="/ui/sources" data-page="sources">Sources</a>
                <a href="/ui/documents" data-page="documents">Documents</a>
                <a href="/ui/settings" data-page="settings">Settings</a>
            </nav>
        </div>
    </header>

    <main class="container" id="main-content">
        <!-- Content dynamically loaded by JavaScript -->
        <div class="loading"><div class="spinner"></div></div>
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
