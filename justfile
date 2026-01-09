# Alloy - Hybrid document indexing MCP server
# Run `just` or `just help` for available commands

# Default recipe: show help
default: help

# Show available commands
help:
    @echo "Alloy - Hybrid document indexing MCP server"
    @echo ""
    @echo "Usage: just <recipe>"
    @echo ""
    @echo "Development:"
    @echo "  build       Build the project"
    @echo "  run         Run the server"
    @echo "  dev         Run with hot reload (cargo watch)"
    @echo "  check       Quick compile check"
    @echo "  test        Run all tests"
    @echo "  test-unit   Run unit tests only"
    @echo "  test-int    Run integration tests only"
    @echo ""
    @echo "Quality:"
    @echo "  fmt         Format code"
    @echo "  lint        Run clippy lints"
    @echo "  audit       Security audit dependencies"
    @echo "  qa          Run all quality checks (fmt, lint, test)"
    @echo ""
    @echo "Release:"
    @echo "  release     Build optimized release binary"
    @echo "  install     Install binary to cargo bin"
    @echo ""
    @echo "Utilities:"
    @echo "  clean       Clean build artifacts"
    @echo "  deps        Update dependencies"
    @echo "  doc         Generate and open documentation"

# Build the project
build:
    cargo build

# Run the server
run *ARGS:
    cargo run -- {{ARGS}}

# Run with hot reload (requires cargo-watch)
dev:
    cargo watch -x run

# Quick compile check
check:
    cargo check

# Run all tests
test:
    cargo test

# Run unit tests only
test-unit:
    cargo test --lib

# Run integration tests only
test-int:
    cargo test --test '*'

# Format code
fmt:
    cargo fmt

# Run clippy lints
lint:
    cargo clippy -- -D warnings

# Security audit dependencies (requires cargo-audit)
audit:
    cargo audit

# Run all quality checks
qa: fmt lint test

# Build optimized release binary
release:
    cargo build --release

# Install binary to cargo bin
install:
    cargo install --path .

# Clean build artifacts
clean:
    cargo clean

# Update dependencies
deps:
    cargo update

# Generate and open documentation
doc:
    cargo doc --open
