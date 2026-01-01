# Multi-stage Dockerfile for portfolio-rag-chat
# Builder stage: compile the Rust application
FROM rust:1.92-slim AS builder

# Install necessary build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    protobuf-compiler \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create a new directory for the app
WORKDIR /build

# Copy manifests from parent context
COPY portfolio-rag-chat/Cargo.toml portfolio-rag-chat/Cargo.lock ./

# Create a dummy main.rs to build dependencies first (caching optimization)
RUN mkdir -p src && \
    echo "fn main() {println!(\"dummy\")}" > src/main.rs && \
    cargo build --release && \
    rm -rf src

# Copy the source code and templates (needed at compile time for Askama)
COPY portfolio-rag-chat/src ./src
COPY portfolio-rag-chat/templates ./templates

# Build the actual application
# Touch main.rs to force rebuild of our code only
RUN touch src/main.rs && cargo build --release

# Ingestion stage: Create the Lance database
FROM debian:trixie-slim AS ingestion

# Install necessary runtime dependencies
RUN apt-get update && apt-get install -y curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy necessary files for ingestion
COPY --from=builder /build/target/release/portfolio-rag-chat ./portfolio-rag-chat
COPY portfolio-rag-chat/static ./static
COPY portfolio-rag-chat/templates ./templates

# Copy the entire portfolio repository (from parent context)
COPY . /portfolio

# Receive build arg and set as env
ARG GEMINI_API_KEY
ENV GEMINI_API_KEY=${GEMINI_API_KEY}

# Set up environment for ingestion
ENV DB_PATH=/app/data/portfolio.lance
ENV RUST_LOG=info
ENV HOST=127.0.0.1
ENV PORT=3000
ENV FASTEMBED_CACHE_DIR=/app/.cache

# Create data directory
RUN mkdir -p /app/data

# Download htmx if needed for templates
RUN test -f /app/static/htmx.min.js || \
    curl -sSL https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js -o /app/static/htmx.min.js

# Pre-download embedding model
RUN mkdir -p /app/.cache && \
    /app/portfolio-rag-chat --warmup

# Run ingestion by starting server, calling ingest endpoint, then stopping
# Note: This requires a valid GEMINI_API_KEY at build time
RUN set -e && \
    (/app/portfolio-rag-chat > /tmp/server.log 2>&1) & \
    SERVER_PID=$! && \
    echo "Waiting for server to start..." && \
    for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do \
        if curl -s http://127.0.0.1:3000/health > /dev/null 2>&1; then \
            echo "Server is ready"; \
            break; \
        fi; \
        echo "Waiting... ($i/30)"; \
        sleep 2; \
    done && \
    curl -s http://127.0.0.1:3000/health > /dev/null || (echo "Server failed to start" && cat /tmp/server.log && exit 1) && \
    echo "Starting ingestion..." && \
    curl -X POST http://127.0.0.1:3000/ingest \
         -H "Content-Type: application/json" \
         -d '{"repo_path": "/portfolio"}' \
         --max-time 600 \
         --retry 1 \
         --fail \
         -v && \
    echo "Ingestion complete, stopping server..." && \
    kill $SERVER_PID 2>/dev/null || true && \
    wait $SERVER_PID 2>/dev/null || true && \
    echo "Server logs:" && \
    cat /tmp/server.log && \
    # Verify database was actually created with tables
    test -d /app/data/portfolio.lance && \
    ls /app/data/portfolio.lance/code_chunks.lance/data/*.lance > /dev/null 2>&1 && \
    echo "Ingestion verified successfully"

# Runtime stage: minimal distroless image
FROM gcr.io/distroless/cc-debian13:nonroot

# Copy the compiled binary
COPY --from=builder /build/target/release/portfolio-rag-chat /app/portfolio-rag-chat

# Copy static assets and templates
COPY --from=ingestion /app/static /app/static
COPY --from=ingestion /app/templates /app/templates

# Copy the pre-ingested Lance database
COPY --from=ingestion /app/data /app/data

# Copy the embedding model cache to /tmp (writable in distroless)
COPY --from=ingestion --chown=65532:65532 /app/.cache /tmp/.cache

# Set working directory
WORKDIR /app

# Set environment variables - /tmp is writable in distroless
ENV HOME=/tmp
ENV FASTEMBED_CACHE_DIR=/tmp/.cache

# Expose the port
EXPOSE 3000

# Run the binary
CMD ["/app/portfolio-rag-chat"]
