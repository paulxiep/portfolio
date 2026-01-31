# Multi-stage Dockerfile for portfolio-rag-chat workspace
# Supports two targets:
#   - chat (default): The query server
#   - raptor: The ingestion CLI
#
# Usage:
#   docker build --target chat -t portfolio-rag-chat .
#   docker build --target raptor -t code-raptor .

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

WORKDIR /build

# Copy workspace manifests
COPY portfolio-rag-chat/Cargo.toml portfolio-rag-chat/Cargo.lock ./
COPY portfolio-rag-chat/crates/coderag-types/Cargo.toml ./crates/coderag-types/
COPY portfolio-rag-chat/crates/coderag-store/Cargo.toml ./crates/coderag-store/
COPY portfolio-rag-chat/crates/code-raptor/Cargo.toml ./crates/code-raptor/

# Create dummy sources for dependency caching
RUN mkdir -p src crates/coderag-types/src crates/coderag-store/src crates/code-raptor/src && \
    echo "fn main() {println!(\"dummy\")}" > src/main.rs && \
    echo "// dummy" > crates/coderag-types/src/lib.rs && \
    echo "// dummy" > crates/coderag-store/src/lib.rs && \
    echo "fn main() {}" > crates/code-raptor/src/main.rs && \
    echo "// dummy" > crates/code-raptor/src/lib.rs

# Build all workspace crates to cache dependencies
RUN cargo build --release --workspace && \
    rm -rf src crates/coderag-types/src crates/coderag-store/src crates/code-raptor/src

# Copy actual source for all crates
COPY portfolio-rag-chat/src ./src
COPY portfolio-rag-chat/crates/coderag-types/src ./crates/coderag-types/src
COPY portfolio-rag-chat/crates/coderag-store/src ./crates/coderag-store/src
COPY portfolio-rag-chat/crates/code-raptor/src ./crates/code-raptor/src
COPY portfolio-rag-chat/templates ./templates

# Rebuild with real source (touch ALL source files to invalidate cache)
RUN touch src/main.rs \
          crates/coderag-types/src/lib.rs \
          crates/coderag-store/src/lib.rs \
          crates/code-raptor/src/lib.rs \
          crates/code-raptor/src/main.rs && \
    cargo build --release --workspace

# ============================================================
# Target: chat - The query server (default)
# ============================================================
FROM gcr.io/distroless/cc-debian13:nonroot AS chat

WORKDIR /app

COPY --from=builder /build/target/release/portfolio-rag-chat ./portfolio-rag-chat
COPY portfolio-rag-chat/static ./static
COPY portfolio-rag-chat/templates ./templates

ENV DB_PATH=/app/data/portfolio.lance
ENV RUST_LOG=info
ENV HOST=0.0.0.0
ENV PORT=3000
ENV HOME=/tmp
ENV FASTEMBED_CACHE_DIR=/tmp/.cache

EXPOSE 3000

CMD ["/app/portfolio-rag-chat"]

# ============================================================
# Target: raptor - The ingestion CLI
# ============================================================
FROM gcr.io/distroless/cc-debian13 AS raptor

WORKDIR /app

COPY --from=builder /build/target/release/code-raptor ./code-raptor

ENV DB_PATH=/app/data/portfolio.lance
ENV RUST_LOG=info
ENV HOME=/tmp
ENV FASTEMBED_CACHE_DIR=/tmp/.cache

# No default CMD - user provides subcommand (ingest, status)
