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

# Runtime stage: minimal distroless image
FROM gcr.io/distroless/cc-debian13:nonroot

WORKDIR /app

# Copy the compiled binary
COPY --from=builder /build/target/release/portfolio-rag-chat ./portfolio-rag-chat

# Copy static assets and templates from source context
COPY portfolio-rag-chat/static ./static
COPY portfolio-rag-chat/templates ./templates

# Set environment variables
# Note: /app/data will be mounted as a volume, /tmp is writable in distroless
ENV DB_PATH=/app/data/portfolio.lance
ENV RUST_LOG=info
ENV HOST=0.0.0.0
ENV PORT=3000
ENV HOME=/tmp
ENV FASTEMBED_CACHE_DIR=/tmp/.cache

# Expose the port
EXPOSE 3000

# Run the binary
CMD ["/app/portfolio-rag-chat"]
