# Ingestion service — multi-stage Rust build
# Build context: invoice_parse/ root
#   docker build -f services/ingestion/Dockerfile .

FROM rust:1.92-slim AS builder

RUN apt-get update && apt-get install -y pkg-config && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY libs/shared-rs libs/shared-rs
COPY services/ingestion services/ingestion
RUN cargo build --release --manifest-path services/ingestion/Cargo.toml

FROM gcr.io/distroless/cc-debian13
COPY --from=builder /build/services/ingestion/target/release/ingestion /app/ingestion
COPY config/ /app/config/
ENV HOME=/tmp
WORKDIR /app
CMD ["/app/ingestion", "serve"]
