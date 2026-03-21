# Ingestion service — multi-stage Rust build
# TODO: implement when service is built

FROM rust:1 AS builder
WORKDIR /app
COPY . .
RUN cargo build --release -p ingestion

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/ingestion /usr/local/bin/
CMD ["ingestion"]
