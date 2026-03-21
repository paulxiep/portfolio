# Output service — multi-stage Rust build
# TODO: implement when service is built

FROM rust:1 AS builder
WORKDIR /app
COPY . .
RUN cargo build --release -p output

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/output /usr/local/bin/
CMD ["output"]
