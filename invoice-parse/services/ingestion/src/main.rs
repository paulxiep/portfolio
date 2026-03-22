mod ingest;

use std::path::PathBuf;
use std::time::Instant;

use log::{error, info};
use shared_rs::adapters::blob_store::LocalFsBlobStore;
use shared_rs::adapters::queue::RedisStreamQueue;
use shared_rs::config::load_config;
use shared_rs::db::create_pool;

use crate::ingest::ingest_file;

const SUPPORTED_EXTENSIONS: &[&str] = &["pdf", "png", "jpg", "jpeg", "webp", "tiff", "bmp"];

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  ingestion <file_or_directory>   Ingest invoice file(s)");
        eprintln!("  ingestion serve                 Start IMAP poll + health endpoint (future)");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "serve" => {
            eprintln!("IMAP polling server not yet implemented. Use file ingestion mode.");
            std::process::exit(1);
        }
        path => {
            run_ingest(path).await;
        }
    }
}

async fn run_ingest(path: &str) {
    let config = load_config(None).expect("Failed to load config");

    let blob_store = LocalFsBlobStore::new(
        config
            .blob_storage
            .base_path
            .as_deref()
            .expect("blob_storage.base_path required"),
    )
    .expect("Failed to create blob store");

    let queue = RedisStreamQueue::new(
        config.queue.url.as_deref().expect("queue.url required"),
        config
            .queue
            .consumer_group
            .as_deref()
            .unwrap_or("invoice_workers"),
    )
    .expect("Failed to create queue");

    let pool = create_pool(&config)
        .await
        .expect("Failed to connect to database");

    let input = PathBuf::from(path);
    let files: Vec<PathBuf> = if input.is_dir() {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&input)
            .expect("Failed to read directory")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| SUPPORTED_EXTENSIONS.contains(&ext.to_lowercase().as_str()))
            })
            .collect();
        entries.sort();
        entries
    } else {
        vec![input]
    };

    if files.is_empty() {
        eprintln!("No supported invoice files found in {}", path);
        std::process::exit(1);
    }

    info!("Ingesting {} file(s) from {}", files.len(), path);
    let start = Instant::now();
    let mut success = 0;
    let mut failed = 0;

    for file_path in &files {
        let filename = file_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        let file_bytes = match std::fs::read(file_path) {
            Ok(bytes) => bytes,
            Err(e) => {
                error!("Failed to read {}: {}", filename, e);
                failed += 1;
                continue;
            }
        };

        match ingest_file(
            filename,
            &file_bytes,
            &pool,
            &blob_store,
            &queue,
            "cli",
            "load_test",
        )
        .await
        {
            Ok(job_id) => {
                info!("  {} → job {}", filename, job_id);
                success += 1;
            }
            Err(e) => {
                error!("  {} → FAILED: {}", filename, e);
                failed += 1;
            }
        }
    }

    let elapsed = start.elapsed();
    println!(
        "\nEnqueued {} jobs in {:.0}ms ({} failed)",
        success,
        elapsed.as_millis(),
        failed,
    );
}
