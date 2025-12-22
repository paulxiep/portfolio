mod ingestion;
mod models;

use ingestion::run_ingestion;

fn main() {
    println!("Portfolio RAG Chat - Ingestion Test");
    println!("====================================\n");

    // Point to your portfolio directory
    let portfolio_path = r"c:\Users\paulx\Documents\portfolio";

    println!("Starting ingestion from: {}", portfolio_path);
    let (code_chunks, readme_chunks) = run_ingestion(portfolio_path);

    println!("\n📊 Ingestion Results:");
    println!("  - Code chunks extracted: {}", code_chunks.len());
    println!("  - README files found: {}", readme_chunks.len());

    // Show some sample output
    if !code_chunks.is_empty() {
        println!("\n📝 Sample Code Chunks (first 3):");
        for (i, chunk) in code_chunks.iter().take(3).enumerate() {
            println!(
                "  {}. {} in {} ({}:{})",
                i + 1,
                chunk.identifier,
                chunk.project_name.as_deref().unwrap_or("unknown"),
                chunk.language,
                chunk.start_line
            );
        }
    }

    if !readme_chunks.is_empty() {
        println!("\n📚 README Files Found:");
        for readme in &readme_chunks {
            println!(
                "  - {} ({} chars)",
                readme.project_name,
                readme.content.len()
            );
        }
    }

    println!("\n✅ Ingestion complete!");
}
