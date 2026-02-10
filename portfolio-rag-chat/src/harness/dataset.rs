use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct TestDataset {
    /// Human-readable purpose of this dataset
    pub description: String,

    /// Schema version — starts at 1, bump on breaking changes
    pub schema_version: u32,

    /// The test cases
    pub cases: Vec<TestCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestCase {
    // --- Identity ---
    /// Unique key for reporting, e.g. "overview-01", "hero-retriever"
    pub id: String,

    /// Natural language query to evaluate
    pub query: String,

    // --- Expectations (all optional via #[serde(default)]) ---
    /// Ground-truth intent classification
    /// Values: "overview" | "implementation" | "relationship" | "comparison"
    /// If None, intent accuracy is not evaluated for this case.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_intent: Option<String>,

    /// Expected file paths in retrieval results.
    /// Matching is **substring**: "retriever.rs" matches "src/engine/retriever.rs".
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub expected_files: Vec<String>,

    /// Expected function/class/struct names in results.
    /// Matching is **exact** against CodeChunk.identifier.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub expected_identifiers: Vec<String>,

    /// Expected chunk types that should appear in results.
    /// Values: "code" | "readme" | "crate" | "module_doc"
    /// At least one result of each listed type must appear.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub expected_chunk_types: Vec<String>,

    /// Expected project names in results.
    /// Matching is **exact** against project_name.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub expected_projects: Vec<String>,

    // --- Soft expectations (pipeline-agnostic) ---
    /// Minimum number of results with relevance > 0.5 expected.
    /// Pipeline-agnostic quality floor — survives routing/chunk-type changes.
    /// If None, this check is not evaluated for this case.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_relevant_results: Option<usize>,

    /// File paths that should NOT appear in results (substring match).
    /// Negative expectations are stable across pipeline evolution.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub excluded_files: Vec<String>,

    // --- Metadata ---
    /// Tags for filtering and categorization.
    /// Conventions: "hero", "edge_case", "no_results", "v1", "v2",
    ///   "overview", "implementation", "relationship", "comparison"
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,

    /// Free-form notes explaining why this test case exists
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

impl TestDataset {
    /// Load and deserialize from a JSON file.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read test dataset from {}", path.display()))?;

        let dataset: TestDataset = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse test dataset from {}", path.display()))?;

        Ok(dataset)
    }

    /// Filter cases by tag (e.g., "hero", "edge_case").
    pub fn filter_by_tag(&self, tag: &str) -> Vec<&TestCase> {
        self.cases
            .iter()
            .filter(|case| case.tags.contains(&tag.to_string()))
            .collect()
    }

    /// Structural validation. Returns warnings, not errors.
    pub fn validate(&self) -> Vec<String> {
        let mut warnings = Vec::new();

        // Check for zero test cases
        if self.cases.is_empty() {
            warnings.push("Dataset has zero test cases".to_string());
            return warnings; // No point in further validation
        }

        // Check for duplicate IDs
        let mut seen_ids = HashSet::new();
        for case in &self.cases {
            if !seen_ids.insert(&case.id) {
                warnings.push(format!("Duplicate test case ID: {}", case.id));
            }
        }

        // Check each test case
        let valid_intents = ["overview", "implementation", "relationship", "comparison"];
        let tier_tags = ["hero", "directional", "smoke"];

        for case in &self.cases {
            // Check for empty ID
            if case.id.is_empty() {
                warnings.push("Test case has empty ID".to_string());
            }

            // Check for unknown intent string
            if let Some(ref intent) = case.expected_intent {
                if !valid_intents.contains(&intent.as_str()) {
                    warnings.push(format!(
                        "Case {}: unknown intent '{}' (valid: {})",
                        case.id,
                        intent,
                        valid_intents.join(", ")
                    ));
                }
            }

            // Warn on overly broad file patterns
            for file in &case.expected_files {
                if file.len() < 8 || !file.contains('/') {
                    warnings.push(format!(
                        "Case {}: expected_file '{}' is short/generic. Consider more specific path.",
                        case.id, file
                    ));
                }
            }

            // Warn on multiple tier tags
            let case_tiers: Vec<_> = case
                .tags
                .iter()
                .filter(|t| tier_tags.contains(&t.as_str()))
                .collect();
            if case_tiers.len() > 1 {
                warnings.push(format!(
                    "Case {} has multiple tier tags: {:?}",
                    case.id, case_tiers
                ));
            }
        }

        warnings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serde_roundtrip() {
        let dataset = TestDataset {
            description: "Test dataset".to_string(),
            schema_version: 1,
            cases: vec![TestCase {
                id: "test-01".to_string(),
                query: "How does X work?".to_string(),
                expected_intent: Some("implementation".to_string()),
                expected_files: vec!["foo.rs".to_string()],
                expected_identifiers: vec!["bar".to_string()],
                expected_chunk_types: vec!["code".to_string()],
                expected_projects: vec!["project-a".to_string()],
                min_relevant_results: Some(3),
                excluded_files: vec!["baz.rs".to_string()],
                tags: vec!["hero".to_string()],
                notes: Some("Test note".to_string()),
            }],
        };

        let json = serde_json::to_string(&dataset).unwrap();
        let deserialized: TestDataset = serde_json::from_str(&json).unwrap();
        assert_eq!(dataset, deserialized);
    }

    #[test]
    fn test_filter_by_tag_hero() {
        let dataset = TestDataset {
            description: "Test".to_string(),
            schema_version: 1,
            cases: vec![
                TestCase {
                    id: "hero-01".to_string(),
                    query: "Q1".to_string(),
                    tags: vec!["hero".to_string()],
                    ..Default::default()
                },
                TestCase {
                    id: "smoke-01".to_string(),
                    query: "Q2".to_string(),
                    tags: vec!["smoke".to_string()],
                    ..Default::default()
                },
                TestCase {
                    id: "hero-02".to_string(),
                    query: "Q3".to_string(),
                    tags: vec!["hero".to_string(), "v2".to_string()],
                    ..Default::default()
                },
            ],
        };

        let heroes = dataset.filter_by_tag("hero");
        assert_eq!(heroes.len(), 2);
        assert_eq!(heroes[0].id, "hero-01");
        assert_eq!(heroes[1].id, "hero-02");
    }

    #[test]
    fn test_filter_by_tag_nonexistent() {
        let dataset = TestDataset {
            description: "Test".to_string(),
            schema_version: 1,
            cases: vec![TestCase {
                id: "test-01".to_string(),
                query: "Q1".to_string(),
                tags: vec!["hero".to_string()],
                ..Default::default()
            }],
        };

        let results = dataset.filter_by_tag("nonexistent");
        assert!(results.is_empty());
    }

    #[test]
    fn test_validate_good_dataset() {
        let dataset = TestDataset {
            description: "Test".to_string(),
            schema_version: 1,
            cases: vec![TestCase {
                id: "test-01".to_string(),
                query: "Q1".to_string(),
                expected_intent: Some("overview".to_string()),
                expected_files: vec!["src/engine/retriever.rs".to_string()],
                tags: vec!["hero".to_string()],
                ..Default::default()
            }],
        };

        let warnings = dataset.validate();
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_validate_empty_id() {
        let dataset = TestDataset {
            description: "Test".to_string(),
            schema_version: 1,
            cases: vec![TestCase {
                id: "".to_string(),
                query: "Q1".to_string(),
                ..Default::default()
            }],
        };

        let warnings = dataset.validate();
        assert!(warnings.iter().any(|w| w.contains("empty ID")));
    }

    #[test]
    fn test_validate_duplicate_ids() {
        let dataset = TestDataset {
            description: "Test".to_string(),
            schema_version: 1,
            cases: vec![
                TestCase {
                    id: "test-01".to_string(),
                    query: "Q1".to_string(),
                    ..Default::default()
                },
                TestCase {
                    id: "test-01".to_string(),
                    query: "Q2".to_string(),
                    ..Default::default()
                },
            ],
        };

        let warnings = dataset.validate();
        assert!(warnings.iter().any(|w| w.contains("Duplicate")));
    }

    #[test]
    fn test_validate_unknown_intent() {
        let dataset = TestDataset {
            description: "Test".to_string(),
            schema_version: 1,
            cases: vec![TestCase {
                id: "test-01".to_string(),
                query: "Q1".to_string(),
                expected_intent: Some("invalid_intent".to_string()),
                ..Default::default()
            }],
        };

        let warnings = dataset.validate();
        assert!(warnings.iter().any(|w| w.contains("unknown intent")));
    }

    #[test]
    fn test_validate_zero_cases() {
        let dataset = TestDataset {
            description: "Test".to_string(),
            schema_version: 1,
            cases: vec![],
        };

        let warnings = dataset.validate();
        assert!(warnings.iter().any(|w| w.contains("zero test cases")));
    }

    #[test]
    fn test_minimal_case() {
        let json = r#"
        {
            "description": "Minimal test",
            "schema_version": 1,
            "cases": [
                {
                    "id": "min-01",
                    "query": "What is X?"
                }
            ]
        }
        "#;

        let dataset: TestDataset = serde_json::from_str(json).unwrap();
        assert_eq!(dataset.cases[0].id, "min-01");
        assert_eq!(dataset.cases[0].query, "What is X?");
        assert!(dataset.cases[0].expected_intent.is_none());
        assert!(dataset.cases[0].expected_files.is_empty());
        assert!(dataset.cases[0].expected_identifiers.is_empty());
        assert!(dataset.cases[0].expected_chunk_types.is_empty());
        assert!(dataset.cases[0].expected_projects.is_empty());
        assert!(dataset.cases[0].min_relevant_results.is_none());
        assert!(dataset.cases[0].excluded_files.is_empty());
        assert!(dataset.cases[0].tags.is_empty());
        assert!(dataset.cases[0].notes.is_none());
    }

    #[test]
    fn test_unknown_fields_ignored() {
        let json = r#"
        {
            "description": "Test",
            "schema_version": 1,
            "unknown_root_field": "ignored",
            "cases": [
                {
                    "id": "test-01",
                    "query": "What is X?",
                    "unknown_case_field": 42
                }
            ]
        }
        "#;

        let result: Result<TestDataset, _> = serde_json::from_str(json);
        assert!(result.is_ok());
    }

    #[test]
    fn test_min_relevant_results_roundtrip() {
        let json = r#"
        {
            "description": "Test",
            "schema_version": 1,
            "cases": [
                {
                    "id": "test-01",
                    "query": "What is X?",
                    "min_relevant_results": 3
                }
            ]
        }
        "#;

        let dataset: TestDataset = serde_json::from_str(json).unwrap();
        assert_eq!(dataset.cases[0].min_relevant_results, Some(3));

        let serialized = serde_json::to_string(&dataset).unwrap();
        let deserialized: TestDataset = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.cases[0].min_relevant_results, Some(3));
    }

    #[test]
    fn test_min_relevant_results_absent() {
        let json = r#"
        {
            "description": "Test",
            "schema_version": 1,
            "cases": [
                {
                    "id": "test-01",
                    "query": "What is X?"
                }
            ]
        }
        "#;

        let dataset: TestDataset = serde_json::from_str(json).unwrap();
        assert!(dataset.cases[0].min_relevant_results.is_none());
    }

    #[test]
    fn test_excluded_files_roundtrip() {
        let json = r#"
        {
            "description": "Test",
            "schema_version": 1,
            "cases": [
                {
                    "id": "test-01",
                    "query": "What is X?",
                    "excluded_files": ["retriever.rs", "parser.rs"]
                }
            ]
        }
        "#;

        let dataset: TestDataset = serde_json::from_str(json).unwrap();
        assert_eq!(dataset.cases[0].excluded_files.len(), 2);
        assert_eq!(dataset.cases[0].excluded_files[0], "retriever.rs");

        let serialized = serde_json::to_string(&dataset).unwrap();
        let deserialized: TestDataset = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.cases[0].excluded_files.len(), 2);
    }

    #[test]
    fn test_excluded_files_absent() {
        let json = r#"
        {
            "description": "Test",
            "schema_version": 1,
            "cases": [
                {
                    "id": "test-01",
                    "query": "What is X?"
                }
            ]
        }
        "#;

        let dataset: TestDataset = serde_json::from_str(json).unwrap();
        assert!(dataset.cases[0].excluded_files.is_empty());
    }

    #[test]
    fn test_smoke_case() {
        let json = r#"
        {
            "description": "Test",
            "schema_version": 1,
            "cases": [
                {
                    "id": "smoke-01",
                    "query": "Overview of system",
                    "min_relevant_results": 3,
                    "excluded_files": ["auth.rs"],
                    "tags": ["smoke"]
                }
            ]
        }
        "#;

        let dataset: TestDataset = serde_json::from_str(json).unwrap();
        assert_eq!(dataset.cases[0].id, "smoke-01");
        assert_eq!(dataset.cases[0].min_relevant_results, Some(3));
        assert_eq!(dataset.cases[0].excluded_files.len(), 1);
        assert!(dataset.cases[0].expected_files.is_empty());
        assert!(dataset.cases[0].expected_identifiers.is_empty());
    }

    #[test]
    fn test_validate_short_file_pattern() {
        let dataset = TestDataset {
            description: "Test".to_string(),
            schema_version: 1,
            cases: vec![TestCase {
                id: "test-01".to_string(),
                query: "Q1".to_string(),
                expected_files: vec!["lib.rs".to_string()],
                ..Default::default()
            }],
        };

        let warnings = dataset.validate();
        assert!(warnings
            .iter()
            .any(|w| w.contains("short/generic") && w.contains("lib.rs")));
    }

    #[test]
    fn test_validate_multiple_tier_tags() {
        let dataset = TestDataset {
            description: "Test".to_string(),
            schema_version: 1,
            cases: vec![TestCase {
                id: "test-01".to_string(),
                query: "Q1".to_string(),
                tags: vec!["hero".to_string(), "smoke".to_string()],
                ..Default::default()
            }],
        };

        let warnings = dataset.validate();
        assert!(warnings
            .iter()
            .any(|w| w.contains("multiple tier tags")));
    }

    #[test]
    fn test_load_test_corpus() {
        let corpus_path = std::path::Path::new("data/test_queries.json");

        // Skip if file doesn't exist (e.g., in CI without data/)
        if !corpus_path.exists() {
            eprintln!("Skipping test_load_test_corpus: data/test_queries.json not found");
            return;
        }

        let dataset = TestDataset::load(corpus_path).unwrap();

        // Verify basic structure
        assert_eq!(dataset.schema_version, 1);
        assert!(dataset.cases.len() >= 30, "Expected at least 30 test cases, got {}", dataset.cases.len());
        assert!(dataset.cases.len() <= 45, "Expected at most 45 test cases, got {}", dataset.cases.len());

        // Verify no validation warnings
        let warnings = dataset.validate();
        assert!(warnings.is_empty(), "Test corpus has validation warnings: {:?}", warnings);

        // Verify required coverage
        let heroes = dataset.filter_by_tag("hero");
        assert!(heroes.len() >= 4, "Expected at least 4 hero queries, got {}", heroes.len());

        let smoke = dataset.filter_by_tag("smoke");
        assert!(smoke.len() >= 5, "Expected at least 5 smoke queries, got {}", smoke.len());

        let edge = dataset.filter_by_tag("edge_case");
        assert!(edge.len() >= 3, "Expected at least 3 edge cases, got {}", edge.len());

        // Verify all IDs are unique (covered by validate but double-check)
        let ids: std::collections::HashSet<_> = dataset.cases.iter().map(|c| &c.id).collect();
        assert_eq!(ids.len(), dataset.cases.len(), "Duplicate IDs found in corpus");
    }
}

// Default implementation for TestCase to make test construction easier
impl Default for TestCase {
    fn default() -> Self {
        Self {
            id: String::new(),
            query: String::new(),
            expected_intent: None,
            expected_files: Vec::new(),
            expected_identifiers: Vec::new(),
            expected_chunk_types: Vec::new(),
            expected_projects: Vec::new(),
            min_relevant_results: None,
            excluded_files: Vec::new(),
            tags: Vec::new(),
            notes: None,
        }
    }
}
