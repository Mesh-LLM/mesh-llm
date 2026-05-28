use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub name: String,
    pub description: String,
    pub github_url: String,
    pub author_email: String,
    pub author_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginCatalog {
    entries: Vec<CatalogEntry>,
}

impl PluginCatalog {
    pub fn parse_jsonl(input: &str) -> Result<Self> {
        let mut entries = Vec::new();
        for (index, line) in input.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let entry: CatalogEntry = serde_json::from_str(line)
                .with_context(|| format!("parse plugins.jsonl line {}", index + 1))?;
            entries.push(entry);
        }
        entries.sort_by(|left, right| left.name.cmp(&right.name));
        ensure_unique_names(&entries)?;
        Ok(Self { entries })
    }

    pub async fn fetch(client: &Client, url: &str) -> Result<Self> {
        let response = client
            .get(url)
            .header(reqwest::header::USER_AGENT, crate::github::USER_AGENT)
            .send()
            .await
            .with_context(|| format!("fetch plugin catalog {url}"))?;
        let status = response.status();
        if !status.is_success() {
            bail!("plugin catalog request failed: {status} {url}");
        }
        let body = response
            .text()
            .await
            .with_context(|| format!("read plugin catalog {url}"))?;
        Self::parse_jsonl(&body)
    }

    pub fn entries(&self) -> &[CatalogEntry] {
        &self.entries
    }

    pub fn find_exact(&self, name: &str) -> Option<&CatalogEntry> {
        self.entries.iter().find(|entry| entry.name == name)
    }

    pub fn search(&self, query: Option<&str>) -> Vec<&CatalogEntry> {
        let Some(query) = query.map(str::trim).filter(|query| !query.is_empty()) else {
            return self.entries.iter().collect();
        };
        let query = query.to_ascii_lowercase();
        self.entries
            .iter()
            .filter(|entry| {
                entry.name.to_ascii_lowercase().contains(&query)
                    || entry.description.to_ascii_lowercase().contains(&query)
                    || entry.author_name.to_ascii_lowercase().contains(&query)
            })
            .collect()
    }
}

fn ensure_unique_names(entries: &[CatalogEntry]) -> Result<()> {
    for pair in entries.windows(2) {
        if pair[0].name == pair[1].name {
            bail!("duplicate plugin catalog entry '{}'", pair[0].name);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_searches_catalog_jsonl() {
        let catalog = PluginCatalog::parse_jsonl(
            r#"{"name":"blackboard","description":"Shared notes","github_url":"https://github.com/mesh-llm/blackboard","author_email":"maintainers@meshllm.cloud","author_name":"Mesh LLM"}
{"name":"notes","description":"Team notes","github_url":"https://github.com/acme/notes","author_email":"dev@example.com","author_name":"Acme"}
"#,
        )
        .unwrap();
        assert_eq!(catalog.entries().len(), 2);
        assert_eq!(
            catalog.find_exact("blackboard").unwrap().author_name,
            "Mesh LLM"
        );
        assert_eq!(catalog.search(Some("team"))[0].name, "notes");
    }

    #[test]
    fn rejects_duplicate_names() {
        let err = PluginCatalog::parse_jsonl(
            r#"{"name":"blackboard","description":"A","github_url":"https://github.com/mesh-llm/blackboard","author_email":"a@example.com","author_name":"A"}
{"name":"blackboard","description":"B","github_url":"https://github.com/mesh-llm/blackboard2","author_email":"b@example.com","author_name":"B"}
"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("duplicate"));
    }
}
