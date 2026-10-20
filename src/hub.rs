//! Tokenizers and datasets from the Hugging Face Hub, over plain HTTPS.
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokenizers::Tokenizer;

/// Largest file we download (datasets can be large; tokenizers are a few MB).
const MAX_DOWNLOAD_BYTES: usize = 2 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepoType {
    Model,
    Dataset,
}

impl RepoType {
    fn url_prefix(self) -> &'static str {
        match self {
            RepoType::Model => "",
            RepoType::Dataset => "datasets/",
        }
    }

    fn cache_dir(self) -> &'static str {
        match self {
            RepoType::Model => "models",
            RepoType::Dataset => "datasets",
        }
    }
}

fn endpoint() -> String {
    std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string())
}

fn hf_home() -> Option<PathBuf> {
    std::env::var_os("HF_HOME").map(PathBuf::from).or_else(|| {
        std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/huggingface"))
    })
}

/// The token saved by `huggingface-cli login`, if any.
pub fn stored_token() -> Option<String> {
    let token = std::fs::read_to_string(hf_home()?.join("token")).ok()?;
    let token = token.trim();
    (!token.is_empty()).then(|| token.to_string())
}

/// Where the CLI caches downloads: `$HF_HOME/inference-benchmarker`.
pub fn default_cache_dir() -> PathBuf {
    hf_home()
        .unwrap_or_else(std::env::temp_dir)
        .join("inference-benchmarker")
}

fn check_part(part: &str, what: &str) -> anyhow::Result<()> {
    let bad = part.is_empty()
        || part.starts_with('/')
        || part
            .split('/')
            .any(|seg| seg.is_empty() || seg == "." || seg == "..");
    if bad {
        return Err(anyhow::anyhow!("invalid {what}: {part:?}"));
    }
    Ok(())
}

/// Download one file of a repository (`main` revision).
pub async fn download(
    repo_type: RepoType,
    repo: &str,
    file: &str,
    token: Option<&str>,
) -> anyhow::Result<Vec<u8>> {
    check_part(repo, "repository")?;
    check_part(file, "file name")?;
    let url = format!(
        "{}/{}{repo}/resolve/main/{file}",
        endpoint(),
        repo_type.url_prefix()
    );
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()?;
    let mut request = client.get(&url);
    if let Some(token) = token {
        request = request.bearer_auth(token);
    }
    let mut response = request.send().await?.error_for_status().map_err(|e| {
        anyhow::anyhow!("download {repo}/{file} failed: {e} (gated repositories need a token)")
    })?;
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if body.len() + chunk.len() > MAX_DOWNLOAD_BYTES {
            return Err(anyhow::anyhow!("{repo}/{file} is larger than 2 GiB"));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

/// Download a file once into `cache_dir` and return its path.
pub async fn download_cached(
    repo_type: RepoType,
    repo: &str,
    file: &str,
    token: Option<&str>,
    cache_dir: &Path,
) -> anyhow::Result<PathBuf> {
    check_part(repo, "repository")?;
    check_part(file, "file name")?;
    let path = cache_dir.join(repo_type.cache_dir()).join(repo).join(file);
    if tokio::fs::try_exists(&path).await? {
        return Ok(path);
    }
    let body = download(repo_type, repo, file, token).await?;
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    // write then rename, so a crash never leaves a truncated file in the cache
    let tmp = path.with_extension(format!("part-{}", uuid::Uuid::new_v4()));
    tokio::fs::write(&tmp, body).await?;
    tokio::fs::rename(&tmp, &path).await?;
    Ok(path)
}

/// Parse a `tokenizer.json`.
pub fn tokenizer_from_bytes(bytes: &[u8]) -> anyhow::Result<Tokenizer> {
    Tokenizer::from_bytes(bytes).map_err(|e| anyhow::anyhow!("Error loading tokenizer: {e}"))
}

/// The `tokenizer.json` of a model repository, cached when `cache_dir` is set.
pub async fn load_tokenizer(
    repo: &str,
    token: Option<&str>,
    cache_dir: Option<&Path>,
) -> anyhow::Result<Tokenizer> {
    let bytes = match cache_dir {
        Some(dir) => {
            let path = download_cached(RepoType::Model, repo, "tokenizer.json", token, dir).await?;
            tokio::fs::read(path).await?
        }
        None => download(RepoType::Model, repo, "tokenizer.json", token).await?,
    };
    tokenizer_from_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_paths_that_escape_the_cache() {
        assert!(check_part("google/gemma-3-12b-it", "repository").is_ok());
        assert!(check_part("data/train.json", "file name").is_ok());
        for bad in ["", "/etc", "a/../b", "a//b", ".."] {
            assert!(check_part(bad, "x").is_err(), "{bad:?}");
        }
    }
}
