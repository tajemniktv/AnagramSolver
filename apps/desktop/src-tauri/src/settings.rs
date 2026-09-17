use anagram_core::policy::DeploymentLimits;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Corpora {
    pub dictionary: String,
    pub unigrams: String,
    pub bigrams: String,
    pub wordnet: String,
    pub phrase: String,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    #[serde(default)]
    pub runtime: Runtime,
    #[serde(default)]
    pub model: String,
    pub schema_version: u32,
    pub theme: String,
    pub corpora: Corpora,
    pub request: Value,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Runtime {
    pub cache_path: String,
    pub cache_entries: u32,
    pub cache_bytes: u32,
    pub custom_limits: bool,
    pub limits: DeploymentLimits,
}
impl Default for Runtime {
    fn default() -> Self {
        Self {
            cache_path: String::new(),
            cache_entries: 32,
            cache_bytes: 32 * 1024 * 1024,
            custom_limits: false,
            limits: limits(),
        }
    }
}
impl Runtime {
    pub fn cache_path(&self, data: &Path) -> PathBuf {
        if self.cache_path.is_empty() {
            data.join("results.sqlite")
        } else {
            PathBuf::from(&self.cache_path)
        }
    }
    pub fn cache_limits(&self) -> anagram_core::cache::StorageLimits {
        anagram_core::cache::StorageLimits {
            max_entries: self.cache_entries,
            max_payload_bytes: self.cache_bytes,
        }
    }
}

pub fn limits() -> DeploymentLimits {
    DeploymentLimits {
        max_input_bytes: Some(4096),
        max_normalized_letters: Some(40),
        max_candidates: Some(250_000),
        max_deep_analyzed: Some(10_000),
        max_beam_width: Some(512),
        max_retained_orders: Some(128),
        timeout_ms: Some(120_000),
    }
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            runtime: Runtime::default(),
            model: String::new(),
            schema_version: 1,
            theme: "dark".into(),
            corpora: Corpora::at(Path::new("corpora")),
            request: json!({ "generation": {"schema_version":1,"text":"","required":[],"hints":[],"excluded":[],
            "exclude_regex":[],"forbid_chars":"","short_policy":"common","extra_short_words":["hi"],
            "min_words":1,"max_words":6,"min_word_length":1,"max_word_length":24,"min_zipf":1.8,
            "candidate_budget":10000,"allow_repeat":false,"strategy":"prefix","hint_mode":"any"},
            "deep_per_group":100,"deep_all":false,"order_mode":"auto","beam_width":128,"exact_max_words":5,
            "retained_orders":56,"phrase_rescore_top":300,"phrase_bonus_max":6.0,"positive_bigrams":true,"result_limit_per_group":10 }),
        }
    }
}
impl Corpora {
    fn at(root: &Path) -> Self {
        let path = |suffix| root.join(suffix).to_string_lossy().into_owned();
        Self {
            dictionary: path("dictionary/normal_user_v2.txt"),
            unigrams: path("ngrams/count_1w.txt"),
            bigrams: path("ngrams/count_2w.txt"),
            wordnet: path("wordnet31/dict"),
            phrase: String::new(),
        }
    }
}
fn defaults(corpora_root: &Path) -> Settings {
    Settings {
        corpora: Corpora::at(corpora_root),
        ..Settings::default()
    }
}
// Migration only: the compiled checkout path is never read or used as a fallback.
pub fn relocate_path(value: &mut String, data: &Path) -> bool {
    let Some(install) = data.parent() else {
        return false;
    };
    let Some(parent) = install.parent() else {
        return false;
    };
    let old = parent.join("AnagramSolver");
    let normalize = |p: &str| p.trim_start_matches(r"\\?\").replace('/', "\\");
    let prefix = format!("{}\\", normalize(&old.to_string_lossy()));
    let current = normalize(value);
    if current
        .to_ascii_lowercase()
        .starts_with(&prefix.to_ascii_lowercase())
    {
        let target = install.join(&current[prefix.len()..]);
        if target.exists() {
            *value = target.to_string_lossy().into_owned();
            return true;
        }
    }
    false
}

fn migrate_repository_paths(settings: &mut Settings, dir: &Path, corpora_root: &Path) -> bool {
    let mut relocated = false;
    for path in [
        &mut settings.corpora.dictionary,
        &mut settings.corpora.unigrams,
        &mut settings.corpora.bigrams,
        &mut settings.corpora.wordnet,
        &mut settings.corpora.phrase,
        &mut settings.model,
        &mut settings.runtime.cache_path,
    ] {
        relocated |= relocate_path(path, dir);
    }
    let legacy_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .unwrap()
        .join(".anagram_data");
    let legacy = Corpora::at(&legacy_root);
    let previous = Corpora::at(&dir.join("corpora"));
    let local = Corpora::at(corpora_root);
    let mut changed = relocated;
    for (current, old, old_local, new) in [
        (
            &mut settings.corpora.dictionary,
            legacy.dictionary,
            previous.dictionary,
            local.dictionary,
        ),
        (
            &mut settings.corpora.unigrams,
            legacy.unigrams,
            previous.unigrams,
            local.unigrams,
        ),
        (
            &mut settings.corpora.bigrams,
            legacy.bigrams,
            previous.bigrams,
            local.bigrams,
        ),
        (
            &mut settings.corpora.wordnet,
            legacy.wordnet,
            previous.wordnet,
            local.wordnet,
        ),
    ] {
        let normalize = |path: &str| path.trim_start_matches(r"\\?\").replace('/', "\\");
        let old_profile_match = std::env::var_os("LOCALAPPDATA").is_some_and(|root| {
            Path::new(&new)
                .strip_prefix(corpora_root)
                .is_ok_and(|suffix| {
                    let old_path = PathBuf::from(root)
                        .join("tv.tajemnik.anagramsolver/corpora")
                        .join(suffix);
                    normalize(current).eq_ignore_ascii_case(&normalize(&old_path.to_string_lossy()))
                })
        });
        if (normalize(current).eq_ignore_ascii_case(&normalize(&old))
            || normalize(current).eq_ignore_ascii_case(&normalize(&old_local))
            || old_profile_match
            || std::env::var_os("LOCALAPPDATA").is_some_and(|root| {
                Path::new(&new)
                    .strip_prefix(corpora_root)
                    .is_ok_and(|suffix| {
                        let old_path = PathBuf::from(root)
                            .join("Programs/TajemnikTV/AnagramSolver/corpora")
                            .join(suffix);
                        normalize(current)
                            .eq_ignore_ascii_case(&normalize(&old_path.to_string_lossy()))
                    })
            }))
            && Path::new(&new).exists()
        {
            *current = new;
            changed = true;
        }
    }
    changed
}
pub fn load(dir: &Path) -> (Settings, Option<String>) {
    let executable = std::env::current_exe().expect("The running executable must have a path");
    load_with_corpora(dir, &executable.parent().unwrap().join("corpora"))
}
pub(super) fn load_with_corpora(dir: &Path, corpora_root: &Path) -> (Settings, Option<String>) {
    let path = dir.join("settings.json");
    // Bound the read itself, not merely deserialization after an unbounded allocation.
    let bytes = fs::File::open(&path).and_then(|file| {
        let mut bytes = Vec::new();
        file.take(1_048_577).read_to_end(&mut bytes)?;
        Ok(bytes)
    });
    match bytes {
        Ok(bytes) if bytes.len() <= 1_048_576 => match serde_json::from_slice::<Settings>(&bytes) {
            Ok(mut settings) if validate(&settings).is_ok() => {
                let warning = if migrate_repository_paths(&mut settings, dir, corpora_root) {
                    save(dir, &settings).err().map(|e| {
                        format!("Corpus paths migrated but settings could not be saved: {e}")
                    })
                } else {
                    None
                };
                (settings, warning)
            }
            _ => (
                defaults(corpora_root),
                Some(format!(
                    "Settings could not be read. Original file preserved at {}. Saving will replace it.",
                    path.display()
                )),
            ),
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (defaults(corpora_root), None),
        _ => (
            defaults(corpora_root),
            Some(format!("Unable to load settings at {}", path.display())),
        ),
    }
}
pub fn validate(settings: &Settings) -> Result<(), String> {
    settings.runtime.limits.validate().map_err(|e| e.message)?;
    if settings.runtime.cache_entries == 0 || settings.runtime.cache_bytes == 0 {
        return Err("Cache bounds must be positive".into());
    }
    if !settings.runtime.cache_path.is_empty()
        && !Path::new(&settings.runtime.cache_path).is_absolute()
    {
        return Err("Cache path must be absolute".into());
    }
    if settings.schema_version != 1
        || !["dark", "light", "system"].contains(&settings.theme.as_str())
    {
        return Err("Unsupported settings version or theme".into());
    }
    let _: anagram_core::solve::Request =
        serde_json::from_value(settings.request.clone()).map_err(|e| e.to_string())?;
    if serde_json::to_vec(settings)
        .map_err(|e| e.to_string())?
        .len()
        > 1_048_576
    {
        return Err("Settings exceed 1 MiB".into());
    }
    Ok(())
}
pub fn save(dir: &Path, settings: &Settings) -> Result<(), String> {
    validate(settings)?;
    let mut file = tempfile::NamedTempFile::new_in(dir).map_err(|e| e.to_string())?;
    serde_json::to_writer_pretty(&mut file, settings).map_err(|e| e.to_string())?;
    file.flush().map_err(|e| e.to_string())?;
    file.as_file().sync_all().map_err(|e| e.to_string())?;
    file.persist(dir.join("settings.json"))
        .map_err(|e| e.to_string())?;
    Ok(())
}
impl Corpora {
    pub fn validate(&self) -> Result<(), String> {
        for (name, path) in [
            ("Dictionary", &self.dictionary),
            ("Unigrams", &self.unigrams),
            ("Bigrams", &self.bigrams),
        ] {
            if !Path::new(path).is_file() {
                return Err(format!(
                    "{name} file is missing. Choose its path in Settings: {path}"
                ));
            }
        }
        for index in ["index.noun", "index.verb", "index.adj", "index.adv"] {
            if !PathBuf::from(&self.wordnet).join(index).is_file() {
                return Err(format!(
                    "WordNet folder is missing {index}. Choose the dict folder in Settings."
                ));
            }
        }
        if !self.phrase.is_empty() && !Path::new(&self.phrase).is_file() {
            return Err(
                "Optional phrase database is missing. Choose a valid file or clear its path."
                    .into(),
            );
        }
        Ok(())
    }
}
