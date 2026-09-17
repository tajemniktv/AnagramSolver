use crate::settings::{self, Settings};
use anagram_core::{cache, contracts::JobStatus, control::Control, solve};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread::JoinHandle,
};

#[derive(Clone, Serialize, Default)]
pub struct Snapshot {
    pub id: u64,
    pub active: bool,
    pub status: Option<JobStatus>,
    pub result: Option<Value>,
    pub error: Option<String>,
}
pub struct Inner {
    pub experiment: crate::experimental::Snapshot,
    pub settings: Settings,
    pub notice: Option<String>,
    pub job: Snapshot,
    pub control: Control,
    pub worker: Option<JoinHandle<()>>,
    pub closing: bool,
}
pub struct Desktop {
    pub inner: Arc<Mutex<Inner>>,
    pub data: PathBuf,
}

fn generate_only(
    request: &solve::Request,
    corpora: &settings::Corpora,
    control: &Control,
    limits: &anagram_core::policy::DeploymentLimits,
    id: &str,
    observer: &mut dyn FnMut(&JobStatus),
) -> Result<Value, anagram_core::request::Error> {
    use anagram_core::{
        contracts::{Exhaustion, Stage},
        lexicon::Unigrams,
        progress::Progress,
        provenance::Snapshot,
        request::Error,
    };
    let control = limits.control(control)?;
    let mut progress = Progress::new(request, limits, &control, id, observer);
    progress.start();
    let result = (|| {
        let fail = |e: std::io::Error| Error::new("corpus_error", e.to_string());
        let dictionary =
            Snapshot::load(Path::new(&corpora.dictionary), "dictionary", &control).map_err(fail)?;
        progress.status.versions.data.push(dictionary.identity);
        let unigrams = if corpora.unigrams.is_empty() {
            None
        } else {
            let source =
                Snapshot::load(Path::new(&corpora.unigrams), "unigrams", &control).map_err(fail)?;
            progress.status.versions.data.push(source.identity);
            Some(Unigrams::load(control.reader(std::io::Cursor::new(source.bytes))).map_err(fail)?)
        };
        progress.status.versions.data_complete = true;
        progress.stage(Stage::Generating);
        let generated = anagram_core::request::generate_controlled(
            &request.generation,
            control.reader(std::io::Cursor::new(dictionary.bytes)),
            unigrams.as_ref(),
            &control,
        )?;
        progress.status.counts.generated = generated.generated;
        progress.status.exhaustion = match generated.stop {
            anagram_core::generation::Stop::Exhausted => Exhaustion::Exhausted,
            anagram_core::generation::Stop::CandidateCap => Exhaustion::Truncated,
            _ => Exhaustion::Unknown,
        };
        control.check().map_err(|e| Error::new(e, e))?;
        progress.status.counts.shown = generated.bags.len();
        serde_json::to_value(generated)
            .map_err(|e| Error::new("serialization_error", e.to_string()))
    })();
    let result = match control.check() {
        Ok(()) => result,
        Err(reason) => Err(Error::new(reason, reason)),
    };
    if result.is_err() {
        progress.status.counts.shown = 0;
    }
    progress.finish(result.as_ref().err());
    result.map(|mut value| {
        value["status"] = serde_json::to_value(progress.status).expect("serializable status");
        value
    })
}
impl Desktop {
    pub fn new(data: PathBuf) -> Result<Self, String> {
        std::fs::create_dir_all(&data).map_err(|e| e.to_string())?;
        let (settings, notice) = settings::load(&data);
        Ok(Self {
            data,
            inner: Arc::new(Mutex::new(Inner {
                settings,
                notice,
                job: Snapshot::default(),
                experiment: Default::default(),
                control: Control::default(),
                worker: None,
                closing: false,
            })),
        })
    }
    #[cfg(test)]
    pub fn start(&self, request: Value) -> Result<u64, String> {
        self.start_with_options(request, false, false)
    }
    #[cfg(test)]
    pub fn start_with_options(
        &self,
        request: Value,
        rebuild: bool,
        extended: bool,
    ) -> Result<u64, String> {
        self.start_with_mode(request, rebuild, extended, false)
    }
    pub fn start_with_mode(
        &self,
        request: Value,
        rebuild: bool,
        extended: bool,
        generation_only: bool,
    ) -> Result<u64, String> {
        let request: solve::Request = serde_json::from_value(request).map_err(|e| e.to_string())?;
        if generation_only {
            anagram_core::request::validate(&request.generation).map_err(|e| e.message)?;
        } else {
            solve::validate(&request).map_err(|e| e.message)?;
        }
        let mut inner = self.inner.lock().map_err(|e| e.to_string())?;
        if !generation_only
            && !extended
            && !inner.settings.runtime.custom_limits
            && request.result_limit_per_group > 100
        {
            return Err("Desktop results are limited to 100 rows per word-count group.".into());
        }
        let limits = if extended {
            anagram_core::policy::DeploymentLimits::default()
        } else if inner.settings.runtime.custom_limits {
            inner.settings.runtime.limits.clone()
        } else {
            settings::limits()
        };
        if generation_only {
            limits.admit_generation(&request.generation)
        } else {
            limits.admit(&request)
        }
        .map_err(|e| e.message)?;
        if inner.closing || inner.job.active || inner.experiment.active {
            return Err("A solve is already running. Cancel it and wait for completion before starting another.".into());
        }
        if !generation_only {
            inner.settings.corpora.validate()?;
        }
        if let Some(worker) = inner.worker.take() {
            let _ = worker.join();
        }
        let mut settings = inner.settings.clone();
        settings.request = serde_json::to_value(&request).map_err(|e| e.to_string())?;
        settings::save(&self.data, &settings)?;
        inner.settings = settings.clone();
        let id = inner.job.id + 1;
        inner.job = Snapshot {
            id,
            active: true,
            ..Default::default()
        };
        inner.control = Control::default();
        let control = inner.control.clone();
        let shared = self.inner.clone();
        let cache_path = settings.runtime.cache_path(&self.data);
        let spawn = std::thread::Builder::new()
            .name("TajsAnagrams-solver".into())
            .spawn(move || {
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let c = settings.corpora;
                    if generation_only {
                        return generate_only(
                            &request,
                            &c,
                            &control,
                            &limits,
                            &format!("desktop-{id}"),
                            &mut |status| {
                                if let Ok(mut inner) = shared.lock() {
                                    inner.job.status = Some(status.clone());
                                }
                            },
                        );
                    }
                    let config = cache::Config {
                        path: &cache_path,
                        limits: settings.runtime.cache_limits(),
                        rebuild,
                    };
                    solve::solve_cached_observed(
                        &request,
                        solve::Paths {
                            dictionary: Path::new(&c.dictionary),
                            unigrams: Path::new(&c.unigrams),
                            bigrams: Path::new(&c.bigrams),
                            wordnet: Path::new(&c.wordnet),
                            phrase: (!c.phrase.is_empty()).then(|| Path::new(&c.phrase)),
                            model: (!settings.model.is_empty()).then(|| Path::new(&settings.model)),
                        },
                        &control,
                        &limits,
                        (&format!("desktop-{id}"), &mut |status| {
                            if let Ok(mut inner) = shared.lock() {
                                inner.job.status = Some(status.clone());
                            }
                        }),
                        Some(&config),
                    )
                    .and_then(|r| {
                        serde_json::to_value(r).map_err(|e| {
                            anagram_core::request::Error::new("serialization_error", e.to_string())
                        })
                    })
                }));
                if let Ok(mut inner) = shared.lock() {
                    match outcome {
                        Ok(Ok(result)) => match serde_json::to_value(result) {
                            Ok(result) => inner.job.result = Some(result),
                            Err(error) => inner.job.error = Some(error.to_string()),
                        },
                        Ok(Err(error)) => {
                            if inner.job.status.is_none() {
                                let mut publish =
                                    |status: &JobStatus| inner.job.status = Some(status.clone());
                                let mut progress = anagram_core::progress::Progress::new(
                                    &request,
                                    &limits,
                                    &control,
                                    &format!("desktop-{id}"),
                                    &mut publish,
                                );
                                progress.finish(Some(&error));
                            }
                            inner.job.error = Some(format!("{}: {}", error.code, error.message))
                        }
                        Err(_) => {
                            inner.job.error = Some(
                                "Solver stopped unexpectedly. You can start a new solve.".into(),
                            )
                        }
                    }
                    inner.job.active = false;
                }
            });
        match spawn {
            Ok(worker) => inner.worker = Some(worker),
            Err(error) => {
                inner.job.active = false;
                inner.job.error = Some(error.to_string());
                return Err(error.to_string());
            }
        }
        Ok(id)
    }
    pub fn shutdown(&self) {
        let worker = if let Ok(mut inner) = self.inner.lock() {
            inner.closing = true;
            inner.control.cancel();
            inner.worker.take()
        } else {
            None
        };
        if let Some(worker) = worker {
            let _ = worker.join();
        }
    }
    pub fn clear_cache(&self) -> Result<(), String> {
        let inner = self.inner.lock().map_err(|e| e.to_string())?;
        if inner.job.active || inner.experiment.active || inner.closing {
            return Err("Wait for the active solve before clearing the cache".into());
        }
        let path = inner.settings.runtime.cache_path(&self.data);
        if !path.exists() {
            return Ok(());
        }
        let mut store = cache::Store::open(
            &path,
            inner.settings.runtime.cache_limits(),
            &Default::default(),
        )
        .map_err(|e| e.message)?;
        store.clear().map_err(|e| e.message)
    }
    pub fn bootstrap(&self) -> Result<Value, String> {
        let inner = self.inner.lock().map_err(|e| e.to_string())?;
        Ok(
            json!({"settings":inner.settings,"notice":inner.notice,"limits":settings::limits(),
            "corpus_error":inner.settings.corpora.validate().err(),"data_directory":self.data,
            "engine_version":env!("CARGO_PKG_VERSION"),"active_jobs":1,"queue_capacity":0}),
        )
    }
}
