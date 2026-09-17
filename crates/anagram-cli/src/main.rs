use anagram_cli::utilities;
use anagram_core::{
    control::Control,
    lexicon::Unigrams,
    request::{Error, GenerateRequest, generate_controlled},
};
use std::{
    fs::File,
    io::{self, BufReader, Read},
    time::{Duration, Instant},
};

fn run() -> Result<serde_json::Value, Error> {
    let mut args: Vec<_> = std::env::args_os().skip(1).collect();
    let cache_path = take_option(&mut args, "--cache")?.map(std::path::PathBuf::from);
    let rebuild = if let Some(index) = args.iter().position(|a| a == "--rebuild-cache") {
        args.remove(index);
        true
    } else {
        false
    };
    let entries = take_option(&mut args, "--cache-max-entries")?;
    let bytes = take_option(&mut args, "--cache-max-bytes")?;
    if cache_path.is_none() && (rebuild || entries.is_some() || bytes.is_some()) {
        return Err(Error::new("usage", "Cache controls require --cache FILE"));
    }
    let parse_bound = |value: Option<std::ffi::OsString>, default| -> Result<u32, Error> {
        match value {
            None => Ok(default),
            Some(value) => value
                .to_str()
                .and_then(|s| s.parse::<u32>().ok())
                .filter(|n| *n > 0)
                .ok_or_else(|| {
                    Error::new(
                        "invalid_cache_limits",
                        "Cache bounds must be positive 32-bit integers",
                    )
                }),
        }
    };
    let cache_limits = anagram_core::cache::StorageLimits {
        max_entries: parse_bound(entries, 128)?,
        max_payload_bytes: parse_bound(bytes, 64 * 1024 * 1024)?,
    };
    let cache = cache_path.as_ref().map(|path| anagram_core::cache::Config {
        path,
        limits: cache_limits,
        rebuild,
    });
    let progress = if let Some(index) = args.iter().position(|a| a == "--progress") {
        args.remove(index);
        true
    } else {
        false
    };
    let control = if let Some(index) = args.iter().position(|a| a == "--timeout-ms") {
        let millis = args
            .get(index + 1)
            .and_then(|v| v.to_str())
            .and_then(|v| v.parse::<u64>().ok())
            .ok_or_else(|| {
                Error::new(
                    "invalid_timeout",
                    "--timeout-ms requires a nonnegative integer",
                )
            })?;
        args.drain(index..=index + 1);
        let deadline = Instant::now()
            .checked_add(Duration::from_millis(millis))
            .ok_or_else(|| {
                Error::new("invalid_timeout", "Deadline exceeds platform clock range")
            })?;
        Control::with_deadline(deadline)
    } else {
        Control::default()
    };
    let limits = if let Some(index) = args.iter().position(|a| a == "--limits") {
        let path = args
            .get(index + 1)
            .ok_or_else(|| Error::new("usage", "--limits requires a policy JSON file"))?;
        let mut input = String::new();
        File::open(path)
            .and_then(|file| file.take(65_537).read_to_string(&mut input))
            .map_err(|e| Error::new("invalid_deployment_limits", e.to_string()))?;
        if input.len() > 65_536 {
            return Err(Error::new(
                "invalid_deployment_limits",
                "Policy exceeds 64 KiB",
            ));
        }
        let limits: anagram_core::policy::DeploymentLimits = serde_json::from_str(&input)
            .map_err(|e| Error::new("invalid_deployment_limits", e.to_string()))?;
        args.drain(index..=index + 1);
        limits
    } else {
        anagram_core::policy::DeploymentLimits::default()
    };
    let control = limits.control(&control)?;
    let result = run_inner(&args, &control, &limits, progress, cache.as_ref());
    if result.is_err() {
        if let Err(reason) = control.check() {
            return Err(Error::new(reason, reason));
        }
    }
    result
}
fn take_option(
    args: &mut Vec<std::ffi::OsString>,
    name: &str,
) -> Result<Option<std::ffi::OsString>, Error> {
    let Some(index) = args.iter().position(|arg| arg == name) else {
        return Ok(None);
    };
    let value = args
        .get(index + 1)
        .filter(|value| !value.to_string_lossy().starts_with("--"))
        .cloned()
        .ok_or_else(|| Error::new("usage", format!("{name} requires a value")))?;
    args.drain(index..=index + 1);
    if args.iter().any(|arg| arg == name) {
        return Err(Error::new("usage", format!("Duplicate option {name}")));
    }
    Ok(Some(value))
}
fn run_inner(
    args: &[std::ffi::OsString],
    control: &Control,
    limits: &anagram_core::policy::DeploymentLimits,
    progress: bool,
    cache: Option<&anagram_core::cache::Config<'_>>,
) -> Result<serde_json::Value, Error> {
    control
        .check()
        .map_err(|reason| Error::new(reason, reason))?;
    let ranked = args.first().is_some_and(|a| a == "solve");
    if cache.is_some() && !ranked {
        return Err(Error::new("usage", "--cache currently applies to solve"));
    }
    if progress && !ranked {
        return Err(Error::new("usage", "--progress currently applies to solve"));
    }
    if let Some(command) = args.first().and_then(|a| a.to_str()) {
        if utilities::is_command(command) {
            return utilities::run(args, control);
        }
    }
    if !(ranked && (5..=7).contains(&args.len())
        || !ranked && (args.len() == 2 || args.len() == 3) && args[0] == "generate")
    {
        return Err(Error::new(
            "usage",
            "anagram-cli generate DICTIONARY [UNIGRAMS] | solve DICTIONARY UNIGRAMS BIGRAMS WORDNET [PHRASE_DB|-] [MODEL] < request.json",
        ));
    }
    let mut input = String::new();
    io::stdin()
        .take(1_048_577)
        .read_to_string(&mut input)
        .map_err(|e| Error::new("input_error", e.to_string()))?;
    if input.len() > 1_048_576 {
        return Err(Error::new(
            "request_too_large",
            "JSON request exceeds 1 MiB",
        ));
    }
    if ranked {
        let request =
            serde_json::from_str(&input).map_err(|e| Error::new("invalid_json", e.to_string()))?;
        let result = anagram_core::solve::solve_cached_observed(
            &request,
            anagram_core::solve::Paths {
                dictionary: std::path::Path::new(&args[1]),
                unigrams: std::path::Path::new(&args[2]),
                bigrams: std::path::Path::new(&args[3]),
                wordnet: std::path::Path::new(&args[4]),
                phrase: args.get(5).filter(|p| *p != "-").map(std::path::Path::new),
                model: args.get(6).map(std::path::Path::new),
            },
            control,
            limits,
            ("local", &mut |status| {
                if progress {
                    eprintln!("{}", serde_json::to_string(status).unwrap());
                }
            }),
            cache,
        )?;
        return serde_json::to_value(result)
            .map_err(|e| Error::new("serialization_error", e.to_string()));
    }
    let request: GenerateRequest =
        serde_json::from_str(&input).map_err(|e| Error::new("invalid_json", e.to_string()))?;
    anagram_core::request::validate(&request)?;
    limits.admit_generation(&request)?;
    let dictionary = File::open(&args[1]).map_err(|e| Error::new("corpus_error", e.to_string()))?;
    let unigrams = args
        .get(2)
        .map(|path| {
            let file = File::open(path)?;
            Unigrams::load(control.reader(BufReader::new(file)))
        })
        .transpose()
        .map_err(|e: io::Error| Error::new("corpus_error", e.to_string()))?;
    let result = generate_controlled(
        &request,
        control.reader(BufReader::new(dictionary)),
        unigrams.as_ref(),
        control,
    )?;
    serde_json::to_value(result).map_err(|e| Error::new("serialization_error", e.to_string()))
}

fn main() {
    match run() {
        Ok(result) => println!("{}", serde_json::to_string(&result).unwrap()),
        Err(error) => {
            println!(
                "{}",
                serde_json::to_string(&anagram_core::request::ErrorResponse {
                    schema_version: 1,
                    error
                })
                .unwrap()
            );
            std::process::exit(2);
        }
    }
}
