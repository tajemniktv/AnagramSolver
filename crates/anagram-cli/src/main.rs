use anagram_core::{
    control::Control,
    lexicon::Unigrams,
    request::{Error, GenerateRequest, generate},
};
use std::{
    fs::File,
    io::{self, BufReader, Read},
    time::{Duration, Instant},
};

fn run() -> Result<serde_json::Value, Error> {
    let mut args: Vec<_> = std::env::args_os().skip(1).collect();
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
    let result = run_inner(&args, &control, &limits);
    if result.is_err() {
        if let Err(reason) = control.check() {
            return Err(Error::new(reason, reason));
        }
    }
    result
}
fn run_inner(
    args: &[std::ffi::OsString],
    control: &Control,
    limits: &anagram_core::policy::DeploymentLimits,
) -> Result<serde_json::Value, Error> {
    control
        .check()
        .map_err(|reason| Error::new(reason, reason))?;
    let ranked = args.first().is_some_and(|a| a == "solve");
    if !(ranked && (args.len() == 5 || args.len() == 6)
        || !ranked && (args.len() == 2 || args.len() == 3) && args[0] == "generate")
    {
        return Err(Error::new(
            "usage",
            "anagram-cli generate DICTIONARY [UNIGRAMS] | solve DICTIONARY UNIGRAMS BIGRAMS WORDNET [PHRASE_DB] < request.json",
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
        let result = anagram_core::solve::solve_with_limits(
            &request,
            anagram_core::solve::Paths {
                dictionary: std::path::Path::new(&args[1]),
                unigrams: std::path::Path::new(&args[2]),
                bigrams: std::path::Path::new(&args[3]),
                wordnet: std::path::Path::new(&args[4]),
                phrase: args.get(5).map(std::path::Path::new),
            },
            control,
            limits,
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
    let result = generate(
        &request,
        control.reader(BufReader::new(dictionary)),
        unigrams.as_ref(),
        control.flag(),
        control.deadline(),
    )?;
    serde_json::to_value(result).map_err(|e| Error::new("serialization_error", e.to_string()))
}

fn main() {
    match run() {
        Ok(result) => println!("{}", serde_json::to_string(&result).unwrap()),
        Err(error) => {
            println!(
                "{}",
                serde_json::json!({"schema_version": 1, "error": error})
            );
            std::process::exit(2);
        }
    }
}
