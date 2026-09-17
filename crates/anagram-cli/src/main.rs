use anagram_core::{
    lexicon::Unigrams,
    request::{Error, GenerateRequest, generate},
};
use std::{
    fs::File,
    io::{self, BufReader, Read},
    sync::atomic::AtomicBool,
};

fn run() -> Result<serde_json::Value, Error> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
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
        let result = anagram_core::solve::solve(
            &request,
            anagram_core::solve::Paths {
                dictionary: std::path::Path::new(&args[1]),
                unigrams: std::path::Path::new(&args[2]),
                bigrams: std::path::Path::new(&args[3]),
                wordnet: std::path::Path::new(&args[4]),
                phrase: args.get(5).map(std::path::Path::new),
            },
        )?;
        return serde_json::to_value(result)
            .map_err(|e| Error::new("serialization_error", e.to_string()));
    }
    let request: GenerateRequest =
        serde_json::from_str(&input).map_err(|e| Error::new("invalid_json", e.to_string()))?;
    let dictionary = File::open(&args[1]).map_err(|e| Error::new("corpus_error", e.to_string()))?;
    let unigrams = args
        .get(2)
        .map(|path| {
            let file = File::open(path)?;
            Unigrams::load(BufReader::new(file))
        })
        .transpose()
        .map_err(|e: io::Error| Error::new("corpus_error", e.to_string()))?;
    let result = generate(
        &request,
        BufReader::new(dictionary),
        unigrams.as_ref(),
        &AtomicBool::new(false),
        None,
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
