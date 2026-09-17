use anagram_core::{
    lexicon::Unigrams,
    request::{Error, GenerateRequest, generate},
};
use std::{
    fs::File,
    io::{self, BufReader, Read},
    sync::atomic::AtomicBool,
};

fn run() -> Result<anagram_core::request::Generated, Error> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() < 2 || args.len() > 3 || args[0] != "generate" {
        return Err(Error::new(
            "usage",
            "anagram-cli generate DICTIONARY [UNIGRAMS] < request.json",
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
    generate(
        &request,
        BufReader::new(dictionary),
        unigrams.as_ref(),
        &AtomicBool::new(false),
        None,
    )
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
