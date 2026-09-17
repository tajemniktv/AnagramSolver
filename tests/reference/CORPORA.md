# Reference data inventory

Reviewed 2026-09-17. Exact bytes and SHA-256 identities are in `manifest.json`;
URLs identify provenance, not immutable releases. Do not refresh data while
comparing implementations. No third-party corpus is committed or authorized for
bundling by this inventory.

| Input | Format and provenance | Distribution decision |
| --- | --- | --- |
| Dictionary | `.anagram_data/dictionary/large.txt`; one word per line, UTF-8 with invalid bytes ignored by the reference. [Feldman download](https://phillipmfeldman.org/English/large.txt). No explicit release number; use the recorded hash. | No license was established from the download. **Do not bundle** without permission or an independently reviewed replacement. |
| Frequencies | `ngrams/count_1w.txt` and `count_2w.txt`; tab-separated token/space-separated pair and integer count. [Norvig provenance](https://www.norvig.com/ngrams/) identifies Google Web Trillion Word Corpus derivatives. | The page's MIT grant names **code**, not these datasets. Data redistribution is unverified; **do not bundle** on the strength of that grant. |
| WordNet | `wordnet31/wn3.1.dict.tar.gz`, extracted `dict/index.*`, `data.*`, exceptions and verb frames. Version 3.1, positional Princeton text formats; morphology, POS and frames are used. | The actual `dict/data.noun` header (lines 1–29) contains the 3.1 license, copyright 2011 Princeton. Preserve copyright, conditions and disclaimer on all copies/modifications; no Princeton-name advertising. The archive has no separately named LICENSE file. Preserve embedded notices and include the full notice in packaging. [Princeton license guidance](https://wordnet.princeton.edu/license-and-commercial-use). |
| Optional phrase index | SQLite `ngrams(text TEXT PRIMARY KEY, n INTEGER, count INTEGER)`; builder also writes `meta`. `build_wikimedia_phrase_index.py` processes Wikimedia title dumps, Wiktionary by default and Wikipedia optionally. Reader uses maximum n and batched text queries. | Not enabled in the real-corpus baseline. Synthetic fixtures contain project-authored phrases/counts only. A real index needs its dump identity, source/attribution and applicable reuse terms reviewed before distribution; the database format does not grant rights to its contents. |
| Optional learned model | UTF-8 JSON, schema `anagram-explicit-ranker-1`, exact ordered 18-feature list and finite numeric weights. `anagram_feature_ranker.py` owns the schema; `train_feature_ranker.py` is the explicit offline trainer. | No trained model is enabled by default. Synthetic fixture weights are project-authored. A shipped model must record model hash, training inputs, configuration and rights to those inputs; no license is inferred from the JSON format. |

Derived normal-user vocabulary and prepared/ranked caches are not independent
corpora: invalidate them when their source identities or code change. Python
pickle caches are private implementation artifacts, never Rust input.

This resolves the inventory, not third-party permissions. Local parity can use
the already-provisioned files. Packaging remains gated on the distribution
decisions above; do not replace the frozen data silently to remove that gate.
