# Dataset sources

## Shakespeare corpus
- **Corpus used:** *Hamlet* by William Shakespeare (Project Gutenberg plain-text edition)
- **Downloaded raw source:** https://www.gutenberg.org/ebooks/1524.txt.utf-8
- **Project page:** https://www.gutenberg.org/ebooks/1524
- **Prepared training file in this package:** `/Users/nicolasmaitland/Library/CloudStorage/OneDrive-DurhamCollege/Agile/Final Project/final_project_submission/technical/data/processed/shakespeare_hamlet_clean.txt`

## Social media corpus
- **Corpus used:** first 3,000 rows from a public tweet subset CSV
- **Raw source file:** https://raw.githubusercontent.com/HarshalSanap/Twitter-Text-Mining/master/twittersubset.csv
- **Repository page:** https://github.com/HarshalSanap/Twitter-Text-Mining
- **Prepared training file in this package:** `/Users/nicolasmaitland/Library/CloudStorage/OneDrive-DurhamCollege/Agile/Final Project/final_project_submission/technical/data/processed/social_media_tweets_clean.txt`

## Tokenization / modeling notes
- Primary comparison models: `shakespeare_word`, `shakespeare_char`, `social_word`, `social_char`
- Extension models: `shakespeare_subword`, `social_subword`
- Strict word models use selective normalization at tokenization time with `min_freq=2`:
  - Shakespeare word model lowercases all-caps alphabetic tokens longer than 3 characters, except Roman numerals.
  - Social word model normalizes `@handle` to `@USER`, digit-bearing tokens to `<NUM>`, lowercases hashtag bodies, and lowercases shouty all-caps tokens longer than 3 characters.
- Subword extension models use per-corpus SentencePiece unigram tokenizers:
  - `shakespeare_subword` uses `vocab_size=1024`.
  - `social_subword` uses `vocab_size=1024`.
