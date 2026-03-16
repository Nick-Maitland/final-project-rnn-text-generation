# Dataset sources

## Shakespeare corpus
- **Corpus used:** *Hamlet* by William Shakespeare (Project Gutenberg plain-text edition)
- **Downloaded raw source:** https://www.gutenberg.org/ebooks/1524.txt.utf-8
- **Project page:** https://www.gutenberg.org/ebooks/1524
- **Prepared training file in this package:** `technical/data/processed/shakespeare_hamlet_clean.txt`

## Social media corpus
- **Corpus used:** first 3,000 rows from a public tweet subset CSV
- **Raw source file:** https://raw.githubusercontent.com/HarshalSanap/Twitter-Text-Mining/master/twittersubset.csv
- **Repository page:** https://github.com/HarshalSanap/Twitter-Text-Mining
- **Prepared training file in this package:** `technical/data/processed/social_media_tweets_clean.txt`

## Notes
- Both corpora were normalized to lowercase and cleaned for CPU-friendly training.
- The social-media corpus retains punctuation, mentions, and informal phrasing to preserve stylistic signals.
