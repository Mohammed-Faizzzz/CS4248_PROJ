"""
Multi-Agent Sentiment Labeling Pipeline
========================================
Classifier (gpt-4o-mini) → Critic (gpt-4o-mini) → Judge (gpt-4o, only when Critic says NO)

Usage:
    python main.py --input elon_tweets.csv --output balanced.csv
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────

TARGET_PER_CLASS = 134
VALID_LABELS = {"Positive", "Neutral", "Negative"}
CLASSIFIER_MODEL = "gpt-5.4-nano"
CRITIC_MODEL = "gpt-5.4-nano"
JUDGE_MODEL = "gpt-5.4-nano"
MAX_RETRIES = 5
RETRY_BASE_DELAY = 1.0
CONCURRENCY = 10  # max parallel tweet pipelines
LOG_EVERY = 50
CACHE_FILE = "labeling_cache.json"
CHECKPOINT_FILE = "checkpoint.json"

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ── Emoji / Emoticon Removal ─────────────────────────────────────────────────

# Comprehensive emoji regex — only actual emoji blocks, no overlap with ASCII/Latin
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # All main emoji blocks (symbols, emoticons, transport, etc.)
    "\U00002600-\U000027BF"  # Misc symbols + dingbats
    "\U0000FE00-\U0000FE0F"  # Variation selectors
    "\U0000200D"             # ZWJ
    "]+",
    flags=re.UNICODE,
)

# Common text emoticons
EMOTICON_PATTERN = re.compile(
    r"(?<!\w)"
    r"(?:"
    r"[:;=8xX][-'^]?[)(\[\]DPp3><|/\\}{@]"
    r"|[)(\[\]DPp><|/\\][-'^]?[:;=8]"
    r"|<3+|</3"
    r"|[><]?[:;=]['\-]?[)(DPpSsOo|/\\]"
    r"|\\[oO]/"
    r"|\^\^;?"
    r"|[oO]_[oO]"
    r"|[>T]_[>T]"
    r"|[Uu]_[Uu]"
    r"|-_-;?"
    r"|[xX]_[xX]"
    r")"
    r"(?!\w)"
)


def strip_emojis(text: str) -> str:
    """Remove all emojis and text emoticons."""
    text = EMOJI_PATTERN.sub("", text)
    text = EMOTICON_PATTERN.sub("", text)
    return text.strip()


# ── Cache ────────────────────────────────────────────────────────────────────

class LabelCache:
    """Disk-backed cache to avoid duplicate API calls."""

    def __init__(self, path: str):
        self.path = path
        self.data: dict = {}
        if os.path.exists(path):
            with open(path, "r") as f:
                self.data = json.load(f)
            log.info(f"Loaded cache with {len(self.data)} entries")

    def key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def get(self, text: str):
        return self.data.get(self.key(text))

    def put(self, text: str, result: dict):
        self.data[self.key(text)] = result

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f)


# ── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(results: list, counters: dict, processed_ids: set):
    """Save progress to allow resumption."""
    data = {
        "results": results,
        "counters": counters,
        "processed_ids": list(processed_ids),
    }
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(data, f)


def load_checkpoint():
    """Load previous progress if available."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            data = json.load(f)
        data["processed_ids"] = set(data["processed_ids"])
        log.info(
            f"Resuming from checkpoint: {len(data['results'])} results, "
            f"counters={data['counters']}"
        )
        return data
    return None


# ── API Helpers ──────────────────────────────────────────────────────────────

async def call_llm(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Call OpenAI with retry + exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "developer", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_completion_tokens=10,
                    temperature=0.0,
                )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            log.warning(f"API error (attempt {attempt+1}/{MAX_RETRIES}): {e}. Retrying in {delay:.1f}s")
            await asyncio.sleep(delay)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


# ── Agent Prompts ────────────────────────────────────────────────────────────

CLASSIFIER_SYSTEM = (
    "You are a sentiment classifier. "
    "Classify sentiment based ONLY on the given tweet text. "
    "Tweets do not contain emojis or emoticons. "
    "Ignore missing tone markers. "
    "Output ONLY one word: Positive, Neutral, or Negative."
)

CRITIC_SYSTEM = (
    "You are a strict sentiment critic. "
    "Given a tweet and its assigned sentiment label, decide if the label is clearly correct. "
    "Answer ONLY YES or NO. Be strict."
)

JUDGE_SYSTEM = (
    "You are a final sentiment judge. "
    "Decide the final sentiment based ONLY on the given tweet text. "
    "Tweets do not contain emojis or emoticons. "
    "Ignore missing tone markers. "
    "Output ONLY one word: Positive, Neutral, or Negative."
)


# ── Pipeline per Tweet ───────────────────────────────────────────────────────

async def classify_tweet(
    client: AsyncOpenAI,
    tweet_text: str,
    cache: LabelCache,
    semaphore: asyncio.Semaphore,
) -> str:
    """Run Classifier → Critic → (optional) Judge pipeline for one tweet."""

    # Check cache
    cached = cache.get(tweet_text)
    if cached:
        return cached["label"]

    # Step 1: Classifier
    raw_label = await call_llm(
        client, CLASSIFIER_MODEL, CLASSIFIER_SYSTEM, tweet_text, semaphore
    )
    label = raw_label.capitalize()
    if label not in VALID_LABELS:
        # Try to extract a valid label
        for v in VALID_LABELS:
            if v.lower() in raw_label.lower():
                label = v
                break
        else:
            label = "Neutral"  # fallback

    # Step 2: Critic
    critic_prompt = f'Tweet: "{tweet_text}"\nLabel: {label}'
    critic_answer = await call_llm(
        client, CRITIC_MODEL, CRITIC_SYSTEM, critic_prompt, semaphore
    )

    # Step 3: Judge (only if Critic says NO)
    if "NO" in critic_answer.upper():
        raw_judge = await call_llm(
            client, JUDGE_MODEL, JUDGE_SYSTEM, tweet_text, semaphore
        )
        label = raw_judge.capitalize()
        if label not in VALID_LABELS:
            for v in VALID_LABELS:
                if v.lower() in raw_judge.lower():
                    label = v
                    break
            else:
                label = "Neutral"

    cache.put(tweet_text, {"label": label})
    return label


# ── Preprocessing ────────────────────────────────────────────────────────────

def preprocess(input_path: str) -> pd.DataFrame:
    """Load, filter, clean, shuffle."""
    log.info(f"Loading {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    log.info(f"  Raw rows: {len(df)}")

    # Normalize column name: accept both clean_txt and clean_text
    if "clean_txt" in df.columns:
        text_col = "clean_txt"
    elif "clean_text" in df.columns:
        text_col = "clean_text"
    else:
        raise ValueError("CSV must have 'clean_txt' or 'clean_text' column")

    # Filter retweets: keep only where isRetweet is False or NaN (not retweeted)
    df["isRetweet"] = df["isRetweet"].astype(str).str.strip().str.lower()
    df = df[df["isRetweet"].isin(["false", "nan", ""])]
    log.info(f"  After retweet filter: {len(df)}")

    # Drop rows with empty / NaN text
    df = df.dropna(subset=[text_col])
    df = df[df[text_col].astype(str).str.strip().ne("")]
    log.info(f"  After dropping empty text: {len(df)}")

    # Remove emojis/emoticons
    df[text_col] = df[text_col].astype(str).apply(strip_emojis)

    # Filter: >1 word
    df = df[df[text_col].str.split().str.len() > 1]
    log.info(f"  After >1 word filter: {len(df)}")

    # Remove duplicates on text
    df = df.drop_duplicates(subset=[text_col])
    log.info(f"  After dedup: {len(df)}")

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    log.info(f"  Final pool: {len(df)} tweets ready for labeling")

    # Rename to canonical column name for downstream
    if text_col == "clean_text":
        df = df.rename(columns={"clean_text": "clean_txt"})

    return df


# ── Main Loop ────────────────────────────────────────────────────────────────

async def run_pipeline(
    input_path: str,
    output_path: str,
    target_per_class: int = TARGET_PER_CLASS,
    concurrency: int = CONCURRENCY,
):
    start_time = time.time()

    # Preprocess
    df = preprocess(input_path)

    # Initialize
    client = AsyncOpenAI()  # uses OPENAI_API_KEY env var
    cache = LabelCache(CACHE_FILE)
    semaphore = asyncio.Semaphore(concurrency)

    # Counters & results
    checkpoint = load_checkpoint()
    if checkpoint:
        results = checkpoint["results"]
        counters = checkpoint["counters"]
        processed_ids = checkpoint["processed_ids"]
    else:
        results = []
        counters = {"Positive": 0, "Neutral": 0, "Negative": 0}
        processed_ids = set()

    total_target = target_per_class * 3  # 399
    processed_count = 0

    def is_complete():
        return all(c >= target_per_class for c in counters.values())

    log.info(f"Target: {target_per_class} per class ({total_target} total)")
    log.info(f"Starting labeling with concurrency={concurrency}...")

    # Process tweets sequentially (as required), but use async for API speed
    for idx, row in df.iterrows():
        if is_complete():
            break

        tweet_id = str(row["id"])
        if tweet_id in processed_ids:
            continue

        tweet_text = row["clean_txt"]
        processed_count += 1

        # Classify
        try:
            label = await classify_tweet(client, tweet_text, cache, semaphore)
        except Exception as e:
            log.error(f"Failed to classify tweet {tweet_id}: {e}")
            continue

        # Check if this class is already full
        if counters.get(label, 0) >= target_per_class:
            continue  # skip, class full

        # Accept
        counters[label] += 1
        processed_ids.add(tweet_id)
        results.append({
            "id": row["id"],
            "url": row["url"],
            "fullText": row["fullText"],
            "clean_txt": tweet_text,
            "label": label,
        })

        # Progress logging — every 50 labelled tweets
        if len(results) % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            log.info(
                f"  Scanned {processed_count} | "
                f"Labelled {len(results)}/{total_target} | "
                f"Positive={counters['Positive']} Neutral={counters['Neutral']} Negative={counters['Negative']} | "
                f"{elapsed:.0f}s"
            )
            # Save checkpoint + cache
            cache.save()
            save_checkpoint(results, counters, processed_ids)

    # Final save
    cache.save()
    elapsed = time.time() - start_time

    log.info("=" * 60)
    log.info(f"DONE in {elapsed:.1f}s")
    log.info(f"  Tweets scanned: {processed_count}")
    log.info(f"  Final counts: P={counters['Positive']} N={counters['Neutral']} Ng={counters['Negative']}")
    log.info(f"  Total accepted: {len(results)}")

    # Build output DataFrame
    out_df = pd.DataFrame(results)
    out_df = out_df[["id", "url", "fullText", "clean_txt", "label"]]
    out_df.to_csv(output_path, index=False)
    log.info(f"  Saved to {output_path}")

    # Verify balance
    label_counts = out_df["label"].value_counts()
    log.info(f"  Label distribution:\n{label_counts.to_string()}")

    # Cleanup checkpoint on success
    if is_complete() and os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        log.info("  Checkpoint cleaned up (run completed successfully)")

    return out_df


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-agent sentiment labeling pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        default="elon_tweets.csv",
        help="Input CSV file path (default: elon_tweets.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        default="balanced.csv",
        help="Output CSV file path (default: balanced.csv)",
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=CONCURRENCY,
        help=f"Max parallel API calls (default: {CONCURRENCY})",
    )
    parser.add_argument(
        "--target", "-t",
        type=int,
        default=TARGET_PER_CLASS,
        help=f"Target count per class (default: {TARGET_PER_CLASS})",
    )
    args = parser.parse_args()

    # Validate API key
    if not os.environ.get("OPENAI_API_KEY"):
        log.error("OPENAI_API_KEY environment variable not set!")
        sys.exit(1)

    asyncio.run(run_pipeline(
        args.input, args.output,
        target_per_class=args.target,
        concurrency=args.concurrency,
    ))


if __name__ == "__main__":
    main()

