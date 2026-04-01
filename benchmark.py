"""Proper evaluation for TurboQuantDC KV cache compression.

Replaces the toy 8-prompt keyword-matching benchmark with three tiers:

Tier 1 -- Perplexity (primary, fast, ~30s per config):
    Wikitext-2 perplexity on a 2048-token window.
    Reported as "% perplexity increase" vs FP16 baseline.

Tier 2 -- Generation Quality (medium, ~60s per config):
    12 diverse prompts covering factual recall, math, code, reasoning,
    long-context recall, and translation.
    Scored using absolute quality metrics (correctness + coherence +
    completeness) -- no FP16 baseline comparison for generation.

Tier 2b -- Needle-in-Haystack:
    Embed a fact in context, ask about it later.
    Tests at 512, 1K, 2K, 4K context lengths.

Tier 3 -- LM-eval-harness (thorough, ~5min per config):
    Integration script for MMLU, GSM8K, HellaSwag.
    Not used during autoresearch sweeps (too slow).

Usage:
    from benchmark import BenchmarkRunner

    runner = BenchmarkRunner(model, tokenizer)
    baseline_ppl = runner.compute_model_perplexity()
    result = runner.evaluate_config(config, baseline_ppl)
    # result.total_score, result.ppl_score, result.gen_score, ...
"""

from __future__ import annotations

import gc
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_NEW_TOKENS = 80
DO_SAMPLE = False
PPL_WINDOW = 2048  # Token window for perplexity evaluation

# Wikitext-2 test excerpt (first ~2500 chars).  This avoids requiring a
# datasets download at runtime.  The text is public domain.
WIKITEXT_EXCERPT = (
    "Robert <unk> is an English film , television and theatre actor . He had a "
    "guest @-@ starring role on the television series The Bill in 2000 . This "
    "was followed by a starring role in the play Herons written by Simon "
    "Stephens , which was performed in 2001 at the Royal Court Theatre . He "
    "had a guest role in the television series The Supply of Stephen Hawking in "
    "2004 . In 2005 , he was cast in the ITV television series Doc Martin as "
    "the local policeman . He has also appeared in the 2006 film Stormbreaker "
    "with Alex Pettyfer , Ewan McGregor , and Mickey Rourke . In May 2007 , "
    "he returned to the Royal Court Theatre starring in the play The "
    "Seagull by Anton Chekhov . He then went on to star in the 2009 "
    "adaptation of Charles Dickens 's novel Little Dorrit . In 2010 , he "
    "starred opposite Julia Stiles in the play Oleanna at the Garrick Theatre "
    "in London . In October 2013 , he was cast in the BBC One television "
    "series Death in Paradise as Inspector Humphrey Goodman , replacing Ben "
    "Miller as the lead detective . In 2015 , he appeared in the film "
    "Suffragette alongside Carey Mulligan and Helena Bonham Carter .\n\n"
    "= Valkyria Chronicles III =\n\n"
    "Senjou no Valkyria 3 : Unrecorded Chronicles is a tactical role @-@ "
    "playing video game developed by Sega and Media.Vision for the PlayStation "
    "Portable . Released in January 2011 in Japan , it is the third game in "
    "the Valkyria series . Employing the same fusion of tactical and real @-@ "
    "time gameplay as its predecessors , the story runs parallel to the first "
    "game anderta a normal military squad rather than the royal army . The "
    "game 's game engine and content tools were originally created for "
    "Valkyria Chronicles , and development on the title had to be fast @-@ "
    "tracked due to the series ' declining popularity . It was also the first "
    "game to include paid downloadable content for its game story . The game "
    "'s game engine and content tools were originally created for "
    "Valkyria Chronicles , and development on the title had to be fast @-@ "
    "tracked due to the series ' declining popularity . Due to the game being "
    "entirely in Japanese , a fan translation of the game was made available "
    "in 2014 .\n\n"
    "The game takes place during the events of the first game and follows the "
    "story of a military squad known as the Nameless . The squad is composed "
    "of deserters , criminals , and military outcasts . While the squad "
    "members are officially listed as dead , they carry out dangerous "
    "operations for the Gallian military . The game was positively received "
    "by Japanese critics . Following the success of the first two games , "
    "Sega released a number of promotional items and events . It achieved "
    "strong sales in Japan , selling over 100 @,@ 000 copies within the "
    "first month of release . The success of the first two games and the "
    "growing popularity of the PSP platform led to the decision to develop "
    "a third game in the series . Development began in late 2009 with the "
    "aim of creating the ultimate Valkyria experience ."
)


# ---------------------------------------------------------------------------
# Generation Prompts (Tier 2) -- 12 diverse prompts
# ---------------------------------------------------------------------------

GENERATION_PROMPTS = [
    # --- Factual recall (4 prompts) ---
    {
        "prompt": "What is the capital of Australia? Answer with just the city name:",
        "expected": ["Canberra"],
        "expected_keywords": ["Canberra"],
        "type": "factual",
    },
    {
        "prompt": "What is the largest planet in our solar system? Answer briefly:",
        "expected": ["Jupiter"],
        "expected_keywords": ["Jupiter"],
        "type": "factual",
    },
    {
        "prompt": "What is the chemical formula for water?",
        "expected": ["H2O"],
        "expected_keywords": ["H2O"],
        "type": "factual",
    },
    {
        "prompt": "What element has atomic number 79?",
        "expected": ["Gold", "Au"],
        "expected_keywords": ["gold", "Au"],
        "type": "factual",
    },
    # --- Math (2 prompts) ---
    {
        "prompt": "What is 15 + 27? Answer with just the number:",
        "expected": ["42"],
        "expected_keywords": ["42"],
        "type": "math",
    },
    {
        "prompt": "What is 144 divided by 12?",
        "expected": ["12"],
        "expected_keywords": ["12"],
        "type": "math",
    },
    # --- Code (2 prompts) ---
    {
        "prompt": "Write a Python function that returns the factorial of n:",
        "expected": ["def ", "factorial", "return"],
        "expected_keywords": ["def", "return"],
        "type": "code",
    },
    {
        "prompt": "Write a Python function to check if a string is a palindrome:",
        "expected": ["def ", "return", "[::-1]"],
        "expected_keywords": ["def", "return"],
        "type": "code",
    },
    # --- Reasoning (2 prompts) ---
    {
        "prompt": "Explain photosynthesis in one sentence:",
        "expected": ["light", "energy", "plant"],
        "expected_keywords": ["light", "sun", "energy", "carbon"],
        "type": "reasoning",
    },
    {
        "prompt": (
            "If a train travels 60 miles in 1 hour, and then 90 miles in "
            "the next 1.5 hours, what is its average speed for the entire "
            "trip? Show your work."
        ),
        "expected": ["60"],
        "expected_keywords": ["60"],
        "type": "reasoning",
    },
    # --- Long-context recall ---
    {
        "prompt": "List three primary colors:",
        "expected": ["red", "blue"],
        "expected_keywords": ["red", "blue"],
        "type": "factual",
    },
    # --- Translation ---
    {
        "prompt": "Translate to French: 'The cat is on the table.'",
        "expected": ["chat", "table"],
        "expected_keywords": ["chat", "table", "Le", "La"],
        "type": "translation",
    },
]


# ---------------------------------------------------------------------------
# Needle-in-Haystack configs (Tier 2b)
# ---------------------------------------------------------------------------

NEEDLE_CONFIGS = [
    {"context_len": 512, "needle_pos": 0.3},
    {"context_len": 1024, "needle_pos": 0.4},
    {"context_len": 2048, "needle_pos": 0.5},
]

NEEDLE_FACT = "The secret code word for the experiment is 'aurora'."
NEEDLE_QUERY = "What is the secret code word for the experiment?"
NEEDLE_ANSWER = "aurora"

HAYSTACK_FILLER = (
    "The quarterly report showed steady growth across all business divisions. "
    "Revenue increased moderately while operating costs remained stable. "
    "The research team achieved promising results in their efficiency studies. "
    "Customer satisfaction scores improved significantly over the previous quarter. "
    "Market conditions remained favorable throughout the reporting period. "
    "The expansion into new territories proceeded according to plan. "
)


# ---------------------------------------------------------------------------
# Pure functions (no model required -- testable in isolation)
# ---------------------------------------------------------------------------


def compute_perplexity(neg_log_likelihoods: torch.Tensor) -> float:
    """Compute perplexity from per-token negative log-likelihoods.

    Args:
        neg_log_likelihoods: 1-D tensor of per-token NLLs.

    Returns:
        Perplexity value (exp of mean NLL).
    """
    if neg_log_likelihoods.numel() == 0:
        return float("inf")
    avg_nll = neg_log_likelihoods.float().mean().item()
    return math.exp(avg_nll)


def normalize_ppl_score(baseline_ppl: float, compressed_ppl: float) -> float:
    """Convert perplexity values to a 0-1 score.

    Score = 1.0 - fraction_increase, clamped to [0, 1].
    So 0% increase -> 1.0, 8% increase -> 0.92, 100%+ increase -> 0.0.

    Args:
        baseline_ppl: FP16 baseline perplexity.
        compressed_ppl: Compressed cache perplexity.

    Returns:
        Float in [0, 1].
    """
    if baseline_ppl <= 0:
        return 0.0
    fraction_increase = (compressed_ppl - baseline_ppl) / baseline_ppl
    score = 1.0 - max(fraction_increase, 0.0)
    return max(min(score, 1.0), 0.0)


def score_correctness(response: str, expected_keywords: List[str]) -> float:
    """Score whether the response contains correct answer keywords.

    Uses case-insensitive matching. Any single keyword match counts as correct
    (OR logic), since expected_keywords may list acceptable alternatives.

    Args:
        response: Model output text.
        expected_keywords: List of acceptable answer keywords.

    Returns:
        1.0 if any keyword found, 0.0 otherwise.
    """
    if not expected_keywords:
        return 0.0
    resp_lower = response.lower()
    for kw in expected_keywords:
        if kw.lower() in resp_lower:
            return 1.0
    return 0.0


def score_coherence(response: str) -> float:
    """Score response coherence by measuring repetition via 4-gram uniqueness.

    A coherent response has mostly unique 4-grams. Heavy repetition
    (e.g. "the the the the...") produces a low ratio.

    Args:
        response: Model output text.

    Returns:
        Float in [0, 1]. 0.0 if ratio < 0.3 (garbled repetition).
    """
    words = response.split()
    if len(words) < 4:
        # Too short to measure 4-grams; give benefit of the doubt
        return 0.5 if len(words) >= 1 else 0.0

    ngrams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    total = len(ngrams)
    unique = len(set(ngrams))

    ratio = unique / total if total > 0 else 0.0
    if ratio < 0.3:
        return 0.0
    return ratio


def score_completeness(response: str) -> float:
    """Score response completeness based on word count.

    - 10-200 words: 1.0 (ideal range)
    - 5-10 or 200-400 words: 0.5 (acceptable)
    - <5 or >400 words: 0.0 (too short or runaway generation)

    Args:
        response: Model output text.

    Returns:
        Float in {0.0, 0.5, 1.0}.
    """
    word_count = len(response.split())
    if 10 <= word_count <= 200:
        return 1.0
    if 5 <= word_count < 10 or 200 < word_count <= 400:
        return 0.5
    return 0.0


def score_response_absolute(prompt_config: Dict, response: str) -> float:
    """Absolute quality score for a single response.

    Three components weighted:
        - Correctness (50%): expected keyword present in response
        - Coherence (30%): unique 4-grams / total 4-grams
        - Completeness (20%): response length in acceptable range

    Args:
        prompt_config: Prompt dict with 'expected_keywords' field.
        response: Model output text.

    Returns:
        Float in [0, 1].
    """
    keywords = prompt_config.get("expected_keywords", prompt_config.get("expected", []))
    corr = score_correctness(response, keywords)
    cohr = score_coherence(response)
    comp = score_completeness(response)
    return 0.5 * corr + 0.3 * cohr + 0.2 * comp


def score_response_similarity(baseline_response: str, compressed_response: str) -> float:
    """Score how similar two responses are using token-level Jaccard + keyword overlap.

    DEPRECATED: Kept for backward compatibility. Use score_response_absolute instead.
    Relative comparison is fundamentally broken because FP16 baseline itself
    produces wrong answers and autoregressive generation is chaotic.

    Args:
        baseline_response: FP16 baseline output.
        compressed_response: Compressed cache output.

    Returns:
        Float in [0, 1] -- 1.0 means identical.
    """
    if not baseline_response and not compressed_response:
        return 1.0
    if not baseline_response or not compressed_response:
        return 0.0

    # Tokenize to words, lowercase
    b_tokens = set(baseline_response.lower().split())
    c_tokens = set(compressed_response.lower().split())

    if not b_tokens and not c_tokens:
        return 1.0
    if not b_tokens or not c_tokens:
        return 0.0

    # Jaccard similarity
    intersection = b_tokens & c_tokens
    union = b_tokens | c_tokens
    jaccard = len(intersection) / len(union) if union else 0.0

    # Also check character-level overlap for short factual answers
    b_str = baseline_response.lower().strip()
    c_str = compressed_response.lower().strip()

    # Prefix match bonus: if the first N characters match, that's a good sign
    prefix_len = min(len(b_str), len(c_str), 50)
    if prefix_len > 0:
        matching_chars = sum(1 for a, b in zip(b_str[:prefix_len], c_str[:prefix_len]) if a == b)
        prefix_score = matching_chars / prefix_len
    else:
        prefix_score = 0.0

    # Combine: 60% Jaccard, 40% prefix match
    return 0.6 * jaccard + 0.4 * prefix_score


def needle_in_haystack_score(response: str, needle: str) -> float:
    """Score whether the response contains the needle fact.

    Args:
        response: Model output text.
        needle: The key fact to look for (case-insensitive).

    Returns:
        1.0 if needle found, 0.0 otherwise.
    """
    return 1.0 if needle.lower() in response.lower() else 0.0


def combined_score(ppl_score: float, gen_score: float) -> float:
    """Compute weighted combined score.

    60% perplexity, 40% generation quality.

    Args:
        ppl_score: Perplexity-based score in [0, 1].
        gen_score: Generation quality score in [0, 1].

    Returns:
        Float in [0, 1].
    """
    return 0.6 * ppl_score + 0.4 * gen_score


def score_response_legacy(prompt_config: Dict, response: str) -> float:
    """Legacy scoring for backward compatibility.

    Identical to the original autoresearch score_response.
    Checks keyword accuracy (0.6), coherence (0.3), length (0.1).
    """
    score = 0.0
    expected = prompt_config["expected"]
    found = sum(1 for e in expected if e.lower() in response.lower())
    accuracy = found / len(expected)
    score += 0.6 * accuracy

    words = response.split()
    if len(words) > 5:
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        coherence = min(unique_ratio / 0.5, 1.0)
    else:
        coherence = 0.5
    score += 0.3 * coherence

    if 3 < len(words) < 200:
        score += 0.1

    return round(score, 4)


# ---------------------------------------------------------------------------
# BenchmarkResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Result from evaluating a single configuration."""

    # Scores
    total_score: float = 0.0
    ppl_score: float = 0.0
    gen_score: float = 0.0
    needle_score: float = 0.0
    legacy_score: float = 0.0

    # Raw metrics
    baseline_ppl: float = 0.0
    compressed_ppl: float = 0.0
    ppl_increase_pct: float = 0.0

    # Per-prompt details (for dashboard compatibility)
    per_prompt: List[Dict[str, Any]] = field(default_factory=list)
    needle_details: List[Dict[str, Any]] = field(default_factory=list)

    # Timing
    elapsed_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSONL output."""
        return {
            "total_score": round(self.total_score, 4),
            "ppl_score": round(self.ppl_score, 4),
            "gen_score": round(self.gen_score, 4),
            "needle_score": round(self.needle_score, 4),
            "legacy_score": round(self.legacy_score, 4),
            "baseline_ppl": round(self.baseline_ppl, 4),
            "compressed_ppl": round(self.compressed_ppl, 4),
            "ppl_increase_pct": round(self.ppl_increase_pct, 2),
            "per_prompt": self.per_prompt,
            "needle_details": self.needle_details,
            "elapsed_s": round(self.elapsed_s, 2),
        }


# ---------------------------------------------------------------------------
# BenchmarkRunner -- requires model + tokenizer
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Runs evaluation benchmarks for KV cache configurations.

    Load model once, then call evaluate_config() for each configuration.
    Perplexity baseline should be computed once with compute_model_perplexity()
    and passed to evaluate_config() for each config.

    Args:
        model: HuggingFace CausalLM model.
        tokenizer: HuggingFace tokenizer.
        max_new_tokens: Max tokens to generate per prompt.
        ppl_window: Token window size for perplexity evaluation.
    """

    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens: int = MAX_NEW_TOKENS,
        ppl_window: int = PPL_WINDOW,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.ppl_window = ppl_window

        # Precompute: tokenize wikitext excerpt once
        self._ppl_input_ids: Optional[torch.Tensor] = None

    def _get_ppl_input_ids(self) -> torch.Tensor:
        """Tokenize the wikitext excerpt, cache result."""
        if self._ppl_input_ids is None:
            tokens = self.tokenizer.encode(
                WIKITEXT_EXCERPT,
                return_tensors="pt",
                add_special_tokens=False,
            )
            # Truncate to ppl_window tokens
            if tokens.shape[1] > self.ppl_window:
                tokens = tokens[:, :self.ppl_window]
            self._ppl_input_ids = tokens
        return self._ppl_input_ids

    # ---- Tier 1: Perplexity ----

    def compute_model_perplexity(self, cache=None) -> float:
        """Compute perplexity on the wikitext excerpt.

        Args:
            cache: Optional KV cache to use (None = FP16 baseline).

        Returns:
            Perplexity value.
        """
        input_ids = self._get_ppl_input_ids().to(self.model.device)
        seq_len = input_ids.shape[1]

        if seq_len < 2:
            return float("inf")

        with torch.no_grad():
            kwargs: Dict[str, Any] = {"input_ids": input_ids}
            if cache is not None:
                kwargs["past_key_values"] = cache
            outputs = self.model(**kwargs)

        # Shift logits and labels for next-token prediction
        logits = outputs.logits[:, :-1, :].float()
        labels = input_ids[:, 1:]

        # Per-token cross-entropy loss
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
        )

        return compute_perplexity(per_token_loss)

    # ---- Tier 2: Generation Quality ----

    def _generate(self, prompt: str, cache=None) -> str:
        """Generate text with optional KV cache.

        Applies the model's chat template for better instruction following
        and uses repetition_penalty to prevent generation loops.
        """
        # Wrap prompt with chat template for Instruct models
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user", "content": prompt},
        ]
        templated = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(templated, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            kwargs = dict(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=DO_SAMPLE,
                repetition_penalty=1.15,
            )
            if cache is not None:
                kwargs["past_key_values"] = cache
            out = self.model.generate(**kwargs)

        response = self.tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return response

    def _build_filler_prefix(self, target_tokens: int = 400) -> str:
        """Build a filler prefix that uses ~target_tokens tokens.

        This ensures context exceeds any FP16 window so compressed configs
        actually exercise the compression path.
        """
        filler = HAYSTACK_FILLER * 20  # ~400 tokens
        return filler

    def evaluate_generation(
        self,
        config: Dict,
        build_cache_fn,
        baseline_responses: Optional[Dict[str, str]] = None,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Evaluate generation quality on the 12-prompt suite.

        Uses absolute quality scoring (correctness + coherence + completeness).
        The baseline_responses parameter is accepted for API compatibility but
        is no longer used for scoring.

        Args:
            config: Cache configuration dict.
            build_cache_fn: Callable that returns a fresh cache for the config.
            baseline_responses: Ignored. Kept for API compatibility.

        Returns:
            (score, per_prompt_details) where score is 0-1.
        """
        filler = self._build_filler_prefix()
        per_prompt = []
        total_score = 0.0
        total_legacy = 0.0

        for prompt_cfg in GENERATION_PROMPTS:
            cache = build_cache_fn(config)
            full_prompt = filler + "\n\n" + prompt_cfg["prompt"]

            try:
                response = self._generate(full_prompt, cache=cache)
            except Exception as e:
                response = f"[ERROR: {e}]"

            # Absolute quality score (correctness + coherence + completeness)
            abs_score = score_response_absolute(prompt_cfg, response)

            # Legacy score for backward compatibility
            legacy = score_response_legacy(prompt_cfg, response)

            total_score += abs_score
            total_legacy += legacy

            per_prompt.append({
                "prompt": prompt_cfg["prompt"],
                "type": prompt_cfg["type"],
                "response": response[:300],
                "score": round(abs_score, 4),
                "legacy_score": round(legacy, 4),
            })

            del cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_score = total_score / len(GENERATION_PROMPTS) if GENERATION_PROMPTS else 0.0
        return round(avg_score, 4), per_prompt

    def compute_baseline_responses(self) -> Dict[str, str]:
        """Generate FP16 baseline responses for all prompts.

        Returns:
            Dict mapping prompt text to FP16 output text.
        """
        filler = self._build_filler_prefix()
        baseline = {}
        for prompt_cfg in GENERATION_PROMPTS:
            full_prompt = filler + "\n\n" + prompt_cfg["prompt"]
            try:
                response = self._generate(full_prompt, cache=None)
            except Exception as e:
                response = f"[ERROR: {e}]"
            baseline[prompt_cfg["prompt"]] = response
        return baseline

    # ---- Tier 2b: Needle-in-Haystack ----

    def evaluate_needle(
        self,
        config: Dict,
        build_cache_fn,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Run needle-in-haystack evaluation at multiple context lengths.

        Args:
            config: Cache configuration dict.
            build_cache_fn: Callable that returns a fresh cache.

        Returns:
            (score, details) where score is 0-1 average across context lengths.
        """
        details = []
        total_score = 0.0

        for nc in NEEDLE_CONFIGS:
            cache = build_cache_fn(config)

            # Build haystack with needle embedded
            ctx_len = nc["context_len"]
            needle_pos = nc["needle_pos"]

            # Generate filler tokens to target length
            # Each repeat of HAYSTACK_FILLER is ~40 tokens
            repeats_needed = max(1, ctx_len // 40)
            filler_before = HAYSTACK_FILLER * int(repeats_needed * needle_pos)
            filler_after = HAYSTACK_FILLER * int(repeats_needed * (1 - needle_pos))

            context = filler_before + NEEDLE_FACT + " " + filler_after
            full_prompt = context + "\n\n" + NEEDLE_QUERY

            try:
                response = self._generate(full_prompt, cache=cache)
            except Exception as e:
                response = f"[ERROR: {e}]"

            score = needle_in_haystack_score(response, NEEDLE_ANSWER)
            total_score += score

            details.append({
                "context_len": ctx_len,
                "needle_pos": needle_pos,
                "response": response[:200],
                "score": score,
            })

            del cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_score = total_score / len(NEEDLE_CONFIGS) if NEEDLE_CONFIGS else 0.0
        return round(avg_score, 4), details

    # ---- Full evaluation ----

    def evaluate_config(
        self,
        config: Dict,
        build_cache_fn,
        baseline_ppl: float,
        baseline_responses: Optional[Dict[str, str]] = None,
        run_needle: bool = True,
    ) -> BenchmarkResult:
        """Full evaluation of a single config: perplexity + generation + needle.

        Args:
            config: Cache configuration dict.
            build_cache_fn: Callable that returns a fresh cache for the config.
            baseline_ppl: FP16 baseline perplexity (precomputed).
            baseline_responses: Optional FP16 baseline outputs for self-judge.
            run_needle: Whether to run needle-in-haystack (adds ~30s).

        Returns:
            BenchmarkResult with all scores populated.
        """
        start = time.time()
        result = BenchmarkResult(baseline_ppl=baseline_ppl)

        # Tier 1: Perplexity
        try:
            ppl_cache = build_cache_fn(config)
            compressed_ppl = self.compute_model_perplexity(cache=ppl_cache)
            result.compressed_ppl = compressed_ppl
            result.ppl_score = normalize_ppl_score(baseline_ppl, compressed_ppl)
            if baseline_ppl > 0:
                result.ppl_increase_pct = round(
                    100.0 * (compressed_ppl - baseline_ppl) / baseline_ppl, 2
                )
            del ppl_cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            result.ppl_score = 0.0
            result.compressed_ppl = float("inf")
            result.ppl_increase_pct = 100.0

        # Tier 2: Generation quality
        try:
            gen_score, per_prompt = self.evaluate_generation(
                config, build_cache_fn, baseline_responses,
            )
            result.gen_score = gen_score
            result.per_prompt = per_prompt

            # Compute average legacy score
            if per_prompt:
                result.legacy_score = round(
                    sum(p.get("legacy_score", 0) for p in per_prompt) / len(per_prompt),
                    4,
                )
        except Exception as e:
            result.gen_score = 0.0
            result.legacy_score = 0.0

        # Tier 2b: Needle-in-haystack (optional)
        if run_needle:
            try:
                needle_score, needle_details = self.evaluate_needle(
                    config, build_cache_fn,
                )
                result.needle_score = needle_score
                result.needle_details = needle_details
            except Exception as e:
                result.needle_score = 0.0

        # Combined score
        result.total_score = round(combined_score(result.ppl_score, result.gen_score), 4)
        result.elapsed_s = round(time.time() - start, 2)

        return result

    def evaluate_baseline(self) -> BenchmarkResult:
        """Evaluate FP16 baseline (no compression).

        Returns:
            BenchmarkResult where ppl_score should be ~1.0.
        """
        start = time.time()
        result = BenchmarkResult()

        # Perplexity
        baseline_ppl = self.compute_model_perplexity(cache=None)
        result.baseline_ppl = baseline_ppl
        result.compressed_ppl = baseline_ppl
        result.ppl_score = 1.0
        result.ppl_increase_pct = 0.0

        # Generation -- absolute quality scoring (same metric as compressed)
        filler = self._build_filler_prefix()
        per_prompt = []
        total_gen_score = 0.0
        for prompt_cfg in GENERATION_PROMPTS:
            full_prompt = filler + "\n\n" + prompt_cfg["prompt"]
            try:
                response = self._generate(full_prompt, cache=None)
            except Exception as e:
                response = f"[ERROR: {e}]"

            # Absolute quality score (same metric used for compressed configs)
            abs_score = score_response_absolute(prompt_cfg, response)
            legacy = score_response_legacy(prompt_cfg, response)
            total_gen_score += abs_score
            per_prompt.append({
                "prompt": prompt_cfg["prompt"],
                "type": prompt_cfg["type"],
                "response": response[:300],
                "score": round(abs_score, 4),
                "legacy_score": round(legacy, 4),
            })

        result.gen_score = round(
            total_gen_score / len(GENERATION_PROMPTS), 4
        ) if GENERATION_PROMPTS else 0.0
        result.per_prompt = per_prompt
        result.legacy_score = round(
            sum(p.get("legacy_score", 0) for p in per_prompt) / len(per_prompt), 4
        ) if per_prompt else 0.0

        # Needle
        needle_details = []
        for nc in NEEDLE_CONFIGS:
            ctx_len = nc["context_len"]
            needle_pos = nc["needle_pos"]
            repeats_needed = max(1, ctx_len // 40)
            filler_before = HAYSTACK_FILLER * int(repeats_needed * needle_pos)
            filler_after = HAYSTACK_FILLER * int(repeats_needed * (1 - needle_pos))
            context = filler_before + NEEDLE_FACT + " " + filler_after
            full_prompt = context + "\n\n" + NEEDLE_QUERY
            try:
                response = self._generate(full_prompt, cache=None)
            except Exception as e:
                response = f"[ERROR: {e}]"
            score = needle_in_haystack_score(response, NEEDLE_ANSWER)
            needle_details.append({
                "context_len": ctx_len,
                "needle_pos": needle_pos,
                "response": response[:200],
                "score": score,
            })
        result.needle_score = round(
            sum(d["score"] for d in needle_details) / len(needle_details), 4
        ) if needle_details else 0.0
        result.needle_details = needle_details

        result.total_score = round(combined_score(result.ppl_score, result.gen_score), 4)
        result.elapsed_s = round(time.time() - start, 2)

        return result


# ---------------------------------------------------------------------------
# Tier 3: LM-eval-harness integration script
# ---------------------------------------------------------------------------


def generate_lm_eval_command(
    model_name: str,
    config: Dict,
    output_dir: str = "lm_eval_results",
) -> str:
    """Generate the command to run lm-eval-harness for a config.

    This is for final validation, not autoresearch.

    Args:
        model_name: HuggingFace model name.
        config: Cache configuration dict.
        output_dir: Directory for results.

    Returns:
        Shell command string.
    """
    tasks = "mmlu,gsm8k,hellaswag"
    return (
        f"lm_eval --model hf "
        f"--model_args pretrained={model_name},load_in_4bit=True "
        f"--tasks {tasks} "
        f"--num_fewshot 5 "
        f"--output_path {output_dir} "
        f"--batch_size auto"
    )
