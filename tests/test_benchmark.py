"""Tests for the benchmark module.

Tests perplexity computation, generation quality scoring, and score
normalization without requiring GPU or model loading.
"""

from __future__ import annotations

import math

import pytest
import torch


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from benchmark import (
    GENERATION_PROMPTS,
    compute_perplexity,
    needle_in_haystack_score,
    normalize_ppl_score,
    combined_score,
    score_response_similarity,
    score_correctness,
    score_coherence,
    score_completeness,
    score_response_absolute,
)


# ---------------------------------------------------------------------------
# Perplexity normalization
# ---------------------------------------------------------------------------


class TestNormalizePplScore:
    """Test the perplexity-to-score mapping."""

    def test_zero_increase_is_one(self):
        """No perplexity increase = perfect score of 1.0."""
        assert normalize_ppl_score(10.0, 10.0) == pytest.approx(1.0)

    def test_eight_percent_increase(self):
        """8% perplexity increase should give 0.92."""
        baseline = 10.0
        compressed = 10.8  # 8% increase
        score = normalize_ppl_score(baseline, compressed)
        assert score == pytest.approx(0.92, abs=0.001)

    def test_fifty_percent_increase(self):
        """50% perplexity increase = 0.5."""
        score = normalize_ppl_score(10.0, 15.0)
        assert score == pytest.approx(0.5, abs=0.001)

    def test_hundred_percent_increase(self):
        """100% perplexity increase should clamp to 0.0."""
        score = normalize_ppl_score(10.0, 20.0)
        assert score == pytest.approx(0.0, abs=0.001)

    def test_decrease_clamps_to_one(self):
        """If compressed is somehow lower, clamp to 1.0."""
        score = normalize_ppl_score(10.0, 9.5)
        assert score == pytest.approx(1.0)

    def test_baseline_zero_returns_zero(self):
        """Edge case: baseline of 0 should not crash."""
        score = normalize_ppl_score(0.0, 5.0)
        assert 0.0 <= score <= 1.0


class TestComputePerplexity:
    """Test the perplexity computation from log-likelihoods."""

    def test_uniform_distribution(self):
        """Uniform distribution over V=100 tokens -> ppl = 100."""
        # log(1/100) = -log(100) for each token
        V = 100
        seq_len = 50
        neg_log_likelihoods = torch.full((seq_len,), math.log(V))
        ppl = compute_perplexity(neg_log_likelihoods)
        assert ppl == pytest.approx(V, rel=0.01)

    def test_perfect_prediction(self):
        """Perfect prediction -> ppl = 1."""
        neg_log_likelihoods = torch.zeros(50)
        ppl = compute_perplexity(neg_log_likelihoods)
        assert ppl == pytest.approx(1.0, abs=0.01)

    def test_higher_loss_gives_higher_ppl(self):
        """Higher average NLL -> higher perplexity."""
        low_loss = torch.full((50,), 1.0)
        high_loss = torch.full((50,), 3.0)
        ppl_low = compute_perplexity(low_loss)
        ppl_high = compute_perplexity(high_loss)
        assert ppl_high > ppl_low


# ---------------------------------------------------------------------------
# Generation quality scoring
# ---------------------------------------------------------------------------


class TestScoreCorrectness:
    """Test correctness scoring (keyword presence)."""

    def test_keyword_found(self):
        """Response containing a keyword should score 1.0."""
        score = score_correctness("The capital is Canberra.", ["Canberra"])
        assert score == pytest.approx(1.0)

    def test_keyword_not_found(self):
        """Response without any keyword should score 0.0."""
        score = score_correctness("The capital is Sydney.", ["Canberra"])
        assert score == pytest.approx(0.0)

    def test_case_insensitive(self):
        """Keyword matching should be case-insensitive."""
        score = score_correctness("gold is element 79", ["Gold", "Au"])
        assert score == pytest.approx(1.0)

    def test_any_keyword_matches(self):
        """Any single keyword match should give 1.0 (OR logic)."""
        score = score_correctness("Au is gold", ["gold", "Au"])
        assert score == pytest.approx(1.0)

    def test_empty_keywords(self):
        """Empty keywords list should score 0.0."""
        score = score_correctness("some response", [])
        assert score == pytest.approx(0.0)

    def test_empty_response(self):
        """Empty response should score 0.0."""
        score = score_correctness("", ["Canberra"])
        assert score == pytest.approx(0.0)


class TestScoreCoherence:
    """Test coherence scoring (4-gram repetition detection)."""

    def test_coherent_text(self):
        """Normal prose should have high coherence."""
        text = (
            "Jupiter is the largest planet in our solar system. "
            "It has a mass more than twice that of all other planets combined."
        )
        score = score_coherence(text)
        assert score > 0.7

    def test_repetitive_text(self):
        """Heavily repeated text should score low."""
        text = "the cat sat the cat sat the cat sat the cat sat the cat sat"
        score = score_coherence(text)
        assert score < 0.5

    def test_extreme_repetition_is_zero(self):
        """Extreme repetition (ratio < 0.3) should score 0.0."""
        text = " ".join(["hello world foo bar"] * 20)
        score = score_coherence(text)
        assert score == pytest.approx(0.0)

    def test_short_text(self):
        """Text shorter than 4 words should return 0.5."""
        score = score_coherence("Hello world")
        assert score == pytest.approx(0.5)

    def test_empty_text(self):
        """Empty text should score 0.0."""
        score = score_coherence("")
        assert score == pytest.approx(0.0)

    def test_unique_text_near_one(self):
        """Fully unique 4-grams should score near 1.0."""
        words = [f"word{i}" for i in range(50)]
        text = " ".join(words)
        score = score_coherence(text)
        assert score > 0.95


class TestScoreCompleteness:
    """Test completeness scoring (word count ranges)."""

    def test_ideal_length(self):
        """10-200 words should score 1.0."""
        text = " ".join(["word"] * 50)
        assert score_completeness(text) == pytest.approx(1.0)

    def test_lower_ideal_boundary(self):
        """Exactly 10 words should score 1.0."""
        text = " ".join(["word"] * 10)
        assert score_completeness(text) == pytest.approx(1.0)

    def test_upper_ideal_boundary(self):
        """Exactly 200 words should score 1.0."""
        text = " ".join(["word"] * 200)
        assert score_completeness(text) == pytest.approx(1.0)

    def test_slightly_short(self):
        """5-9 words should score 0.5."""
        text = " ".join(["word"] * 7)
        assert score_completeness(text) == pytest.approx(0.5)

    def test_slightly_long(self):
        """201-400 words should score 0.5."""
        text = " ".join(["word"] * 300)
        assert score_completeness(text) == pytest.approx(0.5)

    def test_too_short(self):
        """<5 words should score 0.0."""
        text = "hello world"
        assert score_completeness(text) == pytest.approx(0.0)

    def test_too_long(self):
        """>400 words should score 0.0."""
        text = " ".join(["word"] * 500)
        assert score_completeness(text) == pytest.approx(0.0)

    def test_empty(self):
        """Empty string should score 0.0."""
        assert score_completeness("") == pytest.approx(0.0)


class TestScoreResponseAbsolute:
    """Test the combined absolute quality scoring."""

    def test_perfect_response(self):
        """Correct keyword + coherent + ideal length should score high."""
        prompt_cfg = {
            "expected_keywords": ["Jupiter"],
        }
        response = (
            "Jupiter is the largest planet in our solar system. "
            "It is a gas giant with a mass more than twice that of all "
            "other planets combined. Jupiter is known for its Great Red Spot."
        )
        score = score_response_absolute(prompt_cfg, response)
        assert score > 0.8

    def test_wrong_answer(self):
        """Wrong keyword but coherent and complete should cap at 0.5."""
        prompt_cfg = {
            "expected_keywords": ["Canberra"],
        }
        response = (
            "The capital of Australia is Sydney. It is a large city "
            "known for its opera house and harbour bridge attractions."
        )
        score = score_response_absolute(prompt_cfg, response)
        # Correctness = 0, so max is 0.3 * coherence + 0.2 * completeness = 0.5
        assert score <= 0.5

    def test_correct_but_garbled(self):
        """Correct keyword but garbled repetition should score medium."""
        prompt_cfg = {
            "expected_keywords": ["42"],
        }
        response = "42 42 42 42 42 42 42 42 42 42 42 42 42 42 42 42 42 42 42 42"
        score = score_response_absolute(prompt_cfg, response)
        # Correctness = 1.0 (0.5 weight), coherence = 0.0, completeness = 1.0 (0.2 weight)
        assert 0.4 <= score <= 0.75

    def test_empty_response(self):
        """Empty response should score 0.0."""
        prompt_cfg = {"expected_keywords": ["Canberra"]}
        score = score_response_absolute(prompt_cfg, "")
        assert score == pytest.approx(0.0)

    def test_falls_back_to_expected(self):
        """Should use 'expected' if 'expected_keywords' not present."""
        prompt_cfg = {"expected": ["Jupiter"]}
        response = "Jupiter is the largest planet in our solar system and is quite massive."
        score = score_response_absolute(prompt_cfg, response)
        assert score > 0.7

    def test_weights_sum_correctly(self):
        """Verify the 0.5/0.3/0.2 weighting."""
        prompt_cfg = {"expected_keywords": ["test"]}
        # Response: has keyword, perfectly coherent, ideal length
        words = ["test"] + [f"unique{i}" for i in range(29)]
        response = " ".join(words)
        score = score_response_absolute(prompt_cfg, response)
        # correctness=1.0, coherence near 1.0, completeness=1.0
        assert score > 0.9


class TestScoreResponseSimilarity:
    """Test response similarity scoring (deprecated, kept for backward compat)."""

    def test_identical_responses_score_one(self):
        """Identical baseline and compressed responses should score 1.0."""
        text = "The answer is 42."
        score = score_response_similarity(text, text)
        assert score == pytest.approx(1.0)

    def test_empty_responses(self):
        """Both empty should not crash."""
        score = score_response_similarity("", "")
        assert 0.0 <= score <= 1.0

    def test_completely_different(self):
        """Completely different responses should score low."""
        score = score_response_similarity(
            "The capital of France is Paris.",
            "xkcd random noise gibberish qwerty uiop",
        )
        assert score < 0.5

    def test_similar_responses_score_higher_than_garbage(self):
        """Responses with same key facts but different wording should beat garbage."""
        similar_score = score_response_similarity(
            "Jupiter is the largest planet in our solar system.",
            "The largest planet in the solar system is Jupiter.",
        )
        garbage_score = score_response_similarity(
            "Jupiter is the largest planet in our solar system.",
            "xkcd random noise gibberish qwerty uiop",
        )
        assert similar_score > garbage_score


class TestGenerationPrompts:
    """Verify the prompt list is well-formed."""

    def test_minimum_prompt_count(self):
        """Should have at least 12 prompts."""
        assert len(GENERATION_PROMPTS) >= 12

    def test_prompts_have_required_fields(self):
        """Each prompt config must have prompt, type, expected, and expected_keywords."""
        for p in GENERATION_PROMPTS:
            assert "prompt" in p, f"Missing 'prompt' key in {p}"
            assert "type" in p, f"Missing 'type' key in {p}"
            assert "expected" in p, f"Missing 'expected' key in {p}"
            assert "expected_keywords" in p, f"Missing 'expected_keywords' key in {p}"

    def test_no_broken_1984_prompt(self):
        """The broken '1984' prompt should not be present."""
        for p in GENERATION_PROMPTS:
            assert "1984" not in p["prompt"], "The broken 1984 prompt should be removed"

    def test_diverse_types(self):
        """Should cover multiple prompt types."""
        types = {p["type"] for p in GENERATION_PROMPTS}
        assert "factual" in types
        assert "math" in types
        assert "code" in types
        assert "reasoning" in types


# ---------------------------------------------------------------------------
# Combined score
# ---------------------------------------------------------------------------


class TestCombinedScore:
    """Test the combined scoring formula."""

    def test_perfect_scores(self):
        """Both perfect -> combined = 1.0."""
        score = combined_score(ppl_score=1.0, gen_score=1.0)
        assert score == pytest.approx(1.0)

    def test_weights_sum_to_one(self):
        """60% ppl + 40% gen should sum correctly."""
        score = combined_score(ppl_score=0.8, gen_score=0.6)
        expected = 0.6 * 0.8 + 0.4 * 0.6
        assert score == pytest.approx(expected, abs=0.001)

    def test_zero_gen_score(self):
        """Only perplexity contributing."""
        score = combined_score(ppl_score=1.0, gen_score=0.0)
        assert score == pytest.approx(0.6)

    def test_zero_ppl_score(self):
        """Only generation contributing."""
        score = combined_score(ppl_score=0.0, gen_score=1.0)
        assert score == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Needle-in-haystack score helper
# ---------------------------------------------------------------------------


class TestNeedleInHaystack:
    """Test the needle-in-haystack scoring helper."""

    def test_returns_float(self):
        """Should return a float between 0 and 1."""
        score = needle_in_haystack_score(
            response="The secret word is banana.",
            needle="banana",
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_found_needle(self):
        """Response containing the needle should score 1.0."""
        score = needle_in_haystack_score(
            response="The hidden fact is that the capital is Paris.",
            needle="Paris",
        )
        assert score == pytest.approx(1.0)

    def test_missing_needle(self):
        """Response without the needle should score 0.0."""
        score = needle_in_haystack_score(
            response="I don't know the answer to that.",
            needle="Paris",
        )
        assert score == pytest.approx(0.0)
