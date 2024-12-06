import pytest
from main import viterbi


def test_1():
    states = ["rainy", "sunny"]
    observations = ["walk", "shop", "walk"]

    start_probs = {
        "rainy": 0.6,
        "sunny": 0.4
    }

    transition_probs = {
        "rainy": {"rainy": 0.7, "sunny": 0.3},
        "sunny": {"rainy": 0.4, "sunny": 0.6}
    }

    emission_probs = {
        "rainy": {"walk": 0.1, "shop": 0.4},
        "sunny": {"walk": 0.6, "shop": 0.3}
    }

    best_path, best_path_prob = viterbi(observations, states, start_probs, transition_probs, emission_probs)
    expected_best_path = ["sunny", "sunny", "sunny"]
    expected_best_prob = 0.015552

    assert best_path == expected_best_path, f"Expected {expected_best_path}, got {best_path}"
    assert pytest.approx(best_path_prob,
                         0.000001) == expected_best_prob, f"Expected {expected_best_prob}, got {best_path_prob}"

def test_2():
    states = ["s1", "s2", "s3", "s4", "s5"]
    observations = ["obs_a", "obs_b", "obs_b", "obs_a", "obs_c",
                    "obs_c", "obs_a", "obs_b", "obs_c", "obs_a"]

    start_probs = {
        "s1": 0.05,
        "s2": 0.05,
        "s3": 0.8,
        "s4": 0.05,
        "s5": 0.05
    }

    transition_probs = {
        "s1": {"s1": 0.2, "s2": 0.2, "s3": 0.2, "s4": 0.2, "s5": 0.2},
        "s2": {"s1": 0.2, "s2": 0.2, "s3": 0.2, "s4": 0.2, "s5": 0.2},
        "s3": {"s1": 0.05, "s2": 0.05, "s3": 0.8, "s4": 0.05, "s5": 0.05},
        "s4": {"s1": 0.2, "s2": 0.2, "s3": 0.2, "s4": 0.2, "s5": 0.2},
        "s5": {"s1": 0.2, "s2": 0.2, "s3": 0.2, "s4": 0.2, "s5": 0.2}
    }

    emission_probs = {
        "s1": {"obs_a": 0.3, "obs_b": 0.3, "obs_c": 0.4},
        "s2": {"obs_a": 0.3, "obs_b": 0.3, "obs_c": 0.4},
        "s3": {"obs_a": 0.5, "obs_b": 0.4, "obs_c": 0.1},
        "s4": {"obs_a": 0.3, "obs_b": 0.3, "obs_c": 0.4},
        "s5": {"obs_a": 0.3, "obs_b": 0.3, "obs_c": 0.4}
    }

    best_path, best_path_prob = viterbi(observations, states, start_probs, transition_probs, emission_probs)
    expected_best_path = ["s3"] * len(observations)
    expected_best_prob = 4.294967296e-07

    assert best_path == expected_best_path, f"Expected best path {expected_best_path}, got {best_path}"
    assert pytest.approx(best_path_prob,
                         1e-12) == expected_best_prob, f"Expected best path probability ~{expected_best_prob}, got {best_path_prob}"