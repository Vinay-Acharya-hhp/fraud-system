import numpy as np

from src.cost import best_threshold, cost_of_doing_nothing, savings, total_cost

Y = np.array([1, 0, 1, 0])
AMOUNT = np.array([100.0, 50.0, 30.0, 10.0])
SCORES = np.array([0.9, 0.8, 0.2, 0.1])


def test_cost_of_doing_nothing():
    assert cost_of_doing_nothing(Y, AMOUNT) == 130.0


def test_total_cost_and_savings_at_half():
    # flags the first two: misses the 30 fraud and pays 2 reviews of 5
    assert total_cost(Y, SCORES, AMOUNT, 0.5, fee=5) == 40.0
    assert round(savings(Y, SCORES, AMOUNT, 0.5, fee=5), 4) == round(1 - 40 / 130, 4)


def test_best_threshold_flags_both_frauds():
    t = best_threshold(Y, SCORES, AMOUNT, fee=5)
    assert t == 0.2
    assert total_cost(Y, SCORES, AMOUNT, t, fee=5) == 15.0


def test_expensive_reviews_mean_flag_nothing():
    assert best_threshold(Y, SCORES, AMOUNT, fee=100) > 1.0
