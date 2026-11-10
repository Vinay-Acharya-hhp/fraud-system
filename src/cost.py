"""Cost rule: a missed fraud costs its Amount; every flagged transaction costs a review fee."""
import numpy as np


def total_cost(y, scores, amount, threshold, fee):
    """Total cost if we flag every transaction with score >= threshold."""
    y, scores, amount = map(np.asarray, (y, scores, amount))
    flagged = scores >= threshold
    missed_fraud_amount = amount[(y == 1) & ~flagged].sum()
    return missed_fraud_amount + fee * flagged.sum()


def cost_of_doing_nothing(y, amount):
    y, amount = np.asarray(y), np.asarray(amount)
    return amount[y == 1].sum()


def savings(y, scores, amount, threshold, fee):
    """Share of the 'do nothing' cost that the system saves.
    1.0 = perfect, 0 = no gain, below 0 = worse than doing nothing."""
    base = cost_of_doing_nothing(y, amount)
    return 1 - total_cost(y, scores, amount, threshold, fee) / base


def best_threshold(y, scores, amount, fee):
    """Sort transactions from most to least suspicious, try flagging the top 1, top 2, ...
    and keep the cut-off with the lowest total cost."""
    y, scores, amount = map(np.asarray, (y, scores, amount))
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    caught = np.cumsum((amount * (y == 1))[order])  # fraud money caught in the top k
    k = np.arange(1, len(scores) + 1)
    base = cost_of_doing_nothing(y, amount)
    cost = base - caught + fee * k
    best = int(np.argmin(cost))
    if cost[best] >= base:
        return 1.01  # above any probability: flagging anything costs more than it saves
    return float(sorted_scores[best])
