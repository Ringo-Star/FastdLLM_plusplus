"""
Synthetic unit tests for the Fréchet-profile decoding rule.

Test 1: Equal confidences — verify Fréchet selects the same n as factor
         at the expected boundary ((n+1)*eps < 1).
Test 2: Heterogeneous confidences — verify topn never exceeds masked count
         and only updates masked positions.

These run on CPU, no model needed.
"""
import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from generate import get_transfer_index_frechet_profile, get_transfer_index_dynamic


def test_equal_confidences():
    """
    When all masked confidences are equal to (1 - eps), the Fréchet rule
    should select the same n as the factor rule at the boundary.

    Factor rule with factor=1: threshold for position n is 1 - 1/(n+1).
    So for n positions with confidence (1-eps), factor accepts n positions
    iff (1-eps) >= 1 - 1/(n+1), i.e. 1/(n+1) >= eps, i.e. n <= 1/eps - 1.

    Fréchet rule with margin=0: score_n = L_n - U_n
      L_n = n*(1-eps) - (n-1) = 1 - n*eps
      U_n = 1 - (1-eps) = eps
      score_n = 1 - n*eps - eps = 1 - (n+1)*eps
    So score_n > 0 iff (n+1)*eps < 1, i.e. n < 1/eps - 1.

    For eps=0.1: factor allows n=9 (since 10*0.1=1, boundary), Fréchet allows n=9.
    For eps=0.2: factor allows n=4 (since 5*0.2=1, boundary), Fréchet allows n=4.
    """
    print("Test 1: Equal confidences")

    for eps, expected_n in [(0.1, 9), (0.2, 4), (0.05, 19)]:
        B, L = 1, 32
        n_masked = expected_n + 2  # more masked positions than expected selection

        # Build fake logits where softmax gives confidence exactly (1-eps) for the
        # top token at masked positions. Use 2-class effective vocab: set all other
        # logits to -inf so they contribute zero mass after softmax.
        V = 100
        logits = torch.full((B, L, V), -1e9, dtype=torch.float64)
        for pos in range(n_masked):
            # softmax([log(1-eps), log(eps), -inf, ...]) = [1-eps, eps, 0, ...]
            logits[0, pos, 0] = torch.log(torch.tensor(1 - eps, dtype=torch.float64))
            logits[0, pos, 1] = torch.log(torch.tensor(eps, dtype=torch.float64))

        mask_index = torch.zeros(B, L, dtype=torch.bool)
        mask_index[0, :n_masked] = True

        x = torch.zeros(B, L, dtype=torch.long)

        # Run Fréchet with margin=0
        _, transfer_frechet = get_transfer_index_frechet_profile(
            logits, temperature=0, remasking='low_confidence',
            mask_index=mask_index, x=x, margin=0.0
        )
        n_frechet = transfer_frechet.sum().item()

        # Run factor with factor=1
        _, transfer_factor = get_transfer_index_dynamic(
            logits, temperature=0, remasking='low_confidence',
            mask_index=mask_index, x=x, num_transfer_tokens=None, factor=1
        )
        n_factor = transfer_factor.sum().item()

        print(f"  eps={eps}, n_masked={n_masked}: "
              f"frechet selected {n_frechet}, factor selected {n_factor}, "
              f"expected ~{expected_n}")

        # Fréchet and factor should agree at the boundary (within ±1 due to
        # discrete boundary effects and the "at least 1" guarantee)
        assert abs(n_frechet - n_factor) <= 1, (
            f"Fréchet ({n_frechet}) and factor ({n_factor}) diverge too much "
            f"for eps={eps}"
        )
        # Both should be close to expected_n
        assert abs(n_frechet - expected_n) <= 1, (
            f"Fréchet selected {n_frechet}, expected ~{expected_n} for eps={eps}"
        )

    print("  PASSED\n")


def test_heterogeneous_confidences():
    """
    Feed a heterogeneous confidence profile and verify:
    1. topn never exceeds the masked count
    2. transfer_index only has True at masked positions
    3. The number of selected positions is reasonable
    """
    print("Test 2: Heterogeneous confidences")

    B, L = 2, 64
    V = 100

    # Batch 0: 20 masked positions with varying confidence
    # Batch 1: 10 masked positions with varying confidence
    n_masked = [20, 10]

    logits = torch.full((B, L, V), -1e9, dtype=torch.float64)
    mask_index = torch.zeros(B, L, dtype=torch.bool)
    x = torch.zeros(B, L, dtype=torch.long)

    for b in range(B):
        for pos in range(n_masked[b]):
            # Confidence decreases: 0.99, 0.95, 0.90, 0.85, ...
            conf = max(0.99 - 0.05 * pos, 0.05)
            logits[b, pos, 0] = torch.log(torch.tensor(conf, dtype=torch.float64))
            logits[b, pos, 1] = torch.log(torch.tensor(1 - conf, dtype=torch.float64))
            mask_index[b, pos] = True

    for margin in [0.0, 0.01, 0.02, 0.05, 0.1, 0.5]:
        _, transfer_index = get_transfer_index_frechet_profile(
            logits, temperature=0, remasking='low_confidence',
            mask_index=mask_index, x=x, margin=margin
        )

        for b in range(B):
            n_selected = transfer_index[b].sum().item()
            n_mask = mask_index[b].sum().item()

            # Check 1: never exceeds masked count
            assert n_selected <= n_mask, (
                f"batch {b}, margin={margin}: selected {n_selected} > masked {n_mask}"
            )

            # Check 2: only masked positions are selected
            assert (transfer_index[b] & ~mask_index[b]).sum() == 0, (
                f"batch {b}, margin={margin}: selected non-masked positions!"
            )

            # Check 3: at least 1 token selected (guarantee)
            assert n_selected >= 1, (
                f"batch {b}, margin={margin}: selected 0 tokens!"
            )

            print(f"  batch={b}, margin={margin}: selected {n_selected}/{n_mask}")

    print("  PASSED\n")


def test_mutual_exclusion():
    """Verify the assertion fires when multiple decode modes are set."""
    print("Test 3: Mutual exclusion assertion")

    from generate import _select_tokens

    B, L, V = 1, 16, 100
    logits = torch.randn(B, L, V, dtype=torch.float64)
    mask_index = torch.ones(B, L, dtype=torch.bool)
    x = torch.zeros(B, L, dtype=torch.long)

    try:
        _select_tokens(logits, 0, 'low_confidence', mask_index, x,
                       None, threshold=0.9, factor=1.0, frechet_margin=None)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        if "At most one" in str(e):
            print(f"  Correctly caught: {e}")
        else:
            raise

    try:
        _select_tokens(logits, 0, 'low_confidence', mask_index, x,
                       None, threshold=0.9, factor=None, frechet_margin=0.02)
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        if "At most one" in str(e):
            print(f"  Correctly caught: {e}")
        else:
            raise

    print("  PASSED\n")


if __name__ == "__main__":
    test_equal_confidences()
    test_heterogeneous_confidences()
    test_mutual_exclusion()
    print("All synthetic tests PASSED.")
