from presentation_assessment.evaluation import evaluate_pairs


def test_evaluate_pairs_returns_expected_error_metrics() -> None:
    result = evaluate_pairs([80, 60, 90], [82, 63, 87])

    assert result.n == 3
    assert result.mae == 2.6667
    assert result.rmse > 0
    assert result.pearson is not None
