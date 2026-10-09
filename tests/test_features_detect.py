import numpy as np
import pytest

from aiops.detect import IsolationForestDetector, RollingZScore, SeasonalResidual, best_f1_threshold
from aiops.features import SeasonalBaseline, build_features, trailing_mean


def test_trailing_mean_matches_naive_loop():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(2, 50, 3))
    fast = trailing_mean(values, 5)
    for t in (0, 3, 4, 20, 49):
        assert np.allclose(fast[:, t], values[:, max(0, t - 4) : t + 1].mean(axis=1))


def test_seasonal_baseline_standardises_clean_data(dataset, cfg):
    clean = dataset.metrics[:, : cfg.history_start]
    z = SeasonalBaseline.fit(clean).residual_z(clean)
    assert abs(np.median(z)) < 0.2
    assert 0.6 < np.median(np.abs(z)) / 0.6745 < 1.4  # robust spread close to one


def test_baseline_needs_a_full_day(dataset):
    with pytest.raises(ValueError):
        SeasonalBaseline.fit(dataset.metrics[:, :100])


def test_features_are_causal(dataset, cfg):
    clean = dataset.metrics[:, : cfg.history_start]
    baseline = SeasonalBaseline.fit(clean)
    window = dataset.metrics[:, cfg.history_start : cfg.history_start + 300].copy()
    before = build_features(baseline.residual_z(window, cfg.history_start))
    window[:, 200:] *= 5  # change the future
    after = build_features(baseline.residual_z(window, cfg.history_start))
    assert np.array_equal(before[:, :200], after[:, :200])


@pytest.mark.parametrize("make", [RollingZScore, SeasonalResidual,
                                  lambda: IsolationForestDetector(seed=1, max_fit_rows=8000)])
def test_detectors_score_an_injected_fault_higher(dataset, cfg, make):
    clean = dataset.metrics[:, : cfg.history_start]
    detector = make().fit(clean)
    probe = clean[:, -400:].copy()
    probe[0, 300:330, 0] *= 4      # latency spike on the first service
    probe[0, 300:330, 1] += 0.2    # with errors
    offset = cfg.history_start - 400
    scores = detector.score(probe, offset)
    assert scores.shape == probe.shape[:2]
    # Judge the onset: the rolling baseline adapts to a fault that lasts, by design.
    assert scores[0, 300:310].max() > scores[0, 50:290].max()
    assert scores[0, 300:310].max() > scores[1, 300:330].max()
    assert np.isfinite(scores).all() and scores[:, :50].max() < 100  # no warm-up blow-up


@pytest.mark.parametrize("make", [SeasonalResidual, IsolationForestDetector])
def test_unfitted_detector_refuses_to_score(dataset, make):
    with pytest.raises(RuntimeError):
        make().score(dataset.metrics[:, :10])


def test_threshold_separates_clear_classes():
    scores = np.concatenate([np.full(900, 0.1), np.full(100, 5.0)]).reshape(1, -1)
    scores = scores + np.random.default_rng(0).normal(0, 0.01, scores.shape)
    truth = (np.arange(1000) >= 900).reshape(1, -1)
    threshold = best_f1_threshold(scores, truth)
    assert 0.1 < threshold < 5.0
    assert ((scores > threshold) == truth).all()


def test_threshold_handles_ties_and_missing_positives():
    scores = np.array([[0.0, 0.0, 1.0, 1.0, 1.0, 2.0]])
    truth = np.array([[False, False, True, True, True, True]])
    assert best_f1_threshold(scores, truth) == 0.5          # never cuts inside a run of equal scores
    assert best_f1_threshold(scores, np.zeros_like(truth)) == 2.0  # nothing to find: flag nothing


def test_lstm_autoencoder_detects_injected_fault(dataset, cfg):
    pytest.importorskip("torch")
    from aiops.detect_lstm import LSTMAutoencoderDetector

    clean = dataset.metrics[:, : cfg.history_start]
    detector = LSTMAutoencoderDetector(seed=0, epochs=2, train_windows=4000).fit(clean)
    probe = clean[:, -300:].copy()
    probe[2, 200:230, 0] *= 4
    scores = detector.score(probe, cfg.history_start - 300)
    assert scores.shape == probe.shape[:2]
    assert (scores[:, :15] == 0).all()  # not enough history for a full window yet
    assert scores[2, 205:230].mean() > 3 * np.median(scores[2, 20:190])
    with pytest.raises(RuntimeError):
        LSTMAutoencoderDetector().score(probe)
