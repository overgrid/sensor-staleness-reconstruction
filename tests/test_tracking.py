from staleness_pipeline.tracking import NoOpTracker, get_tracker


def test_get_tracker_disabled_returns_noop():
    tracker = get_tracker("http://unused:5000", "unused-experiment", enabled=False)
    assert isinstance(tracker, NoOpTracker)


def test_noop_tracker_run_and_log_dont_raise():
    tracker = NoOpTracker()
    with tracker.run("test-run", tags={"sensor": "aht_temperature"}):
        tracker.log_params({"gap_length_points": 60})
        tracker.log_metrics({"chronos_mae": 0.42})


def test_noop_tracker_log_artifact_does_not_raise():
    tracker = NoOpTracker()
    tracker.log_artifact("some/path/that/does/not/exist.png")