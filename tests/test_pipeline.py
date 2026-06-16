import unittest
from unittest.mock import MagicMock, patch
from ovos_bus_client.message import Message
from ovos_plugin_manager.templates.pipeline import IntentHandlerMatch


def _make_pipeline(config=None, intents=None, renormalize=False):
    """Helper: create a pipeline with a mocked classifier and FakeBus."""
    config = config or {}
    config.setdefault("index_dir", "/fake/index")
    config["renormalize"] = renormalize

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [{}]

    with patch("ovos_hierarchical_knn_pipeline.HierarchicalPairKNNClassifier") as MockCLF, \
         patch("ovos_hierarchical_knn_pipeline.Configuration", return_value={}):
        MockCLF.from_disk.return_value = mock_model
        from ovos_hierarchical_knn_pipeline import HierarchicalKNNIntentPipeline
        from ovos_utils.fakebus import FakeBus
        pipeline = HierarchicalKNNIntentPipeline(bus=FakeBus(), config=config)

    pipeline.model = mock_model
    if intents is not None:
        pipeline.intents = list(intents)
    return pipeline


def _setup_model(pipeline, probs_dict):
    """Set the mock model to return the given label→probability dict."""
    pipeline.model.predict_proba.return_value = [probs_dict]


class TestInit(unittest.TestCase):
    def test_default_conf_thresholds(self):
        p = _make_pipeline()
        self.assertEqual(p.config.get("conf_high", 0.7), 0.7)
        self.assertEqual(p.config.get("conf_medium", 0.5), 0.5)
        self.assertEqual(p.config.get("conf_low", 0.15), 0.15)

    def test_ignore_labels_default_empty(self):
        p = _make_pipeline()
        self.assertEqual(p.ignore_labels, [])

    def test_ignore_labels_from_config(self):
        p = _make_pipeline(config={"index_dir": "/fake", "ignore_intents": ["skill:bad.intent"]})
        self.assertIn("skill:bad.intent", p.ignore_labels)

    def test_missing_index_dir_falls_back_to_hf(self):
        # When index_dir is empty/missing, the plugin should download a
        # pre-built index from HuggingFace via `from_pretrained`.
        with patch("ovos_hierarchical_knn_pipeline.HierarchicalPairKNNClassifier") as MockCLF, \
             patch("ovos_hierarchical_knn_pipeline.Configuration", return_value={}):
            from ovos_hierarchical_knn_pipeline import HierarchicalKNNIntentPipeline
            from ovos_utils.fakebus import FakeBus
            HierarchicalKNNIntentPipeline(bus=FakeBus(), config={"index_dir": ""})
            MockCLF.from_pretrained.assert_called_once()
            MockCLF.from_disk.assert_not_called()


class TestGetAdaptIntents(unittest.TestCase):
    def test_returns_intent_names(self):
        p = _make_pipeline()
        fake_intents = [{"name": "skill_a:intent_one"}, {"name": "skill_b:intent_two"}]
        mock_response = Message("intent.service.adapt.manifest",
                                data={"intents": fake_intents})
        p.bus.wait_for_response = MagicMock(return_value=mock_response)
        result = p._get_adapt_intents()
        self.assertEqual(result, ["skill_a:intent_one", "skill_b:intent_two"])

    def test_filters_ignore_labels(self):
        p = _make_pipeline(config={"index_dir": "/fake", "ignore_intents": ["skill_a:intent_one"]})
        fake_intents = [{"name": "skill_a:intent_one"}, {"name": "skill_b:intent_two"}]
        mock_response = Message("intent.service.adapt.manifest",
                                data={"intents": fake_intents})
        p.bus.wait_for_response = MagicMock(return_value=mock_response)
        result = p._get_adapt_intents()
        self.assertNotIn("skill_a:intent_one", result)
        self.assertIn("skill_b:intent_two", result)

    def test_raises_on_no_response(self):
        p = _make_pipeline()
        p.bus.wait_for_response = MagicMock(return_value=None)
        with self.assertRaises(RuntimeError):
            p._get_adapt_intents()


class TestGetPadatiousIntents(unittest.TestCase):
    def test_returns_intent_names(self):
        p = _make_pipeline()
        fake_intents = ["skill_a:one.intent", "skill_b:two.intent"]
        mock_response = Message("intent.service.padatious.manifest",
                                data={"intents": fake_intents})
        p.bus.wait_for_response = MagicMock(return_value=mock_response)
        result = p._get_padatious_intents()
        self.assertEqual(result, fake_intents)

    def test_filters_ignore_labels(self):
        p = _make_pipeline(config={"index_dir": "/fake", "ignore_intents": ["skill_a:one.intent"]})
        fake_intents = ["skill_a:one.intent", "skill_b:two.intent"]
        mock_response = Message("intent.service.padatious.manifest",
                                data={"intents": fake_intents})
        p.bus.wait_for_response = MagicMock(return_value=mock_response)
        result = p._get_padatious_intents()
        self.assertNotIn("skill_a:one.intent", result)
        self.assertIn("skill_b:two.intent", result)

    def test_raises_on_no_response(self):
        p = _make_pipeline()
        p.bus.wait_for_response = MagicMock(return_value=None)
        with self.assertRaises(RuntimeError):
            p._get_padatious_intents()


class TestHandleSyncIntents(unittest.TestCase):
    def test_debounce_while_syncing(self):
        p = _make_pipeline()
        p._syncing = True
        p._get_adapt_intents = MagicMock()
        p._handle_sync_intents(Message("test"))
        p._get_adapt_intents.assert_not_called()

    def test_syncs_intents(self):
        p = _make_pipeline()
        p._get_adapt_intents = MagicMock(return_value=["skill:adapt_intent"])
        p._get_padatious_intents = MagicMock(return_value=["skill:pad_intent"])
        with patch("ovos_hierarchical_knn_pipeline.time") as mock_time:
            mock_time.sleep = MagicMock()
            p._handle_sync_intents(Message("test"))
        self.assertIn("skill:adapt_intent", p.intents)
        self.assertIn("skill:pad_intent", p.intents)
        self.assertFalse(p._syncing)

    def test_handles_runtime_error_gracefully(self):
        p = _make_pipeline()
        p._get_adapt_intents = MagicMock(side_effect=RuntimeError("bus timeout"))
        with patch("ovos_hierarchical_knn_pipeline.time") as mock_time:
            mock_time.sleep = MagicMock()
            p._handle_sync_intents(Message("test"))
        self.assertFalse(p._syncing)


class TestMatch(unittest.TestCase):
    def test_registered_intent_yielded(self):
        p = _make_pipeline(intents=["skill_a:my.intent"], renormalize=False)
        _setup_model(p, {"skill_a:my.intent": 0.9})
        results = list(p._match("turn on the lights"))
        self.assertEqual(len(results), 1)
        skill_id, label, prob = results[0]
        self.assertEqual(label, "skill_a:my.intent")
        self.assertEqual(skill_id, "skill_a")
        self.assertAlmostEqual(prob, 0.9)

    def test_unregistered_intent_discarded(self):
        p = _make_pipeline(intents=[], renormalize=False)
        _setup_model(p, {"skill_a:my.intent": 0.9})
        results = list(p._match("something"))
        self.assertEqual(results, [])

    def test_empty_string_keys_discarded(self):
        # predict_proba may emit "" for unfilled slots in all_sel
        p = _make_pipeline(intents=["skill_a:my.intent"], renormalize=False)
        _setup_model(p, {"skill_a:my.intent": 0.9, "": 0.1})
        results = list(p._match("test"))
        labels = [label for _, label, _ in results]
        self.assertNotIn("", labels)

    def test_ocp_special_case_bypasses_intents_check(self):
        p = _make_pipeline(intents=[], renormalize=False)
        _setup_model(p, {"ocp:play": 0.95})
        results = list(p._match("play some music"))
        self.assertEqual(len(results), 1)
        skill_id, label, prob = results[0]
        self.assertEqual(skill_id, "ovos.common_play")
        self.assertEqual(label, "ovos.common_play.play_search")

    def test_common_query_special_case_bypasses_intents_check(self):
        p = _make_pipeline(intents=[], renormalize=False)
        _setup_model(p, {"common_query:common_query": 0.8})
        results = list(p._match("what is the capital of France"))
        self.assertEqual(len(results), 1)
        skill_id, label, prob = results[0]
        self.assertEqual(skill_id, "common_query.openvoiceos")
        self.assertEqual(label, "common_query.question")

    def test_stop_special_case_bypasses_intents_check(self):
        p = _make_pipeline(intents=[], renormalize=False)
        _setup_model(p, {"stop:stop": 0.85})
        results = list(p._match("stop"))
        self.assertEqual(len(results), 1)
        skill_id, label, prob = results[0]
        self.assertEqual(skill_id, "stop.openvoiceos")
        self.assertEqual(label, "mycroft.stop")

    def test_multiple_candidates_sorted_by_prob(self):
        p = _make_pipeline(intents=["skill_a:a.intent", "skill_b:b.intent"], renormalize=False)
        _setup_model(p, {"skill_a:a.intent": 0.3, "skill_b:b.intent": 0.6})
        results = list(p._match("test"))
        self.assertEqual(results[0][2], 0.6)
        self.assertEqual(results[1][2], 0.3)

    def test_ignore_labels_not_yielded(self):
        p = _make_pipeline(
            config={"index_dir": "/fake", "ignore_intents": ["skill_a:my.intent"]},
            renormalize=False,
        )
        p.intents = []
        _setup_model(p, {"skill_a:my.intent": 0.9})
        results = list(p._match("test"))
        self.assertEqual(results, [])


class TestNormalization(unittest.TestCase):
    def test_renormalize_true_sums_to_one(self):
        # Two registered intents; unregistered class prob is dropped
        p = _make_pipeline(intents=["skill_a:a.intent", "skill_b:b.intent"], renormalize=True)
        _setup_model(p, {"skill_a:a.intent": 0.6, "skill_b:b.intent": 0.2, "skill_c:c.intent": 0.2})
        results = list(p._match("test"))
        total = sum(prob for _, _, prob in results)
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_renormalize_false_preserves_raw_probs(self):
        p = _make_pipeline(intents=["skill_a:a.intent", "skill_b:b.intent"], renormalize=False)
        _setup_model(p, {"skill_a:a.intent": 0.6, "skill_b:b.intent": 0.2, "skill_c:c.intent": 0.2})
        results = list(p._match("test"))
        probs = sorted([prob for _, _, prob in results], reverse=True)
        self.assertAlmostEqual(probs[0], 0.6, places=6)
        self.assertAlmostEqual(probs[1], 0.2, places=6)

    def test_renormalize_redistributes_masked_probability(self):
        p = _make_pipeline(intents=["skill_a:a.intent"], renormalize=True)
        _setup_model(p, {"skill_a:a.intent": 0.3, "skill_b:b.intent": 0.7})
        results = list(p._match("test"))
        _, _, prob = results[0]
        self.assertAlmostEqual(prob, 1.0, places=6)

    def test_renormalize_no_division_by_zero(self):
        p = _make_pipeline(intents=["skill_a:a.intent"], renormalize=True)
        _setup_model(p, {"skill_a:a.intent": 0.0})
        results = list(p._match("test"))
        _, _, prob = results[0]
        self.assertFalse(prob != prob)  # not NaN
        self.assertAlmostEqual(prob, 0.0)

    def test_special_labels_included_in_normalization(self):
        p = _make_pipeline(intents=["skill_a:a.intent"], renormalize=True)
        _setup_model(p, {"skill_a:a.intent": 0.3, "ocp:play": 0.4, "skill_b:b.intent": 0.3})
        results = list(p._match("test"))
        total = sum(prob for _, _, prob in results)
        self.assertAlmostEqual(total, 1.0, places=6)
        labels = {label for _, label, _ in results}
        self.assertIn("ovos.common_play.play_search", labels)
        self.assertIn("skill_a:a.intent", labels)


class TestMatchConfidence(unittest.TestCase):
    def _setup(self, prob, intents=None):
        p = _make_pipeline(
            intents=["skill_a:my.intent"] if intents is None else intents,
            renormalize=False,
        )
        _setup_model(p, {"skill_a:my.intent": prob})
        msg = Message("recognizer_loop:utterance")
        return p, msg

    def test_match_high_above_threshold(self):
        p, msg = self._setup(0.8)
        result = p.match_high(["turn on lights"], "en", msg)
        self.assertIsInstance(result, IntentHandlerMatch)
        self.assertAlmostEqual(result.match_data["confidence"], 0.8)

    def test_match_high_below_threshold_returns_none(self):
        p, msg = self._setup(0.6)
        result = p.match_high(["turn on lights"], "en", msg)
        self.assertIsNone(result)

    def test_match_medium_above_threshold(self):
        p, msg = self._setup(0.55)
        result = p.match_medium(["turn on lights"], "en", msg)
        self.assertIsInstance(result, IntentHandlerMatch)

    def test_match_medium_below_threshold_returns_none(self):
        p, msg = self._setup(0.4)
        result = p.match_medium(["turn on lights"], "en", msg)
        self.assertIsNone(result)

    def test_match_low_above_threshold(self):
        p, msg = self._setup(0.2)
        result = p.match_low(["turn on lights"], "en", msg)
        self.assertIsInstance(result, IntentHandlerMatch)

    def test_match_low_below_threshold_returns_none(self):
        p, msg = self._setup(0.05)
        result = p.match_low(["turn on lights"], "en", msg)
        self.assertIsNone(result)

    def test_match_returns_none_when_no_intents(self):
        p, msg = self._setup(0.99, intents=[])
        result = p.match_high(["turn on lights"], "en", msg)
        self.assertIsNone(result)

    def test_match_data_contains_utterance(self):
        p, msg = self._setup(0.9)
        result = p.match_high(["hello world"], "en", msg)
        self.assertEqual(result.match_data["utterance"], "hello world")
        self.assertEqual(result.utterance, "hello world")

    def test_custom_conf_high_from_config(self):
        p = _make_pipeline(
            config={"index_dir": "/fake", "conf_high": 0.95},
            intents=["skill_a:my.intent"],
            renormalize=False,
        )
        _setup_model(p, {"skill_a:my.intent": 0.9})
        msg = Message("recognizer_loop:utterance")
        result = p.match_high(["test"], "en", msg)
        self.assertIsNone(result)  # 0.9 < 0.95


class TestInitialIntentSync(unittest.TestCase):
    """Skills that loaded BEFORE the pipeline never emit register_intent.
    The plugin must seed its intent list by querying the manifests on
    construction."""

    def test_seeds_intents_from_manifests_on_init(self):
        from ovos_bus_client.message import Message as _Msg

        adapt_resp = _Msg("intent.service.adapt.manifest",
                          data={"intents": [{"name": "skill_a:foo"}]})
        pad_resp = _Msg("intent.service.padatious.manifest",
                        data={"intents": ["skill_b:bar"]})

        def fake_wait_for_response(msg, reply_type, timeout=1):
            if reply_type == "intent.service.adapt.manifest":
                return adapt_resp
            if reply_type == "intent.service.padatious.manifest":
                return pad_resp
            return None

        mock_model = MagicMock()
        with patch("ovos_hierarchical_knn_pipeline.HierarchicalPairKNNClassifier") as MockCLF, \
             patch("ovos_hierarchical_knn_pipeline.Configuration", return_value={}):
            MockCLF.from_disk.return_value = mock_model
            from ovos_hierarchical_knn_pipeline import HierarchicalKNNIntentPipeline
            from ovos_utils.fakebus import FakeBus
            bus = FakeBus()
            bus.wait_for_response = MagicMock(side_effect=fake_wait_for_response)
            p = HierarchicalKNNIntentPipeline(
                bus=bus, config={"index_dir": "/fake"},
            )

        self.assertIn("skill_a:foo", p.intents)
        self.assertIn("skill_b:bar", p.intents)
        mock_model.set_active_domains.assert_called()

    def test_init_survives_missing_manifests(self):
        mock_model = MagicMock()
        with patch("ovos_hierarchical_knn_pipeline.HierarchicalPairKNNClassifier") as MockCLF, \
             patch("ovos_hierarchical_knn_pipeline.Configuration", return_value={}):
            MockCLF.from_disk.return_value = mock_model
            from ovos_hierarchical_knn_pipeline import HierarchicalKNNIntentPipeline
            from ovos_utils.fakebus import FakeBus
            bus = FakeBus()
            bus.wait_for_response = MagicMock(return_value=None)
            p = HierarchicalKNNIntentPipeline(
                bus=bus, config={"index_dir": "/fake"},
            )
        self.assertEqual(p.intents, [])


class TestSpecialLabelGating(unittest.TestCase):
    """`_allowed_special_labels` gates ocp/stop/common_query by session."""

    def _make(self):
        config = {"index_dir": "/fake", "renormalize": False}
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = [{}]
        with patch("ovos_hierarchical_knn_pipeline.HierarchicalPairKNNClassifier") as MockCLF, \
             patch("ovos_hierarchical_knn_pipeline.Configuration", return_value={}):
            MockCLF.from_disk.return_value = mock_model
            from ovos_hierarchical_knn_pipeline import HierarchicalKNNIntentPipeline
            from ovos_utils.fakebus import FakeBus
            p = HierarchicalKNNIntentPipeline(bus=FakeBus(), config=config)
        p.model = mock_model
        return p

    def _msg(self, pipeline):
        from ovos_bus_client.session import Session
        return Message("test", context={
            "session": Session(session_id="s", pipeline=pipeline).serialize(),
        })

    def test_allows_only_pipelines_present_in_session(self):
        p = self._make()
        msg = self._msg(["ovos-ocp-pipeline-plugin-high"])
        self.assertEqual(p._allowed_special_labels(msg), {"ocp:play"})

    def test_empty_session_pipeline_falls_back_to_all(self):
        p = self._make()
        # Session(pipeline=[]) may silently use class defaults, so we patch
        # SessionManager.get to return a session with an explicitly empty pipeline.
        from ovos_bus_client.session import Session
        empty_sess = Session(session_id="s")
        empty_sess.pipeline = []
        msg = self._msg(["ovos-ocp-pipeline-plugin-high"])  # arbitrary, overridden below
        with patch("ovos_hierarchical_knn_pipeline.SessionManager") as MockSM:
            MockSM.get.return_value = empty_sess
            result = p._allowed_special_labels(msg)
        self.assertEqual(
            result,
            {"ocp:play", "common_query:common_query", "stop:stop"},
        )

    def test_none_message_falls_back_to_all(self):
        p = self._make()
        self.assertEqual(
            p._allowed_special_labels(None),
            {"ocp:play", "common_query:common_query", "stop:stop"},
        )

    def test_match_filters_special_when_session_excludes_it(self):
        p = self._make()
        p.intents = []
        p.model.predict_proba.return_value = [{"ocp:play": 0.95}]
        msg = self._msg(["ovos-hierarchical-knn-pipeline"])  # no ocp
        self.assertEqual(list(p._match("play music", msg)), [])

    def test_match_passes_special_when_session_includes_it(self):
        p = self._make()
        p.intents = []
        p.model.predict_proba.return_value = [{"ocp:play": 0.95}]
        msg = self._msg(["ovos-ocp-pipeline-plugin-high"])
        results = list(p._match("play music", msg))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], "ovos.common_play.play_search")


if __name__ == "__main__":
    unittest.main()

