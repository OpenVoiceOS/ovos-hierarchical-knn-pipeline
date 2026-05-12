"""Live E2E test: run the en-US intent fixture (drawn from
`ovos-localize/data/datasets/classification/en-US.jsonl`, official OVOS
skills only) through the real hierarchical KNN pipeline on a MiniCroft instance.

Gated behind `OVOSCOPE_LIVE=1` because it downloads the model from
HuggingFace (default
`fdemelo/ovos-hierarchical-knn-granite-97m-multilingual-r2`) and
takes a minute to initialise. CI opts in by setting the env var.

For each fixture line (`{"label": "<skill:intent>", "utterance": "..."}`)
the test sets `pipeline.intents` to the union of fixture labels and asserts
that the intent dispatched by `IntentService._emit_match_message` matches
the expected label. Allows up to 20% drift since the model is not pinned.
"""
import json
import os
import threading
import unittest
from pathlib import Path

import pytest

if os.environ.get("OVOSCOPE_LIVE") != "1":
    pytest.skip(
        "Live model test skipped; set OVOSCOPE_LIVE=1 to enable.",
        allow_module_level=True,
    )

pytest.importorskip("ovoscope", reason="ovoscope not installed")

from ovos_bus_client.message import Message  # noqa: E402
from ovos_bus_client.session import Session  # noqa: E402
from ovos_config.config import Configuration  # noqa: E402
from ovoscope import get_minicroft  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "en_us_intents.jsonl"
PIPELINE_ID = "ovos-hierarchical-knn-pipeline"
CONFIG_KEY = "ovos_hierarchical_knn_pipeline"


def _load_fixture():
    cases = []
    with FIXTURE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


class TestLiveFixture(unittest.TestCase):
    """Real model, real bus, real dispatch."""

    @classmethod
    def setUpClass(cls):
        cls.cases = _load_fixture()
        cls.all_labels = sorted({c["label"] for c in cls.cases})

        cfg = Configuration()
        intents_cfg = cfg.setdefault("intents", {})
        cls._orig = intents_cfg.get(CONFIG_KEY)
        intents_cfg[CONFIG_KEY] = {"renormalize": False, "conf_low": 0.0}

        cls.mc = get_minicroft(
            skill_ids=[],
            lang="en-US",
            default_pipeline=[PIPELINE_ID],
            max_wait=300,  # model download can be slow on first run
        )
        cls.pipeline = cls.mc.intents.pipeline_plugins[PIPELINE_ID]
        cls.pipeline.intents = list(cls.all_labels)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.mc.stop()
        finally:
            cfg = Configuration()
            intents_cfg = cfg.get("intents", {})
            if cls._orig is None:
                intents_cfg.pop(CONFIG_KEY, None)
            else:
                intents_cfg[CONFIG_KEY] = cls._orig

    def _emit(self, utterance: str, expected_label: str, timeout: float = 15.0):
        got: list[Message] = []
        done = threading.Event()

        def _capture(msg):
            got.append(msg)
            done.set()

        def _fail(_msg):
            done.set()

        self.mc.bus.on(expected_label, _capture)
        self.mc.bus.on("complete_intent_failure", _fail)
        sess = Session(session_id="live", pipeline=[PIPELINE_ID])
        try:
            self.mc.bus.emit(Message(
                "recognizer_loop:utterance",
                data={"utterances": [utterance], "lang": "en-US"},
                context={"session": sess.serialize()},
            ))
            done.wait(timeout=timeout)
        finally:
            self.mc.bus.remove(expected_label, _capture)
            self.mc.bus.remove("complete_intent_failure", _fail)
        return got[0] if got else None

    def test_fixture_top_intent_matches_label(self):
        misses = []
        for case in self.cases:
            utt, expected = case["utterance"], case["label"]
            msg = self._emit(utt, expected)
            if msg is None:
                misses.append((utt, expected, "no match"))
                continue
            if msg.msg_type != expected:
                misses.append((utt, expected, msg.msg_type))
        # Allow ≤ 20% drift: the model is not pinned and minor regressions
        # in template overlap are acceptable, big ones are not.
        max_misses = max(1, len(self.cases) // 5)
        self.assertLessEqual(
            len(misses), max_misses,
            f"{len(misses)}/{len(self.cases)} fixture cases misclassified:\n"
            + "\n".join(f"  {u!r} expected={e} got={g}" for u, e, g in misses),
        )


if __name__ == "__main__":
    unittest.main()
