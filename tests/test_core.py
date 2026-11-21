"""Fast unit tests for the AVF Sentinel pipeline.

Run:
    PYTHONPATH=src python -m unittest discover -s tests -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from avfsentinel.audio import acoustic_features, log_mel, read_wav, synthesize_bruit, write_wav
from avfsentinel.metrics import (
    choose_sensitivity_threshold,
    choose_threshold,
    compute_metrics,
    fit_temperature,
    triage,
)
from avfsentinel.models import FocalLoss, build_model
from avfsentinel.preprocessing import ArmStats, pair_features, quality_check


class TestAudio(unittest.TestCase):
    def test_log_mel_shape_and_finiteness(self) -> None:
        rng = np.random.default_rng(0)
        y = synthesize_bruit(rng, bruit_type="turbulent", severity=0.8)
        self.assertEqual(y.shape, (160_000,))
        db = log_mel(y)
        self.assertEqual(db.shape, (128, 224))
        self.assertTrue(np.isfinite(db).all())
        # dBFS-like values: neither silent -inf nor wildly positive.
        self.assertGreater(float(np.median(db)), -130.0)
        self.assertLess(float(np.percentile(db, 99)), 20.0)

    def test_acoustic_features(self) -> None:
        rng = np.random.default_rng(1)
        y = synthesize_bruit(rng, bruit_type="laminar", severity=0.5)
        feats = acoustic_features(log_mel(y), y)
        self.assertEqual(feats.shape, (16,))
        self.assertTrue(np.isfinite(feats).all())

    def test_wav_roundtrip(self) -> None:
        rng = np.random.default_rng(2)
        y = synthesize_bruit(rng, bruit_type="absent")[:16_000]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "clip.wav"
            write_wav(path, y)
            loaded = read_wav(path)
        self.assertEqual(loaded.shape, (16_000,))
        self.assertLessEqual(float(np.max(np.abs(loaded))), 1.0)
        self.assertGreater(float(np.std(loaded)), 0.0)


class TestGuardrails(unittest.TestCase):
    def test_dark_capture_is_rejected(self) -> None:
        img = np.full((384, 384, 3), 8, dtype=np.uint8)
        mask = np.ones((384, 384), dtype=np.uint8)
        q = quality_check(img, mask)
        self.assertTrue(q.hard_fail)
        self.assertTrue(any("dark" in s.lower() for s in q.issues))

    def test_overexposed_capture_is_rejected(self) -> None:
        img = np.full((384, 384, 3), 254, dtype=np.uint8)
        mask = np.ones((384, 384), dtype=np.uint8)
        q = quality_check(img, mask)
        self.assertTrue(q.hard_fail)
        self.assertTrue(any("overexposed" in s.lower() or "glare" in s.lower() for s in q.issues))

    def test_blurry_capture_is_rejected(self) -> None:
        import cv2

        rng = np.random.default_rng(3)
        noise = rng.integers(0, 255, (384, 384, 3), dtype=np.uint8)
        img = cv2.GaussianBlur(noise, (0, 0), sigmaX=9, sigmaY=9)
        mask = np.ones((384, 384), dtype=np.uint8)
        q = quality_check(img, mask)
        self.assertTrue(q.hard_fail)
        self.assertTrue(any("blurry" in s.lower() for s in q.issues))

    def test_pair_features_are_finite(self) -> None:
        s = ArmStats(
            angle_deg=3.0,
            arm_length=300.0,
            area_frac=0.22,
            access_width_norm=0.11,
            bulge_contrast=0.08,
            redness=0.12,
            luma=140.0,
            vein_contrast=0.15,
            dark_frac=0.05,
        )
        feats = pair_features(s, s)
        self.assertEqual(feats.shape, (12,))
        self.assertTrue(np.isfinite(feats).all())


class TestMetrics(unittest.TestCase):
    def test_threshold_meets_sensitivity_target(self) -> None:
        y = np.array([1] * 10 + [0] * 90)
        p = np.concatenate([np.linspace(0.90, 0.55, 10), np.linspace(0.50, 0.05, 90)])
        res = choose_threshold(y, p, target_sensitivity=0.90)
        self.assertTrue(res["target_met"])
        self.assertGreaterEqual(res["sensitivity"], 0.90)
        self.assertEqual(res["specificity"], 1.0)

    def test_sensitivity_first_threshold_keeps_margin(self) -> None:
        y = np.array([1] * 10 + [0] * 90)
        p = np.concatenate([np.array([0.4, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]), np.linspace(0.01, 0.3, 90)])
        conservative = choose_threshold(y, p, target_sensitivity=0.90)
        margin = choose_sensitivity_threshold(y, p, target_sensitivity=0.90, min_specificity=0.80)
        self.assertLess(margin["threshold"], conservative["threshold"])
        self.assertGreaterEqual(margin["sensitivity"], 0.90)
        self.assertGreaterEqual(margin["specificity"], 0.80)
        self.assertTrue(margin["target_met"])
        self.assertTrue(margin["specificity_floor_met"])

    def test_unreachable_target_falls_back(self) -> None:
        # Every score sits below the threshold grid floor, so even thresholding
        # at 0.005 detects no positives.
        y = np.array([1] * 5 + [0] * 20)
        p = np.array([0.001] * 5 + [0.0] * 20)
        res = choose_threshold(y, p, target_sensitivity=0.95)
        self.assertFalse(res["target_met"])
        self.assertIn("note", res)

    def test_triage_levels_and_priority(self) -> None:
        thr = {c: 0.8 for c in ("stenosis", "thrombosis", "aneurysm", "infection", "hematoma")}
        amber = {c: 0.4 for c in thr}
        green = triage({c: 0.05 for c in thr}, thr, amber)
        self.assertEqual(green["overall"], "green")
        self.assertIsNone(green["primary"])

        amber_res = triage({**{c: 0.05 for c in thr}, "stenosis": 0.5}, thr, amber)
        self.assertEqual(amber_res["overall"], "amber")

        red_res = triage({**{c: 0.05 for c in thr}, "stenosis": 0.9, "thrombosis": 0.9}, thr, amber)
        self.assertEqual(red_res["overall"], "red")
        self.assertEqual(red_res["primary"], "thrombosis")

    def test_temperature_is_positive(self) -> None:
        rng = np.random.default_rng(4)
        logits = rng.normal(size=(200, 5)).astype(np.float32)
        targets = (rng.random((200, 5)) < 0.3).astype(np.float32)
        t = fit_temperature(logits, targets)
        self.assertGreater(t, 0.0)
        self.assertLess(t, 20.0)

    def test_compute_metrics_smoke(self) -> None:
        rng = np.random.default_rng(5)
        y = (rng.random((60, 5)) < 0.3).astype(np.float32)
        p = np.clip(y * 0.5 + rng.random((60, 5)) * 0.5, 0, 1)
        thr = {c: 0.5 for c in ("stenosis", "thrombosis", "aneurysm", "infection", "hematoma")}
        m = compute_metrics(y, p, thr)
        self.assertEqual(m["n"], 60)
        self.assertIsNotNone(m["macro_auroc"])
        self.assertTrue(0.0 <= m["macro_auroc"] <= 1.0)


class TestModel(unittest.TestCase):
    def test_forward_with_and_without_audio(self) -> None:
        model = build_model(
            arm_backbone="resnet18",
            audio_backbone="resnet18",
            pretrained=False,
            arm_dim=64,
            audio_dim=32,
            hidden=64,
        )
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        spec = torch.randn(2, 2, 64, 64)
        pf = torch.randn(2, 12)
        af = torch.randn(2, 16)
        with torch.inference_mode():
            out = model(x, x, spec, pf, af, has_audio=True)
            out_no_audio = model(x, x, None, pf, None, has_audio=False)
        self.assertEqual(tuple(out.shape), (2, 5))
        self.assertEqual(tuple(out_no_audio.shape), (2, 5))
        self.assertTrue(torch.isfinite(out).all())

    def test_focal_loss_is_finite(self) -> None:
        logits = torch.randn(8, 5, requires_grad=True)
        targets = torch.randint(0, 2, (8, 5)).float()
        loss = FocalLoss(gamma=2.0, alpha=0.65, pos_weight=torch.full((5,), 2.0))(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
