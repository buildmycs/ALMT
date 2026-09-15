import unittest

import numpy as np

from core.intensity_objective import SentimentIntensityObjective


def balance_weights():
    return {
        "ordinal_positive": np.ones(6, dtype=np.float32),
        "ordinal_negative": np.ones(6, dtype=np.float32),
        "class_weights": np.ones(7, dtype=np.float32),
        "class_counts": np.ones(7, dtype=np.int64),
    }


class OrdinalScheduleTest(unittest.TestCase):
    def test_default_schedule_keeps_existing_warmup_behavior(self):
        objective = SentimentIntensityObjective(
            balance_weights(),
            ordinal_weight=0.2,
            contrastive_weight=0.0,
            auxiliary_warmup_epochs=5,
        )

        expected_scales = {1: 0.2, 5: 1.0, 20: 1.0, 50: 1.0}
        for epoch, expected in expected_scales.items():
            objective.set_epoch(epoch)
            self.assertAlmostEqual(objective.ordinal_scale(), expected)

    def test_cosine_decay_reaches_zero_before_training_ends(self):
        objective = SentimentIntensityObjective(
            balance_weights(),
            ordinal_weight=0.2,
            contrastive_weight=0.0,
            auxiliary_warmup_epochs=5,
            ordinal_decay_start_epoch=20,
            ordinal_decay_end_epoch=40,
            ordinal_decay_final_scale=0.0,
        )

        expected_scales = {
            1: 0.2,
            5: 1.0,
            20: 1.0,
            30: 0.5,
            40: 0.0,
            50: 0.0,
        }
        for epoch, expected in expected_scales.items():
            objective.set_epoch(epoch)
            self.assertAlmostEqual(objective.ordinal_scale(), expected)

        objective.set_epoch(25)
        self.assertGreater(objective.ordinal_scale(), 0.0)
        self.assertLess(objective.ordinal_scale(), 1.0)

    def test_invalid_decay_range_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "ordinal_decay_end_epoch must be greater"
        ):
            SentimentIntensityObjective(
                balance_weights(),
                ordinal_decay_start_epoch=20,
                ordinal_decay_end_epoch=20,
            )

    def test_partial_decay_configuration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must either both be set"):
            SentimentIntensityObjective(
                balance_weights(),
                ordinal_decay_start_epoch=20,
            )

    def test_invalid_final_scale_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be in"):
            SentimentIntensityObjective(
                balance_weights(),
                ordinal_decay_start_epoch=20,
                ordinal_decay_end_epoch=40,
                ordinal_decay_final_scale=1.1,
            )


if __name__ == "__main__":
    unittest.main()
