import unittest

import numpy as np

from bank_marketing_knn.knn import ManualKNNClassifier
from bank_marketing_knn.metrics import top_percent_lift


class ManualKNNTests(unittest.TestCase):
    def test_predicts_nearest_majority_class(self):
        x_train = np.array([[0.0], [1.0], [10.0], [11.0]])
        y_train = np.array([0, 0, 1, 1])
        model = ManualKNNClassifier(k=1).fit(x_train, y_train)

        predictions = model.predict(np.array([[0.2], [10.5]]))

        np.testing.assert_array_equal(predictions, np.array([0, 1]))

    def test_predict_proba_uses_neighbor_positive_rate(self):
        x_train = np.array([[0.0], [1.0], [2.0]])
        y_train = np.array([0, 1, 1])
        model = ManualKNNClassifier(k=3).fit(x_train, y_train)

        probabilities = model.predict_proba(np.array([[1.0]]))

        self.assertAlmostEqual(probabilities[0, 1], 2 / 3)


class MetricTests(unittest.TestCase):
    def test_top_percent_lift(self):
        y_true = np.array([0, 0, 1, 1])
        probability = np.array([0.1, 0.2, 0.8, 0.9])

        self.assertEqual(top_percent_lift(y_true, probability, percent=0.5), 2.0)


if __name__ == "__main__":
    unittest.main()
