import unittest

from perceptron import predict, step, train
from main import GATES


class PerceptronTests(unittest.TestCase):
    def test_step_uses_zero_threshold(self):
        self.assertEqual(step(-0.01), 0)
        self.assertEqual(step(0), 1)
        self.assertEqual(step(0.01), 1)

    def test_all_gates_converge_and_predict_every_example(self):
        for name, examples in GATES.items():
            with self.subTest(gate=name):
                result = train(examples)
                self.assertTrue(result.converged)
                self.assertEqual(len(result.errors), result.epochs)
                self.assertEqual(result.errors[-1], 0)
                for inputs, expected in examples:
                    self.assertEqual(predict(inputs, result.weights, result.bias), expected)

    def test_non_separable_xor_does_not_report_convergence(self):
        xor = (((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0))
        result = train(xor, max_epochs=20)
        self.assertFalse(result.converged)
        self.assertEqual(result.epochs, 20)

    def test_invalid_learning_rate_is_rejected(self):
        with self.assertRaises(ValueError):
            train(GATES["AND"], learning_rate=0)


if __name__ == "__main__":
    unittest.main()
