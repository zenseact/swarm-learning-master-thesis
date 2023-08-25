from fleetlearning.server.fleet_aggregators import BaseStrategy
import numpy as np
import pytest


class TestBaseStrategy:
    base_strategy = BaseStrategy()

    @pytest.fixture
    def mock_results(self):
        mock_weights1 = np.array([[1, 1], [2, 2]])
        mock_weights2 = np.array([[2, 2], [3, 3]])
        results = [(mock_weights1, "client1"), (mock_weights2, "client2")]
        return results

    def test_aggregate_fit_fedavg(self, mock_results):
        (
            parameters_aggregated,
            empty_dictionary,
        ) = self.base_strategy.aggregate_fit_fedavg(mock_results)
        print("aggregated: ", parameters_aggregated)

        assert empty_dictionary == {}
        np.testing.assert_array_equal(
            parameters_aggregated, [np.array([1.5, 1.5]), np.array([2.5, 2.5])]
        )
