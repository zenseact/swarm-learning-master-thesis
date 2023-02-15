import unittest
import ray

from utils.training.swarm import SwarmClient 

class TestSwarmClient(unittest.TestCase):
    def setUp(self):
        ray.init()

    def tearDown(self):
        ray.shutdown()

    def test_ping(self):
        # Test that ping() returns the id of the target client's current model
        client1 = SwarmClient.remote(1, [1, 2, 3])
        client2 = SwarmClient.remote(2, [4, 5, 6])
        mid = ray.get(client1.ping.remote(client2))
        self.assertEqual(mid, ray.get(client2.get_model.remote()).mid)

    def test_fetch(self):
        # Test that fetch() returns the model with the given id from the target client
        client1 = SwarmClient.remote(1, [1, 2, 3])
        client2 = SwarmClient.remote(2, [4, 5, 6])
        mid = ray.get(client1.ping.remote(client2))
        model = ray.get(client1.fetch.remote(client2, mid))
        self.assertEqual(model.mid, mid)
        self.assertEqual(model.parameters, [4, 5, 6])

    def test_set_model(self):
        # Test that set_model() sets the current model of the client
        client = SwarmClient.remote(1, [1, 2, 3])
        ray.get(client.set_model.remote([4, 5, 6]))
        model = ray.get(client.get_model.remote())
        self.assertEqual(model.parameters, [4, 5, 6])

    def test_model_history(self):
        # Test that model_history() returns the history of models sorted by creation time
        client = SwarmClient.remote(1, [1, 2, 3], save_all=True)
        ray.get(client.set_model.remote([4, 5, 6]))
        ray.get(client.set_model.remote([7, 8, 9]))
        history = ray.get(client.get_model_history.remote())
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0].parameters, [1, 2, 3])
        self.assertEqual(history[1].parameters, [4, 5, 6])
        self.assertEqual(history[2].parameters, [7, 8, 9])
        
        # The same test but with save_all=False
        client = SwarmClient.remote(1, [1, 2, 3], save_all=False)
        ray.get(client.set_model.remote([4, 5, 6]))
        ray.get(client.set_model.remote([7, 8, 9]))
        history = ray.get(client.get_model_history.remote())
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].parameters, [1, 2, 3])
    
    def test_ping_sequence(self):
         # Create clients
        client1 = SwarmClient.remote(1, [1, 2, 3], save_all=True)
        client2 = SwarmClient.remote(2, [4, 5, 6], save_all=True)
        client3 = SwarmClient.remote(3, [7, 8, 9], save_all=True)

        # Ping clients in sequence
        mid1 = ray.get(client1.ping.remote(client2))
        mid2 = ray.get(client2.ping.remote(client3))
        mid3 = ray.get(client3.ping.remote(client1))

        # Check model history
        history1 = ray.get(client1.get_model_history.remote())
        history2 = ray.get(client2.get_model_history.remote())
        history3 = ray.get(client3.get_model_history.remote())

        # All clients should have 1 model in history (the initial model)
        # Despite being pinged. This is because the initial model
        # Is already saved in history when the client is created
        self.assertEqual(len(history1), 1)
        self.assertEqual(len(history2), 1)
        self.assertEqual(len(history3), 1)
        
    def test_initial_model_in_history(self):
        client1 = SwarmClient.remote(1, [1, 2, 3], save_all=True)
        client2 = SwarmClient.remote(1, [1, 2, 3], save_all=False)
        
        history1 = ray.get(client1.get_model_history.remote())
        history2 = ray.get(client2.get_model_history.remote())

        # one initial model that exists in history
        self.assertEqual(len(history1), 1)
        self.assertEqual(len(history2), 1)

if __name__ == '__main__':
    unittest.main()
