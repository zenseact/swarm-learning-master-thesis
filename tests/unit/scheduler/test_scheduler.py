import socket
import threading
from unittest.mock import patch
import time
from unittest.mock import patch
from fleetlearning.scheduler.scheduler_main import AGX
from fleetlearning.common.socket_utils import recieve_large_message, send_large_message
from fleetlearning.common.static_params import global_configs


global_configs.DEVICE = "cpu"


@patch(
    "fleetlearning.scheduler.scheduler_main.get_parameters",
    return_value="mocked_results",
)
@patch("fleetlearning.scheduler.scheduler_main.train_model")
@patch("fleetlearning.scheduler.scheduler_main.ZodFrames")
def test_scheduler(MockedZodFrames, mock_train_model, mock_get_parameters):
    agx = AGX(agx_ip="127.0.0.1")

    assert MockedZodFrames.call_count == 1

    agx_thread = threading.Thread(target=agx.run_scheduler, args=())
    agx_thread.start()

    time.sleep(0.01)
    fake_client = socket.socket()

    fake_client.connect(("127.0.0.1", 59999))

    message1_to_agx = {"message": "HELLO", "data": {"client_id": 1}}
    send_large_message(fake_client, message1_to_agx)

    time.sleep(0.01)

    message1_from_agx = recieve_large_message(fake_client)

    assert message1_from_agx["message"] == "TASK_SCHEDULED"

    message2_to_agx = {"message": "START", "data": {"client_id": 1}}
    send_large_message(fake_client, message2_to_agx)

    time.sleep(0.01)

    message2_from_agx = recieve_large_message(fake_client)

    fake_client.close()
    agx.run_flag = False
    agx_thread.join()

    assert mock_train_model.call_count == 1
    assert mock_get_parameters.call_count == 1
    assert message2_from_agx["message"] == "RESULTS"
    assert message2_from_agx["data"]["results"] == "mocked_results"
