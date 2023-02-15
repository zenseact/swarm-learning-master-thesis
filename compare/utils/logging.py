import json
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class WriterHandler:
    def __init__(self, config) -> None:
        now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        template = f"""
# Run Configuration
This is an automated report of the configuration used to run this experiment.

**Run at: {now}**

```json
{json.dumps(config, indent=4)}
```
"""
        self.path = "runs/{}".format(now)
        self.writer = SummaryWriter(self.path, comment=config)
        self.writer.add_text("/federated/configuration", template)
