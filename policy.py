from typing import TYPE_CHECKING
from torch import nn

if TYPE_CHECKING:
    from config import Config


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # define your model architecture here

    def forward(self, x):
        # define the forward pass of your model here
        return x


# https://github.com/sony/CaR/blob/main/car/algorithms/qrsac.py


class PolicySync:
    def __init__(self, cfg: "Config"):
        self.cfg = cfg
        self.log = self.cfg.logger.getChild("PolicySync")
        # watch for new policy updates in redis and update the local policy accordingly
        self.cfg.redis.subscribe("policy_updates", self.handle_policy_update)

        self._policy = None

    @property
    def policy(self):
        if self._policy is None:
            # load initial policy from redis
            self._policy = self.cfg.redis.get("current_policy")

    def handle_policy_update(self, message):
        # handle the policy update message and update the local policy
        self.log.info(f"Received policy update: {message['data']}")
        self._policy = message["data"]
