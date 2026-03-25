import random
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from config import Config


MAX_STEPS = 3000


class LaunchStrategy:
    def __init__(self, type: str, data: dict):
        self.type = type
        self.data = data

    @classmethod
    def uniform_jittered(cls, num_cars: int, mps_range: tuple):
        return cls(
            type="uniform_jittered",
            data={
                "num_cars": num_cars,
                "mps_range": mps_range,
            },
        )

    @classmethod
    def eval_launch(cls):
        return cls(
            type="eval_launch",
            data={"num_cars": 1},
        )


class Task:

    def __init__(
        self,
        *,
        launch_strategy: LaunchStrategy,
        random_policy: bool,
        table_name: Optional[str],
        time_out_fn,
        is_eval: bool,
        opponent_policy: Optional[str] = None,
    ):
        self.launch_strategy = launch_strategy
        self.random_policy = random_policy
        self.table_name = table_name
        self.time_out_fn = time_out_fn
        self.is_eval = is_eval
        self.opponent_policy = opponent_policy


class TaskServer:
    def __init__(self, cfg: "Config"):
        self.cfg = cfg

        self.redis = cfg.redis

    @property
    def warm_up(self) -> bool:
        return self.redis.get("warm_up") in [str(True), "1", None]

    @warm_up.setter
    def warm_up(self, value: bool):
        self.redis.set("warm_up", str(value))

    @property
    def needs_eval(self) -> bool:
        return self.redis.get("needs_eval") == str(True)

    @needs_eval.setter
    def needs_eval(self, value: bool):
        self.redis.set("needs_eval", str(value))

    @property
    def mistake_learning_tasks(self) -> List[str]:
        return []

    def mistake_slots_available(self) -> bool:
        return False

    def pop_mistake_learning_task(self) -> Task:
        raise NotImplementedError("No mistake learning tasks implemented yet.")


TASK_SPECS = [
    Task(
        launch_strategy=LaunchStrategy.uniform_jittered(
            num_cars=500, mps_range=(0.0, 0.0)
        ),
        random_policy=False,
        table_name="1v0",
        time_out_fn=lambda steps, obs: steps >= MAX_STEPS,
        is_eval=False,
    ),
]


# class TaskSpec:
#     def __init__(
#         self,
#         name: str,
#         launch_strategy: LaunchStrategy,
#         table_name: str | None,
#         time_out_fn: Callable[[int, "torch.Tensor"], bool],
#         opponent_policy: str | None = None,
#     ):
#         self.name = name
#         self.launch_strategy = launch_strategy
#         self.table_name = table_name
#         self.time_out_fn = time_out_fn
#         self.opponent_policy = opponent_policy

#     @classmethod
#     def random_spec(cls, cfg: "Config") -> "TaskSpec":
#         return random.choice(TASK_SPECS)


def get_task(cfg: "Config") -> Task:
    server = TaskServer(cfg)

    if server.warm_up:
        return Task(
            launch_strategy=LaunchStrategy.uniform_jittered(
                num_cars=400, mps_range=(0.5, 1.0)
            ),
            random_policy=True,
            table_name="1v0",
            time_out_fn=lambda steps, obs: steps >= MAX_STEPS,
            is_eval=False,
        )
    elif server.needs_eval:
        # WTF is a race condition?
        server.needs_eval = False

        return Task(
            launch_strategy=LaunchStrategy.eval_launch(),
            random_policy=False,
            table_name=None,
            time_out_fn=lambda steps, obs: steps >= MAX_STEPS,
            is_eval=True,
        )
    elif len(server.mistake_learning_tasks) > 0 and server.mistake_slots_available():
        return server.pop_mistake_learning_task()
    else:
        # Standard data collection task
        # task_spec = TaskSpec.random_spec(cfg)
        return random.choice(TASK_SPECS)
