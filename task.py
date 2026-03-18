from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config import Config

MAX_STEPS = 1000


class LaunchStrategy:
    def __init__(self, type: str, data: dict):
        self.type = type
        self.data = data

    @classmethod
    def uniform_jittered(cls, num_cars: int, mph_range: tuple):
        return cls(
            type="uniform_jittered",
            data={
                "num_cars": num_cars,
                "mph_range": mph_range,
            },
        )


class Task:
    def __init__(
        self,
        launch_strategy: LaunchStrategy,
        random_policy: bool,
        table_name: str,
        time_out_fn,
        is_eval: bool,
    ):
        self.launch_strategy = launch_strategy
        self.random_policy = random_policy
        self.table_name = table_name
        self.time_out_fn = time_out_fn
        self.is_eval = is_eval

    @classmethod
    def from_config(
        cls,
        launch_strategy: LaunchStrategy,
        random_policy: bool,
        table_name: str,
        time_out_fn,
        is_eval: bool,
    ):
        return cls(
            launch_strategy=launch_strategy,
            random_policy=random_policy,
            table_name=table_name,
            time_out_fn=time_out_fn,
            is_eval=is_eval,
        )


def get_task(cfg: "Config") -> Task:
    return Task.from_config(
        launch_strategy=LaunchStrategy.uniform_jittered(num_cars=20, mph_range=(10, 30)),
        random_policy=True,
        table_name="warmup_1v0",
        time_out_fn=lambda steps, obs: steps >= MAX_STEPS,
        is_eval=False,
    )
