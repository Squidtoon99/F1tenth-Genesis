from typing import TYPE_CHECKING, List
import numpy as np

if TYPE_CHECKING:
    from config import Config

MAX_STEPS = 1000


class MistakeLearningTask:
    def __init__(self, task_spec: dict, key: str):
        self.task_spec = task_spec
        self.key = key


class TaskServer:
    def __init__(self, cfg: "Config"):
        self.warm_up = True
        self.needs_eval = False
        self.cfg = cfg

    @property
    def mistake_learning_tasks(self) -> List[MistakeLearningTask]:
        mistakes = self.cfg.redis.hgetall("mistake_learning_tasks")
        if not mistakes:
            return []
        return [
            MistakeLearningTask(task_spec=v, key=key) for key, v in mistakes.items()  # type: ignore
        ]

    @property
    def warm_up(self) -> bool:
        return self.cfg.redis.get("task_server_warm_up") == "True"

    def acquire_mistake_learning_task(self) -> MistakeLearningTask | None:
        # acquire the lock for global mistake learning task list, pop one task, and release the lock
        lock = self.cfg.redis.lock("mistake_learning_tasks_lock", blocking=False)
        if not lock.acquire(blocking=False):
            return None

        tasks = self.mistake_learning_tasks
        if len(tasks) == 0:
            lock.release()
            return None
        task = tasks[0]
        self.cfg.redis.hdel("mistake_learning_tasks", task.key)
        return task


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

    @classmethod
    def eval_launch(cls):
        return cls(
            type="eval_launch",
            data={},
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

    @classmethod
    def from_mistake_learning_task(cls, mistake_learning_task: MistakeLearningTask):
        # parse the task_spec and return a Task object
        task_spec = mistake_learning_task.task_spec
        return cls(
            launch_strategy=LaunchStrategy(**task_spec["launch_strategy"]),
            random_policy=task_spec["random_policy"],
            table_name=task_spec["table_name"],
            time_out_fn=lambda steps, obs: steps >= MAX_STEPS,
            is_eval=False,
        )


def get_task(cfg: "Config") -> Task:
    # The returned task defines which cars are agent controlled and for each opponent
    # whether to control it with a pre-trained policy, PID, or the built-in AI
    task_server = TaskServer(cfg)
    if task_server.warm_up:
        # During warm up, run 20 cars across the track at once all selecting
        # uniformly random actions
        return Task.from_config(
            launch_strategy=LaunchStrategy.uniform_jittered(
                num_cars=20, mph_range=(10, 30)
            ),
            random_policy=True,
            table_name="1v0",
            time_out_fn=lambda steps, obs: steps >= MAX_STEPS,
            is_eval=False,
        )
    elif len(task_server.mistake_learning_tasks) > 0 and (
        task := task_server.acquire_mistake_learning_task()
    ):
        # acquire the lock for a mistake learning task and return it if available
        return Task.from_mistake_learning_task(task)
    else:
        # Standard data collection setting
        # First we sample a task specification (e.g., 1v0, slipstream, etc.)
        # The TaskSpec is almost a Task, but the launch, opponent policies, and
        # builtin-ai params need to be sampled from a distribution
        # task_spec = np.random.choice(TASK_SPECS, p=cfg.TASK_SPECS_PROBS)
        # return Task(
        #     launch_strategy=task_spec.sample_launch(),
        #     random_policy=False,
        #     agent_controlled_car_ids=task_spec.agent_ids,
        #     car_id_to_opp_policy=task_spec.sample_opp_policies(),
        #     car_id_to_opp_pid=task_spec.pids,
        #     builtin_ai_car_ids=task_spec.builtin_ids,
        #     table_name=task_spec.table_name,
        #     time_out_fn=task_spec.time_out_fn,
        #     is_eval=False,
        # )

        # TODO: Implement this stuff
        raise NotImplementedError("Standard task sampling not implemented yet")
