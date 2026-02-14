import os
import numpy as np
import genesis as gs
import logging
from typing import List, TYPE_CHECKING

from config import Config
from policy import PolicySync

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    logging.warning(
        "python-dotenv not found, skipping .env loading. Make sure to set environment variables manually."
    )


def rollout_loop(cfg: "Config"):
    policy = PolicySync(cfg)


def main():
    logging.info("Initializing Genesis and setting up the scene...")
    # gs.init(backend=gs.gpu)  # type: ignore

    session_id = os.getenv("SESSION_ID", "0")
    cfg = Config(session_id=session_id, redis_uri=os.getenv("REDIS_URI"))


#     scene = gs.Scene(
#         viewer_options=gs.options.ViewerOptions(
#             camera_pos=(0.0, -2.5, 1.5),
#             camera_lookat=(0.4, 0.0, 0.2),
#             camera_fov=35,
#             res=(960, 640),
#             max_FPS=60,
#         ),
#         sim_options=gs.options.SimOptions(dt=0.01),
#         show_viewer=True,
#     )

#     scene.add_entity(gs.morphs.Plane())

#     car = scene.add_entity(
#         gs.morphs.URDF(
#             file=URDF_PATH,
#             pos=(0.0, 0.0, 0.10),
#         )
#     )

#     scene.build(n_envs=100, env_spacing=(1, 1))


if __name__ == "__main__":
    main()
