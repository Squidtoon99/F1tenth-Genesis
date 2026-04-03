from config import Config
from db.models import TrainingSession
from random_word import RandomWords

cfg = Config(session_id="1", bootstrap_db=True)

if cfg.db_session_factory is None:
    raise RuntimeError("Database session factory is not initialized. Cannot proceed.")
# get new session
r = RandomWords()

with cfg.db_session_factory() as session:
    new_session = TrainingSession(
        name=f"{r.get_random_word()} {r.get_random_word()}",
        tracks=[cfg.env["track"]],
    )
    session.add(new_session)
    session.commit()


session_id = new_session.id
print(f"Created new training session with ID: {session_id}")

def start_redis():
    import subprocess
    import sys

    print("Starting Redis server...")
    try:
        # If on windows run the docker command to start redis
        if sys.platform.startswith("win"):
            subprocess.Popen(
                ["docker", "run", "--name", "redis-server", "-p", "6379:6379", "-d", "redis"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("Redis server started successfully using Docker.")
        else:
            # Attempt to start Redis server
            subprocess.Popen(["redis-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Redis server started successfully.")
    except Exception as e:
        print(f"Failed to start Redis server: {e}")
        sys.exit(1)
        
print("Starting up redis")
try:
    cfg._redis_client.ping()
    print("Successfully connected to Redis.")
except Exception as e:
    print(f"Failed to connect to Redis: {e}")
    start_redis()

print("Starting training loop...")
from trainer import train_loop

train_cfg = Config(session_id=str(session_id), bootstrap_db=False)
train_loop(train_cfg)
