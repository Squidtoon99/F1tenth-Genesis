import logging
from redis import Redis
from functools import wraps


class RedisWrapper:
    def __init__(self, redis_client: Redis, session_id: str):
        self.redis_client = redis_client
        self.session_id = session_id

    def _prefix_key(self, key: str) -> str:
        return f"{self.session_id}:{key}"

    def __getattr__(self, name):
        attr = getattr(self.redis_client, name)

        if callable(attr):

            @wraps(attr)
            def wrapper(*args, **kwargs):
                if args and isinstance(args[0], str):
                    args = (self._prefix_key(args[0]),) + args[1:]
                return attr(*args, **kwargs)

            return wrapper
        else:
            return attr


class Config:
    """Configuration class for the application."""

    def __init__(self, session_id: str, redis_uri: str | None):
        self.session_id = session_id
        self.redis_uri = redis_uri or "redis://localhost:6379/0"
        self.redis_client = Redis.from_url(self.redis_uri, decode_responses=True)
        self.redis_binary_client = Redis.from_url(
            self.redis_uri, decode_responses=False
        )
        self.redis = RedisWrapper(self.redis_client, self.session_id)
        self.redis_b = RedisWrapper(self.redis_binary_client, self.session_id)
        self.logger = logging.getLogger("Genesis")
