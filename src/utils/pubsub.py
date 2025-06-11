# src/utils/pubsub.py
import json
import threading

import redis
from loguru import logger

# Connect to Redis
redis_client = redis.Redis(host="localhost", port=6379, db=0)


def publish(channel: str, message: dict):
    """Publishes a message to a Redis channel."""
    logger.info(f"Publishing to channel '{channel}': {message}")
    redis_client.publish(channel, json.dumps(message))


def subscribe(channel: str, callback):
    """Subscribes to a Redis channel and runs a callback for each message."""

    def worker():
        pubsub = redis_client.pubsub()
        pubsub.subscribe(channel)
        logger.info(f"Subscribed to channel '{channel}'.")
        for message in pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                logger.info(f"Received message on channel '{channel}': {data}")
                callback(data)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


def acquire_lock(lock_name: str, timeout: int = 60) -> bool:
    """
    Acquires a lock in Redis. Returns True if the lock was acquired,
    False otherwise. The lock will expire after the timeout.
    """
    return redis_client.set(lock_name, "locked", ex=timeout, nx=True)


def release_lock(lock_name: str):
    """Releases a lock in Redis."""
    redis_client.delete(lock_name)
