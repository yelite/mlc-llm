"""Base cache class."""

import time
import sys

THRESHOLD_BYTES = 128 * 1024


class BaseCache:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cache = {}
        self.total = 0
        self.hit = 0
        self.avg_init_time = 0

    def init_with_timer(self, key):
        start = time.monotonic()
        val = self.init_value(key)
        init_time = time.monotonic() - start
        curr_total = self.total
        new_total = curr_total + 1

        # Update average init time without old_avg * old_total to avoid overflow.
        self.avg_init_time = (init_time / new_total) + (
            curr_total / new_total
        ) * self.avg_init_time

        # Update value
        self.total += 1
        self.cache[key] = val

        return val

    def query(self, key):
        if key in self.cache:
            self.hit += 1
            val = self.cache[key]
        else:
            # Flush the cache is bigger than threshold (128KB)
            if self.get_cache_size() >= THRESHOLD_BYTES:
                print(f"flushing the cache with size: {self.get_cache_size()}")
                self.reset()
            # Init value in the cache
            val = self.init_with_timer(key)

        return val

    def init_value(self, key):
        raise NotImplementedError

    def get_cache_size(self):
        return sys.getsizeof(self.cache)

    def get_cache_hit_rate(self):
        if self.total == 0:
            return 0
        return self.hit / self.total

    def get_avg_init_time(self):
        return self.avg_init_time
