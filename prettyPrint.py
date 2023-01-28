import builtins
import time


class Log:
    def __init__(self, prefix="PrettyPrint"):
        self.prefix = prefix

    def set_prefix(self, prefix):
        self.prefix = prefix

    def get_prefix(self):
        return self.prefix

    def print(self, *objs, **kwargs):
        timestamp = time.strftime("%H:%M:%S")
        builtins.print(f"[{timestamp}][{self.prefix}]", *objs, **kwargs)
