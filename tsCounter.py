from threading import Lock
class tsCounter:
    def __init__(self) -> None:
        self.lock = Lock()
        self.counterValue = 0
    def getValue(self) -> int:
        return self.counterValue
    def increment(self):
        try:
            self.lock.acquire()
            self.counterValue = self.counterValue + 1
        finally:
            self.lock.release()
    def decrement(self):
        try:
            self.lock.acquire()
            self.counterValue = self.counterValue - 1
        finally:
            self.lock.release()
        