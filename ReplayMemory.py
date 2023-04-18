from collections import deque
import random

class Memory:
    def __init__(self, length:int):
        self.memory = deque(maxlen=length)
    
    def __call__(self) -> deque:
        return self.memory

    def append(self, data):
        self.memory.append(data)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    
if __name__ == "__main__":
    memory = Memory(5)
    for i in range(10):
        memory.append(i)
    print(memory())
    print(memory.sample(5))