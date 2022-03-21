from src.policies.random_policy import RandomPolicy
from src.policies.simple_fifo import SimpleFifoPolicy

name_to_policy = {"sfifo": SimpleFifoPolicy, "random": RandomPolicy}
