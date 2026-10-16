import time
from rl_engine.models.muse_glimmer.e2e.mocks import MockTrainer
class SlowMockTrainer(MockTrainer):
    def __init__(self, *a, **k):
        time.sleep(6.0)
        super().__init__(*a, **k)
