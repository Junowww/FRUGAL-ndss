
import torch
import numpy as np
import numpy.random as rd

class BufferArray:
    def __init__(self, memo_max_len, state_dim, action_dim):
        memo_dim = 1 + 1 + state_dim + action_dim + state_dim
        self.memories = np.empty((memo_max_len, memo_dim), dtype=np.float32)

        self.next_idx = 0
        self.is_full = False
        self.max_len = memo_max_len
        self.now_len = self.max_len if self.is_full else self.next_idx

        self.state_idx = 1 + 1 + state_dim  # reward_dim==1, done_dim==1
        self.action_idx = self.state_idx + action_dim

    def add_memo(self, memo_tuple):
        numpy_memo = [m.detach().cpu().numpy() if isinstance(m, torch.Tensor) else m for m in memo_tuple]
        self.memories[self.next_idx] = np.hstack(numpy_memo)
        self.next_idx = self.next_idx + 1
        if self.next_idx >= self.max_len:
            self.is_full = True
            self.next_idx = 0

    def init_after_add_memo(self):
        self.now_len = self.max_len if self.is_full else self.next_idx

    def random_sample(self, batch_size, device):
        # same as:
        indices = rd.randint(self.now_len, size=batch_size)

        memory = self.memories[indices]
        memory_tensor = torch.from_numpy(memory).pin_memory()
        memory = memory_tensor.to(device, non_blocking=True)

        '''convert array into torch.tensor'''
        tensors = (
            memory[:, 0:1],  # rewards
            memory[:, 1:2],  # masks, mark == (1-float(done)) * gamma
            memory[:, 2:self.state_idx],  # states
            memory[:, self.state_idx:self.action_idx],  # actions
            memory[:, self.action_idx:],  # next_states
        )
        return tensors
