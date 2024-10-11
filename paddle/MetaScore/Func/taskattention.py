import torch.nn as nn
import torch.nn.functional as F

class TaskAttention(nn.Module):
    def __init__(self, input_dim):
        super(TaskAttention, self).__init__()
        """
        Initialize the task self-attention mechanism.

        Args:
            input_dim (int): The dimension of task representations.
        """
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, task_embeddings):
        """
        Calculate the weights for each task.

        Args:
            task_embeddings (Tensor): The task representations tensor with shape [meta_batch_size, input_dim].

        Returns:
            Tensor: The task weights tensor with shape [meta_batch_size].
        """
        x = F.relu(self.fc1(task_embeddings))
        x = self.fc2(x)
        weights = F.softmax(x.squeeze(-1), dim=0)
        return weights
