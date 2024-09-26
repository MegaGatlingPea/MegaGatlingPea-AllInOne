import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Model.MetaScore import MetaScore

class MAMLPlusPlus:
    def __init__(self, model, inner_lr, outer_lr, num_inner_steps):
        self.model = model
        self.inner_lr = nn.Parameter(torch.ones(len(list(model.parameters()))) * inner_lr)
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        self.outer_optimizer = optim.Adam(list(self.model.parameters()) + [self.inner_lr], lr=outer_lr)

    def inner_loop(self, support_set):
        temp_model = self.model  # Create a copy of the model for inner loop updates
        
        losses = []
        for step in range(self.num_inner_steps):
            loss = self.compute_loss(temp_model, support_set)
            losses.append(loss)
            grads = torch.autograd.grad(loss, temp_model.parameters(), create_graph=True)
            
            # Update temp_model parameters with per-layer learning rates
            for param, grad, lr in zip(temp_model.parameters(), grads, self.inner_lr):
                param.data = param.data - lr * grad

        return temp_model, losses

    def outer_loop(self, tasks):
        self.outer_optimizer.zero_grad()
        meta_loss = 0
        derivative_reg = 0

        for task in tasks:
            support_set, query_set = task
            updated_model, inner_losses = self.inner_loop(support_set)
            task_loss = self.compute_loss(updated_model, query_set)
            meta_loss += task_loss
            
            # Compute derivative regularization
            for i, loss in enumerate(inner_losses):
                derivative_reg += torch.norm(torch.autograd.grad(loss, updated_model.parameters(), create_graph=True)[0], p=2)

        meta_loss /= len(tasks)
        derivative_reg /= (len(tasks) * self.num_inner_steps)
        total_loss = meta_loss + 0.01 * derivative_reg  # Adjust the regularization strength as needed
        total_loss.backward()
        self.outer_optimizer.step()

        return meta_loss.item(), derivative_reg.item()

    def compute_loss(self, model, data):
        inputs, targets = data
        outputs = model(inputs)
        return nn.functional.mse_loss(outputs, targets)

    def train(self, num_epochs, tasks):
        for epoch in range(num_epochs):
            meta_loss, deriv_reg = self.outer_loop(tasks)
            print(f"Epoch {epoch+1}, Meta Loss: {meta_loss:.4f}, Derivative Regularization: {deriv_reg:.4f}")

metascore = MetaScore(input_dim, hidden_dim, output_dim)
maml_plus_plus = MAML_PlusPlus(metascore, inner_lr, outer_lr, num_inner_steps)

# Assume we have a function to load tasks from clusters
tasks = load_tasks_from_clusters(num_clusters)

maml_plus_plus.train(num_epochs, tasks)