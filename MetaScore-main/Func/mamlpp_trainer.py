import torch
import torch.nn as nn
from torch_geometric.data import Batch
import higher

class MAMLPlusPlusTrainer:
    def __init__(self, model, meta_optimizer, learnable_lr, task_attention, loss_fn, device, num_inner_loops=5, logger=None):
        """
        Initialize the MAML++ trainer.

        Args:
            model (nn.Module): The model to be trained.
            meta_optimizer (torch.optim.Optimizer): The meta optimizer.
            learnable_lr (LearnableLR): The learnable learning rate scheduler.
            task_attention (TaskAttention): The task self-attention mechanism.
            loss_fn (callable): The loss function.
            device (str): The device type.
            num_inner_loops (int): The number of inner loop updates.
            logger (Logger, optional): Logger instance.
        """
        self.model = model.to(device)
        self.meta_optimizer = meta_optimizer
        self.learnable_lr = learnable_lr
        self.task_attention = task_attention
        self.loss_fn = loss_fn
        self.device = device
        self.num_inner_loops = num_inner_loops
        self.logger = logger

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier uniform initialization"""
        for m in self.model.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def train_epoch(self, dataloader):
        """
        Train one epoch.

        Args:
            dataloader (DataLoader): The data loader.

        Returns:
            float: The average meta loss.
        """
        self.model.train()
        meta_loss_epoch = 0.0

        for batch_idx, batch in enumerate(dataloader):
            meta_loss = 0.0
            task_losses = []
            task_embeddings = []

            for task in batch:
                support_set = task['support_set']
                query_set = task['query_set']

                # Move support and query sets to the device
                support_protein_graphs = Batch.from_data_list(
                    [sample[0] for sample in support_set]).to(self.device)
                support_ligand_graphs = Batch.from_data_list(
                    [sample[1] for sample in support_set]).to(self.device)
                support_kd_values = torch.cat(
                    [sample[2] for sample in support_set], dim=0).to(self.device)

                query_protein_graphs = Batch.from_data_list(
                    [sample[0] for sample in query_set]).to(self.device)
                query_ligand_graphs = Batch.from_data_list(
                    [sample[1] for sample in query_set]).to(self.device)
                query_kd_values = torch.cat(
                    [sample[2] for sample in query_set], dim=0).to(self.device)

                # Get the learning rate for the current task
                lr_dict = self.learnable_lr.get_lrs()
                param_groups = []
                lr_list = []
                for name, param in self.model.named_parameters():
                    if name in lr_dict:
                        lr = lr_dict[name].item()
                        param_groups.append({'params': param, 'lr': lr})
                        lr_list.append(lr)
                    else:
                        lr = 1e-4  # Reduce default learning rate
                        param_groups.append({'params': param, 'lr': lr})
                        lr_list.append(lr)

                # Use Adam as the inner loop optimizer
                inner_optimizer = torch.optim.Adam(param_groups)

                with higher.innerloop_ctx(
                    self.model,
                    inner_optimizer,
                    copy_initial_weights=False,
                    override={'lr': lr_list}
                ) as (fmodel, diffopt):
                    # Inner loop: Perform several steps of gradient descent on the support set
                    for inner_step in range(self.num_inner_loops):
                        support_preds = fmodel(
                            support_protein_graphs, support_ligand_graphs)
                        support_loss = self.loss_fn(support_preds, support_kd_values)

                        # Perform parameter update using diffopt
                        diffopt.step(support_loss)

                        # Apply gradient clipping
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # Evaluate the adapted model on the query set
                    query_preds = fmodel(
                        query_protein_graphs, query_ligand_graphs)
                    query_loss = self.loss_fn(query_preds, query_kd_values)

                    # Store the query loss for later calculation of weighted meta loss
                    task_losses.append(query_loss)

                    # Calculate task representation (using support loss)
                    task_embedding = support_loss.detach().unsqueeze(0)  # shape: [1]
                    task_embeddings.append(task_embedding)

            # Stack task representations into a tensor
            task_embeddings = torch.stack(task_embeddings).to(self.device)  # shape: [meta_batch_size, 1]

            # Calculate task weights
            task_weights = self.task_attention(task_embeddings)  # shape: [meta_batch_size]

            # Weighted accumulation of meta loss
            for i in range(len(task_losses)):
                meta_loss += task_weights[i] * task_losses[i]

            # Outer loop: Update global model parameters and learnable learning rate parameters
            self.meta_optimizer.zero_grad()
            meta_loss.backward()

            # Apply gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.meta_optimizer.step()

            # Record meta loss
            meta_loss_epoch += meta_loss.item()

            if self.logger:
                self.logger.info(f"Batch [{batch_idx + 1}/{len(dataloader)}], Meta Loss: {meta_loss.item():.4f}")

        # Calculate average meta loss
        avg_meta_loss = meta_loss_epoch / len(dataloader)
        return avg_meta_loss
