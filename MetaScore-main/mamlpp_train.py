import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool

from Model.MetaScore import MetaScore
from Data.metadata import MetaDataset
from Func.mamlpp_trainer import MAMLPlusPlusTrainer
from Func.lr_config import LearnableLR
from Func.taskattention import TaskAttention

def main():
    # set parameters
    cluster_data_dir = './cluster_data'
    k_shot = 10
    k_query = 15
    meta_batch_size = 4       # number of tasks in a batch
    num_inner_loops = 4       # number of inner loop updates
    init_inner_lr = 0.001     # initial value of inner loop learning rate
    meta_lr = 1e-4            # outer loop learning rate
    num_epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create MetaDataset instance
    dataset = MetaDataset(cluster_data_dir=cluster_data_dir, k_shot=k_shot, k_query=k_query)

    # create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=meta_batch_size,
        shuffle=True,
        collate_fn=lambda x: x,  # not merge, let each element be a task
        num_workers=4,
        pin_memory=True if device == 'cuda' else False
    )

    # initialize MetaScore model
    input_dim_protein = 100  # adjust according to actual data
    input_dim_ligand = 100   # adjust according to actual data
    model = MetaScore(input_dim_protein, input_dim_ligand)

    # define loss function
    loss_fn = nn.MSELoss()

    # initialize learnable learning rate scheduler
    learnable_lr = LearnableLR(model, init_lr=init_inner_lr).to(device)

    # initialize task self-attention mechanism
    task_attention = TaskAttention(input_dim=1).to(device)

    # define meta optimizer
    meta_optimizer = optim.Adam(
        list(model.parameters()) +
        list(learnable_lr.parameters()) +
        list(task_attention.parameters()),
        lr=meta_lr
    )

    # create MAMLPlusPlusTrainer instance
    trainer = MAMLPlusPlusTrainer(
        model=model,
        meta_optimizer=meta_optimizer,
        learnable_lr=learnable_lr,
        task_attention=task_attention,
        loss_fn=loss_fn,
        device=device,
        num_inner_loops=num_inner_loops
    )

    # start training
    for epoch in range(1, num_epochs + 1):
        avg_meta_loss = trainer.train_epoch(dataloader)
        print(f"Epoch [{epoch}/{num_epochs}], Meta Loss: {avg_meta_loss:.4f}")

        # save model every certain epochs
        if epoch % 10 == 0:
            checkpoint_path = f'./checkpoints/mamlpp_epoch_{epoch}.pt'
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'learnable_lr_state_dict': learnable_lr.state_dict(),
                'task_attention_state_dict': task_attention.state_dict(),
                'optimizer_state_dict': meta_optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

if __name__ == '__main__':
    main()
