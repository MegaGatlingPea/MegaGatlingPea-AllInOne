import os
import copy, argparse, json, pickle
import sys
import torch
import numpy as np

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DistributedDataParallel
from time import time as get_time
from torch import autocast
from torch.utils.data import DataLoader
from coati.common import *
from coati.data.dataset import ecloud_dataset

from coati.models.autograd_funs.autograd_funs import all_gather
from coati.models.encoding.tokenizers.trie_tokenizer import TrieTokenizer
from coati.models.encoding.tokenizers import get_vocab
from coati.models.encoding.clip_e2e import clip_ar_xform, e3gnn_smiles_clip_e2e
from coati.models.encoding.clip_e2e import clip_loss as clip_loss_module
from coati.training.logger import COATILogger
from coati.common.util import makedir, utc_epoch_now


def optimizer_to(optim, device):
    """
    将优化器的状态移动到指定的设备上。

    参数：
    - optim: 优化器对象
    - device: 目标设备（如 'cuda' 或 'cpu'）
    """
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def serialize_model(
    train_args,
    # dataset_summary,
    model_state_dict,
    model_kwargs,
    optimizer_state_dict=None,
    **kwargs,
):
    """
    序列化模型和训练状态，以便保存或恢复。

    参数：
    - train_args: 训练参数的字典
    - model_state_dict: 模型的状态字典
    - model_kwargs: 初始化模型时的关键字参数
    - optimizer_state_dict: 优化器的状态字典
    - **kwargs: 其他附加信息

    返回：
    - 序列化后的二进制数据
    """
    d = pickle.dumps(
        {
            "train_args": train_args,
            # "dataset_summary": dataset_summary,
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
            "model_kwargs": model_kwargs,
            **kwargs,
        },
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    print("模型文档大小（MB）：", sys.getsizeof(d) / (1024 * 1024))
    return d


def train_autoencoder(gpu, args):
    """
    训练自动编码器模型。

    参数：
    - gpu: 当前进程使用的GPU索引
    - args: 命令行参数对象
    """
    # 设置当前进程的全局排名（rank）
    rank = 0  # args.nr * args.gpus + gpu
    print(f"训练自动编码器，rank {rank} 已启动。")

    # 获取本地进程的GPU索引，并设置CUDA设备
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # 初始化分布式进程组，使用NCCL后端
    dist.init_process_group(
        backend="nccl"
        # init_method="env://",
        # world_size=args.world_size,  # 由 mp.spawn 调用者计算
        # rank=rank,
    )

    # 如果是主进程（rank == 0），则创建输出目录并保存参数
    if rank == 0:
        output_path = os.path.join(args.output_dir, args.exp_name, args.run_name)
        makedir(output_path)
        with open(os.path.join(output_path, "params.json"), "w") as f:
            json.dump(vars(args), f)

    # 加载数据集，并拆分为训练集和验证集
    dataset = ecloud_dataset(args.data_dir)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [int(len(dataset)) - int(len(dataset) / 10), int(len(dataset) / 10)],
    )
    # 初始化分词器
    tokenizer = TrieTokenizer(n_seq=args.n_seq, **get_vocab(args.tokenizer_vocab))
    token_entropy_unit = np.log(float(len(tokenizer.keys))) / np.log(2.0)

    # 如果是主进程，初始化日志记录器
    if rank == 0:
        logger = COATILogger(
            model_name="e3gnn_smiles_clip_e2e",
            run_time=args.run_name,
            output_path=args.output_dir,
            model_path=args.model_dir,
            args=vars(args),
            dataset="",
        )
        logger.start()

    # 设置设备和数据类型
    device = torch.device(f"cuda:{local_rank}")
    dtype = eval("torch." + args.dtype)
    torch.set_default_dtype(dtype)
    print("使用设备：", device)

    # 定义数据转换函数，用于数据增强和预处理
    xform_routine = lambda X: clip_ar_xform(
        X,
        tokenizer=tokenizer,
        device=device,
        p_dataset=args.p_dataset,
        p_formula=args.p_formula,
        p_fim=args.p_fim,
        p_graph=args.p_graph,
        p_clip=args.p_clip,
        p_clip_cut=args.p_clip_cut,
        p_randsmiles=args.p_randsmiles,
    )

    # 设置模型的初始化参数
    kwargs = {
        "n_layer_xformer": args.n_layer_xformer,
        "n_layer_e3gnn": args.n_layer_e3gnn,
        "n_hidden_e3nn": args.n_hidden_e3nn,
        "n_hidden_xformer": args.n_hidden_xformer,
        "n_embd_common": args.n_embd_common,
        "biases": args.biases,
        "n_head": args.n_head,
        "n_seq": args.max_n_seq,
        "n_tok": tokenizer.n_token,
        "torch_emb": args.torch_emb,
        "norm_clips": args.norm_clips,
        "norm_embed": args.norm_embed,
        "token_mlp": args.token_mlp,
    }

    # 如果不使用CLIP，则不使用点编码器
    if not args.do_clip:
        kwargs["use_point_encoder"] = False

    # 保存模型初始化参数，以便后续恢复
    model_kwargs = kwargs.copy()
    # 将设备信息添加到模型参数中
    kwargs["device"] = device

    # 初始化模型
    model = e3gnn_smiles_clip_e2e(**kwargs)
    if rank == 0:
        print("端到端的CLIP自回归模型：", model)

    # 定义优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
        eps=1e-8,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
    ngrad_updates = 0  # 已进行的梯度更新次数
    offline_losses = {"batch_losses": [], "ar_losses": [], "clip_losses": []}
    n_toks = 0  # 已处理的令牌数

    if rank == 0:
        makedir(args.model_dir)

    # 如果指定了恢复文档，则从检查点加载模型和优化器状态
    if not (args.resume_document is None):
        # 仅主进程具有日志记录器
        with open(args.resume_document, "rb") as f_in:
            model_doc = pickle.load(f_in)

        if "n_toks_processed" in model_doc:
            n_toks = model_doc["n_toks_processed"]
        if "n_grads_processed" in model_doc:
            ngrad_updates = model_doc["n_grads_processed"]
        model_dict_ = model_doc["model"]
        # 去除可能存在的 'module.' 前缀
        new_names = [
            k.replace("module.", "") if k.startswith("module.") else k
            for k in model_dict_.keys()
        ]
        model_dict = {
            new_name: t for new_name, t in zip(new_names, model_dict_.values())
        }
        if args.load_transformer_only:
            print("从检查点仅加载Transformer部分")
            xformer_dict = {
                new_name: t
                for new_name, t in zip(new_names, model_dict.values())
                if new_name.split(".")[0] == "xformer"
            }
            smiles_to_clip_dict = {
                new_name: t
                for new_name, t in zip(new_names, model_dict.values())
                if new_name.split(".")[0] == "smiles_to_clip"
            }
            model.xformer.load_state_dict(xformer_dict, strict=False)
            model.smiles_to_clip.load_state_dict(smiles_to_clip_dict, strict=False)
        else:
            model.load_state_dict(model_dict, strict=False)

        if args.resume_optimizer:
            try:
                optimizer.load_state_dict(model_doc["optimizer"])
                optimizer_to(optimizer, device)
            except Exception as Ex:
                print("无法恢复优化器状态", Ex)
                pass
        else:
            pass
        print("已从检查点加载模型。")

    # 使用DistributedDataParallel包装模型，以支持多GPU并行训练
    model = DistributedDataParallel(
        model, device_ids=[local_rank], find_unused_parameters=True
    )
    # 初始化CLIP损失计算模块
    clip_computer = clip_loss_module()

    def do_epoch(epoch, dataset, partition="train"):
        """
        进行一个训练或验证周期。

        参数：
        - epoch: 当前的周期编号
        - dataset: 使用的数据集
        - partition: 'train' 或 'test'，表示训练或验证
        """
        nonlocal ngrad_updates, n_toks, offline_losses
        res = {"loss": 0, "counter": 0, "loss_arr": []}

        t0 = get_time()  # 记录周期开始时间
        ng = 0  # 当前周期的梯度更新次数

        def do_minibatch(i, batch_data):
            """
            处理一个小批量数据。

            参数：
            - i: 当前批次的索引
            - batch_data: 当前批次的数据
            """
            nonlocal ngrad_updates, ng, t0, res, optimizer, model, epoch, n_toks, offline_losses

            if partition == "train":
                model.train()
                optimizer.zero_grad()
            else:
                model.eval()

            # 将数据移动到设备上
            raw_tokens = torch.Tensor(batch_data["raw_tokens"]).to(device)
            augmented_tokens = torch.Tensor(batch_data["augmented_tokens"]).to(device)
            eclouds = torch.Tensor(batch_data["eclouds"]).to(torch.float).to(device)

            # 前向传播，获取模型输出
            if partition == "train":
                h_e3gnn, h_xformer, logits, bad_rows = model.module.forward_dist(
                    raw_tokens,
                    augmented_tokens,
                    eclouds,
                    tokenizer,
                    p_clip_emb_smi=args.p_clip_emb_smi,
                )
            if partition == "test":
                with torch.no_grad():
                    h_e3gnn, h_xformer, logits, bad_rows = model.module.forward_dist(
                        raw_tokens,
                        augmented_tokens,
                        eclouds,
                        tokenizer,
                        p_clip_emb_smi=args.p_clip_emb_smi,
                    )

            # 聚合所有进程的bad_rows和模型输出
            bad_rows = all_gather(bad_rows)
            all_h_xformer = all_gather(h_xformer)
            all_h_e3gnn = all_gather(h_e3gnn)

            # 准备目标输出，用于计算自回归损失
            y_next = torch.zeros_like(augmented_tokens).to(device)
            y_next[:, :(augmented_tokens.shape[1] - 1)] = augmented_tokens.clone()[:, 1:]
            # 忽略特定的令牌
            y_next[y_next == tokenizer.clip_token] = -1
            y_next[y_next == tokenizer.pad_token] = -1
            y_next[y_next == tokenizer.smiles_token] = -1
            y_next[y_next == tokenizer.unk_token] = -1
            y_next[y_next == tokenizer.suffix_token] = -1
            y_next[y_next == tokenizer.middle_token] = -1

            # 计算自回归损失
            ar_loss_ = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y_next.view(-1).long(),
                ignore_index=-1,
            )
            ar_loss = ar_loss_.mean()

            # 如果使用CLIP，则计算CLIP损失
            if args.do_clip:
                clip_loss_ = clip_computer(all_h_xformer, all_h_e3gnn, bad_rows)
                clip_loss = clip_loss_.mean()
                loss = ar_loss + clip_loss * token_entropy_unit
            else:
                loss = ar_loss

            # 反向传播和优化器更新
            if partition == "train":
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()

            if rank == 0:
                ngrad_updates += batch_data["eclouds"].shape[0]
                ng += batch_data["eclouds"].shape[0]
                n_toks += (batch_data["eclouds"] > 0).sum().item()

            # 定期保存模型
            if ngrad_updates * args.world_size > args.ngrad_to_save and rank == 0:
                ngrad_updates = 0
                if args.data_parallel:
                    msd = model.module.state_dict()
                else:
                    msd = model.state_dict()
                model_doc = serialize_model(
                    train_args=vars(args),
                    # dataset_summary=dataset.summary,
                    model_state_dict=copy.deepcopy(msd),
                    model_kwargs=model_kwargs,
                    optimizer_state_dict=copy.deepcopy(optimizer.state_dict()),
                    n_toks_processed=n_toks,
                    n_grads_processed=ngrad_updates,
                    offline_loss=offline_losses,
                )
                logger.log_pytorch(
                    model_doc,
                    tags={"train_epoch": str(epoch), "dataset_epoch": str(epoch)},
                )
                del model_doc, msd

            # 记录和打印训练信息
            if (i % args.log_batch_loss) == 0 and rank == 0:
                step_batch_loss = logger.log_metric(
                    partition + "_batch_loss",
                    loss.item(),
                    dataset_epoch=epoch,
                    step=i,
                    tags={"n_toks": n_toks},
                )
                step_ar_loss = logger.log_metric(
                    partition + "_ar_loss",
                    ar_loss.item(),
                    dataset_epoch=epoch,
                    step=i,
                    tags={"n_toks": n_toks},
                )
                if args.do_clip:
                    step_clip_loss = logger.log_metric(
                        partition + "_clip_loss",
                        clip_loss.item(),
                        dataset_epoch=epoch,
                        step=i,
                        tags={"n_toks": n_toks},
                    )

                offline_losses["batch_losses"].append(step_batch_loss)
                offline_losses["ar_losses"].append(step_ar_loss)
                if args.do_clip:
                    offline_losses["clip_losses"].append(step_clip_loss)

            res["loss"] += loss.item() * args.batch_size
            res["counter"] += args.batch_size
            res["loss_arr"].append(loss.item())

            prefix = ""
            if partition != "train":
                prefix = ">> %s \t" % partition
            if i % args.log_interval == 0 and rank == 0:
                print(
                    prefix
                    + "Epoch %d | step %d \t toks %im | ar_l: %.4f, clip_l: %.4f, avg_l: %.4f \t grads_ps %.4f"
                    % (
                        epoch,
                        i,
                        int(n_toks / 1e6),
                        ar_loss,
                        clip_loss if args.do_clip else -1,
                        sum(res["loss_arr"][-10:]) / len(res["loss_arr"][-10:]),
                        ng / (get_time() - t0),
                    )
                )

            del batch_data
            return

        # 使用DistributedSampler对数据进行采样
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True
        )
        # 创建数据加载器
        epoch_iter = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            sampler=sampler,
        )
        # 遍历数据集中的每个批次
        for i, batch_data in enumerate(epoch_iter):
            do_minibatch(i, batch_data)

        # 训练周期结束后，更新学习率
        if partition == "train":
            lr_scheduler.step()
        if res["counter"] == 0:
            return

        # 主进程记录周期信息
        if rank == 0:
            print(f"周期完成，共进行了 {ng} 次梯度更新，耗时 {get_time() - t0} 秒")
            logger.log_metric(
                partition + " epoch mean loss",
                res["loss"] / res["counter"],
                dataset_epoch=epoch,
            )
        return res["loss"] / res["counter"]

    # 初始化训练结果的字典
    res = {
        "epochs": [],
        "losses": [],
        "best_test": 1e10,
        "best_epoch": 0,
        "best_model": None,
    }
    # 开始训练循环
    for epoch in range(0, args.n_epochs):
        do_epoch(epoch, train_dataset, partition="train")
        # 定期在验证集上评估模型
        if epoch % args.test_interval == 0 and epoch > 0 and rank == 0:
            test_loss = do_epoch(epoch, val_dataset, partition="test")
            if test_loss is None:
                continue
            res["epochs"].append(epoch)
            res["losses"].append(test_loss)
            if test_loss < res["best_test"]:
                res["best_test"] = test_loss
                res["best_epoch"] = epoch
                if args.data_parallel:
                    msd = model.module.state_dict()
                else:
                    msd = model.state_dict()
                res["best_model"] = copy.deepcopy(msd)
                del msd
            print("验证集损失: %.4f \t 周期 %d" % (test_loss, epoch))
            print(
                "最佳模型: 验证集损失: %.4f \t 周期 %d"
                % (res["best_test"], res["best_epoch"])
            )

    # 训练完成后，保存最佳模型
    if rank == 0:
        model_doc = serialize_model(
            train_args=vars(args),
            # dataset_summary=dataset.summary,
            model_state_dict=res["best_model"],
            model_kwargs=model_kwargs,
            optimizer_state_dict=copy.deepcopy(optimizer.state_dict()),
            n_toks_processed=n_toks,
            n_grads_processed=ngrad_updates,
        )
        logger.log_pytorch(model_doc, tags={"best": "best"})


def do_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="token_transformer")
    parser.add_argument("--exp_name", type=str, default="token_transformer")
    parser.add_argument("--run_name", type=str, default=str(int(utc_epoch_now())))
    parser.add_argument("--output_dir", type=str, default="COATI_outputs")
    parser.add_argument("--model_dir", type=str, default="COATI_models")
    parser.add_argument("--data_dir", type=str, default="COATI_data")

    # 分布式训练选项
    parser.add_argument(
        "-ws", "--world_size", default=1, type=int, help="总的进程数"
    )
    parser.add_argument(
        "-nr", "--nr", default=0, type=int, help="节点内的排名"
    )
    parser.add_argument(
        "-n", "--nodes", default=1, type=int, metavar="N", help="节点数"
    )
    parser.add_argument(
        "-g",
        "--gpus",
        default=torch.cuda.device_count(),
        type=int,
        help="每个节点的GPU数量",
    )

    parser.add_argument(
        "--device", type=str, default="cuda", help="PyTorch后端设备"
    )
    parser.add_argument("--dtype", type=str, default="float", help="默认数据类型")
    parser.add_argument("--log_batch_loss", default=25, help="每多少步记录一次批次损失")
    parser.add_argument(
        "--code_features",
        default=["protein", "secondary", "library"],
        help="一个热编码的附加维度",
    )
    parser.add_argument("--n_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument(
        "--recipe",
        type=list,
        default=[
            {"collection": "geom_drugs", "n_samples": 6_000_000, "filter": {}},
        ],
    )

    parser.add_argument("--n_layer_e3gnn", type=int, default=4)
    parser.add_argument("--n_hidden_e3nn", type=int, default=128)
    parser.add_argument("--msg_cutoff_e3nn", type=float, default=10.0)
    parser.add_argument("--n_hidden_xformer", type=int, default=128)
    parser.add_argument("--n_embd_common", type=int, default=128)
    parser.add_argument("--n_layer_xformer", type=int, default=16)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument(
        "--biases", type=bool, default=True, help="Transformer中是否使用偏置"
    )
    parser.add_argument("--n_seq", type=int, default=200)
    parser.add_argument("--tokenizer_vocab", type=str, default="Jan8")
    parser.add_argument("--torch_emb", type=bool, default=False)
    parser.add_argument(
        "--load_transformer_only",
        type=bool,
        default=False,
        help="仅加载已训练的Transformer部分，使用新的点编码器",
    )

    parser.add_argument("--p_dataset", type=float, default=0.3)
    parser.add_argument("--p_formula", type=float, default=0.3)
    parser.add_argument("--p_fim", type=float, default=0.5)
    parser.add_argument("--p_graph", type=float, default=0.3)
    parser.add_argument("--p_clip", type=float, default=0.3)
    parser.add_argument("--p_clip_cut", type=float, default=0.3)

    parser.add_argument("--p_clip_emb_smi", type=float, default=0.4)
    parser.add_argument("--p_randsmiles", type=float, default=0.5)

    parser.add_argument(
        "--norm_clips", type=bool, default=False, help="是否对CLIP向量进行归一化"
    )
    parser.add_argument(
        "--token_mlp",
        type=bool,
        default=False,
        help="我们是否使用MLP，或者仅使用hclip作为令牌",
    )
    parser.add_argument(
        "--norm_embed", type=bool, default=False, help="嵌入后是否进行LayerNorm"
    )
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--clip_grad", type=float, default=10.0)

    parser.add_argument(
        "--do_clip",
        type=bool,
        default=True,
        help="如果为False，则在训练期间不使用CLIP损失",
    )

    parser.add_argument(
        "--test_frac", type=float, default=0.02, help="测试数据比例"
    )
    parser.add_argument(
        "--valid_frac", type=float, default=0.02, help="验证数据比例"
    )
    parser.add_argument(
        "--test_interval",
        type=int,
        default=1,
        metavar="N",
        help="每多少个周期评估一次测试集",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        metavar="N",
        help="每多少个批次记录一次训练状态",
    )
    parser.add_argument(
        "--ngrad_to_save", default=2e6, help="在模型保存之间的梯度更新次数"
    )

    parser.add_argument(
        "--resume_document", default=None, help="从S3文档恢复模型"
    )
    parser.add_argument(
        "--resume_optimizer",
        type=bool,
        default=False,
        help="从S3文档恢复优化器状态",
    )

    args, unparsed_args = parser.parse_known_args()
    if len(unparsed_args):
        print("警告：未解析的参数：", unparsed_args)
    return args


if __name__ == "__main__":
    args = do_args()
    train_autoencoder(args)
