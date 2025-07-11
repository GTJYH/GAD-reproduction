
import time
import numpy as np
import pandas
import os
import sys
import warnings
warnings.filterwarnings("ignore")


from utils import *
from pretrain_models import *
from e2e_models import *
from edcoders import *

seed_list = list(range(3407, 10000, 10))

def pretrain(model, train_dataloader, args):
    print("start pre-training..")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_rate)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)

    max_epoch = args.epoch_pretrain
    epoch_iter = tqdm(range(max_epoch))
    device = args.device

    for epoch in epoch_iter:
        model.train()
        loss_list = []

        for batched_data in train_dataloader:
            # we don't need labels for pretrain task
            graph = batched_data 
            graph = graph.to(device)
            
            # 预训练时只使用单个图，避免GraphMAE掩码操作的问题
            loss, loss_dict = model(graph, graph.ndata["feature"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        train_loss = np.mean(loss_list)
        loss_dict["lr"] = get_current_lr(optimizer)
        
        print(f"# Epoch {epoch} \n\t train_loss: {train_loss:.4f} \n\t {loss_dict}")

    return model


def work(dataset: Dataset, kernel='gcn', cross_mode='ng2ng', args=None):
    if args is None:
        args = get_args()
    time_cost = 0
    pretrain_model_name = args.pretrain_model
    if '-' in kernel:
        encoder_type, decoder_type = kernel.split('-')
    else:
        encoder_type = kernel
        decoder_type=kernel
        if encoder_type == 'bwgnn':
            decoder_type="gcn"
    encoder_name = kernel
    full_model_name = pretrain_model_name + '-' + encoder_name
    hop = args.khop
    dataset_name = dataset.name.replace('/', '.')
    print(f"preparing the dataset for {args.trials} trials")
    dataset.prepare_dataset(total_trials=args.trials)
    print(f"making the subpooling matrix")
    dataset.make_sp_matrix_graph_list(khop=hop, load_kg=(not args.force_remake_sp))

    ####################### pretrain model start
    pretrain_seed = 42
    set_seed(pretrain_seed)
    # clear gpu mem
    torch.cuda.empty_cache()

    # build pretrain model
    in_dim = dataset.in_dim
    if args.pretrain_model == 'graphmae':
        pretrain_model = GraphMAE(
            in_dim=in_dim, 
            hid_dim=args.hid_dim, 
            num_layer=args.num_layer_pretrain,  
            drop_ratio=args.dropout,
            act=args.act, 
            norm=args.norm, 
            residual=args.residual,
            mask_ratio=args.mask_ratio,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
            replace_ratio=args.replace_ratio,
        ).to(args.device)
    else:
        raise NotImplementedError # FIXME: only graphme now
    
    full_model_name += '-' + decoder_type
    model_path = f"../pretrained_models/{full_model_name}_{dataset_name}_{args.epoch_pretrain}.pt"
    if args.load_model == "":
        # pretrain model from scratch
        print(pretrain_model)
        pretrain_dataloader = dataset.get_pretrain_dataloaders(args.batch_size)
        pretrain_model = pretrain(pretrain_model, pretrain_dataloader, args)
        print(f"model saved to {model_path}")
        if not os.path.exists('../pretrained_models/'):
            os.mkdir('../pretrained_models/')
        torch.save(pretrain_model.state_dict(), model_path)
        del pretrain_dataloader
    else:
        pretrain_model.load_state_dict(torch.load(model_path))
    # clear gpu mem
    torch.cuda.empty_cache()
    #######################  pretrina model end

    # multiple trials
    
    # result list: AUC, PRC, Macro-F1
    result_score_dict_list = []
    for t in range(args.trials):
        print("Dataset {}, Model {}, Trial {}".format(dataset_name, full_model_name, t))
        # reload the state_dict
        pretrain_model.load_state_dict(torch.load(model_path))
        # set seed
        seed = seed_list[t]
        set_seed(seed)
        train_dataloader, val_dataloader, test_dataloader = dataset.get_graph_and_sp_dataloaders(args.batch_size, trial_id=t)
        e2e_model = UnifyMLPDetector(pretrain_model, dataset, (train_dataloader, val_dataloader, test_dataloader), cross_mode=cross_mode, args=args)
        ST = time.time()
        print(f"training...")
        score_test = e2e_model.train()
        result_score_dict_list.append(score_test)

        ED = time.time()
        time_cost += ED - ST
    
    
    model_result = {'dataset name': dataset_name,
                    'model_name': full_model_name,
                    'cross mode': cross_mode,
                    'time cost': time_cost/args.trials}

    # calculate the results across trials
    for k in e2e_model.output_route:
        for metric in ['MacroF1', 'AUROC', 'AUPRC']:
            metric_result_list = [d[k][metric] for d in result_score_dict_list]
            model_result[f'{metric} {NAME_MAP[k]} mean'] = np.mean(metric_result_list)
            model_result[f'{metric} {NAME_MAP[k]} std'] = np.std(metric_result_list)
            print(metric)
            print(metric_result_list)
            print("avg: ", sum(metric_result_list)/len(metric_result_list))

    # save the result to 
    model_result = pandas.DataFrame(model_result, index=[0])
    return model_result



def main():
    args = get_args()
    # parse data
    if args.kernels is not None:
        kernels = args.kernels.split(',')
        print('All kernels: ', kernels)

    if args.datasets is not None:
        if '-' in args.datasets:
            st, ed = args.datasets.split('-')
            dataset_names = DATASETS[int(st):int(ed)+1]
        else:
            dataset_names = [DATASETS[int(t)] for t in args.datasets.split(',')]
        print('All Datasets: ', dataset_names)

    if args.cross_modes is not None:
        cross_modes = args.cross_modes.split(',')
        print('All Cross_modes: ', cross_modes)

    if args.khop == 1:
        sp_type = "star+norm"
    elif args.khop == 2:
        sp_type = "convtree+norm"
    else:
        raise NotImplementedError

    # evaluate all parameters
    for dataset_name in dataset_names:
        # parse dataset
        # dataset_name = DATASETS[dataset_id]
        print('Evaluating dataset: ', dataset_name)

        ### settings
        # load dataset 
        # 使用统一的数据集加载方法，与GADAM和ARC保持一致
        dataset = Dataset(dataset_name, prefix='../../datasets/', sp_type=sp_type, labels_have='ne')
        # metrics to save
        columns = ['name']
        results = pandas.DataFrame(columns=columns)
        
        for kernel in kernels:
            # parse pre-train model and encoder name
            print('Evaluating model: ', args.pretrain_model + kernel)
            # iterate over all cross modes
            for cross_mode in cross_modes:
                # work
                model_result = work(dataset, kernel, cross_mode, args)
                results = pandas.concat([results, model_result])

        # 修改：为每个数据集只保存一个Excel文件，包含所有模型和交叉模式的结果
        save_file_name = f"{args.tag}.{args.act_ft}.dataset_{dataset_name}.preepochs_{args.epoch_pretrain}.hop_{args.khop}.sp_type_{sp_type}.lr_ft_{args.lr_ft}.epochft_{args.epoch_ft}.wd_{args.l2}.mlplayers_{args.stitch_mlp_layers}_{args.final_mlp_layers}.lossweights_{str(args.node_loss_weight)+'-'+str(args.edge_loss_weight)+'-'+str(args.graph_loss_weight)}"
        save_results(results, save_file_name)
        print(results)

if __name__ == "__main__":
    main()