import time
import os
import argparse
import torch
import json
import warnings
from collections import OrderedDict
from torch import nn
from itertools import chain
from data_process_egnn import load_data, process_data, get_drug_molecule_graph, get_target_molecule_graph
from utils_egnn import GraphDataset, collate, model_evaluate
from model_egnn_new import MLC_DTA, PredictModule


def train(model, predictor, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, lr, epoch,
          batch_size):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    predictor.train()
    LOG_INTERVAL = 10
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, chain(model.parameters(), predictor.parameters())), lr=lr, weight_decay=0)
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        # 彻底切除无用参数，只传入图 batch
        drug_embedding, target_embedding = model(drug_graph_batchs, target_graph_batchs)
        output, _ = predictor(data.to(device), drug_embedding, target_embedding)

        # 纯净的 MSE Loss，再无 loss_cl
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item()))


def test(model, predictor, device, loader, drug_graphs_DataLoader, target_graphs_DataLoader):
    model.eval()
    predictor.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    drug_graph_batchs = list(map(lambda graph: graph.to(device), drug_graphs_DataLoader))
    target_graph_batchs = list(map(lambda graph: graph.to(device), target_graphs_DataLoader))
    with torch.no_grad():
        for data in loader:
            drug_embedding, target_embedding = model(drug_graph_batchs, target_graph_batchs)
            output, _ = predictor(data.to(device), drug_embedding, target_embedding)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def train_predict():
    start_time = time.time()
    data_path = './source/data/'

    # ==================== 【保留日志保存到 SAVE 文件夹】 ====================
    import sys
    from datetime import datetime
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVE_DIR = os.path.join(ROOT_DIR, "SAVE")
    os.makedirs(SAVE_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(SAVE_DIR, f"train_log_{timestamp}.txt")

    class Logger:
        def __init__(self, file_path):
            self.terminal = sys.stdout
            self.log = open(file_path, 'w', encoding='utf-8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(log_file)
    print(f"📂 训练日志已保存到：{log_file}\n")
    # ==========================================================================

    print("Data preparation in progress for the {} dataset...".format(args.dataset))
    affinity_mat = load_data(args.dataset)

    # 【修复重点】：严格对齐 data_process_egnn.py 真实的返回参数 (只有 2 个)
    train_data, test_data = process_data(affinity_mat, args.dataset, scenario=args.scenario)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    print(f"Time taken for data preparation: {time.time() - start_time:.2f} seconds")

    # 手动加载图数据字典
    drug_graphs_dict = get_drug_molecule_graph(
        json.load(open(data_path + f'{args.dataset}/drugs.txt'), object_pairs_hook=OrderedDict), dataset=args.dataset)

    drug_graphs_Data = GraphDataset(graphs_dict=drug_graphs_dict, dttype="drug")
    # 替换原本依赖于 affinity_graph 的 batch_size
    drug_graphs_DataLoader = torch.utils.data.DataLoader(drug_graphs_Data, shuffle=False, collate_fn=collate,
                                                         batch_size=len(drug_graphs_dict))

    print(f"Time taken for loading drug molecule graphs: {time.time() - start_time:.2f} seconds")

    target_graphs_dict = get_target_molecule_graph(
        json.load(open(data_path + f'{args.dataset}/targets.txt'), object_pairs_hook=OrderedDict), args.dataset)

    target_graphs_Data = GraphDataset(graphs_dict=target_graphs_dict, dttype="target")
    target_graphs_DataLoader = torch.utils.data.DataLoader(target_graphs_Data, shuffle=False, collate_fn=collate,
                                                           batch_size=len(target_graphs_dict))

    print(f"Time taken for loading target molecule graphs: {time.time() - start_time:.2f} seconds")

    print("Model preparation... ")
    device = torch.device('cuda:{}'.format(args.cuda) if torch.cuda.is_available() else 'cpu')
    dataset = args.dataset

    # 【修复重点】：极简模型实例化
    model = MLC_DTA(d_ms_dims=[78, 78, 78 * 2, 128],
                    t_ms_dims=[54, 54, 54 * 2, 128],
                    embedding_dim=128)

    predictor = PredictModule()
    model.to(device)
    predictor.to(device)

    best_result = [float('inf')]
    best_epoch = -1

    # ==================== 【保留文件名加时间戳】 ====================
    model_name = f'model_wo_Constrative_{args.dataset}_{args.scenario}_{timestamp}'
    predictor_name = f'predictor_wo_Constrative_{args.dataset}_{args.scenario}_{timestamp}'

    print("Start training...")

    patience = 100
    stop_counter = 0

    start_time = time.time()
    trained_path = './new_train/'

    for epoch in range(args.epochs):
        train(model, predictor, device, train_loader, drug_graphs_DataLoader, target_graphs_DataLoader, args.lr,
              epoch + 1, args.batch_size)

        G, P = test(model, predictor, device, test_loader, drug_graphs_DataLoader, target_graphs_DataLoader)
        r = model_evaluate(G, P, dataset=args.dataset)

        if r[0] < best_result[0]:
            best_result = r
            best_epoch = epoch + 1
            stop_counter = 0

            print('mse improved at epoch ', best_epoch, '; best_test_mse', r[0])
            print('best_CI', r[1], '; best_rm2', r[2], '; best_pearson', r[3], '; best_aupr', r[4])
            end_time = time.time()
            print('interval time:', end_time - start_time)

            checkpoint_dir = trained_path + f"pre_trained models/{dataset}/"
            if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
            torch.save(model.state_dict(), checkpoint_dir + model_name + ".pkl", _use_new_zipfile_serialization=False)
            torch.save(predictor.state_dict(), checkpoint_dir + predictor_name + ".pkl",
                       _use_new_zipfile_serialization=False)

        else:
            stop_counter += 1
            print(f'No improvement for {stop_counter} epochs. (Best MSE so far: {best_result[0]:.6f})')

        if stop_counter >= patience:
            print(f'\n[Early Stopping] 已连续 {patience} 轮没有进步，系统判定模型已收敛。')
            print(f'最终最佳 Epoch: {best_epoch}, 最佳 MSE: {best_result[0]}')
            break

    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='davis')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--edge_dropout_rate', type=float, default=0.0)
    parser.add_argument('--scenario', type=str, default='S3', choices=['warm', 'S1', 'S2', 'S3'], help='选择测试场景')
    args, _ = parser.parse_known_args()

    train_predict()