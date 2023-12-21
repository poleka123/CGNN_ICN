import os
import argparse
import torch
import torch.nn as nn
from utils.data_load import Data_load
from utils.data_process import Data_Process, load_matrix
from utils.utils import *
from methods.train import Train
from methods.evaluate import Evaluate
import logger

from model.SCINet import SCINet, InterConvNet
from model.SCI_ODEGCN import ODEGCN
import numpy as np

torch.cuda.current_device()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 3)')
parser.add_argument('--levels', type=int, default=3,
                    help='# of levels (default: 3)')
parser.add_argument('--weight_file', type=str, default='./saved_weights/')
parser.add_argument('--nhid', type=int, default=32,
                    help='number of hidden units per layer (default: 32)')
parser.add_argument('--timesteps_input', type=int, default=12)
parser.add_argument('--timesteps_output', type=int, default=3)
parser.add_argument('--out_channels', type=int, default=32)
parser.add_argument('--spatial_channels', type=int, default=32)
parser.add_argument('--features', type=int, default=1)
parser.add_argument('--alpha', type=int, default=0.2)
parser.add_argument('--nheads', type=int, default=3)
# parser.add_argument('--time_slice', type=list, default=[1, 2, 3])
parser.add_argument('--time_slice', type=int, default=3)

# 生成邻接矩阵
parser.add_argument('--filename', type=str, default='smallscaleaggregation')
parser.add_argument('--sigma1', type=float, default=0.1, help='sigma for the semantic matrix')
parser.add_argument('--sigma2', type=float, default=10, help='sigma for the spatial matrix')
parser.add_argument('--thres1', type=float, default=0.6, help='the threshold for the semantic matrix')
parser.add_argument('--thres2', type=float, default=0.5, help='the threshold for the spatial matrix')

args = parser.parse_args()


if __name__ == '__main__':
    torch.manual_seed(7)
    elogger = logger.Logger('run_log_sci_gcn_timesteps3')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    NATree, data_set = Data_load(args.timesteps_input, args.timesteps_output)
    # data_set, all_a, all_p, all_f, all_mean, all_Kmask = Data_Process("./data_set/SmallScaleAggregation/V_flow_50.csv",
    #                                                                   args.timesteps_input, args.timesteps_output,
    #                                                                   24 * 30 * 2, 128, 10)
    W_nodes = load_matrix('data_set/SmallScaleAggregation/distance.csv')
    sp_matrix = load_matrix('./data_set/SmallScaleAggregation/adjmat_50.csv')
    dtw_matrix = generate_adjmatrix(args)
    dtw_matrix = dtw_matrix.astype('float32')

    Ks = 3 #多项式近似个数
    NATree = np.ones((50, 128))
    Num_of_nodes = NATree.shape[0]
    # MaxNodeNumber = NATree.shape[2]
    # MaxLayerNumber = NATree.shape[1]
    input_channels = 1
    channel_sizes = [args.nhid] * args.levels
    kernel_size = args.ksize

    # 传感器节点编号
    ids = torch.from_numpy(np.arange(0, Num_of_nodes)).to(device)
    # all_Kmask = torch.tensor(all_Kmask, dtype=torch.float32).to(device)
    data_set1 = data_set
    # data_set1['all_Kmask'] = all_Kmask
    data_set1['ids'] = ids

    # 归一化邻接矩阵
    A_sp_wave = torch.from_numpy(get_normalized_adj(sp_matrix)).to(device)
    A_se_wave = torch.from_numpy(get_normalized_adj(dtw_matrix)).to(device)

    model = ODEGCN(
        num_nodes=Num_of_nodes,
        num_features=input_channels,
        num_timesteps_input=args.timesteps_input,
        num_timesteps_output=args.timesteps_output,
        A_sp_hat=A_sp_wave,
        A_se_hat=A_se_wave
    )


    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    L2 = nn.MSELoss()
    for epoch in range(args.epochs):
        print("Train Process")
        permutation = torch.randperm(data_set['train_input'].shape[0])
        epoch_training_losses = []
        loss_mean = 0.0
        # train
        for i in range(0, data_set['train_input'].shape[0], args.batch_size):

            '''importance'''
            model.train()
            optimizer.zero_grad()

            indices = permutation[i:i+args.batch_size]
            X_batch, y_batch = data_set['train_input'][indices], data_set['train_target'][indices]

            if torch.cuda.is_available():
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                std = torch.tensor(data_set['data_std']).to(device)
                mean = torch.tensor(data_set['data_mean']).to(device)
            else:
                std = torch.tensor(data_set['data_std'])
                mean = torch.tensor(data_set['data_mean'])
            pred = model(X_batch)
            pred, y_batch = Un_Z_Score(pred, mean, std), Un_Z_Score(y_batch, mean, std)
            loss = L2(pred, y_batch)

            # 损失反向传播
            loss.backward()
            optimizer.step()
            epoch_training_losses.append(loss.detach().cpu().numpy())
        loss_mean = sum(epoch_training_losses)/len(epoch_training_losses)
        # if i % 50 == 0:
        #     print("Loss Mean "+str(loss_mean))

        # test
        torch.cuda.empty_cache()
        with torch.no_grad():
            print("Evalution Process")
            model.eval()
            eval_input = data_set['eval_input']
            eval_target = data_set['eval_target']
            if torch.cuda.is_available():
                eval_input = eval_input.to(device)
                eval_target = eval_target.to(device)
                std = torch.tensor(data_set['data_std']).to(device)
                mean = torch.tensor(data_set['data_mean']).to(device)
            else:
                std = torch.tensor(data_set['data_std'])
                mean = torch.tensor(data_set['data_mean'])

            pred = model(eval_input)
            val_index = {}
            val_index['MAE'] = []
            val_index['RMSE'] = []
            val_index['sMAPE'] = []
            val_loss = []

            for item in range(1, args.time_slice+1):
                pred_index = pred[:, :, item - 1]
                val_target_index = eval_target[:, :, item - 1]
                pred_index, val_target_index = Un_Z_Score(pred_index, mean, std), Un_Z_Score(val_target_index, mean,
                                                                                             std)

                loss = L2(pred_index, val_target_index)
                val_loss.append(loss)

                filePath = "./results/sci_gcn/"
                if not os.path.exists(filePath):
                    os.makedirs(filePath)
                if ((epoch + 1) % 50 == 0) & (epoch != 0) & (epoch > 200):
                    np.savetxt(filePath + "/pred_" + str(epoch) + ".csv", pred_index.cpu(), delimiter=',')
                    np.savetxt(filePath + "/true_" + str(epoch) + ".csv", val_target_index.cpu(), delimiter=',')
                mae = MAE(val_target_index, pred_index)
                val_index['MAE'].append(mae)

                rmse = RMSE(val_target_index, pred_index)
                val_index['RMSE'].append(rmse)

                smape = SMAPE(val_target_index, pred_index)
                val_index['sMAPE'].append(smape)

        print("---------------------------------------------------------------------------------------------------")
        print("epoch: {}/{}".format(epoch, args.epochs))
        print("Training loss: {}".format(loss_mean))
        elogger.log("Epoch:{}".format(epoch))
        for i in range(1, args.time_slice+1):
            print("time:{}, Evaluation loss:{}, MAE:{}, RMSE:{}, sMAPE:{}"
                  .format(i * 5, val_loss[-((args.time_slice) - i)],
                          val_index['MAE'][-((args.time_slice) - i)],
                          val_index['RMSE'][-((args.time_slice) - i)],
                          val_index['sMAPE'][-((args.time_slice) - i)]))
            elogger.log("time:{}, Evaluation loss:{}, MAE:{}, RMSE:{}, sMAPE:{}"
                        .format(i * 5, val_loss[-((args.time_slice) - i)],
                                val_index['MAE'][-((args.time_slice) - i)],
                                val_index['RMSE'][-((args.time_slice) - i)],
                                val_index['sMAPE'][-((args.time_slice) - i)]))
        elogger.log("-----------")
        print("---------------------------------------------------------------------------------------------------")
