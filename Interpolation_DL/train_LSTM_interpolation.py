import torch
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import dataset_LSTM_interpolation as dli
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import LSTM_interpolation_model
from tqdm import tqdm
from torch import optim
import gc
from rich.progress import track
import evaluation_interpolation_model as eim


if __name__ == '__main__':

    ## init some parameters
    base_path = os.path.dirname(__file__)
    result_path = os.path.join(base_path, 'result')
    train_result_path = os.path.join(result_path, 'train')
    test_result_path = os.path.join(result_path, 'test')

    if not os.path.isdir(result_path):
        os.makedirs(result_path)

    if not os.path.isdir(train_result_path):
        os.makedirs(train_result_path)

    if not os.path.isdir(test_result_path):
        os.makedirs(test_result_path)

    data_path = os.path.join(base_path, '..', 'data', 'data_8080_2_1_25.mat')

    input_size = 4
    batch_size = 24
    epoch = 10

    if torch.cuda.is_available():
        device = 'cuda'
        print("Using cuda")
    else:
        device = 'cpu'
        print("Using CPU")
    
    ## Create data part, using dataset and dataloader
    echoDataset = dli.echoDataset(input_size, data_path)

    train_part = 0.8

    train_size = int(train_part * len(echoDataset))
    test_size = len(echoDataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(echoDataset, [train_size, test_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = LSTM_interpolation_model.varLSTM(input_size=input_size, hidden_size=64, output_size=1, corresponding_feature_size=2, dense_node_size=[16, 32, 32], num_layers=4)
    model = model.to(device)

    # print(model)

    ## determine train method and optim function
    opt = optim.Adam(model.parameters())  # stochastic gradient descent
    loss_function = nn.L1Loss()

    train_loss_list = np.zeros(len(train_loader))
    average_train_loss_list = np.zeros(epoch)

    test_loss_list = np.zeros(len(test_loader))
    average_test_loss_list = np.zeros(epoch)

    train_metrics_list = np.zeros((len(train_dataset), 4)) # [rmse, mae, dtw, mfa]
    average_train_metrics_list = np.zeros((epoch, 4))

    test_metrics_list = np.zeros((len(test_dataset), 4)) # [rmse, mae, dtw, mfa]
    average_test_metrics_list = np.zeros((epoch, 4))


    ## With experiment, we can find that the training process is very slow, so we need to find the bottleneck and we find the bottleneck here is the metrics calculation
    for iter in range(epoch):

        train_metrics_index = 0
        test_metrics_index = 0

        for i, (ts, dis, ts_ans) in track(enumerate(train_loader), description=f"train_epoch{iter} Processing...", total=len(train_loader)):
            # ts, dis, ts_ans = ts.to(device), dis.to(device), ts_ans.to(device)
            ts_ans = ts_ans.to(device)

            '''
            print(ts.shape) # shape is (batch, input_size, seq_len), the model needs (seq_len, batch, input_size)
            print(dis.shape) # shape is (batch, input_size, feature_len)
            print(ts_ans.shape) # shape is (batch, seq_len)
            pause = input('pause')
            '''

            input_ts = ts.permute(1, 2, 0) # change to (input_size, seq_len, batch)

            dis = dis.unsqueeze(-1).expand(-1, -1, -1, model.num_layers) # change to (batch, input_size, seq_len, num_layers)
            input_dis = dis.permute(1, 2, 0, 3) # change to (input_size, faeture_size, batch, num_layers)

            '''
            print(input_ts.shape)
            print(input_dis.shape)
            pause = input('pause')
            '''

            input_ts, input_dis = input_ts.to(device), input_dis.to(device)
            
            predict_ts = model(input_ts, input_dis)

            loss = loss_function(predict_ts, ts_ans)
            train_loss_list[i] = loss



            # # calculate metrics in trainset
            # for j in range(predict_ts.shape[0]):
            #     metrics = eim.evaluation_time_series_txt(ts_ans[j].cpu().detach().numpy(), predict_ts[j].cpu().detach().numpy()) # [rmse, mae, dtw, mfa]
            #     train_metrics_list[train_metrics_index] = metrics
            #     train_metrics_index += 1

            # Using parallel version of evaluation_time_series_txt
            metrics = eim.evaluation_time_series_txt(ts_ans.cpu().detach().numpy(), predict_ts.cpu().detach().numpy(), dim=predict_ts.shape[0]) # [rmse, mae, dtw, mfa]
            # print(metrics)
            train_metrics_list[train_metrics_index:train_metrics_index + predict_ts.shape[0]] = np.array(metrics).reshape(predict_ts.shape[0], 4)
            train_metrics_index += predict_ts.shape[0]

            opt.zero_grad()
            loss.backward()
            opt.step()


            if (i+1) % 100 == 0:

                image_path = os.path.join(train_result_path, 'epoch' + str(iter) + '_' + str(i) + '.png')
                eim.visualEvaluation_time_series_single(ts_ans[0].cpu().detach().numpy(), predict_ts[0].cpu().detach().numpy(), 0.95, image_path)

            if (i+1) % 1000 == 0:
                
                model_parameter_path = os.path.join(result_path, 'epoch' + str(iter) + '_' + str(i) + '.pth')
                torch.save(model.state_dict(), model_parameter_path)

        for i, (ts, dis, ts_ans) in track(enumerate(test_loader), description=f"test_epoch{iter} Processing...", total=len(test_loader)):
            ts_ans = ts_ans.to(device)

            input_ts = ts.permute(1, 2, 0) # change to (input_size, seq_len, batch)

            dis = dis.unsqueeze(-1).expand(-1, -1, -1, model.num_layers) # change to (batch, input_size, seq_len, num_layers)
            input_dis = dis.permute(1, 2, 0, 3) # change to (input_size, faeture_size, batch, num_layers)

            input_ts, input_dis = input_ts.to(device), input_dis.to(device)
            
            predict_ts = model(input_ts, input_dis)

            loss = loss_function(predict_ts, ts_ans)
            test_loss_list[i] = loss

            # # calculate metrics in testset
            # for j in range(predict_ts.shape[0]):
            #     metrics = eim.evaluation_time_series_txt(ts_ans[j].cpu().detach().numpy(), predict_ts[j].cpu().detach().numpy())
            #     test_metrics_list[test_metrics_index] = metrics
            #     test_metrics_index += 1

            # calculate metrics in testset using parallel version
            metrics = eim.evaluation_time_series_txt(ts_ans.cpu().detach().numpy(), predict_ts.cpu().detach().numpy(), dim=predict_ts.shape[0]) # [rmse, mae, dtw, mfa]
            test_metrics_list[test_metrics_index:test_metrics_index + predict_ts.shape[0]] = np.array(metrics).reshape(predict_ts.shape[0], 4)
            test_metrics_index += predict_ts.shape[0]

            for j in range(batch_size):
                image_path = os.path.join(test_result_path, 'epoch' + str(iter) + '_' + str(i) + '_' + str(j) + '.png')
                eim.visualEvaluation_time_series_single(ts_ans[j].cpu().detach().numpy(), predict_ts[j].cpu().detach().numpy(), 0.95, image_path)
        

        average_train_loss_list[iter] = np.mean(train_loss_list)
        average_test_loss_list[iter] = np.mean(test_loss_list)

        average_train_metrics_list[iter] = np.mean(train_metrics_list, axis=0)
        average_test_metrics_list[iter] = np.mean(test_metrics_list, axis=0)

        # save loss image
        fig = plt.figure
        plt.plot(train_loss_list)
        loss_image_path = os.path.join(result_path, 'train_epoch' + str(iter) + '_loss' + '.png')
        plt.savefig(loss_image_path)
        plt.close()

        fig = plt.figure
        plt.plot(test_loss_list)
        loss_image_path = os.path.join(result_path, 'test_epoch' + str(iter) + '_loss' + '.png')
        plt.savefig(loss_image_path)
        plt.close()

        # save loss .txt file
        loss_txt_path = os.path.join(result_path, 'train_epoch' + str(iter) + '_loss' + '.txt')
        np.savetxt(loss_txt_path, train_loss_list)

        # save loss .txt file
        loss_txt_path = os.path.join(result_path, 'test_epoch' + str(iter) + '_loss' + '.txt')
        np.savetxt(loss_txt_path, test_loss_list)

        # save train metrics distribution image
        rmse_image_path = os.path.join(result_path, 'train_epoch' + str(iter) + '_rmse' + '.png')
        mae_image_path = os.path.join(result_path, 'train_epoch' + str(iter) + '_mae' + '.png')
        dtw_image_path = os.path.join(result_path, 'train_epoch' + str(iter) + '_dtw' + '.png')
        mfa_image_path = os.path.join(result_path, 'train_epoch' + str(iter) + '_mfa' + '.png')

        metrics_path = [rmse_image_path, mae_image_path, dtw_image_path, mfa_image_path]
        
        for plot_index in range(len(metrics_path)):

            # plot histogram of metrics
            fig = plt.figure
            plt.hist(train_metrics_list[:, plot_index], edgecolor='black')
            
            ## add label and title

            # get value name from path
            value_name = metrics_path[plot_index].split('_')[-1].split('.')[0]

            plt.xlabel(value_name + '_Value')
            plt.ylabel('Frequency')
            plt.title(value_name + '_Histogram')

            # save image
            plt.savefig(metrics_path[plot_index])
            plt.close()

        
        # save train metrics txt file
        rmse_txt_path = os.path.join(result_path, 'train_epoch' + str(iter) + '_rmse' + '.txt')
        mae_txt_path = os.path.join(result_path, 'train_epoch' + str(iter) + '_mae' + '.txt')
        dtw_txt_path = os.path.join(result_path, 'train_epoch' + str(iter) + '_dtw' + '.txt')
        mfa_txt_path = os.path.join(result_path, 'train_epoch' + str(iter) + '_mfa' + '.txt')

        metrics_path = [rmse_txt_path, mae_txt_path, dtw_txt_path, mfa_txt_path]

        for txt_index in range(len(metrics_path)):
            np.savetxt(metrics_path[txt_index], train_metrics_list[:, txt_index])

        # as same as train, save test metrics histogram and txt file
        rmse_image_path = os.path.join(result_path, 'test_epoch' + str(iter) + '_rmse' + '.png')
        mae_image_path = os.path.join(result_path, 'test_epoch' + str(iter) + '_mae' + '.png')
        dtw_image_path = os.path.join(result_path, 'test_epoch' + str(iter) + '_dtw' + '.png')
        mfa_image_path = os.path.join(result_path, 'test_epoch' + str(iter) + '_mfa' + '.png')

        metrics_path = [rmse_image_path, mae_image_path, dtw_image_path, mfa_image_path]

        for plot_index in range(len(metrics_path)):
            fig = plt.figure
            plt.hist(test_metrics_list[:, plot_index], edgecolor='black')
            
            ## add label and title

            # get value name from path
            value_name = metrics_path[plot_index].split('_')[-1].split('.')[0]

            plt.xlabel(value_name + '_Value')
            plt.ylabel('Frequency')
            plt.title(value_name + '_Histogram')

            # save image
            plt.savefig(metrics_path[plot_index])
            plt.close()
        
        # save test metrics txt file
        rmse_txt_path = os.path.join(result_path, 'test_epoch' + str(iter) + '_rmse' + '.txt')
        mae_txt_path = os.path.join(result_path, 'test_epoch' + str(iter) + '_mae' + '.txt')
        dtw_txt_path = os.path.join(result_path, 'test_epoch' + str(iter) + '_dtw' + '.txt')
        mfa_txt_path = os.path.join(result_path, 'test_epoch' + str(iter) + '_mfa' + '.txt')

        metrics_path = [rmse_txt_path, mae_txt_path, dtw_txt_path, mfa_txt_path]

        for txt_index in range(len(metrics_path)):
            np.savetxt(metrics_path[txt_index], test_metrics_list[:, txt_index])

        
        gc.collect()

        

    # save train average loss image
    fig = plt.figure
    plt.plot(average_train_loss_list)
    loss_image_path = os.path.join(result_path, 'train_average_loss' + '.png')
    plt.savefig(loss_image_path)
    plt.close()

    # save train averge loss .txt file
    loss_txt_path = os.path.join(result_path, 'train_average_loss' + '.txt')
    np.savetxt(loss_txt_path, average_train_loss_list)

    # save test average loss image
    fig = plt.figure
    plt.plot(average_test_loss_list)
    loss_image_path = os.path.join(result_path, 'test_average_loss' + '.png')
    plt.savefig(loss_image_path)
    plt.close()

    # save test averge loss .txt file
    loss_txt_path = os.path.join(result_path, 'test_average_loss' + '.txt')
    np.savetxt(loss_txt_path, average_test_loss_list)

    # save train average metrics image
    rmse_image_path = os.path.join(result_path, 'train_average_rmse' + '.png')
    mae_image_path = os.path.join(result_path, 'train_average_mae' + '.png')
    dtw_image_path = os.path.join(result_path, 'train_average_dtw' + '.png')
    mfa_image_path = os.path.join(result_path, 'train_average_mfa' + '.png')

    metrics_path = [rmse_image_path, mae_image_path, dtw_image_path, mfa_image_path]

    for plot_index in range(len(metrics_path)):
        fig = plt.figure
        plt.plot(average_train_metrics_list[:, plot_index])
        value_name = metrics_path[plot_index].split('_')[-1].split('.')[0]

        ## add label and title
        plt.xlabel('Epoch')
        plt.ylabel(value_name + '_Value')
        plt.title(value_name + '_Epoch')

        # save image
        plt.savefig(metrics_path[plot_index])
        plt.close()

    # save test average metrics image
    rmse_image_path = os.path.join(result_path, 'test_average_rmse' + '.png')
    mae_image_path = os.path.join(result_path, 'test_average_mae' + '.png')
    dtw_image_path = os.path.join(result_path, 'test_average_dtw' + '.png')
    mfa_image_path = os.path.join(result_path, 'test_average_mfa' + '.png')

    metrics_path = [rmse_image_path, mae_image_path, dtw_image_path, mfa_image_path]

    for plot_index in range(len(metrics_path)):
        fig = plt.figure
        plt.plot(average_test_metrics_list[:, plot_index])
        value_name = metrics_path[plot_index].split('_')[-1].split('.')[0]

        ## add label and title
        plt.xlabel('Epoch')
        plt.ylabel(value_name + '_Value')
        plt.title(value_name + '_Epoch')

        # save image
        plt.savefig(metrics_path[plot_index])
        plt.close()

    # save train average metrics txt file
    rmse_txt_path = os.path.join(result_path, 'train_average_rmse' + '.txt')
    mae_txt_path = os.path.join(result_path, 'train_average_mae' + '.txt')
    dtw_txt_path = os.path.join(result_path, 'train_average_dtw' + '.txt')
    mfa_txt_path = os.path.join(result_path, 'train_average_mfa' + '.txt')

    metrics_path = [rmse_txt_path, mae_txt_path, dtw_txt_path, mfa_txt_path]

    for txt_index in range(len(metrics_path)):
        np.savetxt(metrics_path[txt_index], average_train_metrics_list[:, txt_index])

    # save test average metrics txt file
    rmse_txt_path = os.path.join(result_path, 'test_average_rmse' + '.txt')
    mae_txt_path = os.path.join(result_path, 'test_average_mae' + '.txt')
    dtw_txt_path = os.path.join(result_path, 'test_average_dtw' + '.txt')
    mfa_txt_path = os.path.join(result_path, 'test_average_mfa' + '.txt')

    metrics_path = [rmse_txt_path, mae_txt_path, dtw_txt_path, mfa_txt_path]

    for txt_index in range(len(metrics_path)):
        np.savetxt(metrics_path[txt_index], average_test_metrics_list[:, txt_index])



        

            

