from cgi import test
from tracemalloc import start
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from utils import load_evaporation_data, load_humid_data, load_rain_data
import random
import os
import pandas as pd

# Define LSTM Neural Networks
class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers) # utilize the LSTM model in torch.nn 
        self.forwardCalculation = nn.Linear(hidden_size, output_size)
 
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)

        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s*b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1)
        return x

class PreTransformer(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        self.head_conv = nn.Linear(input_size, hidden_size)
        self.transformer = nn.TransformerEncoderLayer(hidden_size, 4, 2*hidden_size, 0.1)
        self.forwardCalculation = nn.Linear(hidden_size, output_size)
 
    def forward(self, _x):

        s, b, h = _x.shape  # x is input, size (seq_len, batch, hidden_size)
        _x = _x.view(s*b, h)
        x = self.head_conv(_x)
        x = x.view(s, b, -1)
        x = self.transformer(x)

        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s*b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1)
        return x

def train_single_model(data, seq_len, train_ratio, batch_size, lr, max_epochs, onehot, model_type='lstm', model_name='rain', model_save_path=''):
    print('train {} model, model type: {}'.format(model_name, model_type))

    len_datas = data.shape[0] - seq_len + 1
    datas = []
    for i in range(len_datas):
        datas.append(data[i:i+seq_len])
    datas = torch.from_numpy(np.stack(datas, axis=1)).cuda().float()
    if model_name != 'humid':
        if not onehot:
                datas_x, datas_y = datas[:, :-1, :], datas[:, 1:, :-1]
        else:
            datas_x, datas_y = datas[:, :-1, :], datas[:, 1:, :-12]
    else:
        datas_x, datas_y = datas[:, :-1, :], datas[:, 1:, :4]
    
    split_channel = datas_x.shape[-1]

    all_data = torch.cat([datas_x, datas_y], dim=-1)
    all_data = [all_data[:, i] for i in range(all_data.shape[1])]
    train_num = int(train_ratio * len(all_data))
    random.shuffle(all_data)
    in_features, out_features = split_channel, all_data[0].shape[-1]-split_channel
    if model_type == 'lstm':
        model = LstmRNN(in_features, 16, output_size=out_features, num_layers=1)
    else:
        model = PreTransformer(in_features, 16, output_size=out_features, num_layers=1)
    model.cuda()
 
    loss_function = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, max_epochs//3, 0.5)
    loss_list, mae_list = [], []
    mae_min = 100000
    for epoch in range(max_epochs):
        random.shuffle(all_data)
        train_data, test_data = all_data[:train_num], all_data[train_num:]
        model.train()
        for i in range(0, (len(train_data)//batch_size)*batch_size, batch_size):
            train_data_x = torch.stack(train_data[i:i+batch_size], dim=1)[:, :, :split_channel]
            train_data_y = torch.stack(train_data[i:i+batch_size], dim=1)[:, :, split_channel:]
            
            output = model(train_data_x)
            loss = loss_function(output, train_data_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            if loss.item() < 1e-4:
                print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch+1, max_epochs, loss.item()))
                print("The loss value is reached")
                break
        loss_list.append(loss.item())
        if (epoch+1) % 100 == 0:
            print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch+1, max_epochs, loss.item()))
        scheduler.step()
        random.shuffle(train_data)

        ## eval
        model = model.eval() # switch to testing model
        test_data_x = torch.stack(test_data, dim=1)[:, :, :split_channel]
        test_data_y = torch.stack(test_data, dim=1)[:, :, split_channel:]
        predictive_y_for_testing = model(test_data_x)
        mae = torch.mean(torch.abs(predictive_y_for_testing - test_data_y))
        if mae < mae_min:
            torch.save(model.state_dict(), os.path.join(model_save_path, '{}_{}.pth'.format(model_name, model_type)))
        mae_list.append(mae.item())
        if (epoch+1) % 100 == 0:
            print('eval result: mae = {}'.format(mae))
    return loss_list, mae_list

def eval_single_model(data, seq_len, model_type='lstm', model_name='rain', model_path='./models', additional_data=None, plot_label_index={}):
    print('evaluating {} model, model type: {}'.format(model_name, model_type))
    all_data0 = data
    all_data = torch.from_numpy(data).cuda()[:, None, :]
    final_month = data[-1, -1] if not onehot else np.argmax(data[-1, -12:]) + 1
    months_2022 = np.arange(final_month+1, 13)
    months_2023 = np.arange(1, 13)
    months_pre = np.concatenate([months_2022, months_2023])
    if onehot:
        months_pre_list = []
        for month in months_pre:
            month_oh = np.zeros(12)
            month_oh[month-1] = 1
            months_pre_list.append(month_oh)
        months_pre = np.stack(months_pre_list)
    else:
        months_pre = months_pre[:, np.newaxis]
    
    months_pre = torch.from_numpy(months_pre).cuda()
    if additional_data is not None:
        # print(additional_data[-months_pre.shape[0]:].shape, months_pre.shape)
        months_pre = torch.cat([additional_data[-months_pre.shape[0]:], months_pre], dim=-1)
    in_channel, out_channel = all_data.shape[-1], all_data.shape[-1]-1 if not onehot else all_data.shape[-1]-months_pre.shape[-1]
    
    if model_type == 'lstm':
        model = LstmRNN(in_channel, 16, out_channel)
    else:
        model = PreTransformer(in_channel, 16, out_channel)
    model.cuda()
    model.load_state_dict(torch.load(os.path.join(model_path, '{}_{}.pth'.format(model_name, model_type))))
    model.eval()

    with torch.no_grad():
        for k in range(months_pre.shape[0]):
            data0 = all_data[-seq_len:, :]
            pre = model(data0.float())
            pre_last = pre[-1, :, :][None, :, :]

            pre_last = torch.cat([pre_last, months_pre[k][None, None, :]], dim=-1)
            all_data = torch.cat([all_data, pre_last], dim=0)
    out_pred = all_data[:, 0, :out_channel].cpu().numpy()
    all_data0 = all_data0[:, :out_channel]
    fig_save_path = os.path.join('./results', model_type)
    if not os.path.exists(fig_save_path):
        os.makedirs(fig_save_path)
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.2,hspace=0.5)
    for i in range(out_channel):
        ax = fig.add_subplot(out_channel,1,i+1)
        ax.set_title(plot_label_index['name'][i], fontsize=8)
        ax.plot(out_pred[:, i], label='prediction')
        ax.plot(all_data0[:, i], label='origin data')
        ax.legend()
        if i == out_channel - 1:
            x_ticks, x_dates = get_month_list(out_pred[:, i])
            ax.tick_params(labelsize=6)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_dates, rotation=45)
            ax.set_xlabel('month', fontsize=8)
        else:
            ax.xaxis.set_visible(False)
        ax.set_ylabel(plot_label_index['name'][i] + '/' + plot_label_index['unit'][i], fontsize=8)
    plt.savefig(os.path.join(fig_save_path, '{}.png'.format(model_name)), dpi=600)   
    return all_data[:, 0, :]

def train(
    rain_data, 
    evaporation_data, 
    humid_data,
    seq_len, 
    batch_size, 
    lr, 
    max_epochs, 
    train_ratio, 
    model_types, 
    onehot=True, 
    is_train=True,
    model_save_path='./models',
    plot_label_indices={}):

    model_save_path = os.path.join(model_save_path, model_types['humid'])
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    loss_list_rain, mae_list_rain, loss_list_evaporation, mae_list_evaporation, loss_list_humid, mae_list_humid = \
        None, None, None, None, None, None
    
    # 1. train rain model
    if is_train or not os.path.exists(os.path.join(model_save_path, '{}_{}.pth'.format('rain', model_types['rain']))):
        loss_list_rain, mae_list_rain = train_single_model(
            rain_data, seq_len, train_ratio, batch_size, lr, max_epochs, onehot, model_types['rain'], 'rain', model_save_path
        )
    predicted_rain_data = eval_single_model(rain_data, seq_len, model_types['rain'], 'rain', model_save_path, plot_label_index=plot_label_indices['rain'])

    # 2. train evaporation model
    if is_train or not os.path.exists(os.path.join(model_save_path, '{}_{}.pth'.format('evaporation', model_types['evaporation']))):
        loss_list_evaporation, mae_list_evaporation = train_single_model(
            evaporation_data, seq_len, train_ratio, batch_size, lr, max_epochs, onehot, model_types['evaporation'], 'evaporation', model_save_path
        )
    predicted_evaporation_data = eval_single_model(evaporation_data, seq_len, model_types['evaporation'], 'evaporation', model_save_path, plot_label_index=plot_label_indices['evaporation'])

    # 3. train humid model
    rain_data = rain_data[:len(humid_data), :]
    if onehot:
        all_data = np.concatenate([humid_data[:, :-12], rain_data[:, :-12], evaporation_data], axis=-1)
    else:
        all_data = np.concatenate([humid_data[:, :-1], rain_data[:, :-1], evaporation_data], axis=-1)
    if is_train or not os.path.exists(os.path.join(model_save_path, '{}_{}.pth'.format('humid', model_types['humid']))):
        loss_list_humid, mae_list_humid = train_single_model(
                all_data, seq_len, train_ratio, batch_size, lr, max_epochs, onehot, model_types['humid'], 'humid', model_save_path
            )
    if onehot:
        additional_data = torch.cat([predicted_rain_data[:, :-12], predicted_evaporation_data[:, :-12]], axis=-1)
    else:
        additional_data = torch.cat([predicted_rain_data[:, :-1], predicted_evaporation_data[:, :-1]], axis=-1)
    predicted_humid_data = eval_single_model(all_data, seq_len, model_types['humid'], 'humid', model_save_path, additional_data, plot_label_index=plot_label_indices['humid'])
    predicted_humid_data = predicted_humid_data[len(humid_data):, :4].cpu().numpy() 
    return loss_list_rain, mae_list_rain, loss_list_evaporation, mae_list_evaporation, loss_list_humid, mae_list_humid, predicted_humid_data

def plot_data_pairs(data_pairs, names1, names2, sup_name, fig_save_path, plot_label_indices):
    fig = plt.figure()
    plt.suptitle(sup_name)
    fig.subplots_adjust(wspace=0.2,hspace=0.3)
    for i in range(len(data_pairs)):
        ax = fig.add_subplot(len(data_pairs),1,i+1)
        ax.set_title(plot_label_indices[names2[i]]['sup_name'], fontsize=8)
        ax.plot(data_pairs[i][0], label=names1[0])
        ax.plot(data_pairs[i][1], label=names1[1])
        ax.legend()
        
        if i == len(data_pairs) - 1:
            x_ticks, x_dates = get_month_list(data_pairs[i][0])
            ax.tick_params(labelsize=6)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_dates, rotation=45)
            ax.set_xlabel('month', fontsize=8)
        else:
            ax.xaxis.set_visible(False)
        ax.set_ylabel(sup_name, fontsize=8)
    plt.savefig(os.path.join(fig_save_path, '{}.png'.format(sup_name)), dpi=600)

def get_month_list(data, init_year=2012, init_month=1, stride=4):
    
    first_year_months = [i for i in range(init_month, 13)]
    out_dates = ['{}/{}'.format(init_year, month) for month in first_year_months]
    current_year = init_year

    all_year_months = [i for i in range(1, 13)]
    while len(out_dates) < len(data):
        current_year += 1
        for month_c in all_year_months:
            out_dates.append('{}/{}'.format(current_year, month_c))
    out_dates = out_dates[:len(data)]
    out_ticks = [i for i in range(len(out_dates))]
    out_dates = [out_dates[i] for i in range(len(out_dates)) if i % stride == 0]
    out_ticks = [i for i in range(len(out_ticks)) if i % stride == 0]
    return out_ticks, out_dates



if __name__ == '__main__':
    ## create database
    is_train = False
    onehot = True
    mode = 'humid+rain+evaporation'
    batch_size = 16
    seq_len = 12
    max_epochs = 10000
    train_ratio = 0.8
    lr=1e-2
    root_humid = ""
    root_evaporation = ""
    root_rain = ""
    evaporation_data = load_evaporation_data(root_evaporation, onehot)
    humid_data = load_humid_data(root_humid, onehot)
    rain_data = load_rain_data(root_rain, onehot)

    model_types1 = {
        'rain': 'trans',
        'evaporation': 'trans',
        'humid': 'trans'
    }

    model_types2 = {
        'rain': 'lstm',
        'evaporation': 'lstm',
        'humid': 'lstm'
    }

    plot_label_indices = {
        'rain': {
            'sup_name': 'rainfall',
            'unit': ['mm'],
            'name': ['rainfall']
        },
        'evaporation': {
            'sup_name': 'evaporation',
            'unit': ['W/m2', 'mm'],
            'name': ['evaporation', 'evaporation']
        },
        'humid': {
            'sup_name': 'moisture',
            'unit': ['kg/m2', 'kg/m2', 'kg/m2', 'kg/m2'],
            'name': ['10cm', '40cm', '100cm', '200cm']
        },
    }

    loss_list_rain2, mae_list_rain2, loss_list_evaporation2, mae_list_evaporation2, loss_list_humid2, mae_list_humid2, predicted_humid_data2 = \
        train(rain_data, evaporation_data, humid_data, seq_len, batch_size, lr, max_epochs, train_ratio, model_types2, onehot, is_train, plot_label_indices=plot_label_indices)

    loss_list_rain1, mae_list_rain1, loss_list_evaporation1, mae_list_evaporation1, loss_list_humid1, mae_list_humid1, predicted_humid_data1 = \
        train(rain_data, evaporation_data, humid_data, seq_len, batch_size, lr, max_epochs, train_ratio, model_types1, onehot, is_train, plot_label_indices=plot_label_indices)

    
    df = pd.DataFrame(predicted_humid_data1)
    df.to_csv('PBMC_pre.csv',index= False, header= False)
    df1 = pd.read_csv('PBMC_pre.csv',header=None)

    if loss_list_rain1 is not None:
        data_pairs1 = [[loss_list_rain1, loss_list_rain2], [loss_list_evaporation1, loss_list_evaporation2], [loss_list_humid1, loss_list_humid2]]
        data_pairs2 = [[mae_list_rain1, mae_list_rain2], [mae_list_evaporation1, mae_list_evaporation2], [mae_list_humid1, mae_list_humid2]]
        names1 = ['transformer', 'lstm']
        names2 = ['rain', 'evaporation', 'humid']

        plot_data_pairs(data_pairs1, names1, names2, 'Loss', 'results', plot_label_indices)
        plot_data_pairs(data_pairs2, names1, names2, 'MAE', 'results', plot_label_indices)

    
