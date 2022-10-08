import pandas as pd
from config import Config
import numpy as np
import seaborn as sns
import matplotlib
print(matplotlib.matplotlib_fname())
print(matplotlib.get_cachedir())
from matplotlib import pyplot as plt
from prophet import Prophet
from autots import AutoTS
from sklearn.svm import SVR
import os
from sklearn.metrics import mean_absolute_error, \
    mean_squared_error, mean_squared_log_error, \
        mean_absolute_percentage_error, median_absolute_error,\
            explained_variance_score, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import StackingRegressor

from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append([*train_seq ,*train_label])
    return inout_seq

def create_inout_sequences_valid(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw+1):
        train_seq = input_data[i:i+tw]
        inout_seq.append(train_seq)
    return inout_seq

if __name__ == "__main__":
    if Config.isAttachment14:
        columns = ["year", "放牧小区（plot）", "放牧强度（intensity）", "SOC土壤有机碳", "SIC土壤无机碳", "STC土壤全碳", "全氮N",	"土壤C/N比"]
        data14 = pd.read_excel(Config.attachment14, usecols=columns)
        
        data14.sort_values(by='year', inplace=True)
        # print(data14.head())

    if Config.by_intensity:
        # 
        # data_by_intensity = data14.groupby(by='放牧强度（intensity）')
        
        for name in ["SOC土壤有机碳", "SIC土壤无机碳", "STC土壤全碳", "全氮N",	"土壤C/N比"]:
            print(matplotlib.matplotlib_fname())
            # sns.relplot(data=data14, kind="line", x="year", y=name, col="放牧小区（plot）", hue="放牧强度（intensity）")
            sns.relplot(data=data14, kind="line", x="year", y=name, hue="放牧小区（plot）", col="放牧强度（intensity）", col_wrap=2, ci=None)
            name = "_".join(name.split('/'))
            plt.savefig(Config.out_dir+"/"+name+".png", dpi=100)

    # if Config.by_plot:
    #     # data14.sort_values(by='year', inplace=True)
    #     # data_by_intensity = data14.groupby(by='放牧强度（intensity）')
        
    #     for name in ["SOC土壤有机碳", "SIC土壤无机碳", "STC土壤全碳", "全氮N",	"土壤C/N比"]:
    #         print(matplotlib.matplotlib_fname())
    #         sns.relplot(data=data14, kind="line", x="year", y=name)
    #         name = "_".join(name.split('/'))
    #         plt.savefig(name+".png")

        # g = sns.FacetGrid(data14[["year", "放牧小区（plot）", "SOC土壤有机碳"]], col="放牧小区（plot）", height=3, col_wrap=3)
        # # g.map(sns.lineplot, "year")
        # g.map_dataframe(sns.lineplot, x="year", y="SOC土壤有机碳")
        # plt.savefig("byplot.png", dpi=100) # , "SIC土壤无机碳", "STC土壤全碳", "全氮N","土壤C/N比"
    
    if Config.forecast_prophet:
        for key, df in data14.groupby(by="放牧小区（plot）"):
            print(key) #, "SIC土壤无机碳": "y2", "STC土壤全碳": "y3", "全氮N": "y4", "土壤C/N比": "y5"
            df_dict = {}
            for name in ["SOC土壤有机碳", "SIC土壤无机碳", "STC土壤全碳", "全氮N",	"土壤C/N比"]:
                prophet = Prophet()
                df['ds'] = df['year'].astype(str).apply(lambda x: x[:4]+"-"+x[4:]+"-01").apply(pd.to_datetime)
                df.rename(columns={name: "y"}, inplace=True)
                prophet.fit(df)
                df.drop(columns=['y'], inplace=True)
                # predit
                future = prophet.make_future_dataframe(periods=4, freq='1Y')
                # print(future)
                forecast = prophet.predict(future)

                # savefigure
                fig = prophet.plot(forecast, include_legend=True, )
                name = "_".join(name.split('/'))
                fig.savefig(Config.out_dir+"/"+key+"_"+name+".png", dip=1000)
                df_dict[name] = forecast['yhat'].values
            df_dict['year'] = forecast['ds'].values
            break

            pd.DataFrame(df_dict).to_csv(Config.out_dir+"/{}.csv".format(key))

    if Config.forecast_prophet_mean:
        # 取均值每年的
        fig_5_12, ax = plt.subplots(12, 5, figsize=(12*3, 5*5))
        ax = ax.reshape(5*12)
        fig_5_12.set_title("predict")
            
        idx = 0
        for key, df in data14.groupby(by="放牧小区（plot）"):
            print(key) #, "SIC土壤无机碳": "y2", "STC土壤全碳": "y3", "全氮N": "y4", "土壤C/N比": "y5"
            df = df[["year", "SOC土壤有机碳", "SIC土壤无机碳", "STC土壤全碳", "全氮N",	"土壤C/N比"]]
            df_mean = df.groupby('year').agg(np.mean).reset_index()
            df_mean['ds'] = df_mean.loc[:, 'year'].astype(str).apply(pd.to_datetime)
            df_dict = {}
            for name in ["SOC土壤有机碳", "SIC土壤无机碳", "STC土壤全碳", "全氮N",	"土壤C/N比"]:
                prophet = Prophet()
                df_mean.rename(columns={name: "y"}, inplace=True)
                prophet.fit(df_mean)
                df_mean.drop(columns=['y'], inplace=True)
                # predit
                future = prophet.make_future_dataframe(periods=3, freq='Y')
                # print(future)
                forecast = prophet.predict(future)
                # savefigure
                fig = prophet.plot(forecast, include_legend=False, xlabel="year", ylabel=name, ax=ax[idx])
                idx += 1
                name = "_".join(name.split('/'))
                df_dict[name] = forecast['yhat'].values
            df_dict['year'] = forecast['ds'].values
        fig.savefig(Config.out_dir+"/"+ "predict.png", dip=1000)
        #     pd.DataFrame(df_dict).to_csv(Config.out_dir+"/{}.csv".format(key))


    if Config.forecast_visual:
        # 取均值每年的
        fig_5_12, ax = plt.subplots(12, 5, figsize=(12*3, 5*5))
        
        for title, axx in zip(data14["放牧小区（plot）"].unique(), ax[:, 0]):
            axx.annotate(title, xy=(0, 0.5), xytext=(-axx.yaxis.labelpad - 5, 0),
                xycoords=axx.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center')

        ax = ax.reshape(5*12)
        idx = 0
        for key, df in data14.groupby(by="放牧小区（plot）"):
            print(key) #, "SIC土壤无机碳": "y2", "STC土壤全碳": "y3", "全氮N": "y4", "土壤C/N比": "y5"
            df = df[["year", "SOC土壤有机碳", "SIC土壤无机碳", "STC土壤全碳", "全氮N",	"土壤C/N比"]]
            df_mean = df.groupby('year').agg(np.mean).reset_index()
            df_mean['ds'] = df_mean.loc[:, 'year'].astype(str).apply(pd.to_datetime)
            df_dict = {}
        
            for name in ["SOC土壤有机碳", "SIC土壤无机碳", "STC土壤全碳", "全氮N",	"土壤C/N比"]:
                prophet = Prophet()
                df_mean.rename(columns={name: "y"}, inplace=True)
                prophet.fit(df_mean)
                df_mean.drop(columns=['y'], inplace=True)
                # predit
                future = prophet.make_future_dataframe(periods=3, freq='Y')
                # print(future)
                forecast = prophet.predict(future)
                # savefigure
                fig = prophet.plot(forecast, include_legend=False, xlabel="year", ylabel=name, ax=ax[idx])
                idx += 1
                name = "_".join(name.split('/'))
                df_dict[name] = forecast['yhat'].values
            df_dict['year'] = forecast['ds'].values
        fig.savefig(Config.out_dir+"/"+ "predict1.png", dip=1000)
        pd.DataFrame(df_dict).to_csv(Config.out_dir+"/{}.csv".format(key))

    if Config.forecast_sk:
        for key, df in data14.groupby(by="放牧小区（plot）"):
            print("dealing...........+++++++++{}".format(key)) #, "SIC土壤无机碳": "y2", "STC土壤全碳": "y3", "全氮N": "y4", "土壤C/N比": "y5"
            df = df[["year", "SOC土壤有机碳", "SIC土壤无机碳", "STC土壤全碳", "全氮N",	"土壤C/N比"]]
            df_mean = df.groupby('year').agg(np.mean).reset_index()
            df_mean['year'] = df_mean.loc[:, 'year'].astype(str).apply(pd.to_datetime)
            df_dict = {}
            for name in ["SOC土壤有机碳", "SIC土壤无机碳", "STC土壤全碳", "全氮N",	"土壤C/N比"]:
                y_ori = df_mean[name].values
                lam1 = np.linspace(0, 8, 5)
                lam2 = np.linspace(0, 8, 9)
                print(lam2)
                print(lam1)

                y_ori = np.interp(lam2, lam1, y_ori)
                print(y_ori)
                #======================时序预测=========================
                y_train = create_inout_sequences(y_ori, tw=3)
                pre_x = np.array(create_inout_sequences_valid(y_ori, tw=3))
                data = np.array(y_train)
                x = data[:, 0:3]
                y = data[:, 3]
                
                print(pre_x)
                print("#=============模型===================")
                if Config.sk_method == "SVR":
                    model = SVR(kernel='rbf', C=1e2, degree=6, max_iter=10, tol=2e-1)
                elif Config.sk_method == "KRR":
                    model = KernelRidge(kernel="rbf", alpha=1e-1)
                elif Config.sk_method == "MLP":
                    model = MLPRegressor(hidden_layer_sizes=(100, 100, 50),activation='relu', solver='lbfgs', alpha=1e-1)
                elif Config.sk_method == "DecisionTreeRegressor":
                    model = DecisionTreeRegressor()
                elif Config.sk_method == "ExtraTreeRegressor":
                    model = ExtraTreeRegressor()
                elif Config.sk_method == "RandomForestRegressor":
                    model = RandomForestRegressor()
                elif Config.sk_method == "AdaBoostRegressor":
                    model = AdaBoostRegressor()
                elif Config.sk_method == "GradientBoostingRegressor":
                    model = GradientBoostingRegressor()
                elif Config.sk_method == "BaggingRegressor":
                    model = BaggingRegressor()
                elif Config.sk_method == "StackingRegressor":
                    estimators = [('dt', AdaBoostRegressor())]
                    model = StackingRegressor(estimators=estimators, final_estimator=SVR(kernel='rbf', C=1e3, degree=6, max_iter=10, tol=1e-4))


                model.fit(x, y)
                pre_y = model.predict(pre_x)

                print("#=============验证===================")
                if Config.sk_valid:
                    model.fit(x[:-2], y[:-2])
                    valid_y = model.predict(x)

                print("#=============绘图===================")
                plt.figure()
                year = sorted(list(set(df['year'].values)))
                year = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
                x_axis = [2015, 2016, 2017, 2018, 2019, 2020, 2021]

                plt.plot(x_axis, pre_y, linewidth=0.8, label="predit")
                plt.plot(year, y_ori, "o:",linewidth=0.8, label="origin", )
                # plt.scatter(year, y_ori, label="origin")
                if Config.sk_valid:
                    x_valid = [2015, 2016, 2017, 2018, 2019, 2020]
                    plt.plot(x_valid, valid_y, linewidth=0.8, label="valid")
                
                plt.legend()
                plt.xlabel("year")
                plt.ylabel(name)
                plt.xticks([2012, 2014, 2016, 2018, 2020, 2022])
                plt.title("{}:year与{}关系图".format(key, name))

                print("#=============保存绘图===================")
                name = "_".join(name.split('/'))
                outdir = os.path.join(Config.out_dir, Config.sk_method)
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
                plt.savefig(outdir + "/" + key + "_" + name + ".png", dpi=100)
                

                print("#=============评价===================")
                if Config.sk_valid:
                    mae = mean_absolute_error(y, valid_y)
                    mse = mean_squared_error(y, valid_y)
                    msle = mean_squared_log_error(y, valid_y)
                    mape = mean_absolute_percentage_error(y, valid_y)
                    med_ae = median_absolute_error(y, valid_y)
                    evs = explained_variance_score(y, valid_y)
                    r2 = r2_score(y, valid_y)
                print("#=============保存===================")
                df_dict["year"] = x_axis
                df_dict[name] = pre_y
                if Config.sk_valid:
                    df_dict['mae_'+name] = mae
                    df_dict['mse_'+name] = mse
                    df_dict['msle_'+name] = msle
                    df_dict['mape_'+name] = mape
                    df_dict['med_ae_'+name] = med_ae
                    df_dict['evs_'+name] = evs
                    df_dict['r2_'+name] = r2
                # print(df_dict)
                if Config.sk_mode == "debug":
                    break
            if Config.sk_mode == "debug":
                break
            pd.DataFrame(df_dict).to_csv(outdir+"/{}.csv".format(key))
            



    if Config.forecast_tsfresh:
            pass

    if Config.forecast_autots:
        for key, df in data14.groupby(by="放牧小区（plot）"):
            print(key) #, "SIC土壤无机碳": "y2", "STC土壤全碳": "y3", "全氮N": "y4", "土壤C/N比": "y5"
            df = df[["year", "SOC土壤有机碳", "SIC土壤无机碳", "STC土壤全碳", "全氮N",	"土壤C/N比"]]
            df_mean = df.groupby('year').agg(np.mean).reset_index()
            df_mean['year'] = df_mean.loc[:, 'year'].astype(str).apply(pd.to_datetime)
            for name in ["SOC土壤有机碳", "SIC土壤无机碳", "STC土壤全碳", "全氮N",	"土壤C/N比"]:
                all_data = df_mean[['year', name]]
                print(all_data)
                autots_model = AutoTS(forecast_length=2, frequency='infer', ensemble='simple', )
                autots_model.fit(all_data, date_col='year', value_col=name, id_col=None)
                prediction = autots_model.predict()
                forecast = prediction.forecast
                print(forecast)
                break
            break