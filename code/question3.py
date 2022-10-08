from audioop import minmax
from holidays import CN
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
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm #最小二乘
from statsmodels.formula.api import ols #加载ols模型

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

                # diff = y_ori[:-1] - y_ori[1:]
                # last = diff.mean()
                # diff_arr = np.concatenate([diff, [last]], axis=0).reshape(-1, 1)
                # y_ori_diff = np.concatenate([y_ori.reshape(-1, 1), diff_arr], axis=1)
                #======================时序预测=========================
                y_train = create_inout_sequences(y_ori, tw=3)
                pre_x = np.array(create_inout_sequences_valid(y_ori, tw=3))
                data = np.array(y_train)
                x = data[:, 0:3]
                y = data[:, 3]
                
                print(y_train)
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
            
    if Config.forecast_lstm_pre:
        other_data = pd.read_csv(Config.attachment, usecols=['植被指数(NDVI)', '高层植被(LAIH,m2/m2)', '低层植被(LAIL,m2/m2)', '绿植覆盖率', '日期'])
        other_data['日期'] = other_data['日期'].apply(lambda x: str(x)[:4])
        other = pd.DataFrame(other_data.groupby('日期').agg({"植被指数(NDVI)": np.mean, "高层植被(LAIH,m2/m2)": np.mean, "低层植被(LAIL,m2/m2)": np.mean, "绿植覆盖率": np.mean}))
        other.to_csv("otherforquestion3.csv")
    
    if Config.interpolate:
        # 返回多项式
        def p(x,a):
            """
            p(x,a)是x的函数，a是各幂次的系数
            """
            s = 0
            for i in range(len(a)):
                s += a[i]*x**i
            return s

        # n次拉格朗日插值
        def lagrange_interpolate(x_list,y_list,x): 
            """
            x_list 待插值的x元素列表
            y_list 待插值的y元素列表
            插值以后整个lagrange_interpolate是x的函数
            """
            if len(x_list) != len(y_list):
                raise ValueError("list x and list y is not of equal length!")
            # 系数矩阵
            A = []
            for i in range(len(x_list)):
                A.append([])
                for j in range(len(x_list)):
                    A[i].append(pow(x_list[i],j))
            b = []
            for i in range(len(x_list)):
                b.append([y_list[i]])
            # 求得各阶次的系数
            
            a = np.linalg.solve(A, b) # 用LU分解法解线性方程组，可以使用numpy的类似函数
            a = np.transpose(a)[0] # change col vec a into 1 dimension
            val = p(x,a)
            # print(x,val)
            return val
        
            #============牛顿插值================
        def difference_list(dlist): # Newton
            if len(dlist)>0:
                print(dlist)
                prev,curr = 0,0
                n = []
                for i in dlist:
                    curr = i
                    n.append(curr - prev)
                    prev = i
                n.pop(0)
                difference_list(n)
        def difference_quotient_list(y_list,x_list = []):
            if x_list == []:
                x_list = [i for i in range(len(y_list))]
            print(y_list)
            prev_list = y_list
            dq_list = []
            dq_list.append(prev_list[0])
            for t in range(1,len(y_list)):
                prev,curr = 0,0  
                m = []
                k = -1
                for i in prev_list:
                    curr = i
                    m.append((curr - prev)/(x_list[k+t]-x_list[k]))
                    prev = i
                    k+=1
                m.pop(0)     
                prev_list = m
                dq_list.append(prev_list[0])
                print(m)
            return dq_list

        def newton_interpolate(x_list,y_list,x):
            coef = difference_quotient_list(y_list,x_list)
            p = coef[0]
            for i in range(1,len(coef)):
                product = 1
                for j in range(i):
                    product *= (x - x_list[j] )
                p += coef[i]*product
            return p

        #==========分段三次埃尔米特插值=============
        import scipy
        outdir = os.path.join(Config.out_dir, Config.interpolate_save)
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        for key, df in data14.groupby(by="放牧小区（plot）"):
            fig, ax = plt.subplots(2, 2, figsize=(2*7, 2*7))
            ax = ax.reshape(-1)
            name = ["SOC土壤有机碳", "SIC土壤无机碳", "STC土壤全碳", "全氮N"]
            df = df[["year", "SOC土壤有机碳", "SIC土壤无机碳", "STC土壤全碳", "全氮N",	"土壤C/N比"]]
            df_mean = df.groupby('year').agg(np.mean).reset_index(drop=True)
            # print(df_mean)
            y_ps = df_mean.values
            # print(y_ps)
            x_p = range(y_ps.shape[0])
            for id in range(y_ps[:, :-1].shape[1]):
                x = np.linspace(0, 4)
                if Config.interpolate_method == "newton":
                    y = list(map(lambda t: newton_interpolate(x_p, y_ps[:, id], t), x))
                elif Config.interpolate_method == "lagrange":
                    y = list(map(lambda t: lagrange_interpolate(x_p, y_ps[:, id], t), x))
                elif Config.interpolate_method == "Pchip":
                    y = list(map(lambda t: scipy.interpolate.PchipInterpolator(x_p,y_ps[:, id])(t), x))
                ax[id].plot(x, y, 'g:')
                ax[id].scatter(x_p, y_ps[:, id])
                ax[id].set_title("{}:{}".format(key,name[id]))
                ax[id].set_ylabel(name[id])
                if Config.generate_result:
                    x = np.linspace(0, 4, 9)
                    y = np.array(list(map(lambda t: newton_interpolate(x_p, y_ps[:, id], t), x)))
                    np.savetxt("{}/{}-{}-{}.txt".format(outdir, key, name[id], Config.interpolate_method), y)

            
            fig.savefig("{}/{}-{}.png".format(outdir, key, Config.interpolate_method))

            # 插值后，获得插值后数据进行预测

            
            




        


    if Config.forecast_lstm:
        from keras.models import Sequential
        from keras.layers.core import Dense, Activation, Dropout
        from keras.layers.rnn import LSTM
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler
        build = tf.sysconfig.get_build_info()
        print(build['cuda_version'])
        print(build['cudnn_version'])
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        for key, df in data14.groupby(by="放牧小区（plot）"):
            print("dealing...........+++++++++{}".format(key)) #, "SIC土壤无机碳": "y2", "STC土壤全碳": "y3", "全氮N": "y4", "土壤C/N比": "y5"
            df = df[["year", "SOC土壤有机碳", "SIC土壤无机碳", "STC土壤全碳", "全氮N",	"土壤C/N比"]]
            df_mean = df.groupby('year').agg(np.mean).reset_index()
            df_mean['year'] = df_mean.loc[:, 'year'].astype(str).apply(pd.to_datetime)
            df_dict = {}
            for name in ["SOC土壤有机碳", "SIC土壤无机碳", "STC土壤全碳", "全氮N",	"土壤C/N比"]:
                y_ori = df_mean[name].values

                diff = y_ori[:-1] - y_ori[1:]
                last = diff.mean()
                diff_arr = np.concatenate([diff, [last]], axis=0).reshape(-1, 1)
                y_ori_diff = np.concatenate([y_ori.reshape(-1, 1), diff_arr], axis=1)
                print(y_ori_diff.shape)
                min_max = MinMaxScaler()
                min_max.fit(y_ori_diff)
                y_ori_diff = min_max.transform(y_ori_diff)
                #======================时序预测=========================
                y_train = create_inout_sequences(y_ori_diff, tw=3)
                pre_x = np.array(create_inout_sequences_valid(y_ori_diff, tw=3))
                data = np.array(y_train)
                
            
                x = data[:, 0:3, :]
                y = data[:, 3, :]
                print(x.shape)
                print(y.shape)
                print("#=============模型===================")
                model = Sequential()
                model.add( LSTM( 100, input_shape=(x.shape[1], x.shape[2]), return_sequences=True) )
                model.add( LSTM( 50, return_sequences=False ) )
                model.add( Dropout( 0.3 ) )
                model.add( Dense( 2 ) )
                model.add( Activation( 'linear' ) )
                model.compile( loss="mse", optimizer="adam" )

                model.fit(x, y, epochs=310, batch_size=2)
                pre_y = model.predict(pre_x)
                pre_y = min_max.inverse_transform(pre_y)
                print(pre_y)
                print("#=============验证===================")
                if Config.lstm_valid:
                    model.fit(x[:-2], y[:-2])
                    valid_y = model.predict(x)

                print("#=============绘图===================")
                plt.figure()
                year = sorted(list(set(df['year'].values)))
                year = [2012, 2014, 2016, 2018,  2020]
                x_axis = [2018, 2020, 2022]

                plt.plot(x_axis, pre_y[:, 0], linewidth=0.8, label="predit")
                plt.plot(year, y_ori, "o:",linewidth=0.8, label="origin", )
                # plt.scatter(year, y_ori, label="origin")
                if Config.lstm_valid:
                    x_valid = [2015, 2016, 2017, 2018, 2019, 2020]
                    plt.plot(x_valid, valid_y, linewidth=0.8, label="valid")
                
                plt.legend()
                plt.xlabel("year")
                plt.ylabel(name)
                plt.xticks([2012, 2014, 2016, 2018, 2020, 2022])
                plt.title("{}:year与{}关系图".format(key, name))

                print("#=============保存绘图===================")
                name = "_".join(name.split('/'))
                outdir = os.path.join(Config.out_dir, Config.lstm_method)
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
                plt.savefig(outdir + "/" + key + "_" + name + ".png", dpi=100)
                

                print("#=============评价===================")
                if Config.lstm_valid:
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
                if Config.lstm_valid:
                    df_dict['mae_'+name] = mae
                    df_dict['mse_'+name] = mse
                    df_dict['msle_'+name] = msle
                    df_dict['mape_'+name] = mape
                    df_dict['med_ae_'+name] = med_ae
                    df_dict['evs_'+name] = evs
                    df_dict['r2_'+name] = r2
                # print(df_dict)
                if Config.lstm_mode == "debug":
                    break
            if Config.lstm_mode == "debug":
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

    

    if Config.visual_by_year:
        fig, ax = plt.subplots(4, 4, figsize=(4*7, 4*7))
        for id, name in  enumerate([ "SOC土壤有机碳", "SIC土壤无机碳", "STC土壤全碳", "全氮N"]):
            for iid, (y, df) in enumerate(data14.groupby('year')):
                if y==2020:
                    break
                for xq in df['放牧小区（plot）'].unique():
                    v = df[df['放牧小区（plot）']==xq][name].sort_index(ascending=True)
                    # print(v)
                    ax[iid, id].plot(range(v.shape[0]), v, 'o-',label=xq)
                ax[iid, id].legend()
                ax[iid, id].set_title(name+":"+str(y))
                from matplotlib.ticker import MaxNLocator
                ax[iid, id].xaxis.set_major_locator(MaxNLocator(integer=True))

                # ax[iid, id].set_xlim(0, 2)


                

        outdir = os.path.join(Config.out_dir, Config.visual_by_year_dir)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        fig.savefig(outdir +"/" + "result.png", dip=1000)


    if Config.visual15:
        # data15_1 = pd.read_excel(Config.attachment15, sheet_name=1, )
        data5 = pd.read_excel(Config.attachment15, sheet_name=5, usecols=['鲜重', '干重(g)', '轮次', '年份'])
        print(data5)
        for year, df in data5.groupby('年份'):
            fig, ax = plt.subplots(1,1)

            df['鲜重'] = df['鲜重'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
            df['干重(g)'] = df['干重(g)'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
            df.fillna(0, inplace=True)
            # df['干重(g)'].str.replace("<", "")
            df_sum = pd.DataFrame(df.groupby('轮次').agg({'鲜重': np.sum, '干重(g)':np.sum}, columns=['鲜重(g)', '干重(g)', '年份']).reset_index())
            print(df_sum)
            width = 0.35
            x = np.arange(len(df_sum["轮次"].unique()))
            ax.bar(x -width/2, df_sum['鲜重'], width, label="鲜重", color='#fdcb6e')
            # ax.plot(x -width/2, df_sum['鲜重'], width, 'o:', color='#fdcb6e')
            ax.bar(x +width/2, df_sum['干重(g)'], width,label="干重", color='#55efc4')
            # ax.plot(x +width/2, df_sum['干重(g)'], width, 'o:', color='#55efc4')
            ax.set_xlabel("轮次")
            ax.set_ylabel("重量(g)")
            ax.set_title("{}的总鲜重和干重".format(year))
            ax.set_xticks(range(len(df_sum["轮次"].unique())), df_sum['轮次'].unique())
            # from matplotlib.ticker import MaxNLocator
            # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            # ax.set_xlim(df_sum['轮次'].unique())
            ax.legend()


            # color = 'tab:green'
            # ax = sns.lineplot(x='轮次', y='鲜重', data = df_sum, color=color)
            # ax.tick_params(axis='y')
            # #twinx共享x轴(类似的语法，如共享y轴twiny)
            # ax1 = ax.twinx()
            # color = 'tab:red'
            # ax1 = sns.lineplot(x='轮次', y='干重(g)', data = df_sum, color=color)



            outdir = os.path.join(Config.out_dir, Config.visual15_dir)
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            plt.savefig('{}/{}.png'.format(outdir, year))

