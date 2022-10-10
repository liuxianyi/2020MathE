from cProfile import label
import pandas as pd
import os

from config import Config
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    attachment3 = Config.attachment3
    attachment4 = Config.attachment4
    attachment5 = Config.attachment5
    attachment6 = Config.attachment6
    attachment7 = Config.attachment7
    attachment8s = Config.attachment8s
    attachment9 = Config.attachment9
    attachment10 = Config.attachment10
    attachment11 = Config.attachment11

    if Config.isAttachment3:
        data3 = pd.read_excel(attachment3)
        data3.loc[:, '日期'] = data3.loc[:, '年份'].astype(str) + data3.loc[:, '月份'].apply(lambda x: "{:0>2}".format(x))
        data3.drop(columns = ['经度(lon)', '纬度(lat)', '月份', '年份'], inplace=True)
    
    
    if Config.isAttachment4:
        data4 = pd.read_excel(attachment4)
        data4.loc[:, '日期'] = data4.loc[:, '年份'].astype(str) + data4.loc[:, '月份'].apply(lambda x: "{:0>2}".format(x))
        data4.drop(columns = ['经度(lon)', '纬度(lat)', '月份', '年份'], inplace=True)


    if Config.isAttachment5:
        data5 = []
        for year in range(2020, 2023):
            data5_ = pd.read_excel(os.path.join(attachment5, "{}绿植覆盖率.xls".format(year)), usecols=['绿植覆盖率', '时间'])
            data5.append(data5_)
        data5 = pd.concat(data5, axis=0)
        data5.loc[:, '日期'] = data5.loc[:, '时间'].apply(pd.to_datetime)
        data5.sort_values(by='日期', inplace=True)
        data5.reset_index(inplace=True, drop=True)

        # 可视化趋势
        data5_visual = data5
        data5_visual.dropna(inplace=True)
        # fig, ax = plt.subplot(111)
        plt.figure()
        print(1111,data5_visual['绿植覆盖率'].values)
        plt.scatter(data5["日期"], data5_visual['绿植覆盖率'].values, label="原始")
        plt.plot(data5["日期"], data5_visual['绿植覆盖率'].values, 'r:', label="补全后")
        plt.title("绿植覆盖率时间关系")
        plt.xlabel("日期")
        plt.ylabel("植被覆盖率")
        plt.legend()

        outdir = Config.mkdir(Config.attachment5_dir)
        Config.savefig(plt, outdir, "{}".format("trend"))

        for idx, rows in enumerate(data5.iterrows()):
            if pd.isna(rows[1]["绿植覆盖率"]):
                print(rows[0])
                print(data5.loc[range(rows[0]-4, rows[0]), :].mean())
                data5.loc[rows[0], "绿植覆盖率"]= data5.loc[range(rows[0]-4, rows[0]), "绿植覆盖率"].mean()
                # data5.loc[rows[0], :].fillna(data5.loc[range(rows[0]-4, rows[0]), :].mean(), inplace=True)
        # plt.figure()
        # plt.scatter(data5["日期"], data5['绿植覆盖率'].values)
        # plt.plot(data5["日期"], data5['绿植覆盖率'].values, 'r:')
        # plt.title("缺失值补全后绿植覆盖率时间关系")
        # plt.xlabel("日期")
        # plt.ylabel("植被覆盖率")
        # Config.savefig(plt, outdir, "{}".format("after_trend"))

        data5["日期"] = data5["日期"].apply(lambda x: "".join(str(x).split('-')[0:2]))
        data5 = pd.DataFrame(data5[["日期", "绿植覆盖率"]].groupby("日期").agg({"绿植覆盖率": np.mean}))
        data5.to_csv("data5.csv")

        


    if Config.isAttachment6:
        data6 = pd.read_excel(attachment6)
        data6.loc[:, '日期'] = data6.loc[:, '年份'].astype(str) + data6.loc[:, '月份'].apply(lambda x: "{:0>2}".format(x))
        # print(data6.head())
        data6.drop(columns = ['经度(lon)', '纬度(lat)', '月份', '年份'], inplace=True)


    if Config.isAttachment7:
        data7_1 = pd.read_excel(attachment7, header=None, usecols=[0, 1])
        data7_2 = pd.read_excel(attachment7, header=None, usecols=[2, 3])
        data7_2 = data7_2.dropna(how='any')
        data7_2.columns = [0, 1]


        data7 = pd.concat([data7_1, data7_2], axis=0).reset_index(drop=True)
        data7 = data7.T
        columns = data7.loc[0, :]
        data7.drop([0], inplace=True)
        data7.columns = columns
        # print(data7)


    if Config.isAttachment8s:
        data8 = []
        for year in range(2012, 2023):
            print(os.path.join(Config.attachment8s, str(year)+"年.xls"))
            data8_ = pd.read_excel(os.path.join(Config.attachment8s, str(year)+"年.xls"))
            data8_.loc[:, '日期'] = data8_.loc[:, '年份'].astype(str) + data8_.loc[:, '月份'].apply(lambda x: "{:0>2}".format(x))
            data8_.drop(columns = ['经度', '纬度', '月份', '年份'], inplace=True)
            data8.append(data8_)
        data8 = pd.concat(data8, axis=0)
        print(data8.head())

    if Config.isAttachment9:
        data9 = pd.read_excel(attachment9)
        data9.loc[:, '日期'] = data9.loc[:, '年份'].astype(str) + data9.loc[:, '月份'].apply(lambda x: "{:0>2}".format(x))
        # print(data9.head())
        data9.drop(columns = ['经度(lon)', '纬度(lat)', '月份', '年份'], inplace=True)


    if Config.isAttachment10:
        data10 = pd.read_excel(attachment10)
        data10['日期'] = data10['日期'].astype(str)
        # print(data10.head())
        data10.drop(columns = ['经度(lon)', '纬度(lat)'], inplace=True)


    if Config.isAttachment11:
        data11 = pd.read_excel(attachment11)
        data11.loc[:, '日期'] = data10.loc[:, '年份'].astype(str) + data11.loc[:, '月份'].apply(lambda x: "{:0>2}".format(x))
        # print(data11.head())
        # data11.drop(['经度(lon)', '纬度(lat)', '月份', '年份'], inplace=True)


    if Config.ismerge:
        data = data3.merge(data4, how='outer', on='日期')
        data = data.merge(data6, how='outer', on='日期')
        # data = data.merge(data7, how='outer', on='日期')
        data = data.merge(data9, how='outer', on='日期')
        data = data.merge(data10, how='outer', on='日期')
        data = data.merge(data8, how='outer', on='日期')
        data = data.merge(data5, how='outer', on='日期')

        new_data7 = pd.DataFrame(np.repeat(data7.values, data.shape[0], axis=0), columns=data7.columns)
        data = pd.concat([data, new_data7], axis=1)

        data.set_index('日期', inplace=True)
        data.sort_index(inplace=True)
        data.to_csv("tmp.csv")
        data.to_excel("tmp.xlsx")

        
        




    