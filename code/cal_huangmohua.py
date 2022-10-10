from tkinter import N
import numpy as np
import os
import xlrd

def load_rain_data(root, years):
    data_dict = {}
    splits = os.listdir(root)
    for split in splits:
        excel = xlrd.open_workbook(os.path.join(root, split))  # 打开excel文件
        sheet = excel.sheet_by_index(0)  # 获取工作薄
        year = sheet.col_values(colx=4, start_rowx=1, end_rowx=None)
        month = sheet.col_values(colx=5, start_rowx=1, end_rowx=None)
        rain_data = sheet.col_values(colx=15, start_rowx=1, end_rowx=None)
        temp_data = sheet.col_values(colx=6, start_rowx=1, end_rowx=None)
        wind_data = sheet.col_values(colx=25, start_rowx=1, end_rowx=None)

        rain_data = [float(k) for i, k in enumerate(rain_data) if int(month[i])>=3 and int(month[i])<=10]
        temp_data = [float(k) for i, k in enumerate(temp_data) if int(month[i])>=3 and int(month[i])<=10]
        wind_data = [float(k) for i, k in enumerate(wind_data) if int(month[i])>=3 and int(month[i])<=10]

        all_years = list(set(year))
        all_years.sort()
        for i in range(len(year)):
            if year[i] in years:
                if year[i] not in data_dict.keys():
                    data_dict[year[i]] = {'rain': np.array(rain_data).sum(), 'temp': np.array(temp_data).mean(), 'wind': np.array(wind_data).mean()}
    hot_rain = np.mean(np.array([data_dict[year]['rain'] for year in data_dict.keys()]))
    hot_temp = np.mean(np.array([data_dict[year]['temp'] for year in data_dict.keys()]))
    hot_wind = np.mean(np.array([data_dict[year]['wind'] for year in data_dict.keys()]))
    
    return hot_rain, hot_temp, hot_wind

def load_cover_data(root, years):
    data_dict = {}
    excel = xlrd.open_workbook(root)  # 打开excel文件
    sheet = excel.sheet_by_index(0)  # 获取工作薄

    year = sheet.col_values(colx=1, start_rowx=1, end_rowx=None)
    month = sheet.col_values(colx=0, start_rowx=1, end_rowx=None)

    cover = sheet.col_values(colx=4, start_rowx=1, end_rowx=None)
    cover = np.array([float(k)*100 for i, k in enumerate(cover) if int(month[i])>=3 and int(month[i])<=10])
    cover[np.where(cover<0)] = 0

    all_years = list(set(year))
    all_years.sort()
    
    for i in range(len(year)):
        if year[i] in years:
            if year[i] not in data_dict.keys():
                data_dict[year[i]] = cover.mean()
    cover = np.mean(np.array([data_dict[year] for year in data_dict.keys()]))
    return cover

def load_water_data(root, years):
    data_dict = {}
    excel = xlrd.open_workbook(root)  # 打开excel文件
    sheet = excel.sheet_by_index(0)  # 获取工作薄

    year = sheet.col_values(colx=1, start_rowx=1, end_rowx=None)
    month = sheet.col_values(colx=0, start_rowx=1, end_rowx=None)

    water = sheet.col_values(colx=5, start_rowx=1, end_rowx=None)
    water = np.array([float(k) / 1e4 for i, k in enumerate(water) if int(month[i])>=6 and int(month[i])<=8])

    all_years = list(set(year))
    all_years.sort()
    
    for i in range(len(year)):
        if year[i] in years:
            if year[i] not in data_dict.keys():
                data_dict[year[i]] = water.mean()
    water = np.mean(np.array([data_dict[year] for year in data_dict.keys()]))
    return water

def index_limits(C):
    limits = [
        [2.61, 1.43],
        [100, 10],
        [35, 30, 25, 5],
        [800, 60],
        [21.47, 1.22],
        [7.7, 4.85],
        [60, 7],
        [600, 300],
        [1100, 450]
    ]
    sign = [1, -1, 0, -1, 2, 1, 1, 1, 1]
    Q = []
    for i in range(len(limits)):
        if sign[i] == 1:
            if C[i] <= limits[i][1]:
                Q.append(0)
            elif C[i] >= limits[i][0]:
                Q.append(1)
            else:
                Q.append((C[i]-limits[i][1]) / (limits[i][0]-limits[i][1]))
        elif sign[i] == -1:
            if C[i] >= limits[i][0]:
                Q.append(0)
            elif C[i] <= limits[i][1]:
                Q.append(1)
            else:
                Q.append((limits[i][0] - C[i]) / (limits[i][0]-limits[i][1]))
        elif sign[i] == 0:
            if C[i] >= limits[i][0] or C[i] <= limits[i][-1]:
                Q.append(1)
            elif C[i] <= limits[i][1] and C[i] >= limits[i][2]:
                Q.append(0)
            elif C[i] <= limits[i][0] and C[i] >= limits[i][1]:
                Q.append((limits[i][0] - C[i]) / (limits[i][0]-limits[i][1]))
            elif C[i] <= limits[i][-2] and C[i] >= limits[i][-1]:
                Q.append((C[i]-limits[i][-1]) / (limits[i][-2]-limits[i][-1]))
        elif sign[i] == 2:
            if C[i] >= limits[i][0]:
                Q.append(0)
            elif C[i] == 0:
                Q.append(1)
            else:
                Q.append((limits[i][0] - C[i]) / (limits[i][0]-limits[i][1]))
    return Q

def apply_weight(Q):
    S = []
    Wc = [0.0931, 0.0576, 0.0354, 0.2455, 0.0743, 0.0675, 0.0692, 0.2920, 0.0821]
    for i in range(len(Wc)):
        S.append(Wc[i] * Q[i])
    return S

if __name__ == '__main__':
    years = ['2021']
    hot_rain, hot_temp, hot_wind = load_rain_data('/比赛/2022/2022年E题/数据集/基本数据/附件8、锡林郭勒盟气候2012-2022', years)
    print(hot_rain, hot_temp, hot_wind)
    # cover = load_cover_data('/2022年E题/数据集/基本数据/附件6、植被指数-NDVI2012-2022年.xls', years)
    water_up = load_water_data('/比赛/2022/2022年E题/数据集/基本数据/附件9、径流量2012-2022年.xlsx', years)
    print(water_up)
    ita = [1.0193, 0.0198]
    n = 9
    hot_wind = hot_wind * 0.514444
    intensity = 800
    if intensity == 0:
        cover = 200
    elif intensity == 200:
        cover = 400
    elif intensity == 400:
        cover = 300
    elif intensity == 800:
        cover = 100

    C = [hot_wind, hot_rain, hot_temp, cover, water_up, 15, 5.2, intensity, 11897]
    assert len(C) == n
    Q = index_limits(C)
    S = apply_weight(Q)
    SM = ita[0] * sum(S) + ita[1]
    print(SM)