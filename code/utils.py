from re import X
import xlrd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_humid_data(root, one_hot=False):
    data_dict = {}
    excel = xlrd.open_workbook(root)  # 打开excel文件
    sheet = excel.sheet_by_index(0)  # 获取工作薄

    year = sheet.col_values(colx=1, start_rowx=1, end_rowx=None)
    month = sheet.col_values(colx=0, start_rowx=1, end_rowx=None)

    humid_10cm = sheet.col_values(colx=4, start_rowx=1, end_rowx=None)
    humid_10cm = np.array([float(k) for k in humid_10cm])

    humid_40cm = sheet.col_values(colx=5, start_rowx=1, end_rowx=None)
    humid_40cm = np.array([float(k) for k in humid_40cm])

    humid_100cm = sheet.col_values(colx=6, start_rowx=1, end_rowx=None)
    humid_100cm = np.array([float(k) for k in humid_100cm])

    humid_200cm = sheet.col_values(colx=7, start_rowx=1, end_rowx=None)
    humid_200cm = np.array([float(k) for k in humid_200cm])
    
    data = np.stack([humid_10cm, humid_40cm, humid_100cm, humid_200cm])

    assert len(year) == len(month) == data.shape[1]

    all_years = list(set(year))
    all_years.sort()

    for i in range(len(year)):
        if year[i] not in data_dict.keys():
            data_dict[year[i]] = {month[i]: data[:, i]}
        else:
            assert month[i] not in data_dict[year[i]].keys()
            data_dict[year[i]][month[i]] = data[:, i]
    
    all_years = list(data_dict.keys())
    all_years.sort()
    sorted_data_list = []
    month_final = []
    for year_ in all_years:
        month_list = list(data_dict[year_].keys())
        month_list = list(set(month_list))
        month_list.sort()
        for month_ in month_list:
            sorted_data_list.append(data_dict[year_][month_])
            if one_hot:
                month_oh = np.zeros(12)
                month_oh[int(month_)-1] = 1
                month_final.append(month_oh)
            else:
                month_final.append(np.ones(1)*float(month_))
    
    assert len(sorted_data_list) == len(year) == len(month)

    data_final = np.stack(sorted_data_list)
    month_final = np.stack(month_final)

    return np.concatenate([data_final, month_final], axis=-1)

def load_evaporation_data(root, one_hot=False):
    data_dict = {}
    excel = xlrd.open_workbook(root)  # 打开excel文件
    sheet = excel.sheet_by_index(0)  # 获取工作薄

    year = sheet.col_values(colx=1, start_rowx=1, end_rowx=None)
    month = sheet.col_values(colx=0, start_rowx=1, end_rowx=None)

    evaporation_wm2 = sheet.col_values(colx=4, start_rowx=1, end_rowx=None)
    evaporation_wm2 = np.array([float(k) for k in evaporation_wm2])

    evaporation_mm = sheet.col_values(colx=5, start_rowx=1, end_rowx=None)
    evaporation_mm = np.array([float(k) for k in evaporation_mm])
    
    data = np.stack([evaporation_wm2, evaporation_mm])

    assert len(year) == len(month) == data.shape[1]

    all_years = list(set(year))
    all_years.sort()

    for i in range(len(year)):
        if year[i] not in data_dict.keys():
            data_dict[year[i]] = {month[i]: data[:, i]}
        else:
            assert month[i] not in data_dict[year[i]].keys()
            data_dict[year[i]][month[i]] = data[:, i]
    
    all_years = list(data_dict.keys())
    all_years.sort()
    sorted_data_list = []
    month_final = []
    for year_ in all_years:
        month_list = list(data_dict[year_].keys())
        month_list = list(set(month_list))
        month_list.sort()
        for month_ in month_list:
            sorted_data_list.append(data_dict[year_][month_])
            if one_hot:
                month_oh = np.zeros(12)
                month_oh[int(month_)-1] = 1
                month_final.append(month_oh)
            else:
                month_final.append(np.ones(1)*float(month_))
    
    assert len(sorted_data_list) == len(year) == len(month)

    data_final = np.stack(sorted_data_list)
    month_final = np.stack(month_final)

    return np.concatenate([data_final, month_final], axis=-1)


def load_rain_data(root, one_hot=False):
    data_dict = {}
    splits = os.listdir(root)
    for split in splits:
        excel = xlrd.open_workbook(os.path.join(root, split))  # 打开excel文件
        sheet = excel.sheet_by_index(0)  # 获取工作薄
        year = sheet.col_values(colx=4, start_rowx=1, end_rowx=None)
        month = sheet.col_values(colx=5, start_rowx=1, end_rowx=None)
        rain_data = sheet.col_values(colx=15, start_rowx=1, end_rowx=None)
        data = np.array([float(k) for k in rain_data])[np.newaxis, :]
        all_years = list(set(year))
        all_years.sort()
        for i in range(len(year)):
            if year[i] not in data_dict.keys():
                data_dict[year[i]] = {month[i]: data[:, i]}
            else:
                assert month[i] not in data_dict[year[i]].keys()
                data_dict[year[i]][month[i]] = data[:, i]
    
    all_years = list(data_dict.keys())
    all_years.sort()
    sorted_data_list = []
    month_final = []
    for year_ in all_years:
        month_list = list(data_dict[year_].keys())
        month_list = list(set(month_list))
        month_list.sort()
        for month_ in month_list:
            sorted_data_list.append(data_dict[year_][month_])
            if one_hot:
                month_oh = np.zeros(12)
                month_oh[int(month_)-1] = 1
                month_final.append(month_oh)
            else:
                month_final.append(np.ones(1)*float(month_))

    data_final = np.stack(sorted_data_list)
    month_final = np.stack(month_final)

    return np.concatenate([data_final, month_final], axis=-1)


if __name__ == '__main__':
    root_humid = "/media/wwk/My Passport/比赛/2022/2022年E题/数据集/基本数据/附件3、土壤湿度2022—2012年.xls"
    root_evaporation = "/media/wwk/My Passport/比赛/2022/2022年E题/数据集/基本数据/附件4、土壤蒸发量2012—2022年.xls"
    root_rain = "/media/wwk/My Passport/比赛/2022/2022年E题/数据集/基本数据/附件8、锡林郭勒盟气候2012-2022"
    evaporation_data, evaporation_month = load_evaporation_data(root_evaporation)
    humid_data, humid_month = load_humid_data(root_humid)
    rain_data, rain_month = load_rain_data(root_rain)
    rain_data = rain_data[:len(humid_data), :]
    
    # assert len(evaporation_data) == len(humid_data) == len(rain_data)
    # plt.figure()
    # # plt.scatter(x=evaporation_data[:, 0], y=humid_data[:, 0])
    # plt.plot(evaporation_data[:, 0])
    # plt.plot(evaporation_data[:, 1])
    # # plt.plot(humid_data[:, 0])
    # # plt.plot(rain_data[:, 0])
    # plt.show()
    
    


    

    