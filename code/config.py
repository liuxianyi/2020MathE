class Config(object):
    out_dir = "/media/HardDisk_A/Workers/goog/2022MathE/train_data"

    attachment3 = "/media/HardDisk_A/Workers/goog/2022MathE/data/基本数据/附件3、土壤湿度2022—2012年.xls"
    isAttachment3 = True
    attachment4 = "/media/HardDisk_A/Workers/goog/2022MathE/data/基本数据/附件4、土壤蒸发量2012—2022年.xls"
    isAttachment4 = True
    attachment6 = "/media/HardDisk_A/Workers/goog/2022MathE/data/基本数据/附件6、植被指数-NDVI2012-2022年.xls"
    isAttachment6 = True
    attachment7 = "/media/HardDisk_A/Workers/goog/2022MathE/data/基本数据/附件7、锡林郭勒土壤基本数据.xls"
    isAttachment7 = True # 单字段，无时间
    attachment8s = "/media/HardDisk_A/Workers/goog/2022MathE/data/基本数据/附件8、锡林郭勒盟气候2012-2022"
    isAttachment8s = True
    attachment9 = "/media/HardDisk_A/Workers/goog/2022MathE/data/基本数据/附件9、径流量2012-2022年.xlsx"
    isAttachment9 = True
    attachment10 = "/media/HardDisk_A/Workers/goog/2022MathE/data/基本数据/附件10、叶面积指数（LAI）2012-2022年.xls"
    isAttachment10 = True
    attachment11 = "/media/HardDisk_A/Workers/goog/2022MathE/data/基本数据/附件11、一些历史数据.xls"
    isAttachment11 = False # 无

    ismerge = True
    
    # question3
    # attachment14 = r"/media/HardDisk_A/Workers/goog/2022MathE/data/监测点数据/附件14：内蒙古自治区锡林郭勒盟典型草原不同放牧强度土壤碳氮监测数据集（2012年8月15日-2020年8月15日）/内蒙古自治区锡林郭勒盟典型草原不同放牧强度土壤碳氮监测数据集（2012年8月15日-2020年8月15日）的副本.xlsx"
    attachment14 = r"/media/HardDisk_A/Workers/goog/2022MathE/data/监测点数据/附件14：内蒙古自治区锡林郭勒盟典型草原不同放牧强度土壤碳氮监测数据集（2012年8月15日-2020年8月15日）/内蒙古自治区锡林郭勒盟典型草原不同放牧强度土壤碳氮监测数据集（2012年8月15日-2020年8月15日）.xlsx"
    isAttachment14 = True

    by_intensity = False
    by_plot = True
    byLGI = False
    byMGI = False
    byHGI = False

    forecast_prophet = False
    forecast_prophet_mean = False
    forecast_visual = False

    forecast_sk = True
    sk_method = "SVR" # "StackingRegressor" #"BaggingRegressor" #"GradientBoostingRegressor" #"AdaBoostRegressor" #"RandomForestRegressor" #"ExtraTreeRegressor" # "DecisionTreeRegressor" #"MLP" # "KRR" # "SVR"
    sk_mode = "debug"

    forecast_tsfresh = False
    forecast_autots = False


