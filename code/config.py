import os
from tabnanny import verbose
from tkinter.dnd import DndHandler

from sqlalchemy import false
from Transformer import PreTransformer
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.rnn import LSTM
import keras_nlp
from keras import backend as K


class Config(object):
    out_dir = "/media/HardDisk_A/Workers/goog/2022MathE/train_data"

    attachment3 = "/media/HardDisk_A/Workers/goog/2022MathE/data/基本数据/附件3、土壤湿度2022—2012年.xls"
    isAttachment3 = True
    attachment4 = "/media/HardDisk_A/Workers/goog/2022MathE/data/基本数据/附件4、土壤蒸发量2012—2022年.xls"
    isAttachment4 = True
    attachment5 = "/media/HardDisk_A/Workers/goog/2022MathE/data/基本数据/附件5、绿植覆盖率（2020-2022）"
    attachment5_dir = "attachment5"
    isAttachment5 = True
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

    #
    by_intensity = False
    by_plot = False
    byLGI = False
    byMGI = False
    byHGI = False


    #
    forecast_prophet = False
    forecast_prophet_mean = False
    forecast_visual = False

    #
    forecast_sk = False
    sk_method = "KRR" 
    load_interpolate_data = "newton"
    sk_interpolate_points = 5+8
    sk_window = 4
    # "StackingRegressor" #"BaggingRegressor" #"GradientBoostingRegressor" 
    # "AdaBoostRegressor" #"RandomForestRegressor" #"ExtraTreeRegressor" 
    # "DecisionTreeRegressor" #"MLP" # "KRR" # "SVR"
    sk_mode = "debug0"
    sk_valid = False

    #
    forecast_tsfresh = False
    forecast_autots = False

    #
    forecast_sm = False
    






    #
    interpolate = False
    interpolate_method = "Pchip"
    # lagrange = True
    # newton = True
    # Pchip
    interpolate_save = "interpolate"
    interpolate_internal = 5+0 # 0 4 8 12
    generate_result = True

    #
    forecast_lstm_pre = True
    attachment = "/media/HardDisk_A/Workers/goog/2022MathE/code/tmp.csv"


    #
    forecast_lstm = False
    lstm_method = "lstm" 
    lstm_get2020_result = False
    # transformer
    # lstm
    load_interpolate_data = "Pchip"
    interpolate_points = 5 + 4
    window = 4
    lstm_mode = "debug"
    lstm_valid = False


    #
    visual_by_year = False # 每年变化单调
    visual_by_year_dir = "visual_by_year"


    # 15 可视化
    visual15 = False
    first = False
    visual15_dir = "visual15"
    attachment15 = "/media/HardDisk_A/Workers/goog/2022MathE/data/监测点数据/附件15：内蒙古自治区锡林郭勒盟典型草原轮牧放牧样地群落结构监测数据集（2016年6月-2020年9月）。/内蒙古自治区锡林郭勒盟典型草原轮牧放牧样地群落结构监测数据集（201.xlsx"


    # 问题5
    question5 = False
    question5_lstm_get2020_result = True


    # last
    last = True
    last_dir = "last"
    last_visual = True



    @staticmethod
    def mkdir(path):
        outdir = os.path.join(Config.out_dir, path)
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        return outdir
    
    @staticmethod
    def savefig(fig, path, name):
        fig.savefig(os.path.join(path, name), dpi=100)

    @staticmethod
    def inverse_transform_col(obj, y, n_col):
        '''scaler是对包含多个feature的X拟合的,y对应其中一个feature,n_col为y在X中对应的列编号.返回y的反归一化结果'''
        y = y.copy()
        y -= obj.min_[n_col]
        y /= obj.scale_[n_col]
        return y
    @staticmethod
    def build_model(name, input):
        import keras
        from keras.layers import Reshape
        if name == "transformer":
            # Create a single transformer encoder layer.
            encoder = keras_nlp.layers.TransformerEncoder(
                intermediate_dim=64, num_heads=8)

            # Create a simple model containing the encoder.
            input = keras.Input(shape=[input.shape[1], input.shape[2]])
            b, s, h = input.shape
            x = Reshape((s, h))(input)
            x = Dense(100, activation='relu')(x)
            x = Reshape((s, -1))(x)
            x = encoder(x)
            
            x = Dense(input.shape[2], activation='linear')(x)
            print(x.shape)
            model = keras.Model(inputs=input, outputs=x)
            model.compile( loss="mse", optimizer="adam")
            return model
        elif name == "lstm":
            model = Sequential()
            model.add( LSTM( 100, input_shape=(input.shape[1], input.shape[2]), return_sequences=True) )
            model.add( LSTM( 50, return_sequences=False ) )
            model.add( Dropout( 0.3 ) )
            model.add( Dense( 4 ) )
            model.add( Activation( 'linear' ) )
            model.compile( loss="mse", optimizer="adam" )
            return model
