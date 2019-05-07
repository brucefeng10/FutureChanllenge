import pandas as pd
from gurobipy import *
from sklearn import linear_model
from sklearn.metrics import accuracy_score, classification_report
import csv


def predict_wind():
    rd = pd.read_csv(r'C:\Bee\ProjectFile\WeatherFlyRouteSeason2\rawdata\ForecastDataforTesting_201802.csv',
                     iterator=True)
    df = rd.get_chunk(210000000)
    # print df
    mean_wind = df['wind'].groupby([df['xid'], df['yid'], df['date_id'], df['hour']]).max()
    mean_rain = df['rainfall'].groupby([df['xid'], df['yid'], df['date_id'], df['hour']]).max()
    mean_wind.to_csv(r'C:\Bee\ProjectFile\WeatherFlyRouteSeason2\predict\wind_max_predict.csv')
    mean_rain.to_csv(r'C:\Bee\ProjectFile\WeatherFlyRouteSeason2\predict\rain_max_predict.csv')


# predict_wind()


def trsp():
    '''get the 10 values of models in row to columns'''
    rd = pd.read_csv(r'C:\Bee\ProjectFile\WeatherFlyRoute\rawdata\ForecastDataforTraining_201712.csv', iterator=True)

    df = rd.get_chunk(210000000)
    # trsp=df.pivot('xid','yid','date_id','hour')
    df.set_index(['xid', 'yid', 'date_id', 'hour', 'model'], inplace=True)
    df_trsp = df.unstack(level=4)

    # print df_trsp
    # print rv

    df_trsp.to_csv(r'C:\Bee\ProjectFile\WeatherFlyRoute\predict\training_wind10_transpose.csv', header=False)


# trsp()


def merge():
    rd = pd.read_csv(r'C:\Bee\ProjectFile\WeatherFlyRoute\predict\training_wind10_transpose.csv', header=None,
                     iterator=True)
    realval = pd.read_csv(r'C:\Bee\ProjectFile\WeatherFlyRoute\rawdata\In_situMeasurementforTraining_201712.csv',
                          iterator=True)
    df_trsp = rd.get_chunk()
    rv = realval.get_chunk()
    df_trsp.columns = ['xid', 'yid', 'date_id', 'hour', 'wind1', 'wind2', 'wind3', 'wind4', 'wind5', 'wind6', 'wind7',
                       'wind8', 'wind9', 'wind10']

    result = pd.merge(df_trsp, rv, how='inner', on=['xid', 'yid', 'date_id', 'hour'])
    print(result)

    result.to_csv(r'C:\Bee\ProjectFile\WeatherFlyRoute\predict\training_wind11_transpose.csv', index=False)


# merge()


def sample():
    rd = pd.read_csv(r'C:\Bee\ProjectFile\WeatherFlyRoute\predict\training_wind11_transpose.csv', iterator=True)
    df = rd.get_chunk(100000)
    df.to_csv(r'C:\Bee\ProjectFile\WeatherFlyRoute\predict\wind11_sample.csv')


# sample()


def lp():
    co1 = [366, 366, 200, 200, 200, 200, 200, 426, 192, 490, 192, 192, 384, 444, 1014, 1034, 362, 722, 128, 128, 296,
           128, 128, 238, 202, 646, 922, 202]
    co2 = [366, 366, 200, 200, 200, 200, 200, 456, 192, 490, 192, 192, 384, 444, 1014, 1034, 362, 722, 128, 128, 304,
           128, 186, 238, 202, 646, 922, 202]
    co3 = [366, 366, 200, 200, 258, 230, 200, 444, 194, 626, 196, 192, 390, 564, 378, 1042, 362, 722, 128, 158, 314,
           164, 186, 202, 202, 202, 202, 202]
    co4 = [0, 0, 200, 200, 200, 200, 200, 426, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 320, 128, 186, 238, 0, 0, 0, 0]
    coe1 = [[128, 348, 200, 342, 512, 398, 578, 1440, 1440, 1440],
            [128, 324, 200, 202, 1440, 366, 1440, 1440, 1440, 1440],
            [128, 192, 592, 202, 394, 366, 1440, 1440, 1440, 1440],
            [128, 246, 200, 362, 1440, 1440, 1440, 1440, 1440, 1440],
            [128, 192, 200, 1440, 1440, 1440, 1440, 1440, 1440, 1440]]
    coe2 = [[128, 348, 200, 342, 512, 398, 578, 1440, 1440, 1440],
            [128, 324, 200, 202, 1440, 366, 440, 1440, 1440, 1440], [128, 192, 592, 202, 394,
                                                                     366, 778, 1440, 1440, 1440],
            [1440, 1440, 1440, 1440, 1440, 1440, 1440, 1440, 1440, 1440],
            [128, 192, 200, 202, 362, 366, 440, 1440, 1440, 1440]]
    coe3 = [[0, 0, 200, 0, 610, 0, 608, 0, 0, 0], [0, 0, 200, 0, 0, 384, 636, 0, 0, 0], [0, 0, 0, 0, 0,
                                                                                         366, 966, 0, 0, 0],
            [0, 0, 0, 0, 674, 0, 598, 0, 0, 0], [0, 0, 200, 0, 362, 0, 440, 0, 0, 0]]
    coe4 = [[128, 348, 200, 342, 610, 398, 608, 0, 0, 0], [128, 324, 200, 202, 0, 384, 636, 0, 0, 0],
            [128, 192, 592, 202, 394, 366, 966, 0, 0, 0], [128, 246, 200, 362, 674, 0, 598, 0, 0, 0],
            [128, 192, 200, 202, 362, 366, 440, 0, 0, 0]]
    p, q = len(coe1), len(coe1[0])
    print(p, q)
    try:
        mod = Model('GetCrash')
        x = mod.addVars(p, q, vtype=GRB.BINARY, name='x')
        obj = 0
        for i in range(p):
            obj += quicksum(x[i, j] for j in range(q))
        # obj=x[2]+x[3]+x[4]+x[5]+x[18]+x[19]+x[21]+x[22]
        mod.setObjective(obj, GRB.MAXIMIZE)

        sum1, sum2 = 0, 0
        sum3, sum4 = 0, 0
        for i in range(p):
            sum1 += quicksum(x[i, j] * coe1[i][j] + 1440 * (1 - x[i, j]) for j in range(q))
            sum2 += quicksum(x[i, j] * coe2[i][j] + 1440 * (1 - x[i, j]) for j in range(q))
            sum3 += quicksum(x[i, j] * coe3[i][j] + 1440 * (1 - x[i, j]) for j in range(q))
            sum4 += quicksum(x[i, j] * coe4[i][j] + 1440 * (1 - x[i, j]) for j in range(q))
            # sumall+=quicksum(x[i,j] for j in range(q))
            # for k in range(7,q):
            #     mod.addConstr(x[i,k]==0)
        # mod.addConstr(sum1==48234)
        # mod.addConstr(sum2 == 50746)
        mod.addConstr(sum4 == 41962)
        # mod.addConstr(sum3==62172)
        # mod.addConstr(quicksum(x[i]*co3[i]+1440*(1-x[i]) for i in range(p))==72000)
        # mod.addConstr(quicksum(x[i] * co4[i] + 1440 * (1 - x[i]) for i in range(p)) == 60648 - 22 * 1440)

        mod.addConstr(x[0, 2] == 1)
        mod.addConstr(x[0, 4] == 1)
        mod.addConstr(x[0, 6] == 1)
        mod.addConstr(x[1, 2] == 1)
        mod.addConstr(x[1, 5] == 1)
        mod.addConstr(x[2, 5] == 1)
        mod.addConstr(x[2, 6] == 1)
        mod.addConstr(x[3, 6] == 1)
        mod.addConstr(x[4, 2] == 1)
        mod.addConstr(x[4, 6] == 1)

        mod.addConstr(x[4, 4] == 0)
        mod.addConstr(x[3, 4] == 0)
        mod.addConstr(x[1, 6] == 0)

        # mod.addConstr(x[0,3]==1)
        # mod.addConstr(x[0,5]==1)
        mod.addConstr(x[1, 1] == 1)

        for i in range(p):
            for j in range(q):
                if coe4[i][j] == 0:
                    mod.addConstr(x[i, j] == 0)

        mod.optimize()
        mod.printAttr('X')

    except GurobiError:
        print('Error reported')


# lp()


def linrrg():
    df = pd.read_csv(r'C:\Bee\ProjectFile\WeatherFlyRoute\predict\wind11_sample.csv')
    x = df[['wind1', 'wind2', 'wind3', 'wind4', 'wind5', 'wind6', 'wind7', 'wind8', 'wind9', 'wind10']]
    y = df['wind']
    linreg = linear_model.LinearRegression()
    linreg.fit(x, y)

    print(linreg.intercept_)
    print(linreg.coef_)
    pred = linreg.predict(x)
    f = open(r'C:\Bee\ProjectFile\WeatherFlyRoute\predict\linreg_pred.csv', 'wb')
    writer = csv.writer(f)
    for i in pred:
        # print i
        writer.writerow([i])
    f.close()


# linrrg()


def lp2():
    c = [342, 398, 324, 546, 246, 362, 604, 192, 362]
    l = len(c)
    try:
        mod = Model('test')
        x = mod.addVars(l, vtype=GRB.BINARY, name='x')
        obj = quicksum(x[i] for i in range(l))
        mod.setObjective(obj, GRB.MAXIMIZE)
        mod.addConstr(quicksum(c[i] * x[i] + 1440 * (1 - x[i]) for i in range(l)) == 68612 - 41 * 1440)
        mod.addConstr(x[0] == 0)
        mod.addConstr(x[1] == 0)
        mod.optimize()
        mod.printAttr('X')

    except GurobiError:
        print('error')


lp2()
