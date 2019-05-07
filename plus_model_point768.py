# -*- coding: utf-8 -*-

import numpy as np
import csv
import time
import pandas as pd
from gurobipy import *

t_beg = time.time()

# starttime=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
starttime = [45, 40, 35, 30, 25, 20, 15, 10, 5, 0]
# starttime6=[45,40,35,30,25,20,15,10,5,0]
pointsqus = [9, 3, 2, 10, 4, 1, 5, 7, 6, 8]
x_expand5 = [[110, 160], [120, 170], [130, 220], [130, 210], [135, 250], [70, 160], [135, 320], [280, 400], [280, 400],
             [290, 440]]
y_expand5 = [[300, 400], [220, 350], [310, 390], [260, 350], [230, 340], [190, 340], [270, 340], [200, 320], [180, 320],
             [230, 320]]
x_expand10 = [[110, 160], [120, 170], [130, 220], [130, 210], [135, 250], [70, 160], [135, 320], [180, 370], [180, 370],
              [185, 425]]
y_expand10 = [[300, 400], [220, 350], [310, 390], [260, 350], [230, 340], [190, 340], [270, 340], [220, 300],
              [200, 300], [250, 290]]
ps = 8

# t_st=starttime[ps]
t_st = 233
print('start time:', t_st)

points = [[142, 328], [84, 203], [199, 371], [140, 234], [236, 241], [315, 281], [358, 207], [363, 237], [423, 266],
          [125, 375], [189, 274]]
mt = 540  # time index
pn = pointsqus[ps]  # destination point index
print('point:', pn)
trs = 10
print('transfer:', trs)
date_id = 6  # date index
print('date:', date_id)
wind = 14.8  # predicted wind speed larger than this value is thought to cause crash
rain = 3.5

i0 = points[trs][0]
j0 = points[trs][1]
i1 = points[pn][0]
j1 = points[pn][1]
# if i0<=i1:
#     x_bd0=i0-1  # the bound of x
#     x_bd1=i1+1
#     xl=i1-i0+3
# else:
#     x_bd0=i1-1
#     x_bd1=i0+1
#     xl=i0-i1+3
# if j0<=j1:
#     y_bd0=j0-1
#     y_bd1=j1+1
#     yl=j1-j0+3
# else:
#     y_bd0=j1-1
#     y_bd1=j0+1
#     yl=j0-j1+3

x_bd0, x_bd1 = x_expand10[ps]
y_bd0, y_bd1 = y_expand10[ps]
print(x_bd0, x_bd1)
print(y_bd0, y_bd1)

t0 = "2018-01-24 03:00:00"
timeArray = time.strptime(t0, "%Y-%m-%d %H:%M:%S")
tint = int(time.mktime(timeArray))


def weather():
    alp = np.zeros([18, 549, 422])
    fl = open(r'C:\Bee\ProjectFile\WeatherFlyRouteSeason2\predict\wind_mean_predict.csv', 'rU')
    fd = csv.reader(fl)
    for val in fd:
        if int(val[2]) == date_id:
            if float(val[-1]) >= wind:
                alp[int(val[3]) - 3, int(val[0]), int(val[1])] = 1
    fl.close()
    fl2 = open(r'C:\Bee\ProjectFile\WeatherFlyRouteSeason2\predict\rain_mean_predict.csv', 'rU')
    fd2 = csv.reader(fl2)
    for val in fd2:
        if int(val[2]) == date_id:
            if float(val[-1]) >= rain:
                alp[int(val[3]) - 3, int(val[0]), int(val[1])] = 1
    fl2.close()
    return alp


print('Finished reading data, Gurobi is running the model...')


def flyroute():
    try:
        alp = weather()
        print('starting weather:', alp[t_st / 30, 142, 328])
        alp[t_st / 30, 142, 328] = 0.
        mod = Model('path')
        # x=mod.addVars(mt,xl,yl,vtype=GRB.BINARY,name='x')
        x = {}
        for t in range(t_st, mt):
            for i in range(x_bd0, x_bd1 + 1):
                for j in range(y_bd0, y_bd1 + 1):
                    x[t, i, j] = mod.addVar(vtype=GRB.BINARY, name="x[%s][%s][%s]" % (t, i, j))
        # s=mod.addVars(mt,vtype=GRB.BINARY,name='s')

        obj = 18 * 60 - t_st * 2 - quicksum(2 * x[t, i1, j1] for t in range(t_st, mt))
        mod.setObjective(obj, GRB.MINIMIZE)

        # adding constraints
        # const1
        for t in range(1 + t_st, mt):
            for i in range(x_bd0 + 1, x_bd1):
                for j in range(y_bd0 + 1, y_bd1):
                    mod.addConstr(
                        x[t, i, j] <= x[t - 1, i, j] + x[t - 1, i - 1, j] + x[t - 1, i + 1, j] + x[t - 1, i, j - 1] + x[
                            t - 1, i, j + 1])
                mod.addConstr(
                    x[t, i, y_bd0] <= x[t - 1, i, y_bd0] + x[t - 1, i - 1, y_bd0] + x[t - 1, i + 1, y_bd0] + x[
                        t - 1, i, y_bd0 + 1])
                mod.addConstr(
                    x[t, i, y_bd1] <= x[t - 1, i, y_bd1] + x[t - 1, i - 1, y_bd1] + x[t - 1, i + 1, y_bd1] + x[
                        t - 1, i, y_bd1 - 1])

            for j in range(y_bd0 + 1, y_bd1):
                mod.addConstr(
                    x[t, x_bd0, j] <= x[t - 1, x_bd0, j] + x[t - 1, x_bd0, j - 1] + x[t - 1, x_bd0, j + 1] + x[
                        t - 1, x_bd0 + 1, j])
                mod.addConstr(
                    x[t, x_bd1, j] <= x[t - 1, x_bd1, j] + x[t - 1, x_bd1, j - 1] + x[t - 1, x_bd1, j + 1] + x[
                        t - 1, x_bd1 - 1, j])

            mod.addConstr(
                x[t, x_bd0, y_bd0] <= x[t - 1, x_bd0, y_bd0] + x[t - 1, x_bd0 + 1, y_bd0] + x[t - 1, x_bd0, y_bd0 + 1])
            mod.addConstr(
                x[t, x_bd0, y_bd1] <= x[t - 1, x_bd0, y_bd1] + x[t - 1, x_bd0 + 1, y_bd1] + x[t - 1, x_bd0, y_bd1 - 1])
            mod.addConstr(
                x[t, x_bd1, y_bd0] <= x[t - 1, x_bd1, y_bd0] + x[t - 1, x_bd1 - 1, y_bd0] + x[t - 1, x_bd1, y_bd0 + 1])
            mod.addConstr(
                x[t, x_bd1, y_bd1] <= x[t - 1, x_bd1, y_bd1] + x[t - 1, x_bd1 - 1, y_bd1] + x[t - 1, x_bd1, y_bd1 - 1])

        # const2
        for t in range(t_st, mt):
            sum1 = 0
            for i in range(x_bd0, x_bd1 + 1):
                sum1 += quicksum(x[t, i, j] for j in range(y_bd0, y_bd1 + 1))
            mod.addConstr(sum1 == 1)

        # const3
        for t in range(t_st, mt):
            for i in range(x_bd0, x_bd1 + 1):
                for j in range(y_bd0, y_bd1 + 1):
                    if i != i1 or j != j1:
                        mod.addConstr(x[t, i, j] <= 1 - alp[
                            t / 30, i, j])  # do not know how to keep vehicle from crash on arrival

        mod.addConstr(x[t_st, i0, j0] == 1)  # the vehicle starts from (i0,j0)
        mod.addConstr(
            quicksum(x[t, i1, j1] for t in range(t_st, mt)) >= 1)  # a vehicle has to arrive at the destination

        mod.Params.TimeLimit = 1000
        # mod.Params.MIPFocus=1
        # mod.Params.ImproveStartGap=0.04
        mod.optimize()

        print('Objective: ', mod.objVal)
        mod.printAttr('X')

        def resultout():
            fout = open(r'C:\Bee\ProjectFile\WeatherFlyRouteSeason2\trans_0208_%d_%d_%d.csv' % (date_id, pn, t_st),
                        'ab')
            writer = csv.writer(fout)
            for v in mod.getVars():
                if v.x != 0:
                    print(v.varName)
                    out = v.varName
                    out = out.strip('x[')
                    out = out.strip(']')
                    out = out.split('][')
                    outr = [pn, date_id, 0, int(out[1]), int(out[2])]
                    tint1 = tint + 60 * 2 * int(out[0])
                    timeArray1 = time.localtime(tint1)
                    outr[2] = time.strftime("%H:%M", timeArray1)
                    writer.writerow(outr)
                    if '[%d][%d]' % (i1, j1) in v.varName:
                        break
            fout.close()

        resultout()
        print('Successful!')
        '''This following part requires that only when the vehicle gets to the destination can we output the result'''
        # ind=False  # indicate if the vehicle arrived
        # for vv in mod.getVars():
        #     if vv.x !=0:
        #         if '[%d][%d]' % (i1, j1) in vv.varName:
        #             resultout()
        #             ind=True
        #             break
        # if ind==False:
        #     print 'Not arrived'



    except GurobiError:
        print('Error Reported')


flyroute()

t_end = time.time()
print('Total time: ', t_end - t_beg)
