# code

## Q1

### Q1.py

```python
import pandas as pd
from getFurnacetemperature import *

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

solder_thickness = 0.015
delta_t = 0.5

def Matrix_A(m, r, h): # 计算左侧的方形矩阵
    delta_z = solder_thickness / m
    A = np.diag([2 * (1 + r)] * (m + 1)) + np.diag([-r] * m, k=1) + np.diag([-r] * m, k=-1)
    A[0][0] = 1 + h * delta_z
    A[0][1] = -1
    A[-1][-2] = -1
    A[-1][-1] = 1 + h * delta_z
    return A

def Matrix_B(m, r): # 计算右侧的方形矩阵
    B = np.diag([2 * (1 - r)] * (m + 1)) + np.diag([r] * m, k=1) + np.diag([r] * m, k=-1)
    B[0][:] = 0
    B[-1][:] = 0
    return B

def CN(m = 100, # 焊接区域厚度的切割份数
       a = np.array([6.683e-4, 8.076e-4, 9.744e-4, 8.493e-4, 5.287e-4]), # 各阶段的a
       h = np.array([24987, 1432, 827.9, 654.8, 1338]), # 各阶段的h
       v = 70/60,# 速度
       draw = 0,# 是否绘图
       temp=[175, 195, 235, 255]):
    delta_z = solder_thickness / m
    r = a * a * delta_t / (delta_z * delta_z)
    # 初值
    right_u = np.array([25] * (m+1))
    time = 0
    t = np.array([0])
    u = np.array([25])
    # 小温区1-5
    while (time * v) < (front_len + 5*furance_len + 4.5*interval_len):
        # 计算右侧的单列矩阵
        right_b = np.array([.0] * (m + 1))
        right_b[0] = h[0] * getFurnacetemperature((time + delta_t) * v,temp) * delta_z
        right_b[-1] = h[0] * getFurnacetemperature((time + delta_t) * v,temp) * delta_z
        left_u = np.linalg.inv(Matrix_A(m, r[0], h[0])) @ (Matrix_B(m, r[0],) @ right_u + right_b)
        time = time + delta_t
        t = np.append(t, time)
        u = np.append(u, left_u[int(m / 2)])
        right_u = left_u
    # 小温区6
    while (time * v) < (front_len + 6*furance_len + 5.5*interval_len):
        # 计算右侧的单列矩阵
        right_b = np.array([.0] * (m + 1))
        right_b[0] = h[1] * getFurnacetemperature((time + delta_t) * v,temp) * delta_z
        right_b[-1] = h[1] * getFurnacetemperature((time + delta_t) * v,temp) * delta_z
        left_u = np.linalg.inv(Matrix_A(m, r[1], h[1])) @ (Matrix_B(m, r[1],) @ right_u + right_b)
        time = time + delta_t
        t = np.append(t, time)
        u = np.append(u, left_u[int(m / 2)])
        right_u = left_u
    # 小温区7
    while (time * v) < (front_len + 7*furance_len + 6.5*interval_len):
        # 计算右侧的单列矩阵
        right_b = np.array([.0] * (m + 1))
        right_b[0] = h[2] * getFurnacetemperature((time + delta_t) * v,temp) * delta_z
        right_b[-1] = h[2] * getFurnacetemperature((time + delta_t) * v,temp) * delta_z
        left_u = np.linalg.inv(Matrix_A(m, r[2], h[2])) @ (Matrix_B(m, r[2],) @ right_u + right_b)
        time = time + delta_t
        t = np.append(t, time)
        u = np.append(u, left_u[int(m / 2)])
        right_u = left_u
    # 小温区8-9
    while (time * v) < (front_len + 9*furance_len + 8.5*interval_len):
        # 计算右侧的单列矩阵
        right_b = np.array([.0] * (m + 1))
        right_b[0] = h[3] * getFurnacetemperature((time + delta_t) * v,temp) * delta_z
        right_b[-1] = h[3] * getFurnacetemperature((time + delta_t) * v,temp) * delta_z
        left_u = np.linalg.inv(Matrix_A(m, r[3], h[3])) @ (Matrix_B(m, r[3],) @ right_u + right_b)
        time = time + delta_t
        t = np.append(t, time)
        u = np.append(u, left_u[int(m / 2)])
        right_u = left_u
    # 小温区10-11
    while (time * v) <= (2*front_len + 11*furance_len + 10*interval_len):
        # 计算右侧的单列矩阵
        right_b = np.array([.0] * (m + 1))
        right_b[0] = h[4] * getFurnacetemperature((time + delta_t) * v,temp) * delta_z
        right_b[-1] = h[4] * getFurnacetemperature((time + delta_t) * v,temp) * delta_z
        left_u = np.linalg.inv(Matrix_A(m, r[4], h[4])) @ (Matrix_B(m, r[4],) @ right_u + right_b)
        time = time + delta_t
        t = np.append(t, time)
        u = np.append(u, left_u[int(m / 2)])
        right_u = left_u
    if(draw):
        return np.array([t, u])
    else:
        sum=0
        tmp=0
        experiment_temperature=np.array(pd.read_csv('附件.csv'))
        for index in range(experiment_temperature.T[1].size):
            while np.array([t,u])[0][tmp]<19:
                tmp+=1
            sum+=(np.array([t, u])[1][tmp+index]-experiment_temperature[index][1])**2
        return sum

def pso():
    num_to_optimize=5
    N=20 # 种群个数
    cnt_max=40 # 最大迭代次数
    xlimit_a=[0.0001,0.001] # a位置界限
    vlimit_a=[-0.0002,0.0002] # a速度界限
    
    xlimit_h=[3000,12000] # h位置界限
    vlimit_h=[-1300,1300] # h速度界限

    # 初始化种群
    x_a=np.array([
        np.random.uniform(low=xlimit_a[0], high=xlimit_a[1], size=N),
        np.random.uniform(low=xlimit_a[0], high=xlimit_a[1], size=N),
        np.random.uniform(low=xlimit_a[0], high=xlimit_a[1], size=N),
        np.random.uniform(low=xlimit_a[0], high=xlimit_a[1], size=N),
        np.random.uniform(low=xlimit_a[0], high=xlimit_a[1], size=N)
    ])
    x_h=np.array([
        np.random.uniform(low=xlimit_h[0], high=xlimit_h[1], size=N),
        np.random.uniform(low=xlimit_h[0], high=xlimit_h[1], size=N),
        np.random.uniform(low=xlimit_h[0], high=xlimit_h[1], size=N),
        np.random.uniform(low=xlimit_h[0], high=xlimit_h[1], size=N),
        np.random.uniform(low=xlimit_h[0], high=xlimit_h[1], size=N)
    ])
    v_a=np.array([
        np.random.uniform(low=-vlimit_a[0], high=vlimit_a[1], size=N),
        np.random.uniform(low=-vlimit_a[0], high=vlimit_a[1], size=N),
        np.random.uniform(low=-vlimit_a[0], high=vlimit_a[1], size=N),
        np.random.uniform(low=-vlimit_a[0], high=vlimit_a[1], size=N),
        np.random.uniform(low=-vlimit_a[0], high=vlimit_a[1], size=N)
    ])
    v_h=np.array([
        np.random.uniform(low=-vlimit_h[0], high=vlimit_h[1], size=N),
        np.random.uniform(low=-vlimit_h[0], high=vlimit_h[1], size=N),
        np.random.uniform(low=-vlimit_h[0], high=vlimit_h[1], size=N),
        np.random.uniform(low=-vlimit_h[0], high=vlimit_h[1], size=N),
        np.random.uniform(low=-vlimit_h[0], high=vlimit_h[1], size=N)
    ])

    w=np.linspace(1.1,0.3,cnt_max) # 惯性系数
    w=np.append(w,np.linspace(0.6,0.6,20))
    cnt_max+=20
    c1=0.6 # 个体学习参数
    c2=0.6 # 全局学习系数
    
    # 个体历史最优
    individual_best=np.zeros(x_a[0].size)
    for index in range(individual_best.shape[0]):
        individual_best[index]=np.copy(CN(100,x_a.T[index],x_h.T[index],70/60))
    # 个体历史最优位置
    individual_best_a=np.copy(x_a)
    individual_best_h=np.copy(x_h)
    # 全局最优解
    global_best=1000000
    # 全局历史最优位置
    global_best_a=np.zeros(num_to_optimize)
    global_best_h=np.zeros(num_to_optimize)

    # 初始化全局最优
    for index in range(x_a[0].size):
        temp=CN(100,x_a.T[index],x_h.T[index],70/60)
        individual_best[index]=temp
        if(global_best > temp):
            global_best_a=np.copy(x_a.T[index])
            global_best_h=np.copy(x_h.T[index])
            global_best=temp
            
    # 进行迭代
    for cnt in range(cnt_max):
        print("*******")
        print("迭代次数: ",cnt+1)
        print("*******")
        r1=np.array(np.random.rand(num_to_optimize))
        r2=np.array(np.random.rand(num_to_optimize))
        for index in range(x_a[0].size):
            # 更新a,v速度
            va_temp=np.copy(w[cnt]*v_a.T[index] + c1*r1*(individual_best_a.T[index]-x_a.T[index]) + c2*r2*(global_best_a-x_a.T[index]))
            vh_temp=np.copy(w[cnt]*v_h.T[index] + c1*r1*(individual_best_h.T[index]-x_h.T[index]) + c2*r2*(global_best_h-x_h.T[index]))
            if not ((va_temp >= vlimit_a[0]).all() and (va_temp <= vlimit_a[1]).all()):
                for pos in range(va_temp.size):
                    if va_temp[pos]>vlimit_a[1]:
                        va_temp[pos]=vlimit_a[1]
                    elif va_temp[pos]<vlimit_a[0]:
                        va_temp[pos]=vlimit_a[0]
            v_a.T[index]=np.copy(va_temp)
            if not ((vh_temp >= vlimit_h[0]).all() and (vh_temp <= vlimit_h[1]).all()):
                for pos in range(vh_temp.size):
                    if vh_temp[pos]>vlimit_h[1]:
                        vh_temp[pos]=vlimit_h[1]
                    elif vh_temp[pos]<vlimit_h[0]:
                        vh_temp[pos]=vlimit_h[0]
            v_h.T[index]=np.copy(vh_temp)
            # 更新a,v位置
            if (x_a.T[index]+v_a.T[index]>=xlimit_a[0]).all() and (x_a.T[index]+v_a.T[index]<=xlimit_a[1]).all() and (x_h.T[index]+v_h.T[index]>=xlimit_h[0]).all() and (x_h.T[index]+v_h.T[index]<=xlimit_h[1]).all():
                # 更新个体历史最优
                x_a.T[index] = np.copy(x_a.T[index]+v_a.T[index])
                x_h.T[index] = np.copy(x_h.T[index]+v_h.T[index])
                tmp = CN(100,x_a.T[index],x_h.T[index],70/60)
                if individual_best[index] > tmp: # 只有优于个体历史最佳时才进行更新
                    individual_best_a.T[index] = np.copy(x_a.T[index])
                    individual_best_h.T[index] = np.copy(x_h.T[index])
                    individual_best[index] = tmp
                # 更新全局历史最优
                if global_best > tmp:
                    print("*******")
                    print("发现全局更优解:",tmp)
                    print("*******")
                    global_best_a = np.copy(x_a.T[index])
                    global_best_h = np.copy(x_h.T[index])
                    global_best = tmp
         
            print("a:",global_best_a)
            print("h:",global_best_h)
            print(global_best)   
    return global_best_a,global_best_h,global_best

if __name__ == "__main__":
    a,h,globalbest=pso()
    test = CN(100, a, h, 70/60, 1,)
    test = test.T
    u = pd.read_csv('附件.csv')
    u = np.array(u)

    plt.figure(figsize=(14, 6))
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(121)
    plt.plot(test[38:-2, 0], test[38:-2, 1], label='模型温度')
    plt.plot(u[:, 0],u[:, 1], label='实验温度')
    plt.legend()
    plt.subplot(122)
    delta = test[38:-1, 1] - u[:, 1]
    F = sum(delta*delta)
    plt.plot(u[:, 0], delta, label='F={}'.format(F))
    plt.legend()
    plt.show()
```

### getFurnacetemperature.py

```python
import pylab as plt
import numpy as np

furance_len = 30.5  # 小温区长度
interval_len = 5  # 间隙长度
front_len = 25  # 炉前区域

def getFurnacetemperature(x, temp=[175, 195, 235, 255]):
    # 炉外区域
    if x <= 0:
        return 25
    # 炉前区域
    elif x <= front_len:
        return 25 + (temp[0]-25)/25 * x
    # 小温区1-5
    elif x <= front_len + 5*furance_len+4*interval_len:
        return temp[0]
    # 过度区域
    elif x <= front_len + 5*(furance_len+interval_len):
        return temp[0] + ((temp[1] - temp[0])/interval_len) * (x-(25 + 5*furance_len+4*interval_len))
    # 小温区6
    elif x <= front_len + 6*furance_len+5*interval_len:
        return temp[1]
    # 过度区域
    elif x <= front_len + 6*(furance_len+interval_len):
        return temp[1] + ((temp[2] - temp[1])/interval_len) * (x-(25 + 6*furance_len+5*interval_len))
    # 小温区7
    elif x <= front_len + 7*furance_len + 6*interval_len:
        return temp[2]
    # 过度区域
    elif x <= front_len + 7*(furance_len+interval_len):
        return temp[2] + ((temp[3] - temp[2])/interval_len) * (x-(25 + 7*furance_len + 6*interval_len))
    # 小温区8-9
    elif x <= front_len + 9*furance_len+8*interval_len:
        return temp[3]
    # 小温区10-11
    elif x <= front_len + 9*(furance_len+interval_len):
        return temp[3] + ((25 - temp[3])/interval_len) * (x-(25 + 9*furance_len+8*interval_len))
    # 炉外区域
    else:
        return 25

if __name__ == "__main__":
    x = np.arange(0, 2*front_len + 11*furance_len + 10*interval_len, 0.1)
    y = np.array([])
    for v in x:
        y = np.append(y, getFurnacetemperature(v))
    plt.plot(x, y, label='炉内温度')
    plt.show()
```

## Q2

### Q2.py
```python
from Q2peakTemperature import *
from Q2slope import *
from Q2temperature150To190 import *
from Q2temperature217 import *

def getmax_v(temp=[173, 198, 230, 257]):
    v_result=np.zeros(4)

    # 最大斜率
    max_slope=3
    # 传送带速度范围
    v_range=[65/60,100/60]

    # 当前速度
    mid=(v_range[0]+v_range[1])/2
    # 最大迭代次数
    cnt_max=6

    # 已知a,h
    a= np.array([0.00067445 ,0.0007394 , 0.00088138 ,0.00071155 ,0.00050576])
    h= np.array([9351.90193919 , 6419.99175218, 11016.59502516 , 9347.90126533,6863.18027028])

    def dichotomy_slope(mid):
        cnt=0
        high=v_range[1]
        low=v_range[0]
        while cnt<cnt_max:
            # print("迭代次数:",cnt+1)
            # print("第",cnt+1,"次mid:",mid)
            if max_slope == Maxslope(mid):
                return mid
            if max_slope > Maxslope(mid):# 加判断条件
                low=mid
                mid=(mid+high)/2
            else:
                high=mid
                mid=(mid+low)/2
            cnt+=1
        # print("+++++++++++++=")
        return mid

    def dichotomy_peak(mid):
        cnt=0
        high=v_range[1]
        low=v_range[0]
        while cnt<cnt_max:
            # print("迭代次数:",cnt+1)
            # print("第",cnt+1,"次mid:",mid)
            if PeakTemperature(mid):# 加判断条件
                low=mid
                mid=(mid+high)/2
            else:
                high=mid
                mid=(mid+low)/2
            cnt+=1
        # print("+++++++++++++=")
        return mid

    def dichotomy_217(mid):
        cnt=0
        high=v_range[1]
        low=v_range[0]
        while cnt<cnt_max:
            # print("迭代次数:",cnt+1)
            # print("第",cnt+1,"次mid:",mid)
            if Temperature217(mid):# 加判断条件
                low=mid
                mid=(mid+high)/2
            else:
                high=mid
                mid=(mid+low)/2
            cnt+=1
        # print("+++++++++++++=")
        return mid

    def dichotomy_150to190(mid):
        cnt=0
        high=v_range[1]
        low=v_range[0]
        while cnt<cnt_max:
            # print("迭代次数:",cnt+1)
            # print("第",cnt+1,"次mid:",mid)
            if Temperature150To190(mid):# 加判断条件
                low=mid
                mid=(mid+high)/2
            else:
                high=mid
                mid=(mid+low)/2
            cnt+=1
        # print("+++++++++++++=")
        return mid

    v_result[0]=dichotomy_slope(mid)
    v_result[1]=dichotomy_peak(mid)
    v_result[2]=dichotomy_217(mid)
    v_result[3]=dichotomy_150to190(mid)
    return np.min(v_result)

if __name__ == "__main__":
   print(getmax_v([182, 203, 237, 254]))
```

### Q2slope.py

```python
from Q1 import *

def Maxslope(speed, temp=[182, 203, 237, 254]):
    maxslope=0
    a= np.array([0.00067445 ,0.0007394 , 0.00088138 ,0.00071155 ,0.00050576])
    h= np.array([9351.90193919 , 6419.99175218, 11016.59502516 , 9347.90126533,6863.18027028])
    ut=CN(100,a,h,speed,1,temp)
    ut=ut.T
    for index in range(ut.T[0].size-1):
        if maxslope < np.abs(ut[index+1][1]-ut[index][1])/0.5:
            maxslope = np.abs(ut[index+1][1]-ut[index][1])/0.5
    return maxslope

if __name__ == "__main__":
    v = np.linspace(65/60, 100/60, 200)
    slope = np.array([])
    for i in range(len(v)):
        slope = np.append(slope, Maxslope(v[i]))
        print(i)
    plt.figure()
    plt.plot(v, slope, label='')
    plt.show()
```

### Q2peakTemperature.py

```python
from Q1 import *

def PeakTemperature(speed, temp=[182, 203, 237, 254]):
    a= np.array([0.00067445 ,0.0007394 , 0.00088138 ,0.00071155 ,0.00050576])
    h= np.array([9351.90193919 , 6419.99175218, 11016.59502516 , 9347.90126533,6863.18027028])
    ut=CN(100,a,h,speed,1,temp)
    u = ut[1]
    # return np.max(u)s
    if np.max(u) >= 240:
        return 1
    else:
        return 0

if __name__ == "__main__":
    v = np.linspace(65/60, 100/60, 200)
    slope = np.array([])
    for i in range(len(v)):
        slope = np.append(slope, PeakTemperature(v[i]))
        print(i)
    plt.figure()
    plt.plot(v, slope, label='')
    plt.show()
```

### Q2temperature217.py

```python
from Q1 import *

def Temperature217(speed, temp=[182, 203, 237, 254]):
    a= np.array([0.00067445 ,0.0007394 , 0.00088138 ,0.00071155 ,0.00050576])
    h= np.array([9351.90193919 , 6419.99175218, 11016.59502516 , 9347.90126533,6863.18027028])
    ut=CN(100,a,h,speed,1,temp)
    u = ut[1]
    a = u[u > 217]
    b = len(a)
    # return b*0.5
    if b*0.5 >= 40:
        return 1
    else:
        return 0

if __name__ == "__main__":
    v = np.linspace(65/60, 100/60, 200)
    slope = np.array([])
    for i in range(len(v)):
        slope = np.append(slope, Temperature217(v[i]))
        print(i)
    plt.figure()
    plt.plot(v, slope, label='')
    plt.show()
```

### Q2temperature150To190.py

```python
from Q1 import *

def Temperature150To190(speed, temp=[182, 203, 237, 254]):
    a= np.array([0.00067445 ,0.0007394 , 0.00088138 ,0.00071155 ,0.00050576])
    h= np.array([9351.90193919 , 6419.99175218, 11016.59502516 , 9347.90126533,6863.18027028])
    ut=CN(100,a,h,speed,1,temp)
    u = ut[1]
    diff_u = np.diff(u)
    a = u[1:][diff_u > 0]
    a = a[(a > 150) & (a < 190)]
    b = len(a)
    # return b*0.5
    if b * 0.5 >= 60:
        return 1
    else:
        return 0


if __name__ == "__main__":
    v = np.linspace(65/60, 100/60, 200)
    slope = np.array([])
    for i in range(len(v)):
        slope = np.append(slope, Temperature150To190(v[i]))
        print(i)
    plt.figure()
    plt.plot(v, slope, label='')
    plt.show()
```
## Q3

### Q3.py

```python
from Q1 import *
from Q3area import *
from Q3judge import *

a= np.array([0.00067445 ,0.0007394 , 0.00088138 ,0.00071155 ,0.00050576])
h= np.array([9351.90193919 , 6419.99175218, 11016.59502516 , 9347.90126533,6863.18027028])

def q3_pso():
    N=5 # 种群个数
    cnt_max=20 # 最大迭代次数
    # 位置界限
    xlimit=np.array([
        [165,185],
        [185,205],
        [225,245],
        [245,265],
        [65/60,90/60]# 为缩短时间将上限减小至90cm/min
    ])
    # 速度界限
    vlimit=np.array([
        [-3,3],
        [-3,3],
        [-3,3],
        [-3,3],
        [-5/60,5/60]
    ])
    
    # 全局最优
    global_best=10000000
    # 全局最优位置
    global_best_x=np.zeros(5)
    # 初始化位置
    x=np.array([
        np.random.uniform(low=xlimit[0][0], high=xlimit[0][1], size=N),
        np.random.uniform(low=xlimit[1][0], high=xlimit[1][1], size=N),
        np.random.uniform(low=xlimit[2][0], high=xlimit[2][1], size=N),
        np.random.uniform(low=xlimit[3][0], high=xlimit[3][1], size=N),
        np.random.uniform(low=xlimit[4][0], high=xlimit[4][1], size=N),
    ]).T
    # 初始化速度
    v=np.array([
        np.random.uniform(low=vlimit[0][0], high=vlimit[0][1], size=N),
        np.random.uniform(low=vlimit[1][0], high=vlimit[1][1], size=N),
        np.random.uniform(low=vlimit[2][0], high=vlimit[2][1], size=N),
        np.random.uniform(low=vlimit[3][0], high=vlimit[3][1], size=N),
        np.random.uniform(low=vlimit[4][0], high=vlimit[4][1], size=N),
    ]).T
    
    # 个人历史最佳
    individual_best=np.zeros(N)
    # 个人历史最佳位置
    individual_best_x=np.copy(x)
    
    # 初始化全局最优
    for index in range(N):
        print("初始化:",index+1)
        ut=CN(100,a,h.T,x[index][4],1,np.array([x[index][0],x[index][1],x[index][2],x[index][3]]))
        # 求是否满足第二问条件
        limit_ran=0
        while not judge(ut):
            # 求出该温度下的上限与下限
            if limit_ran >= 1:
                x[index][0]=np.random.uniform(low=xlimit[0][0], high=xlimit[0][1], size=1)[0]
                x[index][1]=np.random.uniform(low=xlimit[1][0], high=xlimit[1][1], size=1)[0]
                x[index][2]=np.random.uniform(low=xlimit[2][0], high=xlimit[2][1], size=1)[0]
                x[index][3]=np.random.uniform(low=xlimit[3][0], high=xlimit[3][1], size=1)[0]
                x[index][4]=np.random.uniform(low=xlimit[4][0], high=xlimit[4][1], size=1)[0]
                limit_ran=0
            else:
                print("寻找上下限...")
                vrange_temp=getRange(np.array([x[index][0],x[index][1],x[index][2],x[index][3]]))
                if vrange_temp[0]>vrange_temp[1]:
                    print("该温度下符合条件速度不存在!")
                    limit_ran+=1
                    continue
                x[index][4]=np.random.uniform(low=vrange_temp[0], high=vrange_temp[1], size=1)[0]
                ut=CN(100,a,h.T,x[index][4],1,np.array([x[index][0],x[index][1],x[index][2],x[index][3]]))
                limit_ran+=1
        temp=Area(ut)
        # 更新个体最优
        individual_best[index]=temp
        # 更新全局最优
        if global_best>temp:
            global_best=temp
            global_best_x=np.copy(x[index])
        
    w=np.linspace(1.0,0.3,cnt_max) # 惯性系数
    w=np.append(w,np.linspace(0.6,0.6,10))
    cnt_max+=10
    c1=0.5 # 个体学习参数
    c2=0.6 # 全局学习系数
    
    for cnt in range(cnt_max):
        print("*******")
        print("迭代次数:",cnt+1) 
        print("*******")
        r1=np.array(np.random.rand(5))
        r2=np.array(np.random.rand(5))
        # 对每个粒子进行更新
        for index in range(N):
            # 更新a，v速度
            v_temp=np.copy(w[cnt]*v[index]+c1*r1*(individual_best_x[index]-x[index])+c2*r2*(global_best_x-x[index]))
            #判断该速度是否在界限内并且符合该温度设定下的传送速度范围
            if not((v_temp<vlimit.T[1]).all() and (v_temp>vlimit.T[0]).all()):
                for pos in range(v_temp.size):
                    if v_temp[pos]>vlimit[pos][1]:
                        v_temp[pos]=np.copy(vlimit[pos][1])
                    elif v_temp[pos]<vlimit[pos][0]:
                        v_temp[pos]=np.copy(vlimit[pos][0])
            v[index]=np.copy(v_temp)
            if ((x[index]+v[index]>=xlimit.T[0]).all() and (x[index]+v[index]<=xlimit.T[1]).all()):
                x[index]=np.copy(x[index]+v[index])
                ut=CN(100,a,h.T,x[index][4],1,np.array([x[index][0],x[index][1],x[index][2],x[index][3]]))
                 # 求是否满足第二问条件
                limit_ran=0
                while not judge(ut):
                    # 求出该温度下的上限与下限
                    if limit_ran >= 1:
                        x[index][0]=np.random.uniform(low=xlimit[0][0], high=xlimit[0][1], size=1)[0]
                        x[index][1]=np.random.uniform(low=xlimit[1][0], high=xlimit[1][1], size=1)[0]
                        x[index][2]=np.random.uniform(low=xlimit[2][0], high=xlimit[2][1], size=1)[0]
                        x[index][3]=np.random.uniform(low=xlimit[3][0], high=xlimit[3][1], size=1)[0]
                        x[index][4]=np.random.uniform(low=xlimit[4][0], high=xlimit[4][1], size=1)[0]
                        limit_ran=0
                    else:
                        print("寻找上下限...")
                        vrange_temp=getRange(np.array([x[index][0],x[index][1],x[index][2],x[index][3]]))
                        x[index][4]=np.random.uniform(low=vrange_temp[0], high=vrange_temp[1], size=1)[0]
                        ut=CN(100,a,h.T,x[index][4],1,np.array([x[index][0],x[index][1],x[index][2],x[index][3]]))
                        limit_ran+=1
                tmp=Area(ut)
                if individual_best[index] > tmp:
                    individual_best[index] = tmp
                    individual_best_x[index] = np.copy(x[index])
                if global_best>tmp:
                    print("*******")
                    print("发现全局更优解:",tmp)
                    print("*******")
                    global_best=tmp
                    global_best_x=np.copy(x[index])
            print("global best x:",global_best_x)
            print(global_best)  

if __name__ == "__main__":
    q3_pso()
```

### Q3judge.py

```python
from Q1 import *
from Q2 import *

v_result=np.zeros(4)

# 最大斜率
max_slope=3
# 传送带速度范围
v_range=[65/60,100/60]

# 当前速度
mid=(v_range[0]+v_range[1])/2
high=v_range[1]
low=v_range[0]
# 最大迭代次数
cnt_max=6

# 已知a,h
a= np.array([0.00067445 ,0.0007394 , 0.00088138 ,0.00071155 ,0.00050576])
h= np.array([9351.90193919 , 6419.99175218, 11016.59502516 , 9347.90126533,6863.18027028])

# ut=CN(100,a,h,speed,1)
def Maxslope(ut):
    maxslope=0
    ut=ut.T
    for index in range(ut.T[0].size-1):
        if maxslope < np.abs(ut[index+1][1]-ut[index][1])/0.5:
            maxslope = np.abs(ut[index+1][1]-ut[index][1])/0.5
    if maxslope > 3:
        return 0
    else:
        return 1

def PeakTemperature(ut):
    u = ut[1]
    if np.max(u) >= 240 and np.max(u) <= 250:
        return 1
    else:
        return 0
    
def Temperature150To190(ut):
    u = ut[1]
    diff_u = np.diff(u)
    a = u[1:][diff_u > 0]
    a = a[(a > 150) & (a < 190)]
    b = len(a)
    if b * 0.5 >= 60 and b * 0.5 <= 120:
        return 1
    else:
        return 0
    
def Temperature217(ut):
    u = ut[1]
    a = u[u > 217]
    b = len(a)
    if b*0.5 >= 40 and b*0.5 <= 90:
        return 1
    else:
        return 0
    
def judge(ut):
    return Maxslope(ut) and PeakTemperature(ut) and Temperature150To190(ut) and Temperature217(ut)

def getmin_v(temp):
    # 传送带速度范围
    v_range=[65/60,100/60]

    # 当前速度
    mid=(v_range[0]+v_range[1])/2
    high=v_range[1]
    low=v_range[0]
    # 最大迭代次数
    cnt_max=4
   
    def Maxslope_min(speed,temp):
        # 已知a,h
        a= np.array([0.00067445 ,0.0007394 , 0.00088138 ,0.00071155 ,0.00050576])
        h= np.array([9351.90193919 , 6419.99175218, 11016.59502516 , 9347.90126533,6863.18027028])
        maxslope=0
        ut=CN(100,a,h,speed,1,temp)
        ut=ut.T
        for index in range(ut.T[0].size-1):
            if maxslope < np.abs(ut[index+1][1]-ut[index][1])/0.5:
                maxslope = np.abs(ut[index+1][1]-ut[index][1])/0.5
        if maxslope > 0:
            return 1
        else:
            return 0

    def PeakTemperature_min(speed,temp):
        # 已知a,h
        a= np.array([0.00067445 ,0.0007394 , 0.00088138 ,0.00071155 ,0.00050576])
        h= np.array([9351.90193919 , 6419.99175218, 11016.59502516 , 9347.90126533,6863.18027028])
        ut=CN(100,a,h,speed,1,temp)
        u = ut[1]
        if np.max(u) <= 250 :
            return 1
        else:
            return 0
        
    def Temperature150To190_min(speed,temp):
        # 已知a,h
        a= np.array([0.00067445 ,0.0007394 , 0.00088138 ,0.00071155 ,0.00050576])
        h= np.array([9351.90193919 , 6419.99175218, 11016.59502516 , 9347.90126533,6863.18027028])
        ut=CN(100,a,h,speed,1,temp)
        u = ut[1]
        diff_u = np.diff(u)
        a = u[1:][diff_u > 0]
        a = a[(a > 150) & (a < 190)]
        b = len(a)
        if b * 0.5 <= 120:
            return 1
        else:
            return 0
        
    def Temperature217_min(speed,temp):
        # 已知a,h
        a= np.array([0.00067445 ,0.0007394 , 0.00088138 ,0.00071155 ,0.00050576])
        h= np.array([9351.90193919 , 6419.99175218, 11016.59502516 , 9347.90126533,6863.18027028])
        ut=CN(100,a,h,speed,1,temp)
        u = ut[1]
        a = u[u > 217]
        b = len(a)
        if b*0.5 <= 90:
            return 1
        else:
            return 0

    cnt=0
    while cnt<cnt_max:
        # print("迭代次数:",cnt+1)
        if Maxslope_min(mid,temp) and PeakTemperature_min(mid,temp) and Temperature150To190_min(mid,temp) and Temperature217_min(mid,temp):# 加判断条件
            high=mid
            mid=(mid+low)/2
        else:
            low=mid
            mid=(mid+high)/2
        cnt+=1
    return mid
    
def getRange(temp):
    return [getmin_v(temp),getmax_v(temp)]

if __name__ == "__main__":
   print(getmin_v())
```

### Q3area.py

```python
import numpy as np


def Area(ut):
    u = ut[1][:]
    u = u[:np.argmax(u) + 1]
    sum = 0
    for i in range(len(u) - 1):
        if u[i] < 217 and u[i + 1] > 217:
            sum += ((u[i + 1] + 217) / 2 - 217) * 0.5 * ((u[i + 1] - 217) / (u[i + 1] - u[i]))
        if u[i] > 217 and u[i + 1] > 217:
            sum += ((u[i] + u[i + 1]) / 2 - 217) * 0.5
        if u[i] > 217 and u[i + 1] < 217:
            sum += ((u[i] + 217) / 2 - 217) * 0.5 * ((u[i] - 217) / (u[i + 1] - u[i]))
        # print(sum)
    return sum

def Area_right(ut):
    u = ut[1]
    u = u[np.argmax(u):]
    sum = 0
    for i in range(len(u) - 1):
        if u[i] < 217 and u[i + 1] > 217:
            sum += ((u[i + 1] + 217) / 2 - 217) * 0.5 * ((u[i + 1] - 217) / (u[i + 1] - u[i]))
        if u[i] > 217 and u[i + 1] > 217:
            sum += ((u[i] + u[i + 1]) / 2 - 217) * 0.5
        if u[i] > 217 and u[i + 1] < 217:
            sum += ((u[i] + 217) / 2 - 217) * 0.5 * ((u[i] - 217) / (u[i + 1] - u[i]))
        # print(sum)
    return sum

if __name__ == "__main__":
    a = Area(np.array([[1, 1, 1, 1, 1, 1], [200, 223, 256, 258, 260, 255]]))
    print(a)
```

## Q4

### Q4.py

```python
from Q1 import *
from Q4sigma import *
from Q3judge import *

a= np.array([0.00067445 ,0.0007394 , 0.00088138 ,0.00071155 ,0.00050576])
h= np.array([9351.90193919 , 6419.99175218, 11016.59502516 , 9347.90126533,6863.18027028])

def q4_pso():
    N=5 # 种群个数
    cnt_max=20 # 最大迭代次数
    # 位置界限
    xlimit=np.array([
        [165,185],
        [185,205],
        [225,245],
        [245,265],
        [65/60,90/60]# 为缩短时间将上限减小至90cm/min
    ])
    # 速度界限
    vlimit=np.array([
        [-3,3],
        [-3,3],
        [-3,3],
        [-3,3],
        [-5/60,5/60]
    ])
    
    # 全局最优
    global_best=10000000
    # 全局最优位置
    global_best_x=np.zeros(5)
    # 初始化位置
    x=np.array([
        np.random.uniform(low=xlimit[0][0], high=xlimit[0][1], size=N),
        np.random.uniform(low=xlimit[1][0], high=xlimit[1][1], size=N),
        np.random.uniform(low=xlimit[2][0], high=xlimit[2][1], size=N),
        np.random.uniform(low=xlimit[3][0], high=xlimit[3][1], size=N),
        np.random.uniform(low=xlimit[4][0], high=xlimit[4][1], size=N),
    ]).T
    # 初始化速度
    v=np.array([
        np.random.uniform(low=vlimit[0][0], high=vlimit[0][1], size=N),
        np.random.uniform(low=vlimit[1][0], high=vlimit[1][1], size=N),
        np.random.uniform(low=vlimit[2][0], high=vlimit[2][1], size=N),
        np.random.uniform(low=vlimit[3][0], high=vlimit[3][1], size=N),
        np.random.uniform(low=vlimit[4][0], high=vlimit[4][1], size=N),
    ]).T
    
    # 个人历史最佳
    individual_best=np.zeros(N)
    # 个人历史最佳位置
    individual_best_x=np.copy(x)
    
    # 初始化全局最优
    for index in range(N):
        # print("初始化:",index+1)
        ut=CN(100,a,h.T,x[index][4],1,np.array([x[index][0],x[index][1],x[index][2],x[index][3]]))
        # 求是否满足第二问条件
        limit_ran=0
        while not judge(ut):
            # 求出该温度下的上限与下限
            if limit_ran >= 1:
                x[index][0]=np.random.uniform(low=xlimit[0][0], high=xlimit[0][1], size=1)[0]
                x[index][1]=np.random.uniform(low=xlimit[1][0], high=xlimit[1][1], size=1)[0]
                x[index][2]=np.random.uniform(low=xlimit[2][0], high=xlimit[2][1], size=1)[0]
                x[index][3]=np.random.uniform(low=xlimit[3][0], high=xlimit[3][1], size=1)[0]
                x[index][4]=np.random.uniform(low=xlimit[4][0], high=xlimit[4][1], size=1)[0]
                limit_ran=0
            else:
                # print("寻找上下限...")
                vrange_temp=getRange(np.array([x[index][0],x[index][1],x[index][2],x[index][3]]))
                if vrange_temp[0]>vrange_temp[1]:
                    # print("该温度下符合条件速度不存在!")
                    limit_ran+=1
                    continue
                x[index][4]=np.random.uniform(low=vrange_temp[0], high=vrange_temp[1], size=1)[0]
                ut=CN(100,a,h.T,x[index][4],1,np.array([x[index][0],x[index][1],x[index][2],x[index][3]]))
                limit_ran+=1
        temp=Sigma(ut)
        # 更新个体最优
        individual_best[index]=temp
        # 更新全局最优
        if global_best>temp:
            global_best=temp
            global_best_x=np.copy(x[index])
        
    w=np.linspace(1.1,0.2,cnt_max) # 惯性系数
    w=np.append(w,np.linspace(0.4,0.4,20))
    cnt_max+=10
    c1=0.5 # 个体学习参数
    c2=0.6 # 全局学习系数
    
    for cnt in range(cnt_max):
        # print("*******")
        # print("迭代次数:",cnt+1) 
        # print("*******")
        r1=np.array(np.random.rand(5))
        r2=np.array(np.random.rand(5))
        # 对每个粒子进行更新
        for index in range(N):
            # 更新a，v速度
            v_temp=np.copy(w[cnt]*v[index]+c1*r1*(individual_best_x[index]-x[index])+c2*r2*(global_best_x-x[index]))
            #判断该速度是否在界限内并且符合该温度设定下的传送速度范围
            if not((v_temp<vlimit.T[1]).all() and (v_temp>vlimit.T[0]).all()):
                for pos in range(v_temp.size):
                    if v_temp[pos]>vlimit[pos][1]:
                        v_temp[pos]=np.copy(vlimit[pos][1])
                    elif v_temp[pos]<vlimit[pos][0]:
                        v_temp[pos]=np.copy(vlimit[pos][0])
            v[index]=np.copy(v_temp)
            if ((x[index]+v[index]>=xlimit.T[0]).all() and (x[index]+v[index]<=xlimit.T[1]).all()):
                x[index]=np.copy(x[index]+v[index])
                ut=CN(100,a,h.T,x[index][4],1,np.array([x[index][0],x[index][1],x[index][2],x[index][3]]))
                 # 求是否满足第二问条件
                limit_ran=0
                while not judge(ut):
                    # 求出该温度下的上限与下限
                    if limit_ran >= 1:
                        x[index][0]=np.random.uniform(low=xlimit[0][0], high=xlimit[0][1], size=1)[0]
                        x[index][1]=np.random.uniform(low=xlimit[1][0], high=xlimit[1][1], size=1)[0]
                        x[index][2]=np.random.uniform(low=xlimit[2][0], high=xlimit[2][1], size=1)[0]
                        x[index][3]=np.random.uniform(low=xlimit[3][0], high=xlimit[3][1], size=1)[0]
                        x[index][4]=np.random.uniform(low=xlimit[4][0], high=xlimit[4][1], size=1)[0]
                        limit_ran=0
                    else:
                        # print("寻找上下限...")
                        vrange_temp=getRange(np.array([x[index][0],x[index][1],x[index][2],x[index][3]]))
                        x[index][4]=np.random.uniform(low=vrange_temp[0], high=vrange_temp[1], size=1)[0]
                        ut=CN(100,a,h.T,x[index][4],1,np.array([x[index][0],x[index][1],x[index][2],x[index][3]]))
                        limit_ran+=1
                tmp=Sigma(ut)
                if individual_best[index] > tmp:
                    individual_best[index] = tmp
                    individual_best_x[index] = np.copy(x[index])
                if global_best>tmp:
                    # print("*******")
                    # print("发现全局更优解:",tmp)
                    # print("*******")
                    global_best=tmp
                    global_best_x=np.copy(x[index])
    print("global best x:",global_best_x)
    print(global_best)  

if __name__ == "__main__":
    q4_pso()
```

### Q4sigma.py

```python
import numpy as np

from Q3area import *

def Sigma(ut):
    u = ut[1]
    n = np.argmax(u)
    m = np.argmax(u > 217)
    k = m + len(np.where(u > 217)[0]) - 1
    # print("n的值为:", n)
    # print("m的值为:", m)
    # print("k的值为:", k)
    S1 = Area(ut)
    S2 = Area_right(ut)
    sigma = max(abs(S1 - S2)/max(S1, S2), abs(k + m - 2*n)/max(n-m, k-n))
    return sigma

if __name__ == "__main__":
    sigma = Sigma(np.array([[0, 0.5, 1, 1.5, 2, 2.5, 3], [200, 215, 220, 230, 235, 225, 200]]))
    print(sigma)
```