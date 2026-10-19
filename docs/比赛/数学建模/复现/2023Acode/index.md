## mcm2023.py

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 获取某种生物的内禀增长率b，参数为温度（℃），降水量（mm）
def getb(x,y,mu1,mu2,sigma1,sigma2):
    arr=[]
    for index in range(x.size):
        arr.append(3*np.e**(-2*np.log(3)*(
            ((x[index]-mu1)**2/(1*sigma1**2)) + ((y[index]-mu2)**2/(1*sigma2**2))
        )))
    return 10*np.array(arr)

# 生成随机温度、降水量序列
def genRandomWeather(num_of_slices):
    weather = np.array([np.random.randint(20, 35, num_of_slices),np.random.randint(200/12, 1200/12, num_of_slices)])
    return weather

# 生成随机一年内温带大陆气候
def genRandomContinentalWeather(num_of_slices):
    temp=np.array([-13.0,-10.0,-2,7,15,21,27,20,14,6,-3,-10])
    pre=np.array([10.0,11,10,30,40,50,110,120,60,50,23,10])
    for index in range(2):
        temp=np.append(temp,temp)
        pre=np.append(pre,pre)
    temp+=np.random.randint(-3,3,size=num_of_slices)
    pre+=np.random.randint(-8,8,size=num_of_slices)
    # temp[24:36]=25
    pre[24:36]/=10
    return np.array([temp,pre])

def genDroughtWeatherMoreFrequent(num_of_slices):
    temp=np.array([13.0,10.0,12,17,15,21,22,20,14,16,13,10])+10
    pre=np.array([40.0,60,60,67,70,77,110,120,81,66,73,80])
    for index in range(2):
        temp=np.append(temp,temp)
        pre=np.append(pre,pre)
    temp+=np.random.randint(-2,2,size=num_of_slices)
    pre+=np.random.randint(-8,8,size=num_of_slices)
    pre[14:42]/=200
    temp[14:18]+=10
    temp[14:20]+=8
    temp[20:27]+=10
    temp[27:29]+=10
    temp[29:38]+=10
    # temp[14:42]+=np.random.randint(-4,4,size=28)
    # pre[18:20]/=20
    # pre[23:28]/=20
    # pre[15:18]/=20
    return np.array([temp,pre])

# 生成指定长度，固定温度的序列，参数为指定温度、降水量，长度
def genWeather(temperature,precipitation,num_of_slices):
    weather=np.array([np.full(num_of_slices,temperature),np.full(num_of_slices,precipitation)])
    return weather

# 计算其最适宜温度与最适宜年降水量
def calcBestTP(tmin,tmax,pmin,pmax):
    return np.array([(tmin+tmax)/2,(pmin+pmax)/2])

# 获取特定物种适宜温度与降水量
def getCSV(id):
    df=pd.read_csv('species.csv')
    df.values[id][3]/=12
    df.values[id][4]/=12
    return df.values[id]

def getWeatherCSV(filename):
    return pd.read_csv(filename)

def simDroughtWeather(num_of_slices):
    weather = np.array([np.random.randint(34, 36, num_of_slices),np.random.randint(400/12, 600/12, num_of_slices)])
    weather[1][int(num_of_slices/2):]-=33
    # weather[0][int(num_of_slices/2):]+=np.random.randint(0,10)
    return weather

# 计算抗干旱能力
def calcEta(arr :list):
    # arr.T[0].size
    eta=0
    div=arr[0].size
    for index in range(arr[0].size):
        if (arr.T[index][int(arr.T[0].size/2)+1:int(arr.T[0].size/2)+4]<1e-5).any():
            # div-=1
            print("!")
        else:
            eta+=np.sum(arr.T[index][int(arr.T[0].size/2)+1:int(arr.T[0].size/2)+4])/np.sum(arr.T[index][int(arr.T[0].size/2)-4:int(arr.T[0].size/2)-1])
    return eta/div

def drawWeather(t:list,weather:list):
    fig, ax1 = plt.subplots()
    ax1.bar(t,weather[1])
    ax1.set_ylabel("avg precipitation(mm)")
    ax1.tick_params('y')
    ax2 = ax1.twinx()
    ax2.plot(t,weather[0],color="red")
    ax2.scatter(t, weather[0], marker='o',color="r")
    ax2.set_ylabel("avg temperature(℃)")
    ax2.tick_params('y')
    ax2.set_ylim(0, 40)
    ax1.set_xlabel("month")
    plt.show()

if __name__=="__main__":
    a=np.array([33,22,11,23,32,21])
    c=np.array([444,333,222,112,1433,1222])
    print(getb(a,c,10,800,10,900))
```

## Q1.py

```python
from mcm2023 import *
from Q1_SimWeather import *
# 物种数量
N=3
# 切片数
num_of_slices=48

def func(x: list, t, b: list, a: list):
    dx0dt = x[0] * (b[0][int(t)] - a[0][0] * x[0] - a[0][1] * x[1] + a[0][2] * x[2])
    dx1dt = x[1] * (b[1][int(t)] - a[1][0] * x[0] - a[1][1] * x[1] + a[1][2] * x[2])
    dx2dt = x[2] * (b[2][int(t)] + a[2][0] * x[0] + a[2][1] * x[1] - a[2][2] * x[2])
    return [dx0dt, dx1dt, dx2dt]

t = np.linspace(0, num_of_slices-1, num_of_slices)
t = [int(x) for x in t]
x0 = [20, 20, 20]

a=np.full((N,N),0.3)
for index in range(a[0].size):
    a[index][index]=0.5
a*=1.4
b=np.zeros((N,num_of_slices))

# weather=
weather=np.array([[24 ,33, 28 ,24 ,22 ,21, 24 ,26 ,21 ,25 ,28, 30 ,30, 20, 34, 33, 27, 28, 28, 26, 32, 28, 29, 26,
  24 ,34, 32, 23, 24, 27, 29, 33, 20, 32, 31, 29, 28, 30, 34, 20, 32, 25, 23, 27, 27, 25, 28, 27,],
 [69 ,40, 96, 17, 20, 80, 88, 80, 16, 80, 73, 86, 64, 37, 96, 76, 59, 47, 46, 52, 76, 32, 98, 78,
  29 ,29, 33, 28, 68, 89, 21, 95, 96, 65, 21, 33, 24, 76, 26, 20, 99, 24, 19, 64, 48, 92, 20, 73]])
# print(weather)
# weather=genRandomContinentalWeather(num_of_slices)
# weather=genDroughtWeatherMoreFrequent(num_of_slices)
# weather=getColoradoWeather()
# weather=genWeather(24,1000,num_of_slices)
# drawWeather(t,weather)
for index in range(N):
    df=getCSV(index)
    b[index]=getb(weather[0],weather[1],
        (df[1]+df[2])/2,
        (df[3]+df[4])/2,
        (df[2]-df[1]),
        (df[4]-df[3])
    )
    
sol = odeint(func, x0, t, args=(b, a))
print(sol)
plt.plot(t, sol.T[0], label="species 1",color="lightsalmon")
plt.plot(t, sol.T[1], label="species 2",color="lawngreen")
plt.plot(t, sol.T[2], label="species 3",color="lightskyblue")
plt.fill_between(t, sol.T[0], color='lightsalmon', alpha=0.1)
plt.fill_between(t, sol.T[1], color='lawngreen', alpha=0.1)
plt.fill_between(t, sol.T[2], color='blue', alpha=0.1)
# plt.plot(weather[2], sol.T[0], label="species 1")
# plt.plot(weather[2], sol.T[1], label="species 2")
# plt.plot(weather[2], sol.T[2], label="species 3")

plt.xlabel("month")
plt.ylabel("num")
plt.grid()
plt.legend()
plt.show()
```

## Q1_SimWeather.py

```python
from mcm2023 import *
def getColoradoWeather():
    data=getWeatherCSV("Colorado.csv")
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
    data=data[((data['TIMESTAMP'].dt.day == 1) | (data['TIMESTAMP'].dt.day == 15))& (data['TIMESTAMP'].dt.hour == 12)]
    data=data.values.T
    rain_permonth=np.array([])
    temp=25.4*np.array([1.34,1.85,1.1,0.71,0.79,0.59,0.39,0.39,0.75,1.26,1.42,2.36])
    for index in range(8):
        rain_permonth=np.append(rain_permonth,temp)
    return np.array([5+data[17][:96],rain_permonth,data[0][:96]])
    
if __name__=="__main__":
    data=getWeatherCSV("Colorado.csv")
    data['TIMESTAMP'] = pd.to_datetime(data['TIMESTAMP'])
    data=data[(data['TIMESTAMP'].dt.day == 15) & (data['TIMESTAMP'].dt.hour == 12)]
    data=data.values.T
    temp=25.4*np.array([1.34,1.85,1.1,0.71,0.79,0.59,0.39,0.39,0.75,1.26,1.42,2.36])
    rain_permonth=np.array([])
    for index in range(4):
        rain_permonth=np.append(rain_permonth,temp)
    plt.plot(data[0][:96],rain_permonth)
    plt.plot(5+data[0][:96],data[17][:96])
    plt.xlabel('month')
    plt.show()
```

## Q2.py

```python
from mcm2023 import *
from Q1_SimWeather import *
'''
##################
若改变物种数量在此修改
##################
'''
# 物种数量
N=2
# 切片数
num_of_slices=48
# 初始数量
x = np.array([20,20,20,20,20])[:N]
# 种间系数
co=np.array([
    [-1,-1,1,1,1],
    [-1,-1,1,1,1],
    [1,1,-1,-1,1],
    [1,1,-1,-1,1],
    [-1,1,-1,1,-1],
])[:N,:N]
'''
##################
end
##################
'''
def func(x: list, t, b: list, a: list):
    x=x.reshape(1,x.size)
    return (x.T * (a @ (x.T) + (b.T[int(t)]).reshape(N,1))).T.flatten()

t = np.linspace(0, num_of_slices-1, num_of_slices)
t = [int(x) for x in t]
a=np.full((N,N),0.3)
for index in range(a[0].size):
    a[index][index]=0.5
a=a*co
b=np.zeros((N,num_of_slices))

# 生成随机天气
# weather=genRandomWeather(num_of_slices)

# 生成科罗拉多天气
# weather=getColoradoWeather()

# 生成恒定条件天气（slices不能太长？）
# weather=genWeather(24,1000,num_of_slices)

# 模拟干旱天气
# weather=simDroughtWeather(num_of_slices)

# 模拟一组指定的随机干旱天气
weather=np.array([[29, 27, 34, 32, 28, 28, 27, 33, 33, 29, 28, 33, 34, 28, 25, 29,
        26, 32, 31, 27, 32, 29, 30, 33, 27, 29, 25, 26, 25, 32, 33, 25,
        32, 28, 25, 33, 26, 34, 34, 30, 34, 33, 29, 30, 30, 29, 31, 31,
        32, 32, 34, 27, 28, 25, 29, 30, 28, 30, 27, 25, 26, 29, 34, 32,
        31, 34, 30, 27, 33, 27, 33, 26, 29, 31, 28, 26, 29, 33, 28, 32,
        34, 30, 30, 34, 34, 33, 32, 26, 29, 33, 27, 25, 32, 25, 31, 33],
       [38, 45, 44, 34, 40, 46, 36, 47, 35, 36, 38, 38, 40, 35, 35, 45,
        47, 47, 47, 37, 38, 35, 34, 45, 37, 47, 42, 49, 39, 47, 45, 47,
        43, 34, 47, 38, 36, 47, 39, 37, 46, 49, 47, 48, 44, 35, 47, 48,
        14,  0, 15,  1, 11,  7,  3,  4, 11, 16,  8,  2, 13,  1, 12,  5,
         2, 11, 11, 15,  5,  4,  5,  6,  3, 16, 16, 10,  9, 14, 11, 11,
        16,  9,  7,  1, 12,  0, 12, 11, 13, 14,  9,  2, 11,  9,  8,  3]])
weather=genRandomContinentalWeather(num_of_slices)
# 绘制气温与降雨量
drawWeather(t,weather)
for index in range(N):
    df=getCSV(index)
    b[index]=getb(weather[0],weather[1],
        (df[1]+df[2])/2,
        (df[3]+df[4])/2,
        (df[2]-df[1]),
        (df[4]-df[3])
    )
    
sol = odeint(func, x, t, args=(b, a))
print(sol)
print(calcEta(sol))

# 绘制x-t图像
for index in range(N):
    plt.plot(t, sol.T[index], label="%s"%(getCSV(index)[0]))

# for index in range(N):
#     plt.plot(weather[2], sol.T[index], label="species %d"%index)

plt.xlabel("month")
plt.ylabel("num")
plt.legend()
plt.show()
```

## Q2_combination.py

```python
from mcm2023 import *
from Q1_SimWeather import *
def Q2(N:int,combinations:list,getdata=0):
    # 物种数量
    # N=3
    # 切片数
    num_of_slices=96
    # 初始数量
    x = np.array([20,20,20,20,20])[:N]
    # 排列顺序
    # combinations=np.array([0,1,2,3,4])[:N]
    combinations=combinations[:N]
    # 种间系数
    co=np.array([
        [-1,-1,1,1,1],
        [-1,-1,1,1,1],
        [1,1,-1,-1,1],
        [1,1,-1,-1,1],
        [-1,1,-1,1,-1],
    ])
    # 重建种间系数
    co_temp=np.zeros((N,N))
    for row in range(N):
        for col in range(N):
            co_temp[row][col]=np.copy(co[combinations[row]][combinations[col]])
    co=np.copy(co_temp)
    # L-V模型数值求解方程
    def func(x: list, t, b: list, a: list):
        if t>=num_of_slices:
            return
        x=x.reshape(1,x.size)
        return (x.T * (a @ (x.T) + (b.T[int(t)]).reshape(N,1))).T.flatten()

    t = np.linspace(0, num_of_slices-1, num_of_slices)
    t = np.array([int(x) for x in t])
    a=np.full((N,N),0.3)
    for index in range(a[0].size):
        a[index][index]=0.5
    a=a*co
    b=np.zeros((5,num_of_slices))
    # 模拟一组指定的随机干旱天气
    # weather=np.array([[29, 27, 34, 32, 28, 28, 27, 33, 33, 29, 28, 33, 34, 28, 25, 29,
    #         26, 32, 31, 27, 32, 29, 30, 33, 27, 29, 25, 26, 25, 32, 33, 25,
    #         32, 28, 25, 33, 26, 34, 34, 30, 34, 33, 29, 30, 30, 29, 31, 31,
    #         32, 32, 34, 27, 28, 25, 29, 30, 28, 30, 27, 25, 26, 29, 34, 32,
    #         31, 34, 30, 27, 33, 27, 33, 26, 29, 31, 28, 26, 29, 33, 28, 32,
    #         34, 30, 30, 34, 34, 33, 32, 26, 29, 33, 27, 25, 32, 25, 31, 33],
    #     [38, 45, 44, 34, 40, 46, 36, 47, 35, 36, 38, 38, 40, 35, 35, 45,
    #         47, 47, 47, 37, 38, 35, 34, 45, 37, 47, 42, 49, 39, 47, 45, 47,
    #         43, 34, 47, 38, 36, 47, 39, 37, 46, 49, 47, 48, 44, 35, 47, 48,
    #         14,  0, 15,  1, 11,  7,  3,  4, 11, 16,  8,  2, 13,  1, 12,  5,
    #         2, 11, 11, 15,  5,  4,  5,  6,  3, 16, 16, 10,  9, 14, 11, 11,
    #         16,  9,  7,  1, 12,  0, 12, 11, 13, 14,  9,  2, 11,  9,  8,  3]])
    # 模拟干旱天气
    weather=simDroughtWeather(num_of_slices)
    # weather=genRandomContinentalWeather(num_of_slices)
    # 绘制气温与降雨量
    # drawWeather(t,weather)
    cnt=0
    for index in combinations:
        df=getCSV(index)
        b[cnt]=getb(weather[0],weather[1],
            (df[1]+df[2])/2,
            (df[3]+df[4])/2,
            (df[2]-df[1]),
            (df[4]-df[3])
        )
        cnt+=1
    b=b[:N]
    sol = odeint(func, x, t, args=(b, a))
    # print(sol)
    print(calcEta(sol))
    if getdata:
        return calcEta(sol)
    # 绘制x-t图像
    # for index in range(N):
    #     plt.plot(t, sol.T[index], label="%s"%(getCSV(combinations[index])[0]))

    # plt.xlabel("month")
    # plt.ylabel("num")
    # plt.legend()
    # plt.show()
    
if __name__=="__main__":
    str0=[]
    for index in range(5):
        str0.append(Q2(index+1,np.array([4,3,2,1,0]),1))
    print(str0)
    # Q2(5,np.array([0,1,2,3,4]))
```

## Q3.py

```python
from mcm2023 import *
from mpl_toolkits.mplot3d import Axes3D
def Q3(co1,co2):
    # 物种数量
    N=3
    # 切片数
    num_of_slices=48

    def func(x: list, t, b: list, a: list):
        dx0dt = x[0] * (b[0][int(t)] - a[0][0] * x[0] - a[0][1] * x[1] + a[0][2] * x[2])
        dx1dt = x[1] * (b[1][int(t)] - a[1][0] * x[0] - a[1][1] * x[1] + a[1][2] * x[2])
        dx2dt = x[2] * (b[2][int(t)] + a[2][0] * x[0] + a[2][1] * x[1] - a[2][2] * x[2])
        return [dx0dt, dx1dt, dx2dt]

    t = np.linspace(0, num_of_slices-1, num_of_slices)
    t = [int(x) for x in t]
    x0 = [10, 10, 15]

    a=np.full((N,N),0.3)
    for index in range(a[0].size):
        a[index][index]=0.5
    a*=co1
    b=np.zeros((N,num_of_slices))

    weather=genRandomWeather(num_of_slices)
    for index in range(N):
        df=getCSV(index)
        b[index]=co2*getb(weather[0],weather[1],
            (df[1]+df[2])/2,
            (df[3]+df[4])/2,
            (df[2]-df[1]),
            (df[4]-df[3])
        )
        
    sol = odeint(func, x0, t, args=(b, a))
    return np.average(sol.T[0])+np.average(sol.T[1])+np.average(sol.T[2])

if __name__ == "__main__":
    cnt=1
    x=np.linspace(1.01,1.20,20)
    y=np.linspace(0.99,0.80,20)
    z=np.zeros((20,20))
    for co1 in x:
        for co2 in y:
            print(cnt)
            cnt+=1
            z[int(100*(co1-1.01))][int(100*(0.99-co2))]=Q3(co1,co2)
    print(z)
    # 绘制三维热力图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y = np.meshgrid(x, y)
    heat_map = ax.plot_surface(x, y, z, cmap='viridis')
    fig.colorbar(heat_map, ax=ax, label='population sum')
    ax.set_xlabel('X/a coefficient')
    ax.set_ylabel('Y/b coefficient')
    ax.set_zlabel('Z/population sum')
    plt.show()
```