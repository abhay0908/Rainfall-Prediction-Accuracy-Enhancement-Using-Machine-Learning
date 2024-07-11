import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pycwt as wavelets
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor

def initialize(h=8):
    plt.close()
    figprops = dict(figsize=(11,h), dpi=96)
    fig = plt.figure(**figprops)
    return plt.axes()

def plotWavelet(time,power,scales,coi,freqs,title,xlabel,ylabel,yTicks=None,steps=512,lowerLimit=0,upperLimitDelta=0):
    zx = initialize()
    
    # cut out very small powers
    LP2=np.log2(power)
    LP2=np.clip(LP2,0,np.max(LP2))
    
    # draw the CWT
    zx.contourf(time, scales, LP2, steps, cmap=plt.cm.gist_ncar)
    
    # draw the COI
    coicoi=np.clip(coi,0,coi.max())
    zx.fill_between(time,coicoi,scales.max(),alpha=0.2, color='g', hatch='x')

    # Y-AXIS labels
    if (yTicks):
       yt = yTicks 
    else:
        period=1/freqs
        yt = 2 ** np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))

    zx.set_yscale('log')
    zx.set_yticks(yt)
    zx.set_yticklabels(yt)
    zx.grid(b=None, which='major', axis='y', alpha=0.2, antialiased=True, c='k', linestyle='-.')
    
    # exclude some periods from view
    ylim = zx.get_ylim()
    zx.set_ylim(lowerLimit,ylim[1]-upperLimitDelta)
    
    # strings
    zx.set_title(title)
    zx.set_ylabel(ylabel)
    zx.set_xlabel(xlabel)
    # print all
    plt.show()

def calculatewavelet(time,signal,steps_value=32):
    mother_value = wavelets.Morlet(6)
    delta_T = time[1] - time[0]
    dj = 1 / steps_value        
    s0 = 2 * delta_T       
    wavelet, scaleValue, frequency, coi_value, fft_value, fft_freqs = wavelets.cwt(signal, delta_T, dj, s0, -1, mother_value)
    # Normalizing wavelet data
    powerValue = (np.abs(wavelet)) ** 2
    return powerValue,scaleValue,coi_value,frequency


def plotRainfallTimeSeries(time, rainfall, graph_title, x_label, y_label, image_Height=4, interpolate=False, line_width=0.5):
    rainfall_value = savitzky_golay(rainfall,63,3) if interpolate else rainfall
    ax = initialize(image_Height)
    ax.plot(time,rainfall_value,linewidth=line_width,antialiased=True)
    ax.set_title(graph_title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.grid(b=None, which='major', axis='y', alpha=0.2, antialiased=True, c='k', linestyle='-.')
    plt.show()

dataset = pd.read_csv('wavlet.csv',sep=';',usecols=['DATEFRACTION','Rainfall'])
rainfall = dataset['Rainfall']
time= dataset['DATEFRACTION']

#plotRainfallTimeSeries(time,rainfall,'Rainfall Graph','year','Rainfall')
#powerValue,scaleValue,coi_value,frequency = calculatewavelet(time,rainfall,256)
#plotWavelet(time,powerValue,scaleValue,coi_value,frequency,'Rainfall Wavelet','year','period(years)', yTicks=[0.5,1,2,3,4,5,6,8,11,16,22,32,64,80,110,160,220],lowerLimit=0.9,upperLimitDelta=0.5)

dataset = dataset.values
X = dataset[:,0]
Y = dataset[:,1]
X = X.reshape(-1, 1)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.1,random_state=0)

mlp = MLPRegressor(random_state=42, max_iter=100,hidden_layer_sizes=100,alpha=0.00001)
tree = DecisionTreeRegressor(max_depth=50,min_samples_leaf=25,random_state=42,max_features='auto')
vr = VotingRegressor([('mlp', mlp), ('tree', tree)])
vr.fit(X,Y)
tree.fit(X,Y)
prediction = tree.predict(Xtest) 

actual = []
forecast = []
i = len(Ytest)-1
index = 0
while i > 0:
    actual.append(Ytest[i])
    forecast.append(prediction[i])
    print('Day=%d, Forecasted=%f, Actual=%f' % (index+1, prediction[i], Ytest[i]))
    index = index + 1
    i = i - 1
    if len(actual) > 30:
        break

rmse = sqrt(mean_squared_error(Ytest,prediction))
print('\n\nRMSE : ',round(rmse,1))

plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Sales Count')
plt.ylabel('Sales')
plt.plot(actual, 'ro-', color = 'blue')
plt.plot(forecast, 'ro-', color = 'green')
plt.legend(['Actual Sale', 'Forecast Sale'], loc='upper left')
#plt.xticks(wordloss.index)
plt.title('Car Sale Forecasting Graph')
plt.show()








