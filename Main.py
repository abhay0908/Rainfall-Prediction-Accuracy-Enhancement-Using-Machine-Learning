from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

main = tkinter.Tk()
main.title("Rainfall Prediction: Accuracy Enhancement Using Machine Learning")
main.geometry("1200x1200")

global filename
global rainfall, time
global Xtrain, Xtest, Ytrain, Ytest
global dataset

global rmse_arr, accuracy_arr
actual = []
forecast = []



def uploadDataset():
    global filename
    global dataset
    global rainfall, time
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    dataset = pd.read_csv(filename,sep=';',usecols=['DATEFRACTION','Rainfall'])
    text.insert(END,str(dataset.head()))
    rainfall = dataset['Rainfall']
    time= dataset['DATEFRACTION']
    
    
def preprocess():
    global dataset
    global X, Y
    global Xtrain, Xtest, Ytrain, Ytest
    text.delete('1.0', END)
    dataset = dataset.values
    X = dataset[:,0]
    Y = dataset[:,1]
    X = X.reshape(-1, 1)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.1,random_state=0)
    text.insert(END,"Dataset contains total records : "+str(X.shape[0])+"\n")
    text.insert(END,"Totals Records used to train Machine Learning Algorithms : "+str(Xtrain.shape[0])+"\n")
    text.insert(END,"Totals Records used to test ML Algorithms Root Mean Square Error: "+str(Xtest.shape[0])+"\n")
    
def predict(algorithm_name, actual,forecast):
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Day No')
    plt.ylabel('Actual & Predicted Rainfall')
    plt.plot(actual, 'ro-', color = 'blue')
    plt.plot(forecast, 'ro-', color = 'green')
    plt.legend(['Actual Rainfall', 'Predicted Rainfall'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title(algorithm_name+' Rainfall Prediction Graph')
    plt.show()

def runSVM():
    global rmse_arr, accuracy_arr
    rmse_arr = []
    accuracy_arr = []
    text.delete('1.0', END)
    actual.clear()
    forecast.clear()
    global Xtrain, Xtest, Ytrain, Ytest
    svm_cls = SVR(C=1.0, epsilon=0.2)
    svm_cls.fit(Xtrain,Ytrain)
    prediction = svm_cls.predict(Xtest) 
    i = len(Ytest)-1
    while i > 0:
        actual.append(Ytest[i])
        forecast.append(prediction[i])
        i = i - 1
        if len(actual) > 30:
            break
    rmse = sqrt(mean_squared_error(Ytest,prediction))
    acc = svm_cls.score(Xtrain,Ytrain) * 100
    text.insert(END,"SVM RMSE : "+str(round(rmse,1))+"\n")
    text.insert(END,"SVM Accuracy : "+str(acc)+"\n\n")
    rmse_arr.append(rmse)
    accuracy_arr.append(acc)
    for i in range(len(actual)):
        status = "Low"
        if forecast[i] >= 100:
            status = "Very Heavy Rain"
        if forecast[i] >= 80 and forecast[i] < 100:
            status = "Heavy Rain"
        if forecast[i] >= 50 and forecast[i] < 80:
            status = "Moderate Rain"
        if forecast[i] >= 30 and forecast[i] < 50:
            status = "Light Rain"
        if forecast[i] < 30:
            status = "No Rain"
        text.insert(END,"Day "+str((i+1))+" Acutal Rainfall : "+str(actual[i])+" Predicted Rainfall : "+str(forecast[i])+" "+status+"\n\n")
    predict("SVM",actual, forecast)

    
def runRandomForest():
    global rmse_arr, accuracy_arr
    text.delete('1.0', END)
    actual.clear()
    forecast.clear()
    global Xtrain, Xtest, Ytrain, Ytest
    rf_cls = RandomForestRegressor()
    rf_cls.fit(Xtrain,Ytrain)
    prediction = rf_cls.predict(Xtest) 
    i = len(Ytest)-1
    while i > 0:
        actual.append(Ytest[i])
        forecast.append(prediction[i])
        i = i - 1
        if len(actual) > 30:
            break
    rmse = sqrt(mean_squared_error(Ytest,prediction))
    acc = rf_cls.score(Xtrain,Ytrain) * 100
    text.insert(END,"Random Forest RMSE : "+str(round(rmse,1))+"\n")
    text.insert(END,"Random Forest Accuracy : "+str(acc)+"\n\n")
    rmse_arr.append(rmse)
    accuracy_arr.append(acc)
    for i in range(len(actual)):
        status = "Low"
        if forecast[i] >= 100:
            status = "Very Heavy Rain"
        if forecast[i] >= 80 and forecast[i] < 100:
            status = "Heavy Rain"
        if forecast[i] >= 50 and forecast[i] < 80:
            status = "Moderate Rain"
        if forecast[i] >= 30 and forecast[i] < 50:
            status = "Light Rain"
        if forecast[i] < 30:
            status = "No Rain"
        text.insert(END,"Day "+str((i+1))+" Acutal Rainfall : "+str(actual[i])+" Predicted Rainfall : "+str(forecast[i])+" "+status+"\n\n")
    predict("Random Forest",actual, forecast)

def runDecisionTree():
    global rmse_arr, accuracy_arr
    text.delete('1.0', END)
    actual.clear()
    forecast.clear()
    global Xtrain, Xtest, Ytrain, Ytest
    dt_cls = DecisionTreeRegressor()
    dt_cls.fit(Xtrain,Ytrain)
    prediction = dt_cls.predict(Xtest) 
    i = len(Ytest)-1
    while i > 0:
        actual.append(Ytest[i])
        forecast.append(prediction[i])
        i = i - 1
        if len(actual) > 30:
            break
    rmse = sqrt(mean_squared_error(Ytest,prediction))
    acc = dt_cls.score(Xtrain,Ytrain) * 100
    text.insert(END,"Decision Tree RMSE : "+str(round(rmse,1))+"\n")
    text.insert(END,"Decision Tree Accuracy : "+str(acc)+"\n\n")
    rmse_arr.append(rmse)
    accuracy_arr.append(acc)
    for i in range(len(actual)):
        status = "Low"
        if forecast[i] >= 100:
            status = "Very Heavy Rain"
        if forecast[i] >= 80 and forecast[i] < 100:
            status = "Heavy Rain"
        if forecast[i] >= 50 and forecast[i] < 80:
            status = "Moderate Rain"
        if forecast[i] >= 30 and forecast[i] < 50:
            status = "Light Rain"
        if forecast[i] < 30:
            status = "No Rain"
        text.insert(END,"Day "+str((i+1))+" Acutal Rainfall : "+str(actual[i])+" Predicted Rainfall : "+str(forecast[i])+" "+status+"\n\n")
    predict("Decision Tree",actual, forecast)

def runNeuralNetwork():
    global rmse_arr, accuracy_arr, X, Y
    text.delete('1.0', END)
    actual.clear()
    forecast.clear()
    global Xtrain, Xtest, Ytrain, Ytest
    nn = MLPRegressor()
    nn.fit(X, Y)
    prediction = nn.predict(Xtest) 
    i = len(Ytest)-1
    while i > 0:
        actual.append(Ytest[i])
        forecast.append(prediction[i])
        i = i - 1
        if len(actual) > 30:
            break
    rmse = sqrt(mean_squared_error(Ytest,prediction))
    acc = nn.score(Xtrain,Ytrain) * 100
    text.insert(END,"Neural Network RMSE : "+str(round(rmse,1))+"\n")
    text.insert(END,"Neural Network Accuracy : "+str(acc)+"\n\n")
    rmse_arr.append(rmse)
    accuracy_arr.append(acc)
    for i in range(len(actual)):
        status = "Low"
        if forecast[i] >= 100:
            status = "Very Heavy Rain"
        if forecast[i] >= 80 and forecast[i] < 100:
            status = "Heavy Rain"
        if forecast[i] >= 50 and forecast[i] < 80:
            status = "Moderate Rain"
        if forecast[i] >= 30 and forecast[i] < 50:
            status = "Light Rain"
        if forecast[i] < 30:
            status = "No Rain"
        text.insert(END,"Day "+str((i+1))+" Acutal Rainfall : "+str(actual[i])+" Predicted Rainfall : "+str(forecast[i])+" "+status+"\n\n")
    predict("Neural Network",actual, forecast)


def runKNN():
    global rmse_arr, accuracy_arr, X, Y
    text.delete('1.0', END)
    actual.clear()
    forecast.clear()
    global Xtrain, Xtest, Ytrain, Ytest
    knn_cls = KNeighborsRegressor(n_neighbors=2)
    knn_cls.fit(X, Y)
    prediction = knn_cls.predict(Xtest) 
    i = len(Ytest)-1
    while i > 0:
        actual.append(Ytest[i])
        forecast.append(prediction[i])
        i = i - 1
        if len(actual) > 30:
            break
    rmse = sqrt(mean_squared_error(Ytest,prediction))
    acc = knn_cls.score(Xtrain,Ytrain) * 100
    text.insert(END,"KNN RMSE : "+str(round(rmse,1))+"\n")
    text.insert(END,"KNN Accuracy : "+str(acc)+"\n\n")
    rmse_arr.append(rmse)
    accuracy_arr.append(acc)
    for i in range(len(actual)):
        status = "Low"
        if forecast[i] >= 100:
            status = "Very Heavy Rain"
        if forecast[i] >= 80 and forecast[i] < 100:
            status = "Heavy Rain"
        if forecast[i] >= 50 and forecast[i] < 80:
            status = "Moderate Rain"
        if forecast[i] >= 30 and forecast[i] < 50:
            status = "Light Rain"
        if forecast[i] < 30:
            status = "No Rain"
        text.insert(END,"Day "+str((i+1))+" Acutal Rainfall : "+str(actual[i])+" Predicted Rainfall : "+str(forecast[i])+" "+status+"\n\n")
    predict("KNN Algorithm",actual, forecast)    

def graph():
    df = pd.DataFrame([['SVM','RMSE',rmse_arr[0]],['SVM','Accuracy',accuracy_arr[0]],
                       ['Random Forest','RMSE',rmse_arr[1]],['Random Forest','Accuracy',accuracy_arr[1]],
                       ['Decision Tree','RMSE',rmse_arr[2]],['Decision Tree','Accuracy',accuracy_arr[2]],
                       ['Neural Network','RMSE',rmse_arr[3]],['Neural Network','Accuracy',accuracy_arr[3]],
                       ['KNN','RMSE',rmse_arr[4]],['KNN','Accuracy',accuracy_arr[4]],                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.title("All Algorithms RMSE & Accuracy Comparison Graph")
    plt.show()
    
font = ('times', 15, 'bold')
title = Label(main, text='Rainfall Prediction: Accuracy Enhancement Using Machine Learning and Forecasting Techniques')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=5,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Rainfall Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=600,y=100)

processButton = Button(main, text="Preprocess Dataset", command=preprocess)
processButton.place(x=350,y=100)
processButton.config(font=font1)

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM)
svmButton.place(x=50,y=150)
svmButton.config(font=font1)

rfButton = Button(main, text="Run Random Forest", command=runRandomForest)
rfButton.place(x=350,y=150)
rfButton.config(font=font1)

dtButton = Button(main, text="Run Decision Tree Algorithm", command=runDecisionTree)
dtButton.place(x=50,y=200)
dtButton.config(font=font1)

nnButton = Button(main, text="Run Neural Network Algorithm", command=runNeuralNetwork)
nnButton.place(x=350,y=200)
nnButton.config(font=font1)

knnButton = Button(main, text="Run KNN Algorithm", command=runKNN)
knnButton.place(x=50,y=250)
knnButton.config(font=font1)

graphButton = Button(main, text="Accuracy Graph", command=graph)
graphButton.place(x=350,y=250)
graphButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
