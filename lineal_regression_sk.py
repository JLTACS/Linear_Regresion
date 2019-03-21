import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import pandas as pd 
import readData 
import matplotlib.pyplot as plt

def trainMatrix(train_data):
    a_mt = train_data.iloc[:,:len(train_data.columns)-1]
    y_mt = train_data.iloc[:,len(train_data.columns)-1:]

    a_mt.insert(0,'unos',1)

    return a_mt,y_mt

def lineal_regresion(a_mt, y_mt):
    a_matrix = a_mt.to_numpy(dtype=np.float32)
    y_matrix = y_mt.to_numpy(dtype=np.float32)

    regresion = linear_model.LinearRegression()
    regresion.fit(a_matrix,y_matrix)
    
    return regresion

def testEvaluation(test_data,regresion):
    eva_data = test_data.iloc[:,:len(test_data.columns)-1]
    eva_data.insert(0,'unos',1)
    eva_data = eva_data.to_numpy(dtype=np.float32)

    resultado = regresion.predict(eva_data)
    resultado = pd.DataFrame(resultado,columns=['Predicciones'])
    resultado['Real'] = test_data['resultado'].values

    return resultado

def main():
    np.set_printoptions(suppress=True)
    data = readData.sampleData("airfoil_self_noise_.csv",10,["var1","var2","var3","var4","var5","resultado"])
    train,test = readData.separateData(data,75)
    a,y = trainMatrix(train)
    print(a)
    print("-----------------------------------------")
    print(y)
    print("-----------------------------------------")
    regres = lineal_regresion(a,y)
    print(regres.coef_)
    result = testEvaluation(test,regres)
    print(result)
    plot_result = result.to_numpy(dtype=np.float32)
    error = np.sqrt(mean_squared_error(plot_result[1],plot_result[0]))
    print(error)


    plot_result = np.transpose(plot_result)
    plt.scatter(result.index,plot_result[0])
    plt.plot(result.index,plot_result[1])
    plt.show()


main()
