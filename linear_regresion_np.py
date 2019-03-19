import numpy as np 
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
    
    print(a_matrix)

    a_transpose = np.transpose(a_matrix)
    at_a = np.matmul(a_transpose,a_matrix)
    at_y = np.matmul(a_transpose,y_matrix)

    b_mt = np.linalg.solve(at_a,at_y)
    return b_mt

def testEvaluation(test_data,b_vector):
    eva_data = test_data.iloc[:,:len(test_data.columns)-1]
    eva_data.insert(0,'unos',1)
    eva_data = eva_data.to_numpy(dtype=np.float32)

    resultado = np.matmul(eva_data,b_vector)
    resultado = pd.DataFrame(resultado,columns=['Predicciones'])
    resultado['Real'] = test_data['resultado'].values

    return resultado

def rmse(predictions):
    m = len(predictions.index)
    acu = 0
    for index, row in predictions.iterrows():
        acu += ((row['Real']-row['Predicciones'])**2)
    
    error = np.sqrt(acu/m)
    return error


def main():
    np.set_printoptions(suppress=True)
    data = readData.sampleData("airfoil_self_noise_.csv",10,["var1","var2","var3","var4","var5","resultado"])
    train,test = readData.separateData(data,75)
    a,y = trainMatrix(train)
    print(a)
    print("-----------------------------------------")
    print(y)
    print("-----------------------------------------")
    b_mt = lineal_regresion(a,y)
    print(b_mt)
    result = testEvaluation(test,b_mt)
    print(result)
    error = rmse(result)
    print(error)

'''
    plot_result = result.to_numpy(dtype=np.float32)
    plot_result = np.transpose(plot_result)
    plt.scatter(result.index,plot_result[0])
    plt.plot(result.index,plot_result[1])
    plt.show()
'''


main()