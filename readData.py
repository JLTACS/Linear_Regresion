import pandas as pd 
import random

def sampleData (file, sample, nms = None):
    head = 0
    if nms != None:
        head = None

    df = pd.read_csv(file,sep='\t',header=head, names=nms)
    sample = sample/100.0

    for i in range(0,len(df.index)):
        rn = random.random()
        if rn > sample:
            df = df.drop(i)
    df = df.reset_index(drop=True)
    with open("mi_data.csv",mode='w',newline='') as f:
        df.to_csv(f)
    return df

def separateData(data,train):
    df_train = pd.DataFrame(columns = data.columns)
    df_test = pd.DataFrame(columns = data.columns)

    train = train/100.0
    for i in range(0,len(data.index)):
        rn = random.random()
        if rn <= train:
            df_train = df_train.append(data.iloc[[i]])
        else:
            df_test = df_test.append(data.iloc[[i]])
    
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    with open("mi_train.csv",mode='w',newline='') as f:
        df_train.to_csv(f)
    with open("mi_test.csv",mode='w',newline='') as f:
        df_test.to_csv(f)
    return df_train,df_test
    