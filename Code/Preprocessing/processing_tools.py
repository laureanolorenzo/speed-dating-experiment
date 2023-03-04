import pandas as pd
import numpy as np

def rank_cols(col_1=float,col_2=float,col_3=float,col_4=float,col_5=float,col_6=float,keys=list): #No se usó esta versión al final
    '''
    ===========================

    This function takes exactly 6 columns with float values, and their names in the form of a list. It then ranks them according
    the column values (where the column with the highest value is assigned the number 1, and the column with the 
    lowest value the number 6). If there are tied positions, those columns are given the average of the positions they 
    would've been assigned if there was no tie.
    ===========================

    Returns: pandas.Series object
    ===========================

    Arguments:
        -col_1 through col_6: float, the column values.
        -keys: list of strings, the column names.
    ===========================
    
    This function is meant to be used in combination with the pandas method "apply" in order to generate new columns for
    a dataframe.

    '''
    vals = [col_1,col_2,col_3,col_4,col_5,col_6]
    dct = {keys[i]:vals[i] for i in range(len(keys))}
    ser = pd.Series(dct).sort_values(ascending=False).reset_index().rename(columns={0:'value'})
    ser['value'] = ser['value'].astype('float')
    ser['order'] = ser.index + 1
    index = ser.index[-1]
    suma = []
    pos = []
    while index != -1:
        data = ser.loc[index]
        if index == 0:
            if data['value'] == ser.loc[index+1,('value',)]:
                suma.append(data['order'])
                pos.append(index)
                for i in pos:
                    ser.loc[i,('order',)] = np.mean(suma)
            else:
                pass
            break
        if data['value'] == ser.loc[index-1,('value',)]:
            pos.append(index)
            suma.append(data['order'])
        else:
            if len(suma):
                pos.append(index)
                suma.append(data['order'])
                for i in pos:
                    ser.loc[i,('order',)] = np.mean(suma)
                pos = []
                suma = []
        index -= 1
    ser.set_index('index',inplace=True)
    dct = dict(ser['order'])
    new_col_1 = dct[keys[0]]
    new_col_2 = dct[keys[1]]
    new_col_3 = dct[keys[2]]
    new_col_4 = dct[keys[3]]
    new_col_5 = dct[keys[4]]
    new_col_6 = dct[keys[5]]
    return pd.Series([new_col_1,new_col_2,new_col_3,new_col_4,new_col_5,new_col_6],index=keys)

def convert_str(x):
    '''
    ===========================
    Converts objects into int if possible, else float, or returns the object if none of the previous options are possible.
    ===========================
    Returns: Converted obj if possible, else obj.
    ===========================
    Parameters: 
        -x: object.
    ===========================
    If used in combination with pandas applymap, can convert whole columns from a dataframe into numeric, but keeping
    string columns without raising ValueError.
    '''
    try:
        x = np.int8(str(x))
    except:
        try:
            x = float(x) #No pude usar float16.
        except:
            pass
    return x

if __name__ == '__main__':

    #Mini tests to see if everything is working properly
    data = {'col_1': np.random.random_integers(0, 100, 100),
            'col_2': np.random.random_integers(0, 100, 100),
            'col_3': np.random.random_integers(0, 100, 100),
            'col_4': np.random.random_integers(0, 100, 100),
            'col_5': np.random.random_integers(0, 100, 100),
            'col_6': np.random.random_integers(0, 100, 100)}

    df = pd.DataFrame(data)

    # 
    important = ['col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6']
    print('Before:')
    print(df.head())
    print('\n\n\nAfter:')

    df[important] = df.apply(lambda x: rank_cols(x[important[0]], x[important[1]], x[important[2]],
                                                        x[important[3]], x[important[4]], x[important[5]],
                                                        keys=important), axis=1).values.astype('float32')

    print(df)

    df_2 = pd.DataFrame({'col_1':['1','2','3'],'col_2':['1.1','1.2','1.3'],'col_3':['Hello','World',np.nan]})
    print('Before:')
    print(df_2.info())
    print('\n\n\nAfter:')
    print(df_2.applymap(convert_str).info())