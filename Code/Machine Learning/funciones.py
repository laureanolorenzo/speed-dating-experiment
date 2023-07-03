import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

def convert_str(x):
    '''
    This function is used to turn string columns into float/int when necessary. Shouldn't be an issue when saving df as ".parquet"
    '''
    try:
        x = np.int8(x)
    except:
        try:
            x = float(x) 
        except:
            pass
    return x

def cargar_datos_nahuel(path):########################
    '''
    This 
    '''
    df = pd.read_csv(path,delimiter=';',low_memory=False) #Debería especificar dtypes
    old_names = df.columns
    new_names = ['attractive_important', 'sincere_important',
       'intellicence_important', 'funny_important',
       'ambtition_important', 'shared_interests_important']
    df.rename(columns=dict(zip(old_names, new_names)), inplace=True)
    df = df.applymap(lambda x: x.replace(',','.') if type(x) == str else x)
    return df

def pool_with_other(df,categoricas,threshold = 0.05): #Sólo porque hay poco soporte. Sino, podrían usarse los nans como otra categoría
    '''
    Función que agrega categorías con menos del 5% de soporte en "otros"
    '''
    for col in categoricas:
        for pair in zip(df[col].value_counts(normalize=True).index,df[col].value_counts(normalize=True)):
            if pair[1] < threshold:
                df[col] = np.where(df[col] == pair[0],'Other',df[col])
    return df


def filter_cols(df,
                drops = ['gender','has_null','wave',
                    'field','match','race',
                    'race_o','met','interests_correlate']
                    ):
    '''
    Some columns are not useful for our experiment. Columns containing "d_", for example, don't add information to the models. See "female_model.ipynb"
    '''

    drops.extend([i for i in df.columns if i.startswith('d_')])
    quiza_util = ['has_null','wave','field','match','met','interests_correlate'] 
    return df.drop(columns=drops),df[quiza_util] #Just in case we return the remaining columns as well.

def spot_round_cols(df): #Important: use this func. before imputing nans
    '''
    Function to identify columns which should be rounded after null imputation. 
    '''
    cols_to_round = []
    for i in df.columns:
        try:
            df[i].fillna(0).astype('float')
            try:
                df[i].fillna(0).astype('int')
                cols_to_round.append(i)
            except:
                pass
        except:
            pass
    return cols_to_round

def impute_nulls(df,objs = []):
    '''
    Scikit-learn's "Iterative Imputer" implementation
    '''
    objs.append('decision')
    X_train = df.drop(columns=objs).values
    cols_p_despues = df.drop(columns=objs).columns #Si hubiera cols del tipo object, habría que cambiar acá
    imputer = IterativeImputer(
        estimator= RandomForestRegressor(n_jobs=-1),
        max_iter=10,
        min_value=0, #We could see values of up to -1 (correlation columns) in the dataset, but we decide to set a limit at 0 so as to not risk changing other column values.
        **{'verbose':5},
    )
    imputer.fit(X_train)
    objs = df[objs]
    df = pd.DataFrame(imputer.transform(df.drop(columns=objs)),columns = cols_p_despues)
    return pd.concat([df,objs],axis=1)

def separar_por_tipo(df): #Not used currently
    not_useful = df[['met','race' ,'race_o','race','interests_correlate','field',]].copy()
    objetos = []
    for i in df.columns:
        try:
            df[i].fillna(0).astype('float')
        except:
            objetos.append(i)
    #df.drop(columns=['met', 'id_indiv','num_citas','race','race_o','race','interests_correlate'],inplace=True)
    pass
if __name__ == '__main__':
    pass 