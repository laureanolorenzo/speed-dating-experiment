import pandas as pd
import numpy as np
from model_fitting import make_column_from_rule,col_name_from_rule

def get_associated_rules(col,rules):
    '''
    
    Función para recuperar las reglas asociadas al nombre de una columna con una regla. Sólo funciona por la forma en que 
    están escritas las reglas, no es muy generalizable!

    '''
    return [r for r in rules if f"['{col}']" in r]

def calculate_partial_dependence(col,X,model,rules=None):
    '''
    Calcula el promedio en las probabilidad(P(Y=1)) asociado a la respuesta para cada valor único (permutado) de 
    la predictora pasada como argumento. Sólo funciona para var. discretas, de tener una var. contínua, se deberían 
    usar otros métodos.
    Importante: La variable scores contiene las probabilidades asociadas a la 1° clase (de acuerdo al orden 
    de las clases en el modelo, model.classes_), y scores_o las probabilidades asociadas a la segunda.
    '''

    if rules is None:
        rules = []    
    copy = X.copy()
    scores,scores_o = dict(),dict()
    asoc_rules = get_associated_rules(col,rules)
    for i in copy[col].unique():
        copy[col] = i
        for rule in asoc_rules:
            col_name = col_name_from_rule(copy,rule)
            copy[col_name] = make_column_from_rule(copy,rule)
        scores[i] = np.round(np.mean(model.predict_proba(copy)[:,0]),4)
        scores_o[i] = np.round(np.mean(model.predict_proba(copy)[:,1]),4)
    scores_o = pd.DataFrame(scores_o.values(),columns=['Partial Dependence'],index=scores_o.keys())
    scores = pd.DataFrame(scores.values(),columns=['Partial Dependence'],index=scores.keys())
    scores_o.index.name, scores.index.name = col,col
    return scores.sort_index(ascending=True),scores_o.sort_index(ascending=True)

def permutar_col(col,df,features): 
    '''

    Función para intercambiar los valores de una columna, junto con los de toda columna que interactúe con ella.
    Retorna un nuevo dataframe.

    ====================================
    
    Parámetros:
    -col: la columna a ser permutada
    -df: el dataframe original
    -features: Las columnas del dataframe con las que fue entrenado el modelo.

    '''
    df = df.copy()
    interseccion = [c for c in features if col + ' ' in c] #Espacio para evitar interacciones como "attractive_important y attractive", etc
    interseccion.append(col)
    df_per = df.copy()
    for i in interseccion:
        df_per[i] = np.random.permutation(df[i])
    return df_per
def calcular_feat_importance(col,X,y,model,score,features,verbose = 0,n_iter = 50): #Antes hay que agregar las interacciones al test set si se desea usar este
    '''
    Calcula el cambio porcentual en la tasa de error de predicción al permutar una columna y sus interacciones.

    ====================================

    Parámetros:
    -col: la columna a ser permutada
    -X: los datos de las variables predictoras
    -y: los datos de la variable objetivo
    -model: el modelo a usar para las predicciones
    -score: función para el cálculo de la métrica a ser evaluada. Debe aceptar como argumentos las etiquetas verdaderas y las predicciones

    '''
    X = X.copy()
    y_pred = model.predict(X)
    error = 1 - score(y,y_pred)
    errors = []
    for i in range(n_iter): #Para mayor robustez
        if verbose > 0:
            print('Iteration n° {}'.format(i+1))
        X_per = permutar_col(col,X,features)
        y_pred_per = model.predict(X_per)
        error_per = 1 - score(y,y_pred_per)
        errors.append(error_per)
    mean_error = np.mean(errors)
    return np.round(((mean_error - error)/ error) * 100,2)