import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sktools import GradientBoostingFeatureGenerator as GB_generator
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE
from random import sample

def find_best_model(x,y,scorer):
    random_search = RandomizedSearchCV(
    estimator= GradientBoostingClassifier(verbose=5),
        param_distributions= dict(

            n_estimators= [500,1500,3000],
            max_depth = [2,3],
            learning_rate = [0.001,0.005,0.01,0.05],
            max_features = ["sqrt", "log2"],
            min_samples_leaf= [1, 2, 4],
            min_samples_split = [5, 10],   
        ),
        n_iter= 30,
        scoring = scorer,
        verbose = 5,
        cv = 10,
        n_jobs=-1
    )
    random_search.fit(x,y)
    return random_search



def find_path(node_numb, path, x,children_left,children_right): #De stack overflow: https://stackoverflow.com/questions/56334210/how-to-extract-sklearn-decision-tree-rules-to-pandas-boolean-conditions
        path.append(node_numb)
        if node_numb == x:
            return True
        left = False
        right = False
        if (children_left[node_numb] !=-1):
            left = find_path(children_left[node_numb], path, x,children_left,children_right) 
        if (children_right[node_numb] !=-1):
            right = find_path(children_right[node_numb], path, x,children_left,children_right)
        if left or right :
            return True
        path.remove(node_numb)
        return False


def get_rule(path, column_names,children_left,feature,threshold,df_name):#Credits to stack overflow answer: https://stackoverflow.com/questions/56334210/how-to-extract-sklearn-decision-tree-rules-to-pandas-boolean-conditions
    mask = ''
    for index, node in enumerate(path):
        if index!=len(path)-1:
            if (children_left[node] == path[index+1]):
                mask += "(df['{}']<= {}) \t ".format(column_names[feature[node]], np.round(threshold[node],2))
            else:
                mask += "(df['{}']> {}) \t ".format(column_names[feature[node]], np.round(threshold[node],2))
    mask = mask.replace("\t", "&", mask.count("\t") - 1)
    mask = mask.replace("\t", "")
    return mask
def extract_rules(model,data,children_left,children_right,feature,threshold,df_name):
    '''

    From a single tree, pulls out every branch in the form of a "rule" (which then can be evaluated on a dataframe).
    
    =========================
    Parameters:
    -model: A tree from a trained ensemble
    -data: (Unlabeled) data used for training the model
    -children_left,children_right,feature,threshold: contents of each tree
    '''
    leave_id = model.apply(data)

    paths ={}
    for leaf in np.unique(leave_id):
        path_leaf = []
        find_path(0, path_leaf, leaf,children_left,children_right)
        paths[leaf] = np.unique(np.sort(path_leaf))

    rules = {}
    for key in paths:
        rules[key] = get_rule(paths[key], data.columns,children_left,feature,threshold,df_name)
    return rules.values()
def select_random_rules(model,data,number_of_rules = None,branch_lengths = [3],verbose=0,max_trees = None,df_name = 'X_train'): #Se podría hacer más eficiente pero no tarda tanto
    '''
    Gets every rule from each tree until "max_trees" limit is hit. Then returns a sample of size "number_of_rules".
    '''

    if number_of_rules is None:
        number_of_rules = len(model.estimators_)
    rules = np.array([])
    for i in range(len(model.estimators_) if max_trees is None else max_trees):
        tree = model.estimators_[i,0]
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        if verbose != 0 and i % 100 == 0 and i != 0:
            print('Getting rules from tree n°{}'.format(i))
        new_rules = list(extract_rules(model,data,children_left,children_right,feature,threshold,df_name))
        rules = np.append(rules,new_rules)
    print('Selecting random rules')
    rules = [i for i in rules if i.count('&')+1 in branch_lengths]
    try:
        print('Returning rules')
        return sample(sorted(np.unique(rules)),number_of_rules)
    except:
        print('Not enough rules. Returning all available rules')
        return np.unique(rules)


def col_name_from_rule(rule): #2 Individual functions to extract rules and generate columns with it. A pandas error dit not allow me to make it 1 function
    '''
    
    Function to create a more readable column name from an existing rule.
    "ph" simply stands for place holder variable.
    
    
    '''
    ph = rule.split('&')
    ph = [i.split("'")[1:] for i in ph]
    ph = [''.join(x) for x in ph]
    ph = [s.replace(']',' ') for s in ph]
    col_name = '& '.join([g.replace(')','') for g in ph]).strip()
    return col_name

def make_column_from_rule(df,rule):
    '''

    Creates an array associated with the rule passed as an argument.
    
    '''
    df = df.copy()
    return np.select([eval(rule)],[1],default=0)

if __name__ == '__main__':#Testing
    pass 