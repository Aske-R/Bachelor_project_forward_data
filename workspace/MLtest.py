import uproot
import numpy as np
import lightgbm 
import pandas as pd
import shap
import xgboost as xgb
from xgboost import XGBClassifier, DMatrix, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn import model_selection, metrics # additional sklearn functions
from matplotlib import pyplot as plt
from pathlib import Path
from datetime import datetime

#open the root file using uproot
def compute_model(alg,path_to_data,path_to_tree,path_to_weights,name_reqs,name_excepts,useTrainCV=True,small_size=True):
    tree = uproot.open(path_to_data)[path_to_tree]
    # tree = uproot.open("../data/MixedTest.root")[b'tree;1']
    leaf_names=tree.keys()


    subs= name_reqs
    neg_subs=name_excepts
    for a in range(len(subs)):
        leaf_names_p = [i for i in leaf_names if subs[a] in  i]

    for a in range(len(neg_subs)):
        leaf_names_p = [i for i in leaf_names_p if neg_subs[a] not in  i]

    #Get using pandas
 

    train_var , sig_var = tree.pandas.df(leaf_names_p) , tree.pandas.df('p_TruthType')
    weights=pd.DataFrame({'weights':(np.loadtxt(path_to_weights))})
    #n_var=len(train_var.columns)
    #add the previously created weights
    #weights=pd.DataFrame({'weights':(np.loadtxt('../data/total_weights_test'))})
    train_var_w_weights=pd.concat([train_var,weights],ignore_index=False, axis=1)
    train_var_w_weights['weights']=train_var_w_weights['weights'].fillna(1)

    if small_size:
        train_var_w_weights, sig_var =train_var_w_weights.sample(frac=0.4,replace=False, random_state=123),sig_var.sample(frac=0.4,replace=False, random_state=123)
    
    #weights_resized=train_var_w_weights[10]
    #train_var=train_var_w_weights.drop([10],axis=1)

    #split the data into a training sample and a test sample 
    X_train, X_test, y_train, y_test = train_test_split(train_var_w_weights,sig_var,test_size=0.2,random_state=12)
    #save the weights after the randomizing of the tests
    X_train_weights = X_train['weights']
    X_train, X_test = X_train.drop(['weights'],axis=1), X_test.drop(['weights'],axis=1) 


    shap.initjs() #JS visulalization code

    model = alg #lightgbm.LGBMRegressor(objective='binary',random_state=5)
    
    if useTrainCV:
        print(model.get_xgb_params)
        xgb_param = model.get_xgb_params()
        train_var_w_weights_dmatrix=DMatrix(train_var_w_weights,sig_var)
        cvresult = xgb.cv(xgb_param, train_var_w_weights_dmatrix, num_boost_round=model.get_params()['n_estimators'], nfold=5,
        metrics={'auc'}, early_stopping_rounds=50, verbose_eval=None)
        model.set_params(n_estimators=cvresult.shape[0])
        print(model.get_xgb_params)
    

    
    model.fit(X_train,y_train.values.flatten(),sample_weight=X_train_weights)
    #print(lgbm._eval_results)
    y_pred = model.predict(X_test)
    y_predprob = model.predict_proba(X_test)[:,1]
    y_predprob_train = model.predict_proba(X_train)[:,1]
    
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(y_test.values, y_pred))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, y_predprob_train))
    print ("AUC Score (Test): %f" % metrics.roc_auc_score(y_test, y_predprob))
 
    feat_imp = pd.Series(model.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    print(metrics.r2_score(y_test,y_pred))
    return model, X_train, X_test, y_test, y_pred, y_train
def compute_lightgbm(path_to_data,path_to_tree,path_to_weights,name_reqs,name_excepts):
    tree = uproot.open(path_to_data)[path_to_tree]
    leaf_names = tree.keys()
    subs= name_reqs
    neg_subs=name_excepts
    for a in range(len(subs)):
        leaf_names_p = [i for i in leaf_names if subs[a] in  i]
     
    for a in range(len(neg_subs)):
        leaf_names_p = [i for i in leaf_names_p if neg_subs[a] not in  i]
    
    #Get using pandas
    train_var , sig_var = tree.pandas.df(leaf_names_p) , tree.pandas.df('p_TruthType')
    print(sig_var)
    sig_var[sig_var !=2] =0
    print(sig_var)
    # #n_var=len(train_var.columns)
    # #add the previously created weights
    weights=pd.DataFrame({'weights':(np.loadtxt(path_to_weights))})
    #weights=pd.DataFrame({'weights':(np.loadtxt('../data/total_weights_test'))})
    train_var_w_weights=pd.concat([train_var,weights],ignore_index=False, axis=1)

    train_var_w_weights['weights']=train_var_w_weights['weights'].fillna(1)
    #weights_resized=train_var_w_weights[10]
    #train_var=train_var_w_weights.drop([10],axis=1)
    
    # not used for now
    #train_var_w_weights_dmatrix=DMatrix(train_var_w_weights,sig_var)
    #print(train_var_w_weights.values(),train_var_w_weights.index())
    
    
    #split the data into a training sample and a test sample 
    X_train, X_test, y_train, y_test = train_test_split(train_var_w_weights,sig_var,test_size=0.2,random_state=12)
    #save the weights after the randomizing of the tests
    X_train_weights = X_train['weights']
    X_train, X_test = X_train.drop(['weights'],axis=1), X_test.drop(['weights'],axis=1) 
    model=lightgbm.LGBMClassifier(random_state=5)
    start = datetime.now() 
    model.fit(X_train,y_train.values.flatten(),sample_weight=X_train_weights)
    stop = datetime.now()
    execution_time_lgbm = stop-start
    print(execution_time_lgbm)
    y_pred = model.predict(X_test)
    return model, X_train, X_test, y_test, y_pred, y_train

def compute_shap(ML_model,X_train,X_test,get_shap=False):   
    if get_shap:
        # følgende er nok en forkert måde at implementerer shap på
        explainer = shap.TreeExplainer(ML_model) 
        shap_values = explainer.shap_values(X_train)
        # jeg må indrømme at jeg ikke ved hvilken af følgende der skal bruges.

        #X_train_sample = shap.sample(X_train, 100)
        #X_test_sample = shap.sample(X_test, 100)
        #print(X_train_sample)
        #explainer = shap.KernelExplainer(ML_model.predict_proba, X_train_sample, link="logit")
        #shap_values = explainer.shap_values(X_test_sample, nsamples=100)
        #shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X_test.iloc[0,:], link="logit")
        shap.summary_plot(shap_values, X_train, plot_type="bar")
    return

def compute_roc(y_true, y_pred, plot=False):
        """
        TODO
        :param y_true: ground truth
        :param y_pred: predictions 
        :param plot:
        :return:
        """
        fpr, tpr, _ = metrics.roc_curve(y_true, y_pred,pos_label=2)
        #print(fpr,"   and    ",tpr)
        auc_score = metrics.auc(fpr, tpr)
        if plot:
            plt.figure(figsize=(7, 6))
            plt.plot(tpr, fpr,'r,-', markersize=1,
                    label='ROC (AUC = %0.4f)' % auc_score)
            plt.yscale('log')
            plt.grid=(True)
            #plt.gca().invert_yaxis()
            #plt.gca().invert_xaxis()
            plt.legend(loc='upper left')
            plt.title("ROC Curve")
            plt.xlabel("TPR")
            plt.ylabel("FPR")
            plt.savefig('../graphs/First_ROC')
            plt.show()

        return


name_reqs= [b'p_',b'Cluster']
name_excepts=[b'mc',b'weight',b'p_TruthType',b'Truth',b'LHVal']
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
str_to_parent_folder=str(Path(__file__).resolve().parent.parent)
model, X_train, X_test, y_test, y_pred, y_train= compute_lightgbm(str_to_parent_folder+"/forward_MC/user.lehrke.mc16_13TeV.361106.Zee.EGAM8.e3601_e5984_s3126_r10724_r10726_p3648.ePID18_NTUP3_v01_myOutput.root/user.lehrke.17118381._000003.myOutput.root",b'tree;1',str_to_parent_folder+'/weights/weights_MC_03.csv',name_reqs,name_excepts)

param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}

def grid_search_1(X_train,y_train):
    
    gsearch1 = model_selection.GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
    min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
    objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
    param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    gsearch1.fit(X_train,y_train)
    return gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_




compute_roc(y_test,y_pred,plot=True)
compute_shap(model,X_train,X_test,get_shap=True)

#lightdata = lightgbm.Dataset(X_train)
#lightdata.construct()
#print(lightdata.num_feature())

#tree = file["events"]
#tree.keys()