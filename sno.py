import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = "0"

import pandas as pd

df = pd.read_csv("application.csv")
df_train = df.loc[df["label"] == 2]
df_test = df.loc[(df["label"] == 0) | (df["label"] == 1)]
print(df_train)
Y_train = df_train.label.values

ABSTAIN = -1
BLACK = 1

from snorkel.labeling import labeling_function

@labeling_function()
def check_oneorderfraud0(x):
    return BLACK if x.oneorderfraud > 0 else ABSTAIN


@labeling_function()
def check_oneorderfraud1(x):
    return BLACK if x.oneorderfraud > 1 else ABSTAIN


@labeling_function()
def check_oneorderfraudratio0(x):
    return BLACK if x.oneorderfraudratio > 0 else ABSTAIN


@labeling_function()
def check_oneorderfraudratio10(x):
    return BLACK if x.oneorderfraudratio > 0.1 else ABSTAIN


@labeling_function()
def check_oneorderfraudratio35(x):
    return BLACK if x.oneorderfraudratio > 0.35 else ABSTAIN


@labeling_function()
def check_twoorderfraudratio0(x):
    return BLACK if x.twoorderfraudratio > 0 else ABSTAIN


@labeling_function()
def check_twoorderfraudratio5(x):
    return BLACK if x.twoorderfraudratio > 0.05 else ABSTAIN


@labeling_function()
def check_twoorderfraudratio40(x):
    return BLACK if x.twoorderfraudratio > 0.40 else ABSTAIN


@labeling_function()
def check_companyphone1(x):
    return BLACK \
        if x.companyphone > 1 \
           and x.oneorderfraud > 0 \
    else ABSTAIN
    
@labeling_function()
def LF_via_AIA(x):
    return BLACK \
        if x.NeighIdConsistency==false else ABSTAIN

@labeling_function()
def LF_via_AMA(x):
    return BLACK \
        if x.FraudNeighviaMobile==true else ABSTAIN


@labeling_function()
def LF_via_AAA(x):
    return BLACK \
        if x.NeighNumviaRegAdd>3 else ABSTAIN

@labeling_function()
def LF_via_ACA(x):
    return BLACK \
        if x.NeighNumviaCompany>4 else ABSTAIN


@labeling_function()
def check_mobile1(x):
    return BLACK if x.mobiledegree > 1 and x.oneorderfraud > 0 else ABSTAIN


@labeling_function()
def check_companyphone2(x):
    return BLACK if x.companyphone > 1 and x.oneorderfraudratio > 0.1 else ABSTAIN


@labeling_function()
def check_companyphone3(x):
    return BLACK if x.companyphone > 1 and x.oneorderfraudratio > 0.35 else ABSTAIN

from snorkel.labeling import PandasLFApplier

lfs = [check_oneorderfraud0,
       check_oneorderfraud1,
       check_oneorderfraudratio0,
       check_oneorderfraudratio10,
       check_oneorderfraudratio35,
       check_twoorderfraudratio0,
       check_twoorderfraudratio5,
       check_twoorderfraudratio40,
       check_companyphone1,
       check_companyphone2,
       check_companyphone3,
       check_mobile1,
       LF_via_AIA,
       LF_via_AMA, 
       LF_via_AAA,
       LF_via_ACA]
applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df_train)


from snorkel.labeling.model import LabelModel
label_model=LabelModel(cardinality=2,verbose=True)
label_model.fit(L_train=L_train,n_epochs=500,log_freq=100,seed=123)
probs_train=label_model.predict_proba(L=L_train)

from snorkel.labeling import filter_unlabeled_dataframe
df_train_filtered,probs_train_filtered=filter_unlabeled_dataframe(
    X=df_train,y=probs_train,L=L_train
)

from snorkel.utils import probs_to_preds
preds_train=probs_to_preds(probs=probs_train_filtered)

df_train_filtered['label']=preds_train
df_train_filtered.to_csv('app_snorkel.csv')
# preds_train=pd.DataFrame(preds_train,columns=['preds'])


df_train_filtered=df_train_filtered.set_index('newid',drop=True, append=False, inplace=False, verify_integrity=False)
df_train_filtered=df_train_filtered['label']
df.update(df_train_filtered)
df.to_csv('application_snorkel_merge.csv')
# print(df_train_filtered)