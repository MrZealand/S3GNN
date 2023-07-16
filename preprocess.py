import pandas as pd
import numpy as np
from scipy import sparse
import dgl
from utils import get_binary_mask
import torch
from dgl.data.utils import download, get_download_dir, _get_dgl_url
import numpy as np  
import pandas as pd
import gc
from tqdm import tqdm

pd.set_option('display.max_rows', 20)


def MakeCsc(commondata, filepath):
    app_company = pd.read_csv(filepath)
    app_company = commondata.merge(app_company, on=['APPLID'], how='left').sort_values(by='newid', ascending=True)
    print(app_company)

    columns = app_company.columns.values
    APPLID = app_company['newid'].unique()
    app_company_dropna=app_company.dropna(axis=0,how='any')
    print(app_company_dropna)
    COMAID = app_company_dropna[columns[3]].unique()
    print(len(APPLID), len(COMAID))
    df = pd.DataFrame(data=0, index=APPLID, columns=COMAID, dtype=np.int8)

    for i in tqdm(range(len(app_company_dropna))):
        row_i = app_company_dropna.iloc[[i]]
        applid = row_i['newid'].tolist()[0]
        comaid = row_i[columns[3]].tolist()[0]
        df.at[applid, comaid] = 1
        del applid,comaid
        gc.collect()
    print(df)

    Csc = sparse.csc_matrix(np.array(df))
    del df
    gc.collect()
    npzpath, _ = filepath.split('.')
    sparse.save_npz(npzpath + '.npz', Csc)
    del Csc, app_company
    gc.collect()


def preprocess_train():
    app_phone = pd.read_csv("traindata6month/APPLYNO-PHONE-train.csv")
    app_mobile = pd.read_csv("traindata6month/APPLYNO-MOBILE-train.csv")
    app_company = pd.read_csv("traindata6month/APPLYNO-COMPANYNAME-train.csv")
    app_license = pd.read_csv('traindata6month/APPLYNO-LICENSE-train.csv')
    app_address = pd.read_csv('traindata6month/APPLYNO-ADDRESS-train.csv')
    app_idno = pd.read_csv('traindata6month/APPLYNO-IDNO-train.csv')
    app_vehc = pd.read_csv('traindata6month/APPLYNO-VEH_CHASSIS-train.csv')
    app_vehn = pd.read_csv('traindata6month/APPLYNO-VEH_NO-train.csv')

    phone = set(app_phone['APPLID'])
    mobile = set(app_mobile['APPLID'])
    commondata = list(phone & mobile)
    newid = np.array(range(len(commondata)))
    data = {'APPLID': commondata, 'newid': newid}
    commondata = pd.DataFrame(data)
    commondata.to_csv('traindata6month/commondata.csv')
    print(commondata)

    commondata = list(pd.read_csv("traindata6month/commondata.csv")['APPLID'])
    app_phone = app_phone.loc[app_phone['APPLID'].isin(commondata)]
    app_phone.to_csv('traindata6month/app-phone.csv', index=False, encoding='utf-8')
    app_mobile = app_mobile.loc[app_mobile['APPLID'].isin(commondata)]
    app_mobile.to_csv('traindata6month/app-mobile.csv', index=False, encoding='utf-8')
    app_company = app_company.loc[app_company['APPLID'].isin(commondata)]
    app_company.to_csv('traindata6month/app-company.csv', index=False, encoding='utf-8')
    app_license = app_license.loc[app_license['APPLID'].isin(commondata)]
    app_license.to_csv('traindata6month/app-license.csv', index=False, encoding='utf-8')
    app_address = app_address.loc[app_address['APPLID'].isin(commondata)]
    app_address.to_csv('traindata6month/app-address.csv', index=False, encoding='utf-8')
    app_idno = app_idno.loc[app_idno['APPLID'].isin(commondata)]
    app_idno.to_csv('traindata6month/app-idno.csv', index=False, encoding='utf-8')
    app_vehc = app_vehc.loc[app_vehc['APPLID'].isin(commondata)]
    app_vehc.to_csv('traindata6month/app-vehc.csv', index=False, encoding='utf-8')
    app_vehn = app_vehn.loc[app_vehn['APPLID'].isin(commondata)]
    app_vehn.to_csv('traindata6month/app-vehn.csv', index=False, encoding='utf-8')

    app_fasttext = pd.read_csv("traindata6month/fasttext.csv")
    app_oneorder = pd.read_csv("traindata6month/APPLYNO-1order-train.csv")
    app_twoorder = pd.read_csv("traindata6month/APPLYNO-2order-train.csv")
    app_oneorderfraud = pd.read_csv('traindata6month/APPLYNO-1orderfraud-train.csv')
    app_twoorderfraud = pd.read_csv('traindata6month/APPLYNO-2orderfraud-train.csv')
    app_degree = pd.read_csv('traindata6month/APPLYNO-degree-train.csv')

    app = commondata.merge(app_oneorder, on=['APPLID'], how='left').fillna(0)
    app = app.merge(app_twoorder, on=['APPLID'], how='left').fillna(0)
    app = app.merge(app_oneorderfraud, on=['APPLID'], how='left').fillna(0)
    app = app.merge(app_twoorderfraud, on=['APPLID'], how='left').fillna(0)
    app = app.merge(app_degree, on=['APPLID'], how='left').fillna(0)
    app = app.merge(app_idno, on=['APPLID'], how='left').fillna(0)
    app = app.merge(app_address, on=['APPLID'], how='left').fillna(0)

    app = app.sort_values(by='newid', ascending=True)
    app['oneorderfraudratio'] = (app['oneorderfraud'] / app['oneordercount']).fillna(0)
    app['twoorderfraudratio'] = (app['twoorderfraud'] / app['twoordercount']).fillna(0)
    app.to_csv('traindata6month/app.csv', index=False, encoding='utf-8')


    commondata = pd.read_csv('traindata6month/commondata.csv')

    MakeCsc(commondata, 'traindata6month/app-phone.csv')
    MakeCsc(commondata, 'traindata6month/app-mobile.csv')
    MakeCsc(commondata, 'traindata6month/app-company.csv')
    MakeCsc(commondata, 'traindata6month/app-license.csv')
    MakeCsc(commondata, 'traindata6month/app-address.csv')
    MakeCsc(commondata, 'traindata6month/app-idno.csv')
    MakeCsc(commondata, 'traindata6month/app-vehc.csv')
    MakeCsc(commondata, 'traindata6month/app-vehn.csv')


def preprocess_test():
    app_phone = pd.read_csv("testdata7month/APPLYNO-PHONE-test.csv")
    app_mobile = pd.read_csv("testdata7month/APPLYNO-MOBILE-test.csv")
    app_company = pd.read_csv("testdata7month/APPLYNO-COMPANYNAME-test.csv")
    app_license = pd.read_csv('testdata7month/APPLYNO-LICENSE-test.csv')
    app_address = pd.read_csv('testdata7month/APPLYNO-ADDRESS-test.csv')
    app_idno = pd.read_csv('testdata7month/APPLYNO-IDNO-test.csv')
    app_vehc = pd.read_csv('testdata7month/APPLYNO-VEH_CHASSIS-test.csv')
    app_vehn = pd.read_csv('testdata7month/APPLYNO-VEH_NO-test.csv')

    phone = set(app_phone['APPLID'])
    mobile = set(app_mobile['APPLID'])
    company = set(app_company['APPLID'])
    license=set(app_license['APPLID'])
    address=set(app_address['APPLID'])
    idno=set(app_idno['APPLID'])
    vehc=set(app_vehc['APPLID'])
    vehn=set(app_vehn['APPLID'])

    commondata = list(phone &
                      mobile &
                       company &
                       license &
                       address &
                       idno &
                       vehc &
                       vehn
                      )

    newid = np.array(range(len(commondata)))
    data = {'APPLID': commondata, 'newid': newid}
    commondata = pd.DataFrame(data)
    commondata.to_csv('testdata7month/commondata.csv')
    print(commondata)

    commondata = list(pd.read_csv("testdata7month/commondata.csv")['APPLID'])
    app_phone = app_phone.loc[app_phone['APPLID'].isin(commondata)]
    app_phone.to_csv('testdata7month/app-phone.csv', index=False, encoding='utf-8')
    app_mobile = app_mobile.loc[app_mobile['APPLID'].isin(commondata)]
    app_mobile.to_csv('testdata7month/app-mobile.csv', index=False, encoding='utf-8')
    app_company = app_company.loc[app_company['APPLID'].isin(commondata)]
    app_company.to_csv('testdata7month/app-company.csv', index=False, encoding='utf-8')
    app_license = app_license.loc[app_license['APPLID'].isin(commondata)]
    app_license.to_csv('testdata7month/app-license.csv', index=False, encoding='utf-8')
    app_address = app_address.loc[app_address['APPLID'].isin(commondata)]
    app_address.to_csv('testdata7month/app-address.csv', index=False, encoding='utf-8')
    app_idno = app_idno.loc[app_idno['APPLID'].isin(commondata)]
    app_idno.to_csv('testdata7month/app-idno.csv', index=False, encoding='utf-8')
    app_vehc = app_vehc.loc[app_vehc['APPLID'].isin(commondata)]
    app_vehc.to_csv('testdata7month/app-vehc.csv', index=False, encoding='utf-8')
    app_vehn = app_vehn.loc[app_vehn['APPLID'].isin(commondata)]
    app_vehn.to_csv('testdata7month/app-vehn.csv', index=False, encoding='utf-8')

    app_oneorder = pd.read_csv("testdata7month/APPLYNO-1order-test.csv")
    app_twoorder = pd.read_csv("testdata7month/APPLYNO-2order-test.csv")
    app_oneorderfraud = pd.read_csv('testdata7month/APPLYNO-1orderfraud-test.csv')
    app_twoorderfraud = pd.read_csv('testdata7month/APPLYNO-2orderfraud-test.csv')
    app_degree = pd.read_csv('traindata6month/APPLYNO-degree-test.csv')
    
    commondata = pd.read_csv('testdata7month/commondata.csv')
    app = commondata.merge(app_oneorder, on=['APPLID'], how='left').fillna(0)
    app = app.merge(app_twoorder, on=['APPLID'], how='left').fillna(0)
    app = app.merge(app_oneorderfraud, on=['APPLID'], how='left').fillna(0)
    app = app.merge(app_twoorderfraud, on=['APPLID'], how='left').fillna(0)


    app = app.sort_values(by='newid', ascending=True)
    app['oneorderfraudratio'] = (app['oneorderfraud'] / app['oneordercount']).fillna(0)
    app['twoorderfraudratio'] = (app['twoorderfraud'] / app['twoordercount']).fillna(0)
    app['fraudratio'] = (
            (app['oneorderfraud'] + app['twoorderfraud']) / (app['oneordercount'] + app['twoordercount'])).fillna(0)

    app.to_csv('testdata7month/app.csv', index=False, encoding='utf-8')
    del app
    gc.collect()

    commondata = pd.read_csv('testdata7month/commondata.csv')

    MakeCsc(commondata, 'testdata7month/app-vehn.csv')
    MakeCsc(commondata, 'testdata7month/app-phone.csv')
    MakeCsc(commondata, 'testdata7month/app-mobile.csv')
    MakeCsc(commondata, 'testdata7month/app-company.csv')
    MakeCsc(commondata, 'testdata7month/app-license.csv')
    MakeCsc(commondata, 'testdata7month/app-address.csv')
    MakeCsc(commondata, 'testdata7month/app-idno.csv')
    MakeCsc(commondata, 'testdata7month/app-vehc.csv')


preprocess_test()
preprocess_train()
