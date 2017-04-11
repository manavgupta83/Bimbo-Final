
# coding: utf-8

# In[5]:

import pandas as pd
import numpy as np
import random
import requests, zipfile
import time
import xgboost as xgb
import math
from sklearn.cross_validation import train_test_split


start_time = time.time()

print ('data load start')
####IMPORT THE DATA DIRECTLY AS ZIP FILES
train_data = pd.read_csv('train_csv.zip', compression='zip', header=0, sep=',', quotechar='"'
                         ,usecols = ['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID','Semana','Demanda_uni_equil']
                         ,dtype  = {'Semana' : 'int8',
                                    'Agencia_ID' :'int16',
                                    'Canal_ID' : 'int8',
                                    'Ruta_SAK' : 'int16',
                                    'Cliente_ID' : 'int32',
                                    'Producto_ID':'int32',
                                    'Demanda_uni_equil':'int16'}
                        )

train_data['log_Demanda_uni_equil'] = np.log1p(train_data['Demanda_uni_equil'])
train_data = train_data.drop(['Demanda_uni_equil'], axis = 1)

overall_mean = train_data['log_Demanda_uni_equil'].mean()
overall_median = train_data['log_Demanda_uni_equil'].median()

train_data['overall_mean'] = overall_mean
train_data['overall_median'] = overall_median

print ("----------------train file loaded in %s seconds-------------------" % (time.time() - start_time))

start_time = time.time()
test_data = pd.read_csv('test_csv.zip', compression = 'zip', header=0, sep = ',', quotechar = '"'
                        ,dtype  = { 'id' : 'int32',
                                    'Semana' : 'int8',
                                    'Agencia_ID' :'int16',
                                    'Canal_ID' : 'int8',
                                    'Ruta_SAK' : 'int16',
                                    'Cliente_ID' : 'int32',
                                    'Producto_ID':'int32'}
                       )

test_data['overall_mean'] = overall_mean
test_data['overall_median'] = overall_median

print ("----------------test file loaded in %s seconds-------------------" % (time.time() - start_time))
# submission_data = pd.read_csv('sample_submission_csv.zip', compression = 'zip', header=0, sep = ',', quotechar = '"')


###################LOADING TOWN STATE DATA
town_data = pd.read_csv('town_state_csv.zip', compression = 'zip', header=0, sep = ',', quotechar = '"',encoding='utf-8')
unique_s = town_data[['State']].drop_duplicates()   
unique_s['state_id'] = np.arange(len(unique_s))
unique_t = town_data[['Town']].drop_duplicates()   
unique_t['town_id'] = np.arange(len(unique_t))
town_data = town_data.merge(unique_s, 'inner', on = 'State')
town_data = town_data.merge(unique_t, 'inner', on = 'Town')

town_data['state_id'] = "S_"+town_data['state_id'].map(str)
town_data['town_id'] = "T_"+town_data['town_id'].map(str)

town_data = town_data.drop(['Town','State'], axis = 1)

print ('data load complete')

###################### LOAD CLIENT DATA
client_data = pd.read_csv('cliente_tabla_csv.zip', compression = 'zip', header=0, sep = ',', quotechar = '"')
client_data = client_data.drop_duplicates('Cliente_ID')

                    
    
    
                    ######################### PRODUCT INFORMATION EXTRACTION ##################
product_table = pd.read_csv('producto_tabla_csv.zip', compression = 'zip', header=0, sep = ',', quotechar = '"')

product_table['short_name'] = product_table.NombreProducto.str.extract('^(\D*)', expand=False)
w = product_table.NombreProducto.str.extract('(\d+)(Kg|g)', expand=True)
product_table['weight'] = w[0].astype('float')*w[1].map({'Kg':1000, 'g':1})
product_table['pieces'] =  product_table.NombreProducto.str.extract('(\d+)p ', expand=False).astype('float')
product_table['barrita'] = product_table.short_name.str.contains('barri', case = False).astype('int32')
product_table['weigh_per_piece'] = product_table['weight']/product_table['pieces']
product_table['weight'] = product_table['weight'].fillna(0)
product_table['pieces'] = product_table['pieces'].fillna(0)
product_table['weigh_per_piece'] = product_table['weigh_per_piece'].fillna(0)

product_table = product_table.drop(['NombreProducto','short_name'], axis = 1)

# DUMMIFY THE CATEGORICAL FIELDS
all_fields = product_table.columns.values
num_fields = product_table._get_numeric_data()
cat_fields = list(set(all_fields) - set(num_fields))

product_table = pd.get_dummies(product_table, columns = cat_fields)


# In[ ]:

# full_time = time.time()


# ######## 5 WAY ASSOCIATIONS 
# ### median,mean,max,min by sales depot, sales channel, route, client, product 
# start_time = time.time()

# agg_overall_median = train_data['log_Demanda_uni_equil'].median()
# agg_overall_median = train_data['log_Demanda_uni_equil'].mean()

# agg_5w = train_data.groupby(['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID'])\
#                             ['log_Demanda_uni_equil'].agg([('median_5w',np.median),('mean_5w',np.mean),\
#                                                        ('min_5w',np.min),('max_5w',np.max)]).reset_index() 

# ######## 4 WAY ASSOCIATIONS
# agg_4w_1 = train_data.groupby(['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID'])['log_Demanda_uni_equil']\
#                             .agg([('median_4w_1',np.median),('mean_4w_1',np.mean),('max_4w_1',np.max)]).reset_index()

# agg_4w_2 = train_data.groupby(['Agencia_ID','Ruta_SAK','Cliente_ID','Producto_ID'])['log_Demanda_uni_equil']\
#                             .agg([('median_4w_2',np.median),('mean_4w_2',np.mean),('max_4w_2',np.max)]).reset_index()

# agg_4w_3 = train_data.groupby(['Agencia_ID','Canal_ID','Cliente_ID','Producto_ID'])['log_Demanda_uni_equil']\
#                             .agg([('median_4w_3',np.median),('mean_4w_3',np.mean),('max_4w_3',np.max),('min_4w_3',np.min)]).reset_index()

# agg_4w_4 = train_data.groupby(['Agencia_ID','Canal_ID','Ruta_SAK','Producto_ID'])['log_Demanda_uni_equil']\
#                             .agg([('median_4w_4',np.median),('mean_4w_4',np.mean),('max_4w_4',np.max),('min_4w_4',np.min)]).reset_index()


# ######## 3 WAY ASSOCIATIONS
# agg_3w_1 = train_data.groupby(['Agencia_ID','Canal_ID','Ruta_SAK'])['log_Demanda_uni_equil']\
#                             .agg([('median_3w_1',np.median),('mean_3w_1',np.mean)]).reset_index()

# agg_3w_2 = train_data.groupby(['Agencia_ID','Canal_ID','Cliente_ID'])['log_Demanda_uni_equil']\
#                             .agg([('median_3w_2',np.median),('mean_3w_2',np.mean)]).reset_index()

# agg_3w_3 = train_data.groupby(['Agencia_ID','Canal_ID','Producto_ID'])['log_Demanda_uni_equil']\
#                             .agg([('median_3w_3',np.median),('mean_3w_3',np.mean),('max_3w_3',np.max),('min_3w_3',np.min)]).reset_index()

# agg_3w_6 = train_data.groupby(['Agencia_ID','Cliente_ID','Producto_ID'])['log_Demanda_uni_equil']\
#                             .agg([('mean_3w_6',np.mean)]).reset_index()

# agg_3w_8 = train_data.groupby(['Canal_ID','Ruta_SAK','Producto_ID'])['log_Demanda_uni_equil']\
#                             .agg([('median_3w_8',np.median),('mean_3w_8',np.mean)]).reset_index()



# ####### 2 WAY ASSOCIATIONS
# agg_2w_4 = train_data.groupby(['Agencia_ID','Producto_ID'])['log_Demanda_uni_equil']\
#                             .agg([('mean_2w_4',np.mean)]).reset_index()

# agg_2w_7 = train_data.groupby(['Canal_ID','Producto_ID'])['log_Demanda_uni_equil']\
#                             .agg([('median_2w_7',np.median),('mean_2w_7',np.mean)]).reset_index()


# ####### 1 WAY ASSOCIATIONS
# agg_1w_5 = train_data.groupby(['Producto_ID'])['log_Demanda_uni_equil']\
#                     .agg([('median_1w_5',np.median),('mean_1w_5',np.mean),('max_1w_5',np.max)]).reset_index()



# # features_set = features_set.merge(agg_5w, how = 'left', on = ['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID'])
# features_set = agg_5w
# del agg_5w

# features_set = features_set.merge(agg_4w_1, how = 'left', on = ['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID'])
# del agg_4w_1
# features_set = features_set.merge(agg_4w_2, how = 'left', on = ['Agencia_ID','Ruta_SAK','Cliente_ID','Producto_ID'])
# del agg_4w_2
# features_set = features_set.merge(agg_4w_3, how = 'left', on = ['Agencia_ID','Canal_ID','Cliente_ID','Producto_ID'])
# del agg_4w_3
# features_set = features_set.merge(agg_4w_4, how = 'left', on = ['Agencia_ID','Canal_ID','Ruta_SAK','Producto_ID'])
# del agg_4w_4

# features_set = features_set.merge(agg_3w_1, how = 'left', on = ['Agencia_ID','Canal_ID','Ruta_SAK'])
# del agg_3w_1
# features_set = features_set.merge(agg_3w_2, how = 'left', on = ['Agencia_ID','Canal_ID','Cliente_ID'])
# del agg_3w_2
# features_set = features_set.merge(agg_3w_3, how = 'left', on = ['Agencia_ID','Canal_ID','Producto_ID'])
# del agg_3w_3
# features_set = features_set.merge(agg_3w_6, how = 'left', on = ['Agencia_ID','Cliente_ID','Producto_ID'])
# del agg_3w_6

# features_set = features_set.merge(agg_3w_8, how = 'left', on = ['Canal_ID','Ruta_SAK','Producto_ID'])
# del agg_3w_8

# features_set = features_set.merge(agg_2w_7, how = 'left', on = ['Canal_ID','Producto_ID'])
# del agg_2w_7

# features_set = features_set.merge(agg_2w_4, how = 'left', on = ['Agencia_ID','Producto_ID'])
# del agg_2w_4

# features_set = features_set.merge(agg_1w_5, how = 'left', on = ['Producto_ID'])
# del agg_1w_5

# features_set = features_set.merge(product_table, how = 'left', on = ['Producto_ID'])
# features_set = features_set.merge(town_data, how = 'left', on =['Agencia_ID'])


# In[6]:

full_time = time.time()


# agg_overall_median = train_data['log_Demanda_uni_equil'].median()
# agg_overall_mean = train_data['log_Demanda_uni_equil'].mean()

train_data['sample'] = 1
test_data['sample'] = 2

data_shape = 1
i = 0

# ###### loop for slicing data on client starts
# while data_shape > 0:
    
#     #### while loop indicators generate starts
#     if i == 0:
#         size = 5000
    
#     c123 = client_data.sample(n=size, random_state = 1729)
#     client_list = c123.Cliente_ID.tolist()
#     client_data = client_data[~client_data.Cliente_ID.isin(client_list)]

#     data_shape = client_data.shape[0]
    
#     if data_shape < size:
#         size = data_shape
#     #### while loop indicators generate ends
    
    
    ##### ------ DATA PREPARATION STARTS----------
for i in xrange(0,1):
#     size = 5000
#     c123 = client_data.sample(n=size, random_state = 1729)
    client_list = c123.Cliente_ID.tolist()
    
    
    ##actual split starts
    train_x = train_data[train_data.Cliente_ID.isin(client_list)]
    test_x = test_data[test_data.Cliente_ID.isin(client_list)]
    
    id_x = test_x['id']
    test_x = test_x.drop(['id'], axis = 1)  
    
#     train_data = train_data[~train_data.Cliente_ID.isin(client_list)]
#     test_data = test_data[~test_data.Cliente_ID.isin(client_list)] 
    
#     del client_list
    
          
    print ('---------------------------------------Iteration %s starts---------------------------------------' % i)
#     print ('shape of train_x after iteration %s %s' % (i,train_x.shape))
#     print ('shape of train_data after iteration %s %s' % (i,train_data.shape))
#     print ('shape of test_x after iteration %s %s' % (i,test_x.shape))
#     print ('shape of test_data after iteration %s %s' % (i,test_data.shape))
    
    
    ############################ ---------------- TRAINING DATA SUMMARIES-----------------------------############################
#     print (' ------------- TRAINING DATA SUMMARIES STARTS------------')



    ######## 5 WAY ASSOCIATIONS 
    ### median,mean,max,min by sales depot, sales channel, route, client, product 
    start_time = time.time()


        ### add freq of refill
    freq_refill = train_x.groupby(['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID'])                                ['log_Demanda_uni_equil'].agg([('freq_refill',np.size)]).reset_index()
        
    agg_5w = train_x.groupby(['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID'])                                ['log_Demanda_uni_equil'].agg([('median_5w',np.median),('mean_5w',np.mean),                                                           ('min_5w',np.min),('max_5w',np.max)]).reset_index() 

    
    ######## 4 WAY ASSOCIATIONS
    agg_4w_1 = train_x.groupby(['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID'])['log_Demanda_uni_equil']                                .agg([('median_4w_1',np.median),('mean_4w_1',np.mean),('max_4w_1',np.max)]).reset_index()

    agg_4w_2 = train_x.groupby(['Agencia_ID','Ruta_SAK','Cliente_ID','Producto_ID'])['log_Demanda_uni_equil']                                .agg([('median_4w_2',np.median),('mean_4w_2',np.mean),('max_4w_2',np.max)]).reset_index()

    agg_4w_3 = train_x.groupby(['Agencia_ID','Canal_ID','Cliente_ID','Producto_ID'])['log_Demanda_uni_equil']                                .agg([('median_4w_3',np.median),('mean_4w_3',np.mean),('max_4w_3',np.max),('min_4w_3',np.min)]).reset_index()

    agg_4w_4 = train_x.groupby(['Agencia_ID','Canal_ID','Ruta_SAK','Producto_ID'])['log_Demanda_uni_equil']                                .agg([('median_4w_4',np.median),('mean_4w_4',np.mean),('max_4w_4',np.max),('min_4w_4',np.min)]).reset_index()


    ######## 3 WAY ASSOCIATIONS
    agg_3w_1 = train_x.groupby(['Agencia_ID','Canal_ID','Ruta_SAK'])['log_Demanda_uni_equil']                                .agg([('median_3w_1',np.median),('mean_3w_1',np.mean)]).reset_index()

    agg_3w_2 = train_x.groupby(['Agencia_ID','Canal_ID','Cliente_ID'])['log_Demanda_uni_equil']                                .agg([('median_3w_2',np.median),('mean_3w_2',np.mean)]).reset_index()

    agg_3w_3 = train_x.groupby(['Agencia_ID','Canal_ID','Producto_ID'])['log_Demanda_uni_equil']                                .agg([('median_3w_3',np.median),('mean_3w_3',np.mean),('max_3w_3',np.max),('min_3w_3',np.min)]).reset_index()

    agg_3w_6 = train_x.groupby(['Agencia_ID','Cliente_ID','Producto_ID'])['log_Demanda_uni_equil']                                .agg([('mean_3w_6',np.mean)]).reset_index()

    agg_3w_8 = train_x.groupby(['Canal_ID','Ruta_SAK','Producto_ID'])['log_Demanda_uni_equil']                                .agg([('median_3w_8',np.median),('mean_3w_8',np.mean)]).reset_index()



    ####### 2 WAY ASSOCIATIONS
    agg_2w_4 = train_x.groupby(['Agencia_ID','Producto_ID'])['log_Demanda_uni_equil']                                .agg([('mean_2w_4',np.mean)]).reset_index()

    agg_2w_7 = train_x.groupby(['Canal_ID','Producto_ID'])['log_Demanda_uni_equil']                                .agg([('median_2w_7',np.median),('mean_2w_7',np.mean)]).reset_index()


    ####### 1 WAY ASSOCIATIONS
    agg_1w_5 = train_x.groupby(['Producto_ID'])['log_Demanda_uni_equil']                        .agg([('median_1w_5',np.median),('mean_1w_5',np.mean),('max_1w_5',np.max)]).reset_index()



    features_set = freq_refill
    del freq_refill
    
    features_set = features_set.merge(agg_5w, how = 'left', on = ['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID'])
    del agg_5w

    features_set = features_set.merge(agg_4w_1, how = 'left', on = ['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID'])
    del agg_4w_1
    features_set = features_set.merge(agg_4w_2, how = 'left', on = ['Agencia_ID','Ruta_SAK','Cliente_ID','Producto_ID'])
    del agg_4w_2
    features_set = features_set.merge(agg_4w_3, how = 'left', on = ['Agencia_ID','Canal_ID','Cliente_ID','Producto_ID'])
    del agg_4w_3
    features_set = features_set.merge(agg_4w_4, how = 'left', on = ['Agencia_ID','Canal_ID','Ruta_SAK','Producto_ID'])
    del agg_4w_4

    features_set = features_set.merge(agg_3w_1, how = 'left', on = ['Agencia_ID','Canal_ID','Ruta_SAK'])
    del agg_3w_1
    features_set = features_set.merge(agg_3w_2, how = 'left', on = ['Agencia_ID','Canal_ID','Cliente_ID'])
    del agg_3w_2
    features_set = features_set.merge(agg_3w_3, how = 'left', on = ['Agencia_ID','Canal_ID','Producto_ID'])
    del agg_3w_3
    features_set = features_set.merge(agg_3w_6, how = 'left', on = ['Agencia_ID','Cliente_ID','Producto_ID'])
    del agg_3w_6

    features_set = features_set.merge(agg_3w_8, how = 'left', on = ['Canal_ID','Ruta_SAK','Producto_ID'])
    del agg_3w_8

    features_set = features_set.merge(agg_2w_7, how = 'left', on = ['Canal_ID','Producto_ID'])
    del agg_2w_7

    features_set = features_set.merge(agg_2w_4, how = 'left', on = ['Agencia_ID','Producto_ID'])
    del agg_2w_4

    features_set = features_set.merge(agg_1w_5, how = 'left', on = ['Producto_ID'])
    del agg_1w_5

    features_set = features_set.merge(product_table, how = 'left', on = ['Producto_ID'])
    features_set = features_set.merge(town_data, how = 'left', on =['Agencia_ID'])
#     print (' ------------- TRAINING DATA SUMMARIES CREATED IN %s SECONDS' % (time.time() - start_time))
    
    
    #############################---------------- THIS IS THE SECTION FOR CLIENT PRODUCT SUMMARIES-------------##################
#     print (' ------------- CLIENT PRODUCT SUMMARIES STARTED-----------')
    


    
    data_all = train_x.append(test_x, ignore_index = True)
    del train_x
    del test_x
    
    data_all['log_Demanda_uni_equil'] = data_all['log_Demanda_uni_equil'].fillna(0) ##this to make demand for test data as zero. Would help with cumsum

       

    agg_dem_each_week = data_all.groupby(['Semana','Cliente_ID','Producto_ID'])['log_Demanda_uni_equil']                            .agg([('median_lag_each_week',np.median),('mean_lag_each_week',np.mean)]).reset_index()

    agg_dem_each_week = agg_dem_each_week.sort_values(['Cliente_ID','Producto_ID','Semana'])

    #####******* loop for 'lag' summaries starts
    for w in xrange(3,12): #run this for week 3 to 11
        data_week = data_all[data_all['Semana'] == w]

        if w < 11:
            agg_lag1 = agg_dem_each_week[agg_dem_each_week['Semana'] == w-1]
            agg_lag1 = agg_lag1.rename(columns = {'median_lag_each_week':'med_lag1','mean_lag_each_week':'mean_lag1'})
            agg_lag1 = agg_lag1.drop(['Semana'], axis = 1)

            agg_lag2 = agg_dem_each_week[agg_dem_each_week['Semana'] == w-2]
            agg_lag2 = agg_lag2.rename(columns = {'median_lag_each_week':'med_lag2','mean_lag_each_week':'mean_lag2'})
            agg_lag2 = agg_lag2.drop(['Semana'], axis = 1)

            agg_lag3 = agg_dem_each_week[agg_dem_each_week['Semana'] == w-3]
            agg_lag3 = agg_lag3.rename(columns = {'median_lag_each_week':'med_lag3','mean_lag_each_week':'mean_lag3'})
            agg_lag3 = agg_lag3.drop(['Semana'], axis = 1)                           

            data_week = data_week.merge(agg_lag1, how = 'left', on = ['Cliente_ID','Producto_ID'])
            data_week = data_week.merge(agg_lag2, how = 'left', on = ['Cliente_ID','Producto_ID'])
            data_week = data_week.merge(agg_lag3, how = 'left', on = ['Cliente_ID','Producto_ID'])
        else:
            agg_lag1 = agg_dem_each_week[agg_dem_each_week['Semana'] == w-2]
            agg_lag1 = agg_lag1.rename(columns = {'median_lag_each_week':'med_lag1','mean_lag_each_week':'mean_lag1'})
            agg_lag1 = agg_lag1.drop(['Semana'], axis = 1)

            agg_lag2 = agg_dem_each_week[agg_dem_each_week['Semana'] == w-3]
            agg_lag2 = agg_lag2.rename(columns = {'median_lag_each_week':'med_lag2','mean_lag_each_week':'mean_lag2'})
            agg_lag2 = agg_lag2.drop(['Semana'], axis = 1)

            agg_lag3 = agg_dem_each_week[agg_dem_each_week['Semana'] == w-4]
            agg_lag3 = agg_lag3.rename(columns = {'median_lag_each_week':'med_lag3','mean_lag_each_week':'mean_lag3'})
            agg_lag3 = agg_lag3.drop(['Semana'], axis = 1)                           

            data_week = data_week.merge(agg_lag1, how = 'left', on = ['Cliente_ID','Producto_ID'])
            data_week = data_week.merge(agg_lag2, how = 'left', on = ['Cliente_ID','Producto_ID'])
            data_week = data_week.merge(agg_lag3, how = 'left', on = ['Cliente_ID','Producto_ID'])
        
        del agg_lag1 
        del agg_lag2
        del agg_lag3
        
        #append each week processed for every client and product combination
        if w == 3:
            client_product_summ = data_week
            del data_week
        else:
            client_product_summ = client_product_summ.append(data_week, ignore_index = True)
            del data_week
                
        client_product_summ['med_lag1'] = client_product_summ['med_lag1'].fillna(0)
        client_product_summ['med_lag2'] = client_product_summ['med_lag2'].fillna(0)
        client_product_summ['med_lag3'] = client_product_summ['med_lag3'].fillna(0)
        client_product_summ['mean_lag1'] = client_product_summ['mean_lag1'].fillna(0)
        client_product_summ['mean_lag2'] = client_product_summ['mean_lag2'].fillna(0)
        client_product_summ['mean_lag3'] = client_product_summ['mean_lag3'].fillna(0)
        
        
        ##### cumulative sum of mean lags
        client_product_summ[['cum_med_lag1','cum_med_lag2','cum_med_lag3']] = client_product_summ.                                                                   groupby(['Agencia_ID','Canal_ID','Cliente_ID','Producto_ID','Ruta_SAK'])                                                                    ['med_lag1','med_lag2','med_lag3'].cumsum()

        ##### cumulative sum of mean lags
        client_product_summ[['cum_mean_lag1','cum_mean_lag2','cum_mean_lag3']] = client_product_summ.                                                                   groupby(['Agencia_ID','Canal_ID','Cliente_ID','Producto_ID','Ruta_SAK'])                                                                    ['mean_lag1','mean_lag2','mean_lag3'].cumsum()

        ##### cumulative sum of mean lags
        client_product_summ['cum_sum_demand'] = client_product_summ.groupby(['Agencia_ID','Canal_ID','Cliente_ID','Producto_ID','Ruta_SAK'])                                        ['log_Demanda_uni_equil'].cumsum()

        
        
        
    #####******* loop for 'lag' summaries ends
    
    
    ### take only client product summaries starting week 6
    client_product_summ = client_product_summ[client_product_summ['Semana'] > 5] 
    
#     print (' ------------- CLIENT PRODUCT SUMMARIES CREATED IN %s SECONDS' % (time.time() - start_time))   
    
    
    #### merge the client product summary data to the actual data 
    client_product_summ = client_product_summ.drop(['sample','log_Demanda_uni_equil'],axis = 1) # dropping because already exist in main data
    data_all = data_all.merge(client_product_summ, how = 'left', on = ['Semana','Agencia_ID','Canal_ID','Cliente_ID','Producto_ID','Ruta_SAK'])
    del client_product_summ
    
    ########### ---------------SECTION FOR CLIENT PRODUCT SUMMARIES IS NOW OVER-----------------
    
    
    #### Merge the overall features previously created
    data_all = data_all.merge(features_set, how = 'left', on = ['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID'])
    del features_set
    
#     data_all = data_all.merge(freq_refill, how = 'left', on = ['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID'])
#     del freq_refill
    
#     data_all['agg_overall_median'] = agg_overall_median
#     data_all['agg_overall_mean'] = agg_overall_mean
    
    town_client_count = data_all.groupby(['Cliente_ID'])['town_id'].agg([('town_client_count',pd.Series.nunique)]).reset_index()
    state_client_count = data_all.groupby(['Cliente_ID'])['state_id'].agg([('state_client_count',pd.Series.nunique)]).reset_index()
    agency_client_count = data_all.groupby(['Cliente_ID'])['Agencia_ID'].agg([('agency_client_count',pd.Series.nunique)]).reset_index()
    route_client_count = data_all.groupby(['Cliente_ID'])['Ruta_SAK'].agg([('route_client_count',pd.Series.nunique)]).reset_index()
    channel_client_count = data_all.groupby(['Cliente_ID'])['Canal_ID'].agg([('channel_client_count',pd.Series.nunique)]).reset_index()
    
    data_all = data_all.merge(town_client_count, how = 'left', on = ['Cliente_ID'])
    del town_client_count
    data_all = data_all.merge(state_client_count, how = 'left', on = ['Cliente_ID'])
    del state_client_count
    data_all = data_all.merge(agency_client_count, how = 'left', on = ['Cliente_ID'])
    del agency_client_count
    data_all = data_all.merge(route_client_count, how = 'left', on = ['Cliente_ID'])
    del route_client_count
    data_all = data_all.merge(channel_client_count, how = 'left', on = ['Cliente_ID'])
    del channel_client_count
    
    ###### merge week 3 demand to week 7, week 4 demand to week 8, week 5 demand to 9, week 6 demand to 10
    data_aggs = data_all[data_all.Semana.isin([3,4,5,6])]
    aggs_4week_ago = data_aggs.groupby(['Semana','Cliente_ID','Producto_ID'])['log_Demanda_uni_equil']                                .agg([('med_demand_4wk_ago',np.median),('mean_demand_4wk_ago',np.mean),                                      ('max_demand_4wk_ago',np.max)]).reset_index()
    del data_aggs
    
    aggs_4week_ago['Semana'] = aggs_4week_ago['Semana']+4
    
    data_all = data_all.merge(aggs_4week_ago, how = 'left', on = ['Semana','Cliente_ID','Producto_ID'])
    
    del aggs_4week_ago
    
    data_all = data_all.fillna(0)
    ### take data of weeks greater than 5
    data_all = data_all[data_all['Semana'] > 5]
    
    train_x = data_all[data_all['sample'] == 1]
    test_x = data_all[data_all['sample'] == 2]
    
    del data_all
    
    train_x = train_x.drop(['sample','town_id','state_id'], axis = 1) #don't drop Semana here
    test_x = test_x.drop(['Semana','sample','log_Demanda_uni_equil','town_id','state_id'], axis = 1)
    
    # frequency of refill has to be increased by one count for test data
    test_x['freq_refill'] = test_x['freq_refill'].fillna(0) + 1
    
        
#     print ('-------- Modeling Starts --------')

    ##### training all weeks together
#     X_train, X_val, y_train, y_val = train_test_split(train_x.drop(['log_Demanda_uni_equil','Semana'], axis = 1)\
#                                                       , train_x['log_Demanda_uni_equil'], test_size=0.2, random_state=1729)
#     del train_x
    
    

    ##### MODEL STEP 1 : training 3 weeks combination and testing on the left out week - THIS TO CHECK THE PERFORMANCE OF MODEL
#     for week in xrange(6,10):
    week = 9
    train_final = train_x[train_x['Semana'] != week]
    val_final = train_x[train_x['Semana'] == week]
    
   
    y_train = train_final['log_Demanda_uni_equil']
    X_train = train_final.drop(['log_Demanda_uni_equil','Semana'], axis = 1)
    del train_final
    
    y_val = val_final['log_Demanda_uni_equil']
    X_val = val_final.drop(['log_Demanda_uni_equil','Semana'], axis = 1)
    del val_final
   
    ROUNDS=150

    params = {}
    params["objective"] = "reg:linear"
    params["eta"] = 0.02
    params["min_child_weight"] = 1
    params["subsample"] = 0.7
    params["colsample_bytree"] = 0.8
    params["colsample_bylevel"] = 0.9
    params["max_depth"] = 5
    params["eval_metric"] = "rmse"
    params["silent"] = 1

    xgtrain   = xgb.DMatrix(X_train, label=y_train)
    xgval     = xgb.DMatrix(X_val, label=y_val)
    del X_train
    del X_val

    plst      = list(params.items())
    watchlist = [(xgtrain, 'train'),(xgval, 'val')]

    model     = xgb.train(plst, xgtrain, ROUNDS, watchlist,early_stopping_rounds=5,verbose_eval = 49)

    del xgtrain
    del xgval

    #### MODEL STEP 2: now train the model on 6,7,8 & 9 to make predictions on week 10 and 11
    y_train = train_x['log_Demanda_uni_equil']
    X_train = train_x.drop(['log_Demanda_uni_equil','Semana'], axis = 1)
    del train_x
    
    xgtrain   = xgb.DMatrix(X_train, label=y_train)
    del X_train
    
    plst      = list(params.items())
    watchlist = [(xgtrain, 'train')]
    model     = xgb.train(plst, xgtrain, ROUNDS, watchlist,early_stopping_rounds=5,verbose_eval = 49)
        
    ###### MODEL STEP 3: predict on actual test data   

    xgb_test = xgb.DMatrix(test_x)   
    del test_x

    preds_test = model.predict(xgb_test)

    del xgb_test
    
    if i == 0:
        submission_final = pd.DataFrame({'id':id_x, 'log_Demanda_uni_equil': preds_test})
        del preds_test
    else:
        submission = pd.DataFrame({'id':id_x, 'log_Demanda_uni_equil': preds_test})
        submission_final = submission_final.append(submission)
        del preds_test
        del submission

    print ('----------------------------Iteration %s ends----------------------------' % i)
    
    i = i+1 ###  important since this determines the loop ID
    ###### loop for slicing data on client ends



print (' ---- OVERALL TIME TAKEN %s SECONDS ---- ' % (time.time() - full_time))

# feat_importance = pd.DataFrame(model.get_fscore().items(),columns=['feature', 'value'])
# feat_importance = feat_importance.sort_values(['value'], ascending = False)
# print feat_importance


# In[7]:

submission_final = submission_final.sort_values(['id'])
submission_final['Demanda_uni_equil'] = np.expm1(submission_final['log_Demanda_uni_equil'])
cols = ['id','Demanda_uni_equil']
submission_final = submission_final[cols]

submission_final.ix[submission_final.Demanda_uni_equil < 0, 'Demanda_uni_equil'] = 0
submission_final.to_csv('submission_data.csv', index = False)


# In[4]:

######################## TESTING BLOCKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK--------------------


# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1729)

# X.head(10)
# X_train.head(10)

# import pandas as pd
# train_data = pd.read_csv('train_csv.zip', compression='zip', header=0, sep=',', quotechar='"'
#                          ,usecols = ['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID','Semana','Demanda_uni_equil']
#                          ,dtype  = {'Semana' : 'int8',
#                                     'Agencia_ID' :'int16',
#                                     'Canal_ID' : 'int8',
#                                     'Ruta_SAK' : 'int16',
#                                     'Cliente_ID' : 'int32',
#                                     'Producto_ID':'int32',
#                                     'Demanda_uni_equil':'int16'}
#                         )

# train_data['log_Demanda_uni_equil'] = np.log1p(train_data['Demanda_uni_equil'])
# train_data = train_data.drop(['Demanda_uni_equil'], axis = 1)

# c123 = train_data[train_data['Producto_ID'] < 0]
# c123.head(10)

# product_table = pd.read_csv('producto_tabla_csv.zip', compression = 'zip', header=0, sep = ',', quotechar = '"')
# c123 = product_table[product_table['Producto_ID'] < 0]
# c123.head(10)


# feat_importance = pd.DataFrame(model.get_fscore().items(), columns=['Feat', 'values'])
# feat_importance = feat_importance.sort_values(['values'], ascending = False)
# feat_importance.head(100)
# from sklearn.cross_validation import train_test_split

# y = train_data['log_Demanda_uni_equil']

# X = train_data.drop(['log_Demanda_uni_equil','Semana'],axis =1)

# X_train, X_val, y_train, y_val = train_test_split(X, y,test_size=0.2, random_state=1729)
# train_data = pd.read_csv('train_csv.zip', compression='zip', header=0, sep=',', quotechar='"'
#                          ,usecols = ['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID','Semana','Demanda_uni_equil']
#                          ,dtype  = {'Semana' : 'int8',
#                                     'Agencia_ID' :'int16',
#                                     'Canal_ID' : 'int8',
#                                     'Ruta_SAK' : 'int16',
#                                     'Cliente_ID' : 'int32',
#                                     'Producto_ID':'int32',
#                                     'Demanda_uni_equil':'int16'}
#                         )


# t123 = train_data[(train_data['Agencia_ID'] == 1110) & (train_data['Canal_ID'] == 7) & (train_data['Ruta_SAK'] == 3301)\
#                 & (train_data['Cliente_ID'] == 24080) & (train_data['Producto_ID'] == 2233)]
# t123.head(10)


# t456 = test_data[(test_data['Agencia_ID'] == 1110) & (test_data['Canal_ID'] == 7) & (test_data['Ruta_SAK'] == 3301)\
#                 & (test_data['Cliente_ID'] == 24080) & (test_data['Producto_ID'] == 2233)]

# t456.head(10)
###################### LOAD CLIENT DATA

# client_data = pd.read_csv('cliente_tabla_csv.zip', compression = 'zip', header=0, sep = ',', quotechar = '"')
# client_data = client_data.drop_duplicates('Cliente_ID')

# data_shape = 0

# while data_shape == 0:
#     rows = random.sample(client_data.index, 5000)
#     c123 = client_data.ix[rows]
#     c_c = c123[c123['Cliente_ID'] == 24080]
#     data_shape = c_c.shape[0]
# c_c.head(10)


# result_check = submission_final[submission_final.id.isin([6355862,6972947])]
# result_check.head(10)

# data_shape = 1
# i = 0
# while data_shape > 0:
    
#     #### while loop indicators generate starts
#     if i == 0:
#         size = 50000
    
#     c123 = client_data.sample(n=size, random_state = 1729)
#     client_list = c123.Cliente_ID.tolist()
#     client_data = client_data[~client_data.Cliente_ID.isin(client_list)]
    
#     data_shape = client_data.shape[0]
#     print ('size %s' % data_shape)
    
#     if data_shape < size:
#         size = data_shape
    
#     i = i + 1

# agg_1w_5 = train_data['log_Demanda_uni_equil'].median()
# agg_1w_5

# train_data['median'] = agg_1w_5
# train_check = train_x.ix[:250000,:]
# test_check = test_x.ix[:450000,:]


# train_check.to_csv('train_check.csv')
# test_check.to_csv('test_check.csv')
# import pandas as pd
# import numpy as np

# train_data = pd.read_csv('train_csv.zip', compression='zip', header=0, sep=',', quotechar='"'
#                          ,usecols = ['Agencia_ID','Canal_ID','Ruta_SAK','Cliente_ID','Producto_ID','Semana','Demanda_uni_equil']
#                          ,dtype  = {'Semana' : 'int8',
#                                     'Agencia_ID' :'int16',
#                                     'Canal_ID' : 'int8',
#                                     'Ruta_SAK' : 'int16',
#                                     'Cliente_ID' : 'int32',
#                                     'Producto_ID':'int32',
#                                     'Demanda_uni_equil':'int16'}
#                         )

# train_data['log_Demanda_uni_equil'] = np.log1p(train_data['Demanda_uni_equil'])
# train_data = train_data.drop(['Demanda_uni_equil'], axis = 1)
# t123 = train_data[(train_data['Agencia_ID'] == 1110) & (train_data['Canal_ID'] == 7) & (train_data['Ruta_SAK'] == 3301)\
#                 & (train_data['Cliente_ID'] == 24080) & (train_data['Producto_ID'] == 2233)]
# del train_data


##### cumulative sum of mean lags
# t123['cum_sum_demand'] = t123.groupby(['Agencia_ID','Canal_ID','Cliente_ID','Producto_ID','Ruta_SAK'])\
#                                 ['log_Demanda_uni_equil'].cumsum()
# t123.head(10)

