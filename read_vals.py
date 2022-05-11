import numpy as np

#for experiment in ['order_by_increasing_t1','order_by_decreasing_t1','order_by_increasing_r0','order_by_decreasing_r0','order_by_increasing_qp','order_by_decreasing_qp','order_by_increasing_d0','order_by_decreasing_d0','order_by_increasing_z0','order_by_decreasing_z0']:
#for experiment in ['baseline_20k','MinMax_d0','MinMax_z0d0','MinMax_all']:
#for experiment in ['oldhighnoise/new_features_xpyp_decreasing_t1_highnoise_20k','oldhighnoise/baseline_highnoise_20k','new_features_xpyp_decreasing_t1_highnoise_20k','baseline_highnoise_20k']:
for experiment in ['baseline_highnoise_280k','bestmodel_highnoise_280k','baseline_280k','bestmodel_280k']:
    vals=[]
    for trial in ['trial1','trial2','trial3','trial4','trial5']:
        file = experiment+'/'+trial+'/'+'history.npy'
        print(file)
        ting=np.load(file,allow_pickle='TRUE').item()
        val=np.min(ting['val_loss'])
        print(val)
        vals.append(val)
    #print(vals)
    print(experiment+': {} \pm {}'.format(np.mean(vals),np.std(vals)))

