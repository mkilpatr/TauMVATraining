from root_numpy import root2array, array2root
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# samples
sig_file = 'TauTrain_2017_tree.root'
bkg_file = 'TauTrain_2017_tree.root'

# output file with MVA scores (for both training and testing events)
output_file = 'result.root'

train_file =  'tauMVA_training.h5'

#model_file =  'tauMVA-xgb.model'
model_file =  'tauMVA-xgb'

# baseline selection
basesel = 'Pass_MET & Pass_Baseline'
sig_sel = basesel + ' & gentaumatch'
bkg_sel = basesel + ' & (not gentaumatch)'
wgtvar = 'Stop0l_evtWeight'


# observer vars: not used in training but will be saved in the hdf5 file
obs_vars = [
 'Stop0l_evtWeight',
 'nJet',
 'gentaumatch'
]

# training vars: float
varsF = [
 'pt',
 'abseta',
 'chiso0p1',
 'chiso0p2',
 'chiso0p3',
 'chiso0p4',
 'totiso0p1',
 'totiso0p2',
 'totiso0p3',
 'totiso0p4',
 'neartrkdr',
 'contjetdr',
 'contjetcsv',
]

input_vars = varsF

# preprocessing of the samples

def prepSamp(filepath, is_signal):
    if is_signal:
        selection, target = sig_sel, 1
    else:
        selection, target = bkg_sel, 0
    df = pd.DataFrame(root2array(filepath, treename='Events'))
    df.query(selection, inplace=True)
    #df = df.loc[:, input_vars + obs_vars].reset_index(drop=True)
    df = df.loc[:, input_vars + obs_vars].reindex()
    df['target'] = target
    return df

def convertToHD5(output):
    sig_df = prepSamp(sig_file, True)
    bkg_df = prepSamp(bkg_file, False)
    sig_sum_wgt = sig_df[wgtvar].sum()
    bkg_sum_wgt = bkg_df[wgtvar].sum()
    print 'Signal events: %d, Background events: %d' % (len(sig_df), len(bkg_df))
    print 'Signal weight: %f, Background weight: %f' % (sig_sum_wgt, bkg_sum_wgt)
    sig_df['train_weight'] = sig_df[wgtvar] * (1000. / sig_sum_wgt)
    bkg_df['train_weight'] = bkg_df[wgtvar] * (1000. / bkg_sum_wgt)
    print 'After normalization:'
    print 'Signal weight: %f, Background weight: %f' % (sig_df['train_weight'].sum(), bkg_df['train_weight'].sum())
    full_df = pd.concat([sig_df, bkg_df], ignore_index=True)
    full_df.to_hdf(output, 'Events', format='table')
    return full_df

def loadSample():
    try:
        return pd.read_hdf(train_file)
    except IOError:
        print 'Sample %s does not exist! Creating from root files...' % train_file
        return convertToHD5(train_file)


# Load samples
train_samp = loadSample()
train_samp = train_samp.sample(frac=1).reset_index(drop=True)  # shuffle the dataset

# Load variables
X = train_samp[input_vars]
y = train_samp['target']
W = train_samp['train_weight']
wgt0 = train_samp['Stop0l_evtWeight']

# split samples for training and testing
test_size = 0.3  # size of the validtion sample
isTrain = np.random.uniform(size=len(train_samp)) > test_size
isTest = np.logical_not(isTrain)
train_samp['isTraining'] = isTrain

X_train, X_test = X[isTrain], X[isTest]
y_train, y_test = y[isTrain], y[isTest]
W_train, W_test = W[isTrain], W[isTest]
wgt0_train, wgt0_test = wgt0[isTrain], wgt0[isTest]

# from sklearn.cross_validation import train_test_split
# index = np.arange(len(y))
# idx_train, idx_test, X_train, X_test, W_train, W_test, wgt0_train, wgt0_test, y_train, y_test = train_test_split(index, X, W, wgt0, y, test_size=0.3)
# X_train, X_val, W_train, W_val, y_train, y_val = train_test_split(X_dev, W_dev, y_dev, test_size=0.2)

# load into xgboost DMatrix
import xgboost as xgb
d_train = xgb.DMatrix(X_train, label=y_train, weight=W_train)
d_test = xgb.DMatrix(X_test, label=y_test, weight=W_test)

# setup parameters for xgboost
param = {}
param['objective'] = 'binary:logistic'
param['eval_metric'] = ['error', 'auc', 'logloss']
# param['min_child_weight'] = 1
# param['gamma']=0.01
param['eta'] = 0.3
param['max_depth'] = 10
# param['colsample_bytree'] = 0.8
# param['subsample'] = 0.8

print 'Starting training...'
print 'Using %d vars:' % len(input_vars)
print input_vars

watchlist = [ (d_train, 'train'), (d_test, 'eval') ]
num_round = 1000  # max allowed no. of trees

# start training
#bst = xgb.train(param, d_train, num_round, watchlist, early_stopping_rounds=20)
bst = xgb.train(param, d_train, num_round, watchlist)

##matt adding functions to find output result
#evals_result = {}
#bst = xgb.train(param, d_train, num_round, watchlist, evals_result=evals_result, early_stopping_rounds=20)
#
#print('Access logloss metric directly from evals_result:')
#print(evals_result['eval']['logloss'])
#
#print('')
#print('Access metrics through a loop:')
#for e_name, e_mtrs in evals_result.items():
#    print('- {}'.format(e_name))
#    for e_mtr_name, e_mtr_vals in e_mtrs.items():
#        print('   - {}'.format(e_mtr_name))
#        print('      - {}'.format(e_mtr_vals))
#
#print('')
#print('Access complete dictionary:')
#print(evals_result)

# print prediction
print "prediction: ", bst.predict(d_test)

# save the model
bst.save_model(model_file+'_nvar%d_eta%f_maxdepth%d.model' % (len(input_vars), param['eta'], param['max_depth']))
# dump the model
bst.dump_model('dump.raw.txt')

# print out variable ranking
scores = bst.get_score()
ivar = 1
for k in sorted(scores, key=scores.get, reverse=True):
    print "%2d. %24s: %s" % (ivar, k, str(scores[k]))
    ivar = ivar + 1

# fill MVA scores and make root file
dmat = xgb.DMatrix(X, label=y)
preds = bst.predict(dmat)
train_samp['score'] = preds

print('Write prediction file to %s' % output_file)
array2root(train_samp.to_records(index=False), filename=output_file, treename='Events', mode='RECREATE')

print "dmat: ", dmat
print "preds: ", preds
print "train_samp: ", train_samp

####### convert to TMVA compatible XML file #######
model = bst.get_dump()

import re
import xml.etree.cElementTree as ET
regex_float_pattern = r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?'

def build_tree(xgtree, base_xml_element, var_indices):
    parent_element_dict = {'0':base_xml_element}
    pos_dict = {'0':'s'}
    for line in xgtree.split('\n'):
        if not line: continue
#         print line
#         print parent_element_dict
#         print pos_dict
        if ':leaf=' in line:
            # leaf node
            result = re.match(r'(\t*)(\d+):leaf=({0})$'.format(regex_float_pattern), line)
            if not result:
                print line
            depth = result.group(1).count('\t')
            inode = result.group(2)
            res = result.group(3)
            node_elementTree = ET.SubElement(parent_element_dict[inode], "Node", pos=str(pos_dict[inode]),
                                             depth=str(depth), NCoef="0", IVar="-1", Cut="0.0e+00", cType="1", res=str(res), rms="0.0e+00", purity="0.0e+00", nType="-99")
        else:
            # \t\t3:[var_topcand_mass<138.19] yes=7,no=8,missing=7
            result = re.match(r'(\t*)([0-9]+):\[(?P<var>.+)<(?P<cut>{0})\]\syes=(?P<yes>\d+),no=(?P<no>\d+)'.format(regex_float_pattern), line)
            if not result:
                print line
            depth = result.group(1).count('\t')
            inode = result.group(2)
            var = result.group('var')
            cut = result.group('cut')
            lnode = result.group('yes')
            rnode = result.group('no')
            pos_dict[lnode] = 'l'
            pos_dict[rnode] = 'r'
            node_elementTree = ET.SubElement(parent_element_dict[inode], "Node", pos=str(pos_dict[inode]),
                                             depth=str(depth), NCoef="0", IVar=str(var_indices[var]), Cut=str(cut),
                                             cType="1", res="0.0e+00", rms="0.0e+00", purity="0.0e+00", nType="0")
            parent_element_dict[lnode] = node_elementTree
            parent_element_dict[rnode] = node_elementTree

def convert_model(model, input_variables, output_xml):
    NTrees = len(model)
    var_list = input_variables
    var_indices = {}

    # <MethodSetup>
    MethodSetup = ET.Element("MethodSetup", Method="BDT::BDT")

    # <Variables>
    Variables = ET.SubElement(MethodSetup, "Variables", NVar=str(len(var_list)))
    for ind, val in enumerate(var_list):
        name = val[0]
        var_type = val[1]
        var_indices[name] = ind
        Variable = ET.SubElement(Variables, "Variable", VarIndex=str(ind), Type=val[1],
            Expression=name, Label=name, Title=name, Unit="", Internal=name,
            Min="0.0e+00", Max="0.0e+00")

    # <GeneralInfo>
    GeneralInfo = ET.SubElement(MethodSetup, "GeneralInfo")
    Info_Creator = ET.SubElement(GeneralInfo, "Info", name="Creator", value="xgboost2TMVA")
    Info_AnalysisType = ET.SubElement(GeneralInfo, "Info", name="AnalysisType", value="Classification")

    # <Options>
    Options = ET.SubElement(MethodSetup, "Options")
    Option_NodePurityLimit = ET.SubElement(Options, "Option", name="NodePurityLimit", modified="No").text = "5.00e-01"
    Option_BoostType = ET.SubElement(Options, "Option", name="BoostType", modified="Yes").text = "Grad"

    # <Weights>
    Weights = ET.SubElement(MethodSetup, "Weights", NTrees=str(NTrees), AnalysisType="1")

    for itree in range(NTrees):
        BinaryTree = ET.SubElement(Weights, "BinaryTree", type="DecisionTree", boostWeight="1.0e+00", itree=str(itree))
        build_tree(model[itree], BinaryTree, var_indices)

    tree = ET.ElementTree(MethodSetup)
    tree.write(output_xml)


use_vars = [(k, 'F') for k in varsF]

convert_model(model, input_variables=use_vars, output_xml='xgboost.xml')

####### convert to TMVA compatible XML file #######


# plot roc curve

def plotROC(y_score, X_input, y_true, sample_weight=None):
    from sklearn.metrics import auc, roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)
    roc_auc = auc(fpr, tpr, reorder=True)

#     fpr_tau21, tpr_tau21, _ = roc_curve(y_true, 1-X_input['fj_tau21'], sample_weight=sample_weight)
#     tau21_auc = auc(fpr_tau21, tpr_tau21, sample_weight=sample_weight)

    plt.figure()
    plt.plot(tpr, 1 - fpr, label='MVA (area = %0.3f)' % roc_auc)
#     plt.plot(tpr_tau21, fpr_tau21, label=r'$\tau_{21}$ (area = %0.3f)' % tau21_auc)
    plt.plot([0, 1], [1, 0], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tagging eff.')
    plt.ylabel('Mistag rate')
#     plt.title('Receiver operating characteristic example')
    plt.legend(loc='best')
    plt.grid()


plotROC(bst.predict(d_test), X_test, y_test, wgt0_test)  # here we use the natural weight as in the sample
plt.ylim(0.8, 1)
plt.savefig('roc_xgb_nvar%d_eta%f_maxdepth%d.pdf' % (len(input_vars), param['eta'], param['max_depth']))

