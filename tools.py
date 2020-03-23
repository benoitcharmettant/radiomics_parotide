from os.path import join

from pandas import read_csv, DataFrame
from numpy import zeros, where
from scipy.stats import ttest_ind

from sklearn import metrics

from anonymization_tools import get_date_exam

def get_meta_data(data_path):
    return read_csv(join(data_path, 'overview.csv'), delimiter=";")

def parse_features(dir_path, exam_id, exam_type, return_date):
    
    file_path = join(dir_path, f"{exam_id}_{exam_type}.csv")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()[18:126]
        
    features = {} 
    
    for l in lines:
        l = l.split(";")
        value = l[-1] if not l[-1] == '\n' else l[-2]
        
        features[l[0].replace(' ','')] = float(value)
    
    if return_date:
        return features, get_date_exam(file_path)
    return features
        
    
def load_features(df_meta, data_path, verbose=True, type_to_include=['gado', 'diff', 't1', 't2']):
    
    ls_exams = []
    
    for line in df_meta.values:
        
        id_exam = line[0]
        sexe = line[1]
        age = line[2]
        tesla = line[3]
        multiclass_label = line[4]
        binary_label = line[5]
        
        
        try:
            gado_features, d1 = parse_features(join(data_path, 'exams'), id_exam, 'GADO', return_date=True)
            diff_features, d2 = parse_features(join(data_path, 'exams'), id_exam, 'DIFF', return_date=True)
            t1_features, d3 = parse_features(join(data_path, 'exams'), id_exam, 'T1', return_date=True)
            t2_features, d4 = parse_features(join(data_path, 'exams'), id_exam, 'T2', return_date=True)
        except FileNotFoundError as e:
            if verbose:
                print(f'Exam {id_exam} wasn\'t found or incomplete')
            continue
            
        # Sanity check 1 (4 dates must be equal)
        assert d1 == d2
        assert d1 == d3
        assert d1 == d4
        
        # Sanity check 2 (features key must be equal)
        assert sorted(gado_features.keys()) == sorted(diff_features.keys())
        assert sorted(gado_features.keys()) == sorted(t1_features.keys())
        assert sorted(gado_features.keys()) == sorted(t2_features.keys())
        
        exam = {'id':id_exam,
                'sexe':sexe,
                'age':age,
                'tesla':tesla,
                'multiclass_label':multiclass_label,
                'binary_label':binary_label,
                'gado_features':gado_features,
                'diff_features':diff_features,
                't1_features':t1_features,
                't2_features':t2_features,
                'features_keys':t2_features.keys()}
        ls_exams.append(exam)
        
    key_to_id = {}
    id_to_key = {}
    
    i = 0
    for mri_type in type_to_include:
        for key in ls_exams[0]['features_keys']:
            unique_key = f"{mri_type}_{key}"
            
            key_to_id[unique_key] = i
            id_to_key[i] = unique_key
            
            i+=1
    
    return ls_exams, id_to_key, key_to_id
        
        
def format_exam(exam, key_to_id, label="binary_label", meta_fields=['sexe', 'age', 'tesla']):
    
    label = exam[label]
    features = zeros(len(key_to_id))
    
    meta = zeros(len(meta_fields))
    
    for i, field in enumerate(meta_fields):
        meta[i] = exam[field]
    
    for k, i in key_to_id.items():
        mri_type = k.split('_')[0]
        feature = k.split('_')[1]
        
        features[i] = exam[f'{mri_type}_features'][feature]
        
    return label, features, meta

        
def order_dict(d, reverse=True):
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=reverse)}


def feature_auc(features, labels, id_to_feat, regressor):
    n_features = features.shape[1]
    
    aucs = zeros(n_features)
    ls_features_names = zeros(n_features).astype(str)
    
    feature_significance_auc = {}
    
    for i in range(n_features):
        feature_name = id_to_feat[i]

        regressor.fit(features[:, i].reshape(-1,1), labels)
        predictions = regressor.predict_proba(features[:, i].reshape(-1,1))

        auc = metrics.roc_auc_score(labels, predictions[:,1])

        aucs[i] = auc 

        ls_features_names[i] = feature_name

        feature_significance_auc[feature_name] = auc
        
    return feature_significance_auc
    
def feature_t_test(features, labels, id_to_feat):
    
    n_features = features.shape[1]
    n_exams = features.shape[0]

    labels_0 = where(labels == 0)
    labels_1 = where(labels == 1)

    p_values = zeros(n_features)
    ls_features_names = zeros(n_features).astype(str)

    feature_significance_p = {}

    for i in range(n_features):
        _, p = ttest_ind(features[labels_0, i].tolist()[0], features[labels_1, i].tolist()[0])
        feature_name = id_to_feat[i]

        p_values[i] = p
        ls_features_names[i] = feature_name

        feature_significance_p[feature_name] = p

    return feature_significance_p

def choose_features_from_dict(feature_dict, n_features, feat_to_id, reverse=False):
    """
    Selects the best features from a dictionnary
    """
    selected_features_id = []

    ranked_dictionnary = order_dict(feature_dict, reverse=reverse) # Ranked dictionnary according to AUC
    
    for i, f in enumerate(ranked_dictionnary.keys()):
        if i < n_features:
            selected_features_id.append(feat_to_id[f])
            
    return selected_features_id