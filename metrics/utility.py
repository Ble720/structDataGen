import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

def evaluate_ml_utility(train_synth, train_real, test_real, target_col):
    clf_synth = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_synth.fit(train_synth.drop(target_col, axis=1), train_synth[target_col])
    synth_preds = clf_synth.predict(test_real.drop(target_col, axis=1))
    tstr_score = f1_score(test_real[target_col], synth_preds, average='macro')

    clf_real = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_real.fit(train_real.drop(target_col, axis=1), train_real[target_col])
    real_preds = clf_real.predict(test_real.drop(target_col, axis=1))
    trtr_score = f1_score(test_real[target_col], real_preds, average='macro')

    utility_retention = (tstr_score / trtr_score) * 100

    print(f"TRTR (Baseline) F1: {trtr_score:.4f}")
    print(f"TSTR (Synthetic) F1: {tstr_score:.4f}")
    print(f"Utility Retention: {utility_retention:.2f}%")
    
    return utility_retention

def evaluate_correlation_utility(real_df, synth_df):
    real_corr = real_df.corr(numeric_only=True).fillna(0)
    synth_corr = synth_df.corr(numeric_only=True).fillna(0)
    
    diff = np.abs(real_corr.values - synth_corr.values)
    mean_corr_err = np.mean(diff)
    
    max_corr_err = np.max(diff)
    
    print(f"Mean Correlation Error: {mean_corr_err:.4f}")
    print(f"Max Correlation Error:  {max_corr_err:.4f}")
    
    return mean_corr_err

def evaluate_js_divergence(real_col, synth_col, num_categories):
    p = np.histogram(real_col, bins=np.arange(num_categories + 1), density=True)[0]
    q = np.histogram(synth_col, bins=np.arange(num_categories + 1), density=True)[0]
    
    # Add a tiny epsilon to avoid true zeros if necessary, 
    # though scipy's jensenshannon handles it.
    return jensenshannon(p, q) ** 2

def evaluate_wasserstein(real_col, synth_col):
    return wasserstein_distance(real_col, synth_col)