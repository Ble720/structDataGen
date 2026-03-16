import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
 
def evaluate_discriminator(real_df, synth_df):
    real_df = real_df.copy()
    synth_df = synth_df.copy()
    real_df['is_synthetic'] = 0
    synth_df['is_synthetic'] = 1

    n_samples = min(len(real_df), len(synth_df))
    combined_df = pd.concat([
        real_df.sample(n_samples, random_state=42),
        synth_df.sample(n_samples, random_state=42)
    ])
    
    X = combined_df.drop('is_synthetic', axis=1)
    y = combined_df['is_synthetic']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    
    probs = clf.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, probs)
    
    print(f"--- Discriminator Turing Test ---")
    print(f"Discriminator AUC: {auc_score:.4f}")
    
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop features used to identify synthetic rows:")
    print(importances.head(3))
    
    return auc_score, importances