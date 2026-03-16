from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import numpy as np

def evaluate_proximity_metrics(real_df, synthetic_df):
    nn = NearestNeighbors(n_neighbors=2, metric='euclidean', n_jobs=-1)
    nn.fit(real_df.values)
    
    distances, _ = nn.kneighbors(synthetic_df.values)
    
    d1 = distances[:, 0]
    d2 = distances[:, 1]
    
    exact_matches = np.sum(d1 < 1e-8)
    avg_dcr = np.mean(d1)
    
    nndr_scores = d1 / (d2 + 1e-10)
    avg_nndr = np.mean(nndr_scores)
    
    print(f"--- Proximity Privacy Report ---")
    print(f"Exact Matches Found: {exact_matches}")
    print(f"Average DCR:         {avg_dcr:.4f}")
    print(f"Average NNDR:        {avg_nndr:.4f}")
    
    high_risk_rows = np.sum(nndr_scores < 0.1)
    print(f"High-Risk Rows (NNDR < 0.1): {high_risk_rows}")
    
    return {
        "dcr": d1,
        "nndr": nndr_scores,
        "exact_matches": exact_matches
    }

def evaluate_mia(real_train, real_test, synth_df):
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1)
    nn.fit(synth_df.values)
    
    dist_mem, _ = nn.kneighbors(real_train.values)
    dist_non, _ = nn.kneighbors(real_test.values)
    
    y_true = np.concatenate([np.ones(len(dist_mem)), np.zeros(len(dist_non))])
    
    # 4. Probabilities (Closer distance = Higher probability of being a member)
    # We use negative distance as a 'confidence score'
    scores = np.concatenate([-dist_mem.flatten(), -dist_non.flatten()])
    
    roc_auc = roc_auc_score(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)
    
    # 6. Attacker Advantage (Privacy Risk Score)
    # Advantage = 2 * (ROC_AUC - 0.5)
    advantage = max(0, 2 * (roc_auc - 0.5))

    print(f"--- MIA Report ---")
    print(f"ROC AUC: {roc_auc:.4f} (Ideal: 0.5000)")
    print(f"PR AUC:  {pr_auc:.4f} (Baseline: 0.5000)")
    print(f"Attacker Advantage: {advantage:.4f} (Ideal: 0.0000)")
    
    if advantage > 0.1:
        print("WARNING: Significant Privacy Leakage Detected.")
    else:
        print("PASS: Privacy within acceptable DP bounds.")
        
    return {
        "roc_auc": roc_auc, 
        "pr_auc": pr_auc, 
        "advantage": advantage
    }