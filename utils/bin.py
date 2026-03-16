import numpy as np

def get_num_bins(data, min_bin):
    n_examples = len(data)
    q75, q25 = np.percentile(data, [75 ,25])
    iqr = q75 - q25
    if iqr == 0:
        return int(np.sqrt(n_examples))
    
    bin_width = 2 * iqr / (n_examples ** (1/3))
    num_bins = int((data.max() - data.min()) / bin_width)
    
    return max(min_bin, min(num_bins, 100))