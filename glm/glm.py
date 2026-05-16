import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import jensenshannon
import torch
import torch.nn as nn
import random

# get the labels y
def get_y(data_file, n_bins=50):
    dataset = pd.read_parquet(data_file)

    arr_fpts = np.array([row for row in dataset['fpts']])
    
    where_arr_fpts_nonzero = np.where(arr_fpts != 0)
    arr_logfpts = np.log10(arr_fpts[where_arr_fpts_nonzero])

    binedges_logfpt = np.linspace(np.min(arr_logfpts), np.max(arr_logfpts), n_bins+1)
    
    arr_hist_logfpts = np.zeros((dataset.shape[0], n_bins))
    for index, row in dataset.iterrows():
        fpts = row['fpts']
        where_fpts_nonzero = np.where(fpts != 0)
        logfpts = np.log10(fpts[where_fpts_nonzero])
        hist_logfpts, _ = np.histogram(logfpts, binedges_logfpt, density=True)
        hist_logfpts /= np.sum(hist_logfpts) # normalize so that values add to 1; this does not reflect the probability density
        arr_hist_logfpts[index] = hist_logfpts
    
    return arr_hist_logfpts, binedges_logfpt

# get the features X
def get_X(data_file):
    dataset = pd.read_parquet(data_file)
    
    X = dataset.drop(columns=['filepath', 'sequence', 'dot_bracket_from_file', 'fpts', 'mfe_structure', 'mfe'])
    for i in range(1, 21):
        X = X.drop(columns=['min_'+str(i)+'_structure'])
        X = X.drop(columns=['min_'+str(i)+'_energy'])
        X = X.drop(columns=['min_'+str(i)+'_bp_dist'])
        X = X.drop(columns=['min_'+str(i)+'_tree_dist'])
    
    feature_names = X.columns.values.tolist()
    return np.array(X), feature_names

class GLM(nn.Module):
    def __init__(self, input_dim, n_bins=50):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_bins)
    
    def forward(self, x):
        return self.linear(x)  # return logits; apply log_softmax in loss

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_val_test_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    return X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t

def train_model(X_train_t, X_val_t, y_train_t, y_val_t, model, learning_rate=1e-3):
    set_seed(42)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.KLDivLoss(reduction='batchmean')  # soft targets

    train_losses, val_losses = [], []

    for epoch in range(500):
        model.train()
        log_probs = torch.log_softmax(model(X_train_t), dim=-1)
        loss = loss_fn(log_probs, y_train_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            log_probs_val = torch.log_softmax(model(X_val_t), dim=-1)
            val_loss = loss_fn(log_probs_val, y_val_t)
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if epoch % 100 == 0:
            print(f'epoch {epoch}: train_loss = {train_losses[-1]}, val_loss = {val_losses[-1]}')

    return train_losses, val_losses

def plot_training_curve(train_losses, val_losses, file_name):
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='Train')
    ax.plot(val_losses, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL Divergence')
    ax.legend()
    plt.savefig(file_name+'.pdf', bbox_inches='tight')
    plt.savefig(file_name+'.png', bbox_inches='tight')

def main():
    datafile = '../kinpfn_testing_set/parquet_parsing/test_val_dataset.parquet' 
    
    print('Getting features X from datafile '+datafile+' ...')
    X, feature_names = get_X(datafile)
    print('Done.\n')

    print('Getting labels y from datafile '+datafile+' ...')
    y = get_y(datafile)
    print('Done.\n')

    print('Performing train val test split ...')
    X_train_t, X_val_t, X_test_t, y_train_t, y_val_t, y_test_t = train_val_test_split(X, y)
    print('Done.\n')
    
    model = GLM(input_dim=X_train_t.shape[1])
    train_losses, val_losses = train_model(X_train_t, X_val_t, y_train_t, y_val_t,
                                         model, learning_rate=1e-3)
    
    torch.save(model.state_dict(), 'glm_weights.pth')

    #np.savetxt()

if __name__ == "__main__":
    main()