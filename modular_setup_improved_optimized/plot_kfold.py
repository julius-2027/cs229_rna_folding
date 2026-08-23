"""
plot_kfold.py

Plot the results of k-fold cross-validation
"""

import numpy as np
import json
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 20

models = ["glm_baseline", "mean_pool_mlp", "lstm", "lstm_rna_fm", "struct_lstm_rna_fm", "bilstm_rna_fm_proj", "transformer_rna_fm", "loc_rna_fm", "all_local_mlp"]

for i in range(len(models)):
    model = models[i]

    with open('cross_val_results/cross_val_metrics_'+model+'_v2_k5.json', encoding='utf-8-sig') as f:
        results_dict = json.load(f)
    #print(results_dict['train_loss_mean'])
    #epochs = results_dict['epochs']
    epochs = 15

    plt.errorbar(np.arange(1, epochs+1), results_dict['train_loss_mean'][:epochs],
                 yerr=results_dict['train_loss_std'][:epochs], capsize=5,
                 linestyle='--', color='C'+str(i))
    
    model_label = model
    if model=='lstm':
        model_label='lstm'

    plt.errorbar(np.arange(1, epochs+1), results_dict['val_loss_mean'][:epochs],
                 yerr=results_dict['val_loss_std'][:epochs], capsize=5,
                 color='C'+str(i), label=model_label)
    
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(fontsize=10)

plt.savefig('kfold_plot.png', bbox_inches='tight')
plt.savefig('kfold_plot.pdf', bbox_inches='tight')

plt.show()