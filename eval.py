import torch
import numpy as np
from torch.utils.data import DataLoader # Import the appropriate DataLoader implementation
from visualize import *
from sklearn.cluster import KMeans
from sklearn import metrics
from data_utils import unravel
import xgboost as xgb
import csv
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def evaluate_model(model, dataset, device,checkpoint_dir, visualize=False):

    num_classes = dataset.shape[0]
    sims_per_class = dataset.shape[1]

    unravelled_magic = []
    output_embeddings = []
    for i in dataset:
        for j in range(sims_per_class):
            sim = i[j]
            ravioli = unravel(sim)
            unravelled_magic.append(ravioli)


    # Create labels for the dataset
    labels = torch.tensor([i for i in range(num_classes) for _ in range(sims_per_class)])
    
    model.eval()
    for i in unravelled_magic:
        i = i.to(device)
        output_embeddings.append(model(i.unsqueeze(0)).detach().cpu().numpy())


    output_embeddings = np.array(output_embeddings)
    output_embeddings = np.squeeze(output_embeddings, axis=1)
    
    ari, ami = evaluate_clustering_metrics(
        output_embeddings, labels, len(np.unique(labels)))


    print(f"ARI: {ari}, AMI: {ami}\n")
    

    if visualize:
        
        visualize_embeddings(output_embeddings, labels, checkpoint_dir)
        visualize_pca_kmeans(output_embeddings,num_classes , checkpoint_dir)


def xgb_regress(model, data_path, save_path, device):
    parameter_values = []
    unwrapped_pizza = []
    output_embeddings = []
    csv_file_path = "./data/params.csv"
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            parameter_values.append(list(map(float, row)))

    parameter_values_tensor = torch.tensor(parameter_values, dtype=torch.float32)
    final_param_labels = parameter_values_tensor.unsqueeze(1).repeat(1, 10, 1).reshape(-1, 3)

    datas = torch.load(data_path)
    reshaped_data = datas.view(datas.shape[0]*datas.shape[1], datas.shape[2],datas.shape[3]).transpose(2,1)
    
    for i in range(reshaped_data.shape[0]):
        ax = reshaped_data[i]
        ravioli = unravel(ax)
        unwrapped_pizza.append(ravioli)
    
    model.eval()
    for i in unwrapped_pizza:
         i=i.to(device)
         output_embeddings.append(model(i.unsqueeze(0)).detach().cpu().numpy())
    output_embeddings=np.array(output_embeddings)
    output_embeddings = np.squeeze(output_embeddings, axis=1)
    output_embeddings.shape
    final_param_labels_np = final_param_labels.numpy()
    X = output_embeddings
    y = final_param_labels_np
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.9, random_state=42)


    model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


    # def calc_adj_r2(r2, n=1000, p=1):
    #     return 1 - ((1 - r2) * (n - 1)) / (n - p)


    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}\n")
    r2 = r2_score(y_test, y_pred)
    print(f"R2 Score: {r2:.4f}\n")
    #print(f"Adjusted R2 Score: {calc_adj_r2(r2, p=3):.4f}\n")

    
    label_r2_scores = r2_score(y_test, y_pred, multioutput='raw_values')
    for i, score in enumerate(label_r2_scores):
        print(f"Label {i+1} R2 Score: {score:.4f}")
        #print(f"Label {i+1} Adjusted R2 Score: {calc_adj_r2(score):.4f}\n")
    gamma_gt = y_test[:, 0]
    gamma_pred = y_pred[:, 0]
    reg_param_gt = y_test[:, 1]
    reg_param_pred = y_pred[:, 1]
    adh_gt = y_test[:, 2]
    adh_pred = y_pred[:, 2]


    data = {
        'Gamma_GT': gamma_gt,
        'Gamma_Pred': gamma_pred,
        'Reg_param_GT': reg_param_gt,
        'Reg_param_Pred': reg_param_pred,
        'Adh_GT': adh_gt,
        'Adh_Pred': adh_pred
    }

    df = pd.DataFrame(data)

    csv_filename =(f'{save_path}/pred_params.csv')
    df.to_csv(csv_filename, index=False)

    print(f'Parameters saved to {csv_filename}')


