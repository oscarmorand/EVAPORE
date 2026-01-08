# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: graph-neural-networks
#     language: python
#     name: python3
# ---

# %%
import networkx as nx
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch import nn
import torch_geometric as pyg
import numpy as np
import PIL.Image as Image

# %%
img = np.array(Image.open("/home/morand/afs/datasets/FIVES/train/Ground truth/1_A.png"))[:,:,0]
#img = np.array(Image.open("/home/morand/afs/tests/New Piskel.png"))[:,:,0]
#img = np.array(Image.open("/home/morand/afs/tests/simple_loop_two.png"))[:,:,0]
#img = np.array(Image.open("/home/morand/afs/tests/error_loop_branch.png"))[:,:,0]

print(img.shape)

plt.imshow(img, cmap='gray')
plt.show()

# %%
from torch_geometric.utils.convert import from_networkx
from graph.graph_creation import img_to_graph

G = img_to_graph(img, clean=True, closing_radius=1, return_pixel_graph=False)

# %%
from graph.graph_io import save_graph_to_dot

save_graph_to_dot(G, "/home/morand/afs/tests/debug_graph.dot")

# %%
pyg_graph = from_networkx(G)

# %%
print(pyg_graph.edge_index)

# %%
print(pyg_graph.edge_attr)


# %%
def get_position_features(G):
    coords = []
    for node_id, node_data in G.nodes(data=True):
        coords.append(node_data['pos'])
    return torch.tensor(coords, dtype=torch.float)

def get_radius_features(G):
    radii = []
    for node_id, node_data in G.nodes(data=True):
        radii.append([node_data.get('radius', 0.0)])
    return torch.tensor(radii, dtype=torch.float)

features_func = {
    'pos': get_position_features,
    'radius': get_radius_features
}

def generate_node_features(G, wanted_features):
    features = None
    for feature_name in wanted_features:
        if feature_name in features_func:
            feature_tensor = features_func[feature_name](G)
            if features is None:
                features = feature_tensor
            else:
                features = torch.cat((features, feature_tensor), dim=1)
        else:
            raise ValueError(f"Feature '{feature_name}' is not recognized.")
    return features


# %%
wanted_features = ['pos', 'radius']

x = generate_node_features(G, wanted_features)
print(x.shape)

# %%
data = pyg_graph
data.x = x

# %%
print(G.number_of_edges() * 2)

print(data.edge_index.shape)

# %%
import os
from torch_geometric.data import Data

graphs = []
wanted_features = ['pos', 'radius']
base_dir = "/home/morand/afs/datasets/FIVES/train/Ground truth/"

paths = os.listdir(base_dir)
paths.sort()
paths_indexes = [int(p.split('_')[0]) for p in paths]
print(paths_indexes)
real_paths = [None] * len(paths_indexes)
for i, idx in enumerate(paths_indexes):
    real_paths[idx - 1] = paths[i]
print(real_paths)

# %%
networkx_graphs = []
imgs = []

for i, path in enumerate(real_paths):
    if i >= 20:
        break
    full_path = os.path.join(base_dir, path)
    img = np.array(Image.open(full_path))[:,:,0]
    imgs.append(img)
    G = img_to_graph(img, clean=True, closing_radius=1, return_pixel_graph=False)
    networkx_graphs.append(G)

    pyg_graph = from_networkx(G)
    x = generate_node_features(G, wanted_features)

    pyg_graph.x = x.float()
    pyg_graph.y = torch.ones(pyg_graph.x.shape[0], dtype=torch.float)

    edge_set = set()
    for edge in pyg_graph.edge_index.T.tolist():
        if (edge[1], edge[0]) not in edge_set and (edge[0], edge[1]) not in edge_set:
            edge_set.add((edge[0], edge[1]))
    edge_index_list = list(edge_set)
    pyg_graph.edge_index = torch.tensor(edge_index_list, dtype=torch.int64).T

    clean_graph = Data(x=pyg_graph.x, edge_index=pyg_graph.edge_index, y=pyg_graph.y)

    graphs.append(clean_graph)

print(graphs)

# %%
from torch_geometric.transforms import RandomLinkSplit

train, val, test = [], [], []
for data in graphs:
    edge_transform = RandomLinkSplit(num_val=0.2, 
                                     num_test=0.2, 
                                     key='edge_label', 
                                     is_undirected=True, 
                                     add_negative_train_samples=True,
                                     neg_sampling_ratio=1.0,
                                     )
    train_data, val_data, test_data = edge_transform(data)

    print("Train data:", train_data)
    print("Val data:", val_data)
    print("Test data:", test_data)

    train.append(train_data)
    val.append(val_data)
    test.append(test_data)

# %%
train_loader = pyg.loader.DataLoader(train, batch_size=4, num_workers=7)
val_loader = pyg.loader.DataLoader(val, batch_size=4, num_workers=7)
test_loader = pyg.loader.DataLoader(test, batch_size=4)

# %%
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score
from lightning.pytorch.loggers import WandbLogger

class SimpleGCN(pl.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels, lr=0.01):
        super(SimpleGCN, self).__init__()
        self.save_hyperparameters()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.activation = nn.ReLU()
        self.criterion = nn.BCEWithLogitsLoss()
    
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.conv3(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        out = self.decode(z, edge_index).view(-1)
        return out

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
    
    def training_step(self, batch, batch_idx):
        out = self.forward(batch.x, batch.edge_index)
        loss = self.criterion(out, batch.edge_label.float())
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def _compute_auc(self, batch):
        out = self.forward(batch.x, batch.edge_label_index)
        pred = torch.sigmoid(out)
        auc = roc_auc_score(batch.edge_label.cpu(), pred.cpu())
        return auc
    
    def _compute_acc(self, batch):
        out = self.forward(batch.x, batch.edge_label_index)
        pred = (torch.sigmoid(out) > 0.5).float()
        correct = (pred == batch.edge_label).sum().item()
        acc = correct / batch.edge_label.size(0)
        return acc

    def validation_step(self, batch, batch_idx):
        val_auc = self._compute_auc(batch)
        val_acc = self._compute_acc(batch)

        self.log('val_acc', val_acc, prog_bar=True)
        self.log('val_auc', val_auc, prog_bar=True)
        
        return val_auc
    
    def test_step(self, batch, batch_idx):
        test_auc = self._compute_auc(batch)
        test_acc = self._compute_acc(batch)

        self.log('test_acc', test_acc, prog_bar=True)
        self.log('test_auc', test_auc, prog_bar=True)

        return test_auc
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
model = SimpleGCN(in_channels=3, hidden_channels=512, out_channels=64)

# %%
import wandb

wandb.finish()
wandb_logger = WandbLogger(project='simple_gcn_test', log_model='all')

trainer = pl.Trainer(max_epochs=50, logger=wandb_logger, enable_checkpointing=False)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# %%
trainer.test(model, dataloaders=test_loader)

# %%
# inference on a single graph from the test set

out = model.forward(test[0].x, test[0].edge_label_index)
pred = (torch.sigmoid(out) > 0.5).float()

print(len(pred), pred)

# %%
import networkx as nx

def save_test_graph_to_dot(test_input_graph, pred, path:str) -> None:
    G = nx.Graph()

    for i in range(test_input_graph.x.shape[0]):
        G.add_node(i)

    for i, edge in enumerate(test_input_graph.edge_index.T.tolist()):
        G.add_edge(edge[0], edge[1], color='black', style='solid')

    diff = test_input_graph.edge_label - pred
    for i, edge in enumerate(test_input_graph.edge_label_index.T.tolist()):
        if diff[i] == 1:
            G.add_edge(edge[0], edge[1], color='red', style='dashed')
        elif diff[i] == 0:
            G.add_edge(edge[0], edge[1], color='green', style='solid')

    save_graph_to_dot(G, path)


# %%
save_test_graph_to_dot(test[0], pred, "/home/morand/afs/tests/test_graph_prediction.dot")

# %%
import networkx as nx

def save_split_graph_to_dot(train_data, val_data, test_data, path: str) -> None:
    """
    Save a split NetworkX graph to DOT format.

    Parameters:
        - graph (NetworkX.Graph): graph to save
        - path (str): Output file path (e.g., 'graph.dot')
    """

    split_graph = nx.MultiGraph()

    edge_set = set()
    for edge in train_data.edge_index.T.tolist():
        edge = tuple(sorted((edge[0], edge[1])))
        if edge not in edge_set:
            edge_set.add(edge)
            split_graph.add_edge(edge[0], edge[1], color='red', style='solid')

    for i, edge in enumerate(val_data.edge_label_index.T.tolist()):
        if val_data.edge_label[i] == 1:
            edge = tuple(sorted((edge[0], edge[1])))
            if edge not in edge_set:
                edge_set.add(edge)
                split_graph.add_edge(edge[0], edge[1], color='blue', style='solid')

    for i, edge in enumerate(test_data.edge_label_index.T.tolist()):
        if test_data.edge_label[i] == 1:
            edge = tuple(sorted((edge[0], edge[1])))
            if edge not in edge_set:
                edge_set.add(edge)
                split_graph.add_edge(edge[0], edge[1], color='green', style='solid')

    nx.nx_pydot.write_dot(split_graph, path)


# %%
save_split_graph_to_dot(train[0], val[0], test[0], "/home/morand/afs/tests/split_graph.dot")
