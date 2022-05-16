from utils import *
from models import GCN
import warnings
warnings.filterwarnings("ignore")

#Parameters:
mask_size = 0.8
mask_value = 10 #inicialización de los nodos desconocidos, valor más o menos arbitrario (se estudirá en detalle más adelante)
gnn_epochs = 1000
message_passing_iterations = 50

bg, edges = build_graph_and_node_features('data/graph_mix_2019.csv')
visualize(bg, 'pollution')

mask = np.random.choice(a=[True, False], size=(bg.num_nodes()), p=[mask_size, 1-mask_size])
bg_original = bg.ndata['pollution'].clone().detach() 
    
bg.ndata['pollution'][mask] = mask_value
bg.ndata['pollution'][remove] = -1

MS_1layer(bg, mask, feature_name = 'pollution', max_iters = message_passing_iterations, history = True, original_feat = bg_original, show = False, save = True)


normalized_y = normalize_feature(bg.ndata['y'])
normalized_x = normalize_feature(bg.ndata['x'])
normalized_noise = normalize_feature(bg.ndata['noise'])
normalized_pollution = normalize_feature(bg.ndata['pollution'])

node_features = torch.stack([normalized_x, normalized_y, normalized_noise, normalized_pollution], axis=1).squeeze(2).float()
node_labels = bg_original

train_size = 0.8
train_mask = np.random.choice(a=[True, False], size=(bg.num_nodes()), p=[train_size, 1-train_size])
train_mask = torch.tensor(train_mask)
valid_mask = torch.tensor(~train_mask)
n_features = node_features.shape[1]

model = GCN(in_feats=n_features, hid_feats=50)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(trainable_params, ' trainable parameters')
print(total_params, ' total parameters')


opt = torch.optim.Adam(model.parameters(), lr = 0.01)
best_val = 1000.
earlystop = 0 #parar entrenamiento si el performance no mejora en x epocas

for epoch in range(gnn_epochs):
    
    model.train()
    # forward propagation by using all nodes
    logits = model(bg, node_features)
    # compute loss
    loss = F.mse_loss(logits[train_mask], node_labels[train_mask])
    # compute validation loss
    val_loss = F.mse_loss(logits[valid_mask], node_labels[valid_mask])
    # backward propagation
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch%500==0:
        print('Epoch: ', epoch)
        print('Train loss: ', loss.item())
        print('Validation loss: ', val_loss.item())
        print('Total graph loss: ', F.mse_loss(logits, bg_original))
    if val_loss.item() < best_val:
        best_val = val_loss.item()
        earlystop = 0
    else:
        earlystop +=1
    if earlystop == 500:
        break

    # Save model if necessary.  Omitted in this example.
    
    
logits = model(bg, node_features)
u, v = torch.tensor(edges.FID_x), torch.tensor(edges.FID_y)
g = dgl.graph((u, v))
bg_reconstructed = dgl.to_bidirected(g)
bg_reconstructed.ndata['pollution'] = logits.detach()
visualize(bg_reconstructed, 'pollution')
print('Final error: ', mean_absolute_percentage_error(bg_reconstructed.ndata['pollution'], bg_original))
