from utils import *

#Parameters:
mask_size = 0.7
mask_value = 10 #inicialización de los nodos desconocidos, valor más o menos arbitrario (se estudirá en detalle más adelante)
message_passing_iterations = 50

bg, edges = build_graph_from_csv('data/graph_mix_2019.csv')
visualize(bg)

mask = np.random.choice(a=[True, False], size=(bg.num_nodes()), p=[mask_size, 1-mask_size])
bg_original = bg.ndata['x'].clone().detach() 
    
bg.ndata['x'][mask] = mask_value
bg.ndata['x'][remove] = -1
visualize(bg)

MS_1layer(bg, mask,feature_name = 'x', max_iters = message_passing_iterations, history = True, original_feat = bg_original, show = False, save = True)

print('Final error: ', mean_absolute_percentage_error(bg.ndata['x'], bg_original))