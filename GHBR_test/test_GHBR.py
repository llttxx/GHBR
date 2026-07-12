import contextlib
import os
from collections import Counter

from copy import deepcopy
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import *
from torch.cuda.amp import autocast, GradScaler
from train_utils import Bn_Controller
from torch.autograd import Function
import matplotlib.pyplot as plt
import cv2
from img2graph.img2graph import granular_balls_generate
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from datasets.dataset import GHBR_Dataset
from datasets.data_utils import get_data_loader
from ghbrmodel import R50_R50
from models.GBR import GCN_8_plus

from torch_geometric.data import Data,DataLoader
from Data_loader.transforms import Normalize



def save_anomaly_map(anomaly_map,image,save_path,file_name):
    if anomaly_map.shape != image.shape:
        anomaly_map = cv2.resize(anomaly_map, (image.shape[0], image.shape[1]))
    anomaly_map_norm = min_max_norm(anomaly_map)
    # anomaly map on image
    heatmap = cvt2heatmap(anomaly_map_norm * 255)
    hm_on_img = heatmap_on_image(heatmap,image)

    # save images
    cv2.imwrite(os.path.join(save_path,file_name),hm_on_img)

def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr

def specificity_score(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    TN = (y_true[y_score == 0] == 0).sum()
    N = (y_true == 0).sum()
    return TN / N

def testimg(save_visual=True, visualize_tsne=False):

    model = R50_R50(img_size=256,
                    train_encoder=True,
                    stop_grad=True,
                    reshape=True,
                    bn_pretrain=False,
                    )
    GBmodel = GCN_8_plus(num_features=13,initdim=256,inithead=8).cuda()
    print('model',model)
    # model loading parameters
    checkpoint1 = torch.load(r'E:\GHBR\saved_models\3799model')
    model.cuda()
    model.load_state_dict(checkpoint1['model'])
    model.eval()

    print('module load finish')

    checkpoint2 = torch.load(r'E:\GHBR\saved_models\3799GBmodel')
    GBmodel.load_state_dict(checkpoint2['GBmodel'])
    GBmodel.eval()
    print('GBmodule load finish')

    # define eval_loader
    data_dir = r"E:/Br35H"
    eval_dset = GHBR_Dataset(name='mri',train=False,data_dir=data_dir)
    eval_dset = eval_dset.get_dset()
    eval_loader = get_data_loader(eval_dset,64,num_workers=1,drop_last=False,pin_memory=False)
    print('eval_loader is ready')
    total_num = 0.0
    total_loss = 0.0
    y_true = []
    y_prob = []
    y1_prob = []
    y2_prob = []
    y3_prob = []

    with torch.no_grad():  
        for _, x, r, xo, y, file_names in eval_loader:
            x, y = x.cuda(), y.cuda().float()
            num_batch = x.shape[0]
            total_num += num_batch            
            bz = r.shape[0]
            datalist = []
            count = 0
            # granular rectangle data
            for imager in r:
                xr,adj,edge_attr,center_index = granular_balls_generate(imager.numpy())
                nodes = torch.as_tensor(np.array(xr),dtype=torch.float)  
                adj = torch.as_tensor(np.array(adj),dtype=torch.long)  
                edge_attr = torch.as_tensor(np.array(edge_attr),dtype=torch.float) 
                labelr = y[count]
               
                # normalization
                mean = torch.tensor([[3.2000e+01,3.2000e+01,3.7249e-01,9.5431e-01,9.6399e+01,1.5960e+03,
                                      9.6353e+01,1.5959e+03,9.6287e+01,1.5959e+03,8.6175e+01,1.0976e+02,8.6026e+01]])
                std = torch.tensor([[3.2000e+01,3.2000e+01,9.8542e-01,1.1391e+00,5.9946e+01,2.6235e+03,
                                     5.9925e+01,2.6236e+03,5.9927e+01,2.6235e+03,6.1715e+01,4.8584e+01,4.9315e+01]])

                normalize = Normalize(mean,std)
                nodes = normalize(nodes)

                data = Data(x=nodes,edge_index=adj,edge_attr=edge_attr,y=labelr)
                datalist.append(data)
                
                count = count + 1
            GRloader = DataLoader(datalist,batch_size=bz,shuffle=False)
            
            for step,sample_batched in enumerate(GRloader):  
                sample_batched = sample_batched.cuda()
                gr = GBmodel(sample_batched)      
            
            result = model(x,gr)
            amap_reduction = 'max'
            if  amap_reduction == 'mean':  
                p_img = result['p_all'].flatten(1).mean(1)
                p1_img = result['p1'].flatten(1).mean(1)
                p2_img = result['p2'].flatten(1).mean(1)
                p3_img = result['p3'].flatten(1).mean(1)
            elif isinstance(amap_reduction, float):  
                anomaly_map = result['p_all'].flatten(1)
                p_img = torch.sort(anomaly_map, dim=1, descending=True)[0][:,
                        :int(anomaly_map.shape[1] * amap_reduction)].mean(dim=1)
                anomaly_map = result['p1'].flatten(1)
                p1_img = torch.sort(anomaly_map, dim=1, descending=True)[0][:,
                         :int(anomaly_map.shape[1] * amap_reduction)].mean(dim=1)
                anomaly_map = result['p2'].flatten(1)
                p2_img = torch.sort(anomaly_map, dim=1, descending=True)[0][:,
                         :int(anomaly_map.shape[1] * amap_reduction)].mean(dim=1)
                anomaly_map = result['p3'].flatten(1)
                p3_img = torch.sort(anomaly_map, dim=1, descending=True)[0][:
                         ,:int(anomaly_map.shape[1] * amap_reduction)].mean(dim=1)
            else:  
                p_img = result['p_all'].flatten(1).max(1)[0]
                p1_img = result['p1'].flatten(1).max(1)[0]
                p2_img = result['p2'].flatten(1).max(1)[0]
                p3_img = result['p3'].flatten(1).max(1)[0]

            y_true.extend(y.cpu().tolist())
            y_prob.extend(p_img.cpu().tolist())
            y1_prob.extend(p1_img.cpu().tolist())
            y2_prob.extend(p2_img.cpu().tolist())
            y3_prob.extend(p3_img.cpu().tolist())

            total_loss += result['loss'].detach().item() * num_batch

            
            if save_visual:
                save_path = r'E:\2025testimg'                       
           
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                anomaly_maps = F.interpolate(result['p_all'], size=xo.shape[1:3], mode='bilinear')                
                recon_img = F.interpolate(result['d1'][:,0:1,:,:],size=xo.shape[1:3],mode='bilinear')
                                
                for i in range(xo.shape[0]):
                    image = xo[i].numpy().astype('uint8')
                    anomaly_map = anomaly_maps[i].cpu().permute(1, 2, 0).numpy()
                    reconimg = recon_img[i].permute(1, 2, 0).detach().cpu().numpy()                    
                    file_name = file_names[i]
                    cv2.imwrite(os.path.join(save_path,file_name),anomaly_map)
                    
                print('save all img')
            torch.cuda.empty_cache()
             
        # Visualize in 3D with t-SNE
        # if visualize_tsne:
        #     features = np.concatenate(feature_list, axis=0)
        #     self.visualize_tsne_3d(features, y_true, save_path=os.path.join(args.save_dir, args.save_name, 'features_tsne_3d.png'))

        # # Visualize using t-SNE
        # if visualize_tsne:
        #     self.visualize_tsne(features, y_true, save_path=os.path.join(args.save_dir, args.save_name, 'features_tsne.png'))

        thresh = return_best_thr(y_true, y_prob)
        acc = accuracy_score(y_true, y_prob >= thresh)
        f1 = f1_score(y_true, y_prob >= thresh)
        recall = recall_score(y_true, y_prob >= thresh)
        specificity = specificity_score(y_true, y_prob >= thresh)

        AUC = roc_auc_score(y_true, y_prob)
        AUC1 = roc_auc_score(y_true, y1_prob)
        AUC2 = roc_auc_score(y_true, y2_prob)
        AUC3 = roc_auc_score(y_true, y3_prob)

        print('eval/loss:', total_loss / total_num, 'eval/f1:',f1, 'eval/recall:', recall,
            'eval/specificity:', specificity, 'eval/acc:', acc,
            'eval/AUC:', AUC, 'eval/AUC1:', AUC1, 'eval/AUC2:', AUC2, 'eval/AUC3:', AUC3)

if __name__ == "__main__":
    testimg()
    


