import os
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.nn.functional as F


from posenet_model import get_pose_net
import datasets
from loss import GeodesicLoss, HeadPoseLoss

from MVPose import MVPoseNet


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--num_epochs', dest='num_epochs',
        help='Maximum number of training epochs.',
        default=50, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=64, type=int)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.00008, type=float)
    parser.add_argument(
        '--dataset', dest='dataset', help='Dataset type.',
        default='DevEvMV', type=str) #Pose_300W_LP
    parser.add_argument(
        '--data_dir', dest='data_dir', help='Directory path for data.',
        default='/nfs/hpc/cn-gpu5/DevEv/headpose_dataset/bodyhead_dataset_new/', type=str)#BIWI_70_30_train.npz
    parser.add_argument(
        '--filename_list', dest='filename_list',
        help='Path to text file containing relative paths for every example.',
        default='/nfs/hpc/cn-gpu5/DevEv/headpose_dataset/gt_body_new.txt', type=str) #BIWI_70_30_train.npz #300W_LP/files.txt
    parser.add_argument(
        '--output_string', dest='output_string',
        help='String appended to output snapshots.', default='Eval_DevEvMat', type=str)
    parser.add_argument(
        '--setup', dest='setup',
        help='Mat view or Room view (room/mat)', default='room', type=str)
    parser.add_argument(
        '--timestamps', dest='timestamps',
        help='Path to timestamps file', default="../DevEvData_2024-02-02.csv", type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='../BodyPose/infant_w48_384x288.pth', type=str)

    args = parser.parse_args()
    return args


def get_ignored_params(model):
    b = [model.layer0]
    #b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_non_ignored_params(model):
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_fc_params(model):
    b = [model.linear_reg]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param


def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

def prep_features(features, views, aug = True):
    B, S = views.size()
    _, D = features.size()
    

    multiview_features = torch.zeros((B,8,D)).float().to(features.device)
    start, end = 0, 0
    for b in range(B):
        L =  (views[b] == 1).sum()
        end += L
        multiview_features[b, views[b] > 0] = features[start:end]
        start = end.clone()
        
        if aug and np.random.rand() > 0.5 and L > 1:
            # Find the indices of ones in the tensor
            ones_indices = torch.nonzero(views[b] == 1).squeeze()
            # Randomly choose one index from the ones_indices
            index_to_change = torch.randint(0, ones_indices.size(0), (1,)).item()
            # Set the chosen index to zero
            multiview_features[b, ones_indices[index_to_change]] *= 0


    
    return multiview_features

def get_angular_error(gt, pred):
    similarity = (pred*gt).sum(-1).detach().cpu().numpy()
    theta = (np.arccos(similarity)*180/np.pi)
    return theta

def test(test_loader, model, model_R):
    err = []
    for i, (images, gt_mat, att_dir, views) in enumerate(test_loader):
        images = torch.Tensor(images).cuda(gpu)
        gt_mat = gt_mat.cuda(gpu)
        att_dir = F.normalize(att_dir, dim=-1).cuda(gpu)
        views = views.cuda(gpu)

        # Forward pass
        with torch.no_grad():
            input = None
            for b, im in enumerate(images):
                if input is None: input = im[views[b] > 0]
                else:
                    input = torch.cat([input, im[views[b] > 0] ], dim = 0)
            features = model.forward_feature(input)
            features = prep_features(features, views, aug = False)
            pred_mat = model_R(features, views)
            #pred_mat = torch.matmul(pred_mat, torch.FloatTensor([0,0,1]).cuda(gpu))
            
        theta = get_angular_error(att_dir, pred_mat)
        err.append(theta)
    
    err = np.concatenate(err, axis = 0)
    mu, sigma = err.mean(), err.std()
    
    err[err >= 90] = 90
    bins = np.arange(0.0, max(err) + 1, 5)
    hist, bin_edges = np.histogram(err, bins=bins)
    # Normalize by total area (bin width)
    hist_normalized = hist / len(err)
    print(hist_normalized)
    # Plot the normalized histogram
    plt.bar(bin_edges[:-1] + 2.5, hist_normalized, width=5, edgecolor='black', alpha=0.75)
    plt.xlabel('Angular Error (deg)')
    plt.ylabel('Density')
    plt.xticks(bin_edges, rotation=45, ha="right")
    plt.yticks(np.arange(0, max(hist_normalized) + 0.05, 0.1))
    plt.title("Angular Error Hist - {:.3f} +/- {:.3f}".format(mu, sigma))
    plt.tight_layout() 
    plt.savefig("output/snapshots/error_hist_ang.png")   
    plt.close()


    return mu


if __name__ == '__main__':

    args = parse_args()
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    gpu = args.gpu_id

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    #model = SixDRepNet(backbone_name='RepVGG-B1g2',
    #                   backbone_file='',
    #                   deploy=True,
    #                   pretrained=False)
    
    model = get_pose_net(is_train=False)

    model_R = MVPoseNet(feature_dim=384+17*2) 
    
    if not args.snapshot == '':
        saved_state_dict = torch.load(args.snapshot)
        # Load snapshot

        if 'model_state_dict' in saved_state_dict:
            model.load_state_dict(saved_state_dict['model_state_dict'])
        else:
            model.load_state_dict(saved_state_dict)
        print("Loaded model snapshot...")

    print('Loading data.')

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    transformations = transforms.Compose([transforms.Resize((224,224)), #transforms.Resize((224,224)),
                                          #transforms.RandomCrop(224),
                                          transforms.ToTensor(),
                                          normalize])

    pose_dataset_train = datasets.getDataset(
        args.dataset, args.data_dir, args.filename_list, transformations, train_mode = True, setup = args.setup, timestamps=args.timestamps)

    pose_dataset_test = datasets.getDataset(
        args.dataset, args.data_dir, args.filename_list, transformations, train_mode = False, setup = args.setup, timestamps=args.timestamps)

    train_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1)

    test_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1)
    
    model.cuda(gpu)
    model_R.cuda(gpu)
    crit =  GeodesicLoss().cuda(gpu) #torch.nn.MSELoss().cuda(gpu)
    crit2 = HeadPoseLoss()

    optimizer = torch.optim.Adam(model_R.parameters(), lr=args.lr, weight_decay=0.0004)

    #milestones = np.arange(num_epochs)
    milestones = [20, 30, 35, 40]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.7)


    model = model.train()
    error_list = []
    test_error, best_error = [], 1000
    print('Starting training.', len(pose_dataset_train))
    for epoch in range(1, num_epochs+1):
        loss_sum = .0
        iter = 0
        err = []
        for i, (images, gt_mat, att_dir, views) in enumerate(train_loader):
            iter += 1
            images = torch.Tensor(images).cuda(gpu)
            gt_mat = gt_mat.cuda(gpu)
            att_dir = F.normalize(att_dir, dim=-1).cuda(gpu)
            views = views.cuda(gpu)

            # Forward pass
            with torch.no_grad():
                input = None
                for b, im in enumerate(images):
                    if input is None: input = im[views[b] > 0]
                    else:
                        input = torch.cat([input, im[views[b] > 0] ], dim = 0)
                features = model.forward_feature(input)

            features = prep_features(features, views)
            pred_mat = model_R(features, views)
            # Calc loss
            #loss = 1.0*crit(gt_mat, pred_mat) #+ crit2(gt_mat, pred_mat)
            loss = F.mse_loss(att_dir, pred_mat)
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                     
            err.append(get_angular_error(att_dir, pred_mat).mean())
            loss_sum += loss.item()
            if i % 20 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: '
                      '%.6f' % (
                          epoch,
                          num_epochs,
                          i+1,
                          len(pose_dataset_train)//batch_size,
                          loss.item(),
                      )
                      )

        scheduler.step()
        
        error_list.append([loss_sum/(i+1),  sum(err)/len(err)])
        print("Epoch {}, error {:.3f} degrees".format(epoch, error_list[-1][-1]))
        #print("Error Yaw: {:.2f}, Pitch: {:.2f}, Roll: {:.2f}".format(err_y.item(), err_p.item(), err_r.item()))              
        error = test(test_loader, model, model_R) 
        print("Test Error:", error)
        test_error.append([epoch, error])
        # Save models at numbered epochs.
        if best_error >= error and epoch < num_epochs:
            print('Taking snapshot...',
                  torch.save({
                      'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'model_R_state_dict': model_R.state_dict(),
                  }, 'output/snapshots/' + args.output_string +
                      'MV_best.pth')
                  )
            best_error = error
        
    #############
    print('Taking snapshot...',
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_R_state_dict': model_R.state_dict(),
            }, 'output/snapshots/' + args.output_string +
                'MV_epoch_' + str(epoch) + '.pth')
            )
                
    error_list = np.array(error_list)
    test_error = np.array(test_error)
    
    best = np.argmin(test_error[:,1:].sum(-1))
    plt.figure()
    plt.plot(test_error[:,0], test_error[:,1], label='Test Error, Best:{:.2f} deg'.format(test_error[best,1]))
    plt.legend(frameon=False)
    plt.savefig('output/snapshots/' + args.output_string + "_MV_test.png")
    plt.close()
    
    plt.figure()
    plt.plot(error_list[:,0], label='Training loss')
    plt.legend(frameon=False)
    plt.savefig('output/snapshots/' + args.output_string + "_MV_training_log.png")
    plt.close()
    
    best = np.argmin(error_list[:,1:].sum(-1))
    plt.figure()
    plt.plot(error_list[:,1], label='Error, Best:{:.2f} deg'.format(error_list[best,1]))
    plt.legend(frameon=False)
    plt.savefig('output/snapshots/' + args.output_string + "_MV_error_log.png")
    plt.close()
    
    
