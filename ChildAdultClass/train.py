import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Subset

    
def load_split_train_test(datadir, batch_size):
    train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                        transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.RandomErasing(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                       ])
    test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                      ])
    
    train_data = datasets.ImageFolder(datadir + "train/",       
                    transform=train_transforms)
    test_data = datasets.ImageFolder(datadir + "test/",
                    transform=test_transforms)
    
    
    idx = [i for i in range(len(train_data)) if train_data.imgs[i][1] != train_data.class_to_idx['adults'] and train_data.imgs[i][1] != train_data.class_to_idx['children']]
    # build the appropriate subset
    train_data = Subset(train_data, idx)
    idx = [i for i in range(len(test_data)) if test_data.imgs[i][1] != test_data.class_to_idx['adults'] and test_data.imgs[i][1] != test_data.class_to_idx['children']]
    # build the appropriate subset
    test_data = Subset(test_data, idx)
    
    print("Training set size:", len(train_data), "Test set size:",len(test_data))

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    return trainloader, testloader

def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
    
    
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(nn.Linear(512, 128),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(128, 2),
                                 #nn.Linear(128, 3),
                                 nn.LogSoftmax(dim=1))

    """model = models.resnet34(pretrained=True)
    model.fc = nn.Sequential(nn.Linear(512,128),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(128, 2),
                                 nn.LogSoftmax(dim=1))   """


    """model = models.swin_s(weights='DEFAULT')
    model.head = nn.Sequential(nn.Linear(512, 2),
                                 nn.LogSoftmax(dim=1))"""

    return model, device

def modify_labels(y):
    # 'adults': 0, 'adultsDevEv': 1, 'children': 2, 'childrenDevEv': 3
    y_ = torch.zeros_like(y)
    y_[y==2] = 1
    y_[y==3] = 1
    return y_

def train(data_dir, output_dir):
    model, device = get_model()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    model.to(device)

    trainloader, testloader = load_split_train_test(data_dir, 64)

    num_epochs = 50
    steps = 0
    running_loss = 0
    print_every = 10
    train_losses, test_losses = [], []
    accuracy_list = []
    for epoch in range(num_epochs):
        for inputs, labels in trainloader:
            labels = modify_labels(labels)
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        labels = modify_labels(labels)
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()
                        
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                train_losses.append(running_loss/len(trainloader))
                test_losses.append(test_loss/len(testloader))   
                accuracy_list.append(accuracy/len(testloader))                 
                print(f"Epoch {epoch+1}/{num_epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(testloader):.3f}.. "
                    f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                model.train()
                if len(accuracy_list) >= 10:
                    if accuracy_list[-1] == max(accuracy_list):
                        torch.save(model, os.path.join(output_dir,'childclassifier_best.pth'))

        scheduler.step()
    train_losses = np.array(train_losses)
    train_losses[train_losses > 1.5] = 1.5
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Test loss')
    plt.plot(accuracy_list, label='Test Accuracy Best:{:.1f}'.format(max(accuracy_list)*100))
    plt.legend(frameon=False)
    plt.savefig(os.path.join(output_dir, "training_log.png"))
    torch.save(model, os.path.join(output_dir,'childclassifier.pth'))

    return


def test(data_dir, checkpath='output/childclassifier_best.pth'):
    model, device = get_model()
    model = torch.load(checkpath)
    model.to(device)
    model.eval()

    _, testloader = load_split_train_test(data_dir, 64)
        
    total, accuracy = 0, 0
    for inputs, labels in testloader:
        labels = modify_labels(labels)
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            logps = model.forward(inputs)

        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += equals.type(torch.FloatTensor).sum().item()
        total += len(inputs)
               
    print(f"Test accuracy: {accuracy/total:.3f}")
    return

if __name__ == "__main__":
    data_dir = 'AdultChildDataset/'
    output_dir = "output/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    train(data_dir, output_dir)
    test(data_dir)
    