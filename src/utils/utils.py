import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np


def train_duq(model,epoch,device,train_loader,optimizer,lambda_):
    # train duq model for one epoch. applies gradient
    # penalty and updates centroids 

    model.train()
    for i, (data,targets) in enumerate(train_loader):

        data, targets = data.to(device), targets.to(device)
        data.requires_grad_(True)
        targets = F.one_hot(targets,num_classes=10).float()
        optimizer.zero_grad()
        Wx = model(data)
        K = model.kernel(Wx)
        batch_loss = model.loss(K,targets)

        data_grad = lambda_ * grad_penalty(K,data)
        batch_loss += data_grad

        batch_loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.update_centroids(Wx,targets)

    print('done epoch: {}'.format(epoch))


def test_duq(model,epoch,batch_size,device,test_loader):
    # evaluate model for classification accuracy 

    model.eval()
    correct = 0
    with torch.no_grad():
        for data,targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            Wx = model(data)
            K  = model.kernel(Wx)
            pred = K.argmax(dim = 1)
            correct += pred.eq(targets).sum().item()

    print('Test accuracy: {:.4f}'.
            format(correct/(len(test_loader)*batch_size)))


def grad_penalty(K,data):
    # computes the gradient penalty of the sum of the 
    # kernal distances wrt the data

    K_sum = K.sum(-1)
    dk_dx = torch.autograd.grad(
        outputs=K_sum,
        inputs=data,
        grad_outputs=torch.ones_like(K_sum),
        create_graph=True,
        retain_graph=True)[0]

    dk_dx = dk_dx.flatten(start_dim=1)
    grad_norm_sq = (dk_dx**2).sum(dim=1)
    grad_penalty = ((grad_norm_sq - 1) ** 2).mean()
    return grad_penalty


def ood_detection_eval(model,device,ood_loader):
    # runs inference on duq model. scores
    # are defined as negative max kernel distance

    model.eval()
    eval_scores = []
    with torch.no_grad():
        for data,_ in ood_loader:
            data = data.to(device)
            Wx = model(data)
            K  = model.kernel(Wx)
            scores = -torch.max(K,dim = 1)[0]
            eval_scores.append(scores.cpu().numpy())
    
    return np.concatenate(eval_scores)


def train_standard(model,loss,epoch,device,train_loader,optimizer):
    # train one epoch for standard classification model

    model.train()
    running_loss = 0
    for i, (data,targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(data)
        batch_loss = loss(output,targets)
        batch_loss.backward()
        optimizer.step()
        running_loss += batch_loss.item()

    print('done epoch: {}'.format(epoch))


def test_standard(model,loss,epoch,batch_size,device,test_loader):
    # evaluation for standard model 

    model.eval()
    correct = 0
    with torch.no_grad():
        for data,targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            pred = output.argmax(dim = 1)
            correct += pred.eq(targets).sum().item()

    print('Test accuracy: {:.4f}'.format(correct/(len(test_loader)*batch_size)))
