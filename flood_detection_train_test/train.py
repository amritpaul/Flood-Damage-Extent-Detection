'''
Authors - Amrit Paul, Karan Pathak
Code to train flood multiclass semantic segmentation for the following classes
    0- Background
    1- Water/Flood
    2- Non-Flooded Buildings
    3- Flooded Buildings
with mixed-precision training
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torchsummary import summary
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
import albumentations as A
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

import config
from dataset import HarveyDataset, HarveyDatasetSiamese, HarveyDatasetMeta
from network import UNet, UNetSiamese, UNetMeta

import wandb

wandb.config = {
  "learning_rate": config.LEARNING_RATE,
  "epochs": config.NUM_EPOCHS,
  "batch_size": config.BATCH_SIZE,
}

wandb.init(project="Flood Damage Extent Detection", entity="flood_damage_extent_detection", config=wandb.config)
max_acc = 0.0

def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = y_true.view(-1)
    y_pred_pos = y_pred.view(-1)
    true_pos = (y_true_pos * y_pred_pos).sum()
    false_neg = (y_true_pos * (1 - y_pred_pos)).sum()
    false_pos = ((1 - y_true_pos) * y_pred_pos).sum()
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

def check_accuracy(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    '''
    intersection = (inputs * targets).sum()                            
    dice = (2.*intersection)/(inputs.sum() + targets.sum() + smooth)  
    '''
    with torch.no_grad():
        if isinstance(loader, tuple):
            x,y = loader
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            softmax = nn.Softmax(dim=1)
            preds = torch.argmax(softmax(model(x)),axis=1)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

        else:        
            for x, y in loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                softmax = nn.Softmax(dim=1)
                preds = torch.argmax(softmax(model(x)),axis=1)
                num_correct += (preds == y).sum()
                num_pixels += torch.numel(preds)
                dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    acc = num_correct/num_pixels*100
    dice_score = dice_score/len(loader)
    model.train()
    return acc, dice_score

def check_accuracy_siamese(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    '''
    intersection = (inputs * targets).sum()                            
    dice = (2.*intersection)/(inputs.sum() + targets.sum() + smooth)  
    '''
    with torch.no_grad():
        if isinstance(loader, tuple):
            x_pre, x_post, y = loader
            x_pre = x_pre.to(DEVICE)
            x_post = x_post.to(DEVICE)
            y = y.to(DEVICE)
            softmax = nn.Softmax(dim=1)
            print(x_pre.shape)
            print(x_post.shape)
            preds = torch.argmax(softmax(model(x_pre, x_post)),axis=1)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

        else:        
            for x_pre, x_post, y in loader:
                x_pre = x_pre.to(DEVICE)
                x_post = x_post.to(DEVICE)
                y = y.to(DEVICE)
                softmax = nn.Softmax(dim=1)
                preds = torch.argmax(softmax(model(x_pre, x_post)),axis=1)
                num_correct += (preds == y).sum()
                num_pixels += torch.numel(preds)
                dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    acc = num_correct/num_pixels*100
    dice_score = dice_score/len(loader)
    model.train()
    return acc, dice_score

def check_accuracy_meta(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    '''
    intersection = (inputs * targets).sum()                            
    dice = (2.*intersection)/(inputs.sum() + targets.sum() + smooth)  
    '''
    with torch.no_grad():
        if isinstance(loader, tuple):
            x_pre, x_post, y = loader
            x_pre = x_pre.to(DEVICE)
            x_post = x_post.float().to(DEVICE)
            y = y.to(DEVICE)
            softmax = nn.Softmax(dim=1)
            preds = torch.argmax(softmax(model(x_pre, x_post)),axis=1)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            acc = tversky(y, preds, smooth=1, alpha=0.7)

        else:        
            for x_pre, x_post, y in loader:
                x_pre = x_pre.to(DEVICE)
                x_post = x_post.float().to(DEVICE)
                y = y.to(DEVICE)
                softmax = nn.Softmax(dim=1)
                preds = torch.argmax(softmax(model(x_pre, x_post)),axis=1)
                num_correct += (preds == y).sum()
                num_pixels += torch.numel(preds)
                dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
                acc = tversky(y, preds, smooth=1, alpha=0.7)

    
    # acc = num_correct/num_pixels*100
    dice_score = dice_score/len(loader)
    model.train()
    return acc, dice_score

def check_accuracy_custom(loader, model):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    '''
    intersection = (inputs * targets).sum()                            
    dice = (2.*intersection)/(inputs.sum() + targets.sum() + smooth)  
    '''
    with torch.no_grad():
        if isinstance(loader, tuple):
            x,y = loader
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            softmax = nn.Softmax(dim=1)
            preds = torch.argmax(softmax(model(x)),axis=1)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            acc = tversky(y, preds, smooth=1, alpha=0.7)

        else:        
            for x, y in loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                softmax = nn.Softmax(dim=1)
                preds = torch.argmax(softmax(model(x)),axis=1)
                num_correct += (preds == y).sum()
                num_pixels += torch.numel(preds)
                dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
                acc = tversky(y, preds, smooth=1, alpha=0.7)

    # acc = num_correct/num_pixels*100
    dice_score = dice_score/len(loader)
    model.train()
    return acc, dice_score


def visualize_during_training(train_loader):
    for img,mask in train_loader:
        img1 = np.transpose(img[0,:,:,:],(1,2,0))
        mask1 = np.array(mask[0,:,:])
        img2 = np.transpose(img[1,:,:,:],(1,2,0))
        mask2 = np.array(mask[1,:,:])
        img3 = np.transpose(img[2,:,:,:],(1,2,0))
        mask3 = np.array(mask[2,:,:])
        _ , ax =  plt.subplots(3, 2, figsize=(18, 18))
        ax[0][0].imshow(img1)
        ax[0][1].imshow(mask1)
        ax[1][0].imshow(img2)
        ax[1][1].imshow(mask2)
        ax[2][0].imshow(img3)
        ax[2][1].imshow(mask3)
        plt.show()
        break

def unet_acc(train_loader, test_loader, model):
    print("Calculating Accuracy on Training Dataset...")    
    acc_train, dice_score = check_accuracy(train_loader, model)
    print(f'Train Accuracy: {acc_train.item():.2f}, Dice Score: {dice_score.item()}')

    print("Calculating Accuracy on Test Dataset...")    
    acc_test, dice_score = check_accuracy(test_loader, model)
    print(f'Test Accuracy: {acc_test.item():.2f}, Dice Score: {dice_score.item()}')

def unet_trainer_engine(loop):
    global max_acc
    for _,(data, targets) in loop:
        loop.set_description(f"Epoch {epoch+1}/{num_epochs} ")
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)
        targets = targets.type(torch.long)
        
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        display_acc = check_accuracy((data,targets), model)[0]
        acc.append(display_acc.item())
        
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item(), accuracy=display_acc.item())

    # update wandb
    wandb.log({"loss": loss})
    wandb.log({"epochs": epoch+1})
    wandb.log({"accuracy": display_acc.item()})

    # print(np.mean(acc))
    if np.mean(acc)>max_acc:
        max_acc = np.mean(acc)
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, f'./best_model.pt')
        print(f'Model saved at {epoch+1}!')

    if epoch+1 == num_epochs:
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, f'./latest_model.pt')
    
def unet_test_engine(x, y, out_dir, model):
    try:
        x = x.to(DEVICE)
        fig , ax =  plt.subplots(3, 3, figsize=(18, 18))
        softmax = nn.Softmax(dim=1)
        preds = torch.argmax(softmax(model(x)),axis=1).to('cpu')
        img1 = np.transpose(np.array(x[0,:,:,:].to('cpu')),(1,2,0))
        preds1 = np.array(preds[0,:,:])
        mask1 = np.array(y[0,:,:])
        img2 = np.transpose(np.array(x[1,:,:,:].to('cpu')),(1,2,0))
        preds2 = np.array(preds[1,:,:])
        mask2 = np.array(y[1,:,:])
        img3 = np.transpose(np.array(x[2,:,:,:].to('cpu')),(1,2,0))
        preds3 = np.array(preds[2,:,:])
        mask3 = np.array(y[2,:,:])
        ax[0,0].set_title('Image')
        ax[0,1].set_title('Prediction')
        ax[0,2].set_title('Mask')
        ax[1,0].set_title('Image')
        ax[1,1].set_title('Prediction')
        ax[1,2].set_title('Mask')
        ax[2,0].set_title('Image')
        ax[2,1].set_title('Prediction')
        ax[2,2].set_title('Mask')
        ax[0][0].axis("off")
        ax[1][0].axis("off")
        ax[2][0].axis("off")
        ax[0][1].axis("off")
        ax[1][1].axis("off")
        ax[2][1].axis("off")
        ax[0][2].axis("off")
        ax[1][2].axis("off")
        ax[2][2].axis("off")
        ax[0][0].imshow(img1)
        ax[0][1].imshow(preds1)
        ax[0][2].imshow(mask1)
        ax[1][0].imshow(img2)
        ax[1][1].imshow(preds2)
        ax[1][2].imshow(mask2)
        ax[2][0].imshow(img3)
        ax[2][1].imshow(preds3)
        ax[2][2].imshow(mask3)
        fig.savefig(f"{out_dir}/{batch_idx}.png")
    except:
        print(f'Skipped Last Image.')
        pass

def visualize_during_training_siamese(train_loader):
    for img_pre, img_post, mask in train_loader:
        print(img_pre.numpy().shape, img_post.numpy().shape, mask.numpy().shape) #(8, 3, 512, 512) (8, 3, 512, 512) (8, 512, 512)
        img_pre1 = np.transpose(img_pre[0,:,:,:],(1,2,0))
        img_post1 = np.transpose(img_post[0,:,:,:],(1,2,0))
        mask1 = np.array(mask[0,:,:])

        img_pre2 = np.transpose(img_pre[1,:,:,:],(1,2,0))
        img_post2 = np.transpose(img_post[1,:,:,:],(1,2,0))
        mask2 = np.array(mask[1,:,:])

        img_pre3 = np.transpose(img_pre[2,:,:,:],(1,2,0))
        img_post3 = np.transpose(img_post[2,:,:,:],(1,2,0))
        mask3 = np.array(mask[2,:,:])

        fig , ax =  plt.subplots(3, 3, figsize=(18, 18))
        ax[0][0].imshow(img_pre1)
        ax[0][1].imshow(img_post1)
        ax[0][2].imshow(mask1)

        ax[1][0].imshow(img_pre2)
        ax[1][1].imshow(img_post2)
        ax[1][2].imshow(mask2)

        ax[2][0].imshow(img_pre3)
        ax[2][1].imshow(img_post3)
        ax[2][2].imshow(mask3)
        plt.show()
        break

def unetsiamese_trainer_engine(loop):
    global max_acc
    for batch_idx,(img_pre, img_post, targets) in loop:
        loop.set_description(f"Epoch {epoch+1}/{num_epochs} ")
        img_pre = img_pre.to(DEVICE)
        img_post = img_post.to(DEVICE)
        targets = targets.to(DEVICE)
        targets = targets.type(torch.long)

        
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(img_pre, img_post)
            loss = loss_fn(predictions, targets)

        display_acc = check_accuracy_siamese((img_pre, img_post,targets), model)[0]
        acc.append(display_acc.item())
        
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item(), accuracy=display_acc.item())

    # update wandb
    wandb.log({"loss_siamese": loss})
    wandb.log({"epochs_siamese": epoch+1})
    wandb.log({"accuracy_siamese": display_acc.item()})

    # print(np.mean(acc))
    if np.mean(acc)>max_acc:
        max_acc = np.mean(acc)
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, f'./best_model_siamese.pt')
        print(f'Model saved at {epoch+1}!')
    if epoch+1 == num_epochs:
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, f'./latest_model_siamese.pt')

def unetsiamese_acc(train_loader, test_loader, model):
    print("Calculating Accuracy on Training Dataset...")    
    acc_train, dice_score = check_accuracy_siamese(train_loader, model)
    print(f'Train Accuracy: {acc_train.item():.2f}, Dice Score: {dice_score.item()}')

    print("Calculating Accuracy on Test Dataset...")    
    acc_test, dice_score = check_accuracy_siamese(test_loader, model)
    print(f'Test Accuracy: {acc_test.item():.2f}, Dice Score: {dice_score.item()}')

def unetsiamese_test_engine(img_pre, img_post, y, out_dir, model):
    try:
        img_pre = img_pre.to(DEVICE)
        img_post = img_post.to(DEVICE)
        y = y.to(DEVICE)

        fig , ax =  plt.subplots(3, 4, figsize=(18, 18))
        softmax = nn.Softmax(dim=1)
        preds = torch.argmax(softmax(model(img_pre, img_post)),axis=1).to('cpu')

        img_pre1 = np.transpose(np.array(img_pre[0,:,:,:].to('cpu')),(1,2,0))
        img_post1 = np.transpose(np.array(img_post[0,:,:,:].to('cpu')),(1,2,0))
        preds1 = np.array(preds[0,:,:])
        mask1 = np.array(y[0,:,:].to('cpu'))
        img_pre2 = np.transpose(np.array(img_pre[1,:,:,:].to('cpu')),(1,2,0))
        img_post2 = np.transpose(np.array(img_post[1,:,:,:].to('cpu')),(1,2,0))
        preds2 = np.array(preds[1,:,:])
        mask2 = np.array(y[1,:,:].to('cpu'))
        img_pre3 = np.transpose(np.array(img_pre[2,:,:,:].to('cpu')),(1,2,0))
        img_post3 = np.transpose(np.array(img_post[2,:,:,:].to('cpu')),(1,2,0))
        preds3 = np.array(preds[2,:,:])
        mask3 = np.array(y[2,:,:].to('cpu'))

        ax[0,0].set_title('Pre Image')
        ax[0,1].set_title('Post Image')
        ax[0,2].set_title('Prediction')
        ax[0,3].set_title('Mask')
        ax[1,0].set_title('Pre Image')
        ax[1,1].set_title('Post Image')
        ax[1,2].set_title('Prediction')
        ax[1,3].set_title('Mask')
        ax[2,0].set_title('Pre Image')
        ax[2,1].set_title('Post Image')
        ax[2,2].set_title('Prediction')
        ax[2,3].set_title('Mask')
        ax[0][0].axis("off")
        ax[0][1].axis("off")
        ax[0][2].axis("off")
        ax[0][3].axis("off")
        ax[1][0].axis("off")
        ax[1][1].axis("off")
        ax[1][2].axis("off")
        ax[1][3].axis("off")
        ax[2][0].axis("off")
        ax[2][1].axis("off")
        ax[2][2].axis("off")
        ax[2][3].axis("off")
        ax[0][0].imshow(img_pre1)
        ax[0][1].imshow(img_post1)
        ax[0][2].imshow(preds1)
        ax[0][3].imshow(mask1)
        ax[1][0].imshow(img_pre2)
        ax[1][1].imshow(img_post2)
        ax[1][2].imshow(preds2)
        ax[1][3].imshow(mask2)
        ax[2][0].imshow(img_pre3)
        ax[2][1].imshow(img_post3)
        ax[2][2].imshow(preds3)
        ax[2][3].imshow(mask3)
        fig.savefig(f"{out_dir}/{batch_idx}.png")
    except:
        print(f'Skipped Last Image.')
        pass

def unetcustom_trainer_engine(loop):
    global max_acc
    for _,(data, targets) in loop:
        loop.set_description(f"Epoch {epoch+1}/{num_epochs} ")
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)
        targets = targets.type(torch.long)
        
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        display_acc = check_accuracy((data,targets), model)[0]
        acc.append(display_acc.item())
        
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item(), accuracy=display_acc.item())

    # update wandb
    wandb.log({"loss": loss})
    wandb.log({"epochs": epoch+1})
    wandb.log({"accuracy": display_acc.item()})

    # print(np.mean(acc))
    if np.mean(acc)>max_acc:
        max_acc = np.mean(acc)
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, f'./best_model_custom.pt')
        print(f'Model saved at {epoch+1}!')

    if epoch+1 == num_epochs:
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, f'./latest_model_custom.pt')

def unetcustom_acc(train_loader, test_loader, model):
    print("Calculating Accuracy on Training Dataset...")    
    acc_train, dice_score = check_accuracy_custom(train_loader, model)
    print(f'Train Accuracy: {acc_train.item():.2f}, Dice Score: {dice_score.item()}')

    print("Calculating Accuracy on Test Dataset...")    
    acc_test, dice_score = check_accuracy_custom(test_loader, model)
    print(f'Test Accuracy: {acc_test.item():.2f}, Dice Score: {dice_score.item()}')

def unetmeta_trainer_engine(loop):
    global max_acc
    for batch_idx,(img_pre, img_meta, targets) in loop:
        loop.set_description(f"Epoch {epoch+1}/{num_epochs} ")
        img_pre = img_pre.to(DEVICE)
        img_meta = img_meta.float().to(DEVICE)
        targets = targets.to(DEVICE)
        targets = targets.type(torch.long)

        
        # forward
        
        predictions = model(img_pre, img_meta)
        loss = loss_fn(predictions, targets)

        display_acc = check_accuracy_meta((img_pre, img_meta, targets), model)[0]
        acc.append(display_acc.item())
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       

        # update tqdm loop
        loop.set_postfix(loss=loss.item(), accuracy=display_acc.item())

    # update wandb
    wandb.log({"loss_meta": loss})
    wandb.log({"epochs_meta": epoch+1})
    wandb.log({"accuracy_meta": display_acc.item()})

    # print(np.mean(acc))
    if np.mean(acc)>max_acc:
        max_acc = np.mean(acc)
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, f'./best_model_meta.pt')
        print(f'Model saved at {epoch+1}!')
    if epoch+1 == num_epochs:
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, f'./latest_model_meta.pt')

def unetmeta_acc(train_loader, test_loader, model):
    print("Calculating Accuracy on Training Dataset...")    
    acc_train, dice_score = check_accuracy_meta(train_loader, model)
    print(f'Train Accuracy: {acc_train.item():.2f}, Dice Score: {dice_score.item()}')

    print("Calculating Accuracy on Test Dataset...")    
    acc_test, dice_score = check_accuracy_meta(test_loader, model)
    print(f'Test Accuracy: {acc_test.item():.2f}, Dice Score: {dice_score.item()}')

def unetmeta_test_engine(img_pre, img_meta, y, out_dir, model, batch_idx):
    try:
        img_pre = img_pre.to(DEVICE)
        img_meta = img_meta.float().to(DEVICE)
        y = y.to(DEVICE)

        fig , ax =  plt.subplots(3, 4, figsize=(18, 18))
        softmax = nn.Softmax(dim=1)
        preds = torch.argmax(softmax(model(img_pre, img_meta)),axis=1).to('cpu')

        img_pre1 = np.transpose(np.array(img_pre[0,:,:,:].to('cpu')),(1,2,0))
        img_post1 = np.transpose(np.array(img_meta[0,:,:,:].to('cpu')),(1,2,0))
        preds1 = np.array(preds[0,:,:])
        mask1 = np.array(y[0,:,:].to('cpu'))
        
        ax[0][0].imshow(img_pre1)
        ax[0][1].imshow(img_post1)
        ax[0][2].imshow(preds1)
        ax[0][3].imshow(mask1)
    
        fig.savefig(f"{out_dir}/{batch_idx}.png")
    except:
        print(f'Skipped Last Image.')
        pass

class FocalLoss(nn.Module):
    '''
    Multi-class Focal loss implementation
    '''
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flood Semantic Segmentation')
    parser.add_argument('--model', type=str, default='UNet', help='models (UNet, UNetSiamese, UNetCustom, UNetMeta)')
    parser.add_argument('--mode', type=str, default='train', help='modes (train, test)')
    parser.add_argument('--vis', type=str, default='False', help='Visualize (True/False)')
    parser.add_argument('--images_pre_path', type=str, default='../dataset/raw/pre_img', help='path to pre-disaster images')
    parser.add_argument('--images_post_path', type=str, default='../dataset/raw/post_img', help='path to post-disaster images')
    parser.add_argument('--images_mask_path', type=str, default='../dataset/raw/post_msk_colored', help='path to masks')
    parser.add_argument('--output_dir', type=str, default='./predictions_custom_nonmeta', help='output dir path to save pred images')
    parser.add_argument('--batch_size', type=str, default='', help='batch size for training/inference')
    parser.add_argument('--num_epochs', type=str, default='', help='number of epochs for training')
    parser.add_argument('--lr', type=str, default='', help='learning rate for training')
    args = parser.parse_args()

    if args.batch_size == '':
        batch_size = config.BATCH_SIZE
    else:
        batch_size = int(args.batch_size)
    
    if args.num_epochs == '':
        num_epochs = config.NUM_EPOCHS
    else:
        num_epochs = int(args.num_epochs)
    
    if args.lr == '':
        learning_rate = config.LEARNING_RATE
    else:
        learning_rate = float(args.lr)
    
    out_dir = args.output_dir
    
    DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'

    if args.model == 'UNet':
        model = UNet().to(DEVICE)    
        image_paths = args.images_post_path
        seg_paths = args.images_mask_path
        train_transform = A.Compose([
            # A.HorizontalFlip(p=0.5),
            # A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
            A.RandomCrop(height=512, width=512, always_apply=True),
            # A.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
            # A.GaussNoise(p=0.2),
            # A.Perspective(p=0.5),
            # A.OneOf([A.CLAHE(p=1), A.RandomBrightnessContrast(p=1), A.RandomGamma(p=1)], p=0.9),
            # A.OneOf([A.Sharpen(p=1), A.Blur(blur_limit=3, p=1), A.MotionBlur(blur_limit=3, p=1)], p=0.9),
            # A.OneOf([A.RandomBrightnessContrast(p=1), A.HueSaturationValue(p=1)], p=0.9),
            A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
            ])

        data = HarveyDataset(image_paths, seg_paths, transform = train_transform)
        train_dataset, test_dataset = random_split(data, [int(0.8 * len(data)), len(data) - int(0.8 * len(data))])
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True)

        if args.vis == 'True':
            visualize_during_training(train_loader)
        
        print(summary(model, (3, 512, 512)))
    
    if args.model == 'UNetSiamese':
        model = UNetSiamese().to(DEVICE)    
        image_paths_pre = args.images_pre_path
        image_paths_post = args.images_post_path
        seg_paths = args.images_mask_path

        train_transform = A.Compose([
            A.RandomCrop(height=512, width=512, always_apply=True),
            A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
            ], additional_targets={'image0': 'image'})

        data = HarveyDatasetSiamese(image_paths_pre, image_paths_post, seg_paths, transform = train_transform)
        train_dataset, test_dataset = random_split(data, [int(0.8 * len(data)), len(data) - int(0.8 * len(data))])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        if args.vis == 'True':
            visualize_during_training_siamese(train_loader)
        
        print(summary(model, [(3, 512, 512),(3, 512, 512)]))

    if args.model == 'UNetCustom':
        model = UNet().to(DEVICE)    
        image_paths = args.images_pre_path
        seg_paths = args.images_mask_path
        train_transform = A.Compose([
            # A.HorizontalFlip(p=0.5),
            # A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
            A.Resize(512, 512, always_apply=True),
            # A.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
            # A.GaussNoise(p=0.2),
            # A.Perspective(p=0.5),
            # A.OneOf([A.CLAHE(p=1), A.RandomBrightnessContrast(p=1), A.RandomGamma(p=1)], p=0.9),
            # A.OneOf([A.Sharpen(p=1), A.Blur(blur_limit=3, p=1), A.MotionBlur(blur_limit=3, p=1)], p=0.9),
            # A.OneOf([A.RandomBrightnessContrast(p=1), A.HueSaturationValue(p=1)], p=0.9),
            A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
            ])

        data = HarveyDataset(image_paths, seg_paths, transform = train_transform)
        train_dataset, test_dataset = random_split(data, [int(0.8 * len(data)), len(data) - int(0.8 * len(data))])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        if args.vis == 'True':
            visualize_during_training(train_loader)
        
        print(summary(model, (3, 512, 512)))   

    if args.model == 'UNetMeta':
        model = UNetMeta().to(DEVICE)

        image_paths = args.images_pre_path
        elevation_path = r'..\dataset\custom_nonmeta\masks_elevation_meta'
        seg_paths = args.images_mask_path

        train_transform = A.Compose([
            # A.HorizontalFlip(p=0.5),
            # A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
            A.Resize(512, 512, always_apply=True),
            # A.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
            # A.GaussNoise(p=0.2),
            # A.Perspective(p=0.5),
            # A.OneOf([A.CLAHE(p=1), A.RandomBrightnessContrast(p=1), A.RandomGamma(p=1)], p=0.9),
            # A.OneOf([A.Sharpen(p=1), A.Blur(blur_limit=3, p=1), A.MotionBlur(blur_limit=3, p=1)], p=0.9),
            # A.OneOf([A.RandomBrightnessContrast(p=1), A.HueSaturationValue(p=1)], p=0.9),
            A.augmentations.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
            ])

        data = HarveyDatasetMeta(image_paths, elevation_path, seg_paths, transform = train_transform)
        train_dataset, test_dataset = random_split(data, [int(0.8 * len(data)), len(data) - int(0.8 * len(data))])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        # if args.vis == 'True':
        #     visualize_during_training(train_loader)
        
        print(summary(model, [(3, 512, 512), (3, 512, 512)]))
        
    if args.model == 'UNetCustom' or args.model  == 'UNetMeta':
        # Weight Decay
        # n_samples/(n_classes/number of pixels)
        # weight -> background, low, medium, high
        loss_fn = FocalLoss(weight=torch.Tensor([8.020853664571524, 3197.085640419458, 3106.601780670556, 1925.1822158951368]).to(DEVICE))
    else:    
        loss_fn = config.LOSS_FN

    optimizer = Adam(model.parameters(), lr=learning_rate)
    scaler = config.scaler
    

    if args.mode == 'train' and args.model == 'UNet':
        print(f'Training UNet Model...')
        for epoch in range(num_epochs):
            loop = tqdm(enumerate(train_loader), unit="batch", total=len(train_loader))
            acc = []
            unet_trainer_engine(loop)

        unet_acc(train_loader, test_loader, model)

    if args.mode == 'test' and args.model == 'UNet':
        # Load the Model
        print(f'Loaded UNet Model...')
        optimizer = Adam(model.parameters(), lr=learning_rate)
        checkpoint = torch.load(f'./latest_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        model.eval()
            
        for batch_idx, (x,y) in enumerate(test_loader):
            unet_test_engine(x, y, out_dir, model)

    if args.mode == 'train' and args.model == 'UNetSiamese':
        print(f'Training UNetSiamese Model...')
        for epoch in range(num_epochs):
            loop = tqdm(enumerate(train_loader), unit="batch", total=len(train_loader))
            acc = []
            unetsiamese_trainer_engine(loop)
            

        unetsiamese_acc(train_loader, test_loader, model)
        print("Calculating Accuracy on Training Dataset...")    
        acc_train, dice_score = check_accuracy_siamese(train_loader, model)
        print(f'Train Accuracy: {acc_train.item():.2f}, Dice Score: {dice_score.item()}')

        print("Calculating Accuracy on Test Dataset...")    
        acc_test, dice_score = check_accuracy_siamese(test_loader, model)
        print(f'Test Accuracy: {acc_test.item():.2f}, Dice Score: {dice_score.item()}')

    if args.mode == 'test' and args.model == 'UNetSiamese':
        # Load the Model
        print(f'Loaded UNetSiamese Model...')
        optimizer = Adam(model.parameters(), lr=learning_rate)
        checkpoint = torch.load(f'./latest_model_siamese.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        model.eval()

        for batch_idx, (img_pre, img_post, y) in enumerate(test_loader):
            unetsiamese_test_engine(img_pre, img_post, y, out_dir, model)

    if args.mode == 'train' and args.model == 'UNetCustom':
        print(f'Training UNetCustom Model...')
        for epoch in range(num_epochs):
            loop = tqdm(enumerate(train_loader), unit="batch", total=len(train_loader))
            acc = []
            unetcustom_trainer_engine(loop)

        unetcustom_acc(train_loader, test_loader, model)
    
    if args.mode == 'test' and args.model == 'UNetCustom':
        # Load the Model
        print(f'Loaded UNetCustom Model...')
        optimizer = Adam(model.parameters(), lr=learning_rate)
        checkpoint = torch.load(f'./latest_model_custom.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        model.eval()
            
        for batch_idx, (x,y) in enumerate(test_loader):
            unet_test_engine(x, y, out_dir, model)
    
    if args.mode == 'train' and args.model == 'UNetMeta':
        print(f'Training UNetMeta Model...')
        for epoch in range(num_epochs):
            loop = tqdm(enumerate(train_loader), unit="batch", total=len(train_loader))
            acc = []
            unetmeta_trainer_engine(loop)

        unetmeta_acc(train_loader, test_loader, model)

    if args.mode == 'test' and args.model == 'UNetMeta':
        # Load the Model
        print(f'Loaded UNetMeta Model...')
        optimizer = Adam(model.parameters(), lr=learning_rate)
        checkpoint = torch.load(f'./latest_model_meta.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        model.eval()

        for batch_idx, (img_pre, img_meta, y) in enumerate(test_loader):
            unetmeta_test_engine(img_pre, img_meta, y, out_dir, model, batch_idx)
