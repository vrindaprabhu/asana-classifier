from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy
import glob
from PIL import Image
import pickle
import seaborn as sns
import pandas as pd
from tqdm import tqdm

from cf_matrix import make_confusion_matrix
import pickle

class YogaInference():
    def __init__(self, model_paths, class_names):
        self.class_names = class_names
            
        self.model_ft, self.model_conv = self.load_models(*model_paths)
        
        self.data_transforms =  transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
    def load_models(self, finetune_path, conv_feature_path):
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        # Here the size of each output is generalized
        model_ft.fc = nn.Linear(num_ftrs, len(self.class_names))
        model_ft = model_ft.to("cpu")
        model_ft.load_state_dict(torch.load(finetune_path,map_location=torch.device('cpu')))
        _ = model_ft.eval()
        
        
        model_conv = torchvision.models.resnet18(pretrained=True)
        for param in model_conv.parameters():
            param.requires_grad = False

        # Parameters of newly constructed modules have requires_grad=True by default
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, len(self.class_names))

        model_conv.load_state_dict(torch.load(conv_feature_path,map_location=torch.device('cpu')))
        _ = model_conv.eval()

        return model_ft, model_conv
    
    
    @torch.no_grad()
    def validate_batch(self, test_path,model="fine_tune"):
        test_dataset = datasets.ImageFolder( test_path, self.data_transforms)
        test_dl = DataLoader(test_dataset, 4, num_workers=3, pin_memory=True)
        pred_batch_probs = []
        actual_batch_vals = []
        for xb, label in tqdm(test_dl):
            if model == "fine_tune":
                preds = self.model_ft(xb)
                
            if model == "conv_feature":
                preds = self.model_conv(xb)
                
            pred_batch_probs.append(preds)
            actual_batch_vals.append(label)
        
        pred_batch_probs = torch.cat(pred_batch_probs)
        #The below lines are to macth the validation output index with the actuals
        actual_batch_vals = torch.cat(actual_batch_vals)
        batch_classnames = [test_dl.dataset.classes[i] for i in actual_batch_vals]
        actual_batch_vals = [self.class_names.index(i) for i in batch_classnames]
        
        return torch.tensor(actual_batch_vals), pred_batch_probs
        
        
    @torch.no_grad()
    def get_output(self, image_names, model):
        output = None
        for idx, image_name in enumerate(image_names):
            image = Image.open(image_name)
            image = self.data_transforms(image).float()
            image = image.clone().detach()
            image = image.unsqueeze(0)
            if model == "fine_tune":
                preds = self.model_ft(image)
            
            if model == "conv_feature":
                preds = self.model_conv(image)

            if idx == 0:
                output = preds.numpy()
            else:
                output = np.vstack([output ,preds.numpy()])
        return output
    
    
    def classify_asana(self, images, model="fine_tune", batch=False, raw_out=False):
        if batch:
            output = self.get_output(images, model)
        else:
            output = self.get_output([images], model)
            
        asanas = []
        idx = np.argmax(output,axis=1)
        asanas = [self.class_names[i] for i in idx]
        
        if raw_out:
            return asanas, output
        return asanas, None


def get_top_3(file_name):
    _, predictions = yoga_inf.classify_asana(file_name, model="fine_tune",batch=False, raw_out=True)
    page = torch.nn.functional.softmax(torch.Tensor(predictions),dim=-1)
    top_prob, top_indices = torch.topk(page,3,axis=1)

    top_3_preds = []
    for i in range(3):
        top_3_preds.append([class_names[top_indices[0][i].item()],top_prob[0][i].item()])

    return top_3_preds


def compute_confusion_matrix(class_names, true, pred):
    '''Computes a confusion matrix using numpy for two np.arrays
    true and pred.

    Results are identical (and similar in computation time) to: 
    "from sklearn.metrics import confusion_matrix"

    However, this function avoids the dependency on sklearn.'''

    K = len(np.unique(class_names)) # Number of classes 
    result = np.zeros((K, K))
    
    for i in range(len(true)):
        try:
            result[true[i]][pred[i]] += 1
        except IndexError:
            print ("The predicted value is outside the true value in the batch")
            continue

    return result


def analyse_preds(actuals, pred, error_only=True):    
    page = torch.nn.functional.softmax(pred, dim=-1)
    top_prob, top_indices = torch.topk(page,3,axis=1)   
    
    if error_only:
        non_matched = torch.where(act!=top_indices[:,0])
    else:
        non_matched = list(range(len(page)))
        
    df = pd.DataFrame()
    df['actuals'] = [class_names[i.item()] for i in act[non_matched]]
    df['predicted_1'] = [class_names[i.item()] for i in top_indices[non_matched][:,0]]
    df['probs_1'] = top_prob[non_matched][:,0]
    
    df['predicted_2'] = [class_names[i.item()] for i in top_indices[non_matched][:,1]]
    df['probs_2'] = top_prob[non_matched][:,1]
    
    return df


def analyse_training(pickle_path = "fine_tune.pkl"):
    with open(pickle_path, "rb") as op:
        x = pickle.load(op)
        
    df = pd.DataFrame(x)
    df.columns = ['phase', 'loss', 'accuracy']
    
    return df  


def plot_metadata(df, info="accuracy", trans_method="fine_tune"):
    val_scores = df[df["phase"] == "val"][info]
    train_scores = df[df["phase"] == "train"][info]
    plt.plot(val_scores,'r', label='validation')
    plt.plot(train_scores,'b', label='training')
    plt.xlabel('epoch')
    plt.ylabel(info)
    plt.legend()
    plt.title(info + ' vs. no. of epochs')
    plt.savefig(os.path.join(out_dir,trans_method+"_"+ info+".png"))
    plt.close()


## Required Paths
model_paths = ["../models/finetuned_model.pth", "../models/convfeat_model.pth"]
classnames_path = "../models/class_names.txt"

single_images = "inference_images/unseen_images/"
batch_validation = "inference_images/batch_validation"

out_dir = "output"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

images = glob.glob(os.path.join(single_images,"*.jpeg"))

with open(classnames_path,"r") as op:
    class_names = [i.strip() for i in op.readlines()]


## Training plots
df = analyse_training(pickle_path="../models/fine_tune.pkl")
plot_metadata(df, info="accuracy", trans_method="fine_tune")
plot_metadata(df, info="loss", trans_method="fine_tune") 

df = analyse_training(pickle_path="../models/conv_feature.pkl")
plot_metadata(df, info="accuracy", trans_method="conv_feat")
plot_metadata(df, info="loss", trans_method="conv_feat") 

print ("*"*50)

## Inference
yoga_inf = YogaInference(model_paths, class_names)

### Unknown class verification
#Getting predictions for all images of a folder
print ("Predicting for all images in a input folder")
out, _ = yoga_inf.classify_asana(images, model="fine_tune",batch=True)
for i in out:
    print (i)

#Getting top-3 predictions for a single image 
out = get_top_3(images[0])
for i in out:
    print (i[0], ":", i[1])

print ("*"*50)

### Know class validation
print ("Validating on a batch of images (similar to model validation)")
act, pred = yoga_inf.validate_batch(batch_validation, model="fine_tune")

#Getting the confusion matrix
print ("Saving confusion matrix...")
cm = compute_confusion_matrix(class_names, act.numpy(), torch.argmax(pred, axis=1).numpy())
conf_mat = make_confusion_matrix(cm, categories=class_names,figsize=(10,10))
conf_mat.savefig(os.path.join(out_dir, "confuse_mat.png"), bbox_inches="tight")
plt.close()

#Getting analysis report
print ("Saving error report...")
df = analyse_preds(act, pred, error_only=False)
df.to_csv(os.path.join(out_dir, "analysis.csv"), index=False)

print ("*"*50)

