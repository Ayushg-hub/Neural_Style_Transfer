import torch
from torchvision import models
from PIL import Image
from torchvision import transforms as T
import numpy as np
import matplotlib.pyplot as plt
from torch import optim


def preprocess_img(path,max_size = 500):
    image = Image.open(path).convert('RGB')
    size = min(max(image.size),max_size)

    img_tranforms = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225])
    ])

    image = img_tranforms(image)
    image = image.unsqueeze(0)
    return image

def deprocess_img(tensor):
    image = tensor.to('cpu').clone()
    image = image.numpy()
    image = image.squeeze(0)
    image = image.transpose(1,2,0)
    image = image*np.array([0.229,0.224,0.225]) + np.array([0.229,0.224,0.225])
    image = image.clip(0,1)

    return image

def get_features(image,model):
    layers = {
        '0' : 'conv1_1',
        '5' : 'conv2_1',
        '10' : 'conv3_1',
        '19' : 'conv4_1',
        '21' : 'conv4_2',
        '28' : 'conv5_1'
    }
    x = image

    features = {}

    for name,layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x 

    return features

def gram_matrix(tensor):
    d,c,h,w = tensor.size()
    tensor =tensor.view(c,h*w)
    gram = torch.mm(tensor,tensor.t())
    return gram

def content_loss(target_conv4_2,content_conv4_2):
    loss = torch.mean((target_conv4_2-content_conv4_2)**2)
    return loss

def style_loss(style_weights,target_features,style_grams):
    loss =0
    for layer in style_weights:
        target_f = target_features[layer]
        target_gram = gram_matrix(target_f)
        style_gram = style_grams[layer]
        b,c,h,w = target_f.shape
        layer_loss = style_weights[layer]*torch.mean((target_gram - style_gram)**2)
        loss += layer_loss/(c*h*w)

    return loss

def total_loss(c_loss,s_loss,alpha,beta):
    loss = alpha*c_loss + beta*s_loss
    return loss




our_model = models.vgg19(pretrained = True)
our_model = our_model.features #removing the classifier

#freeze the layers i.e no need to change their weights and biases during training
for para in our_model.parameters():
    para.requires_grad_(False)

content_p = preprocess_img("Project-NST/content11.jpg")
style_p = preprocess_img("Project-NST/style12.jpg")
#use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if(device != 'cpu'):
    our_model.to(device)
    content_p.to(device)
    style_p.to(device)


print("Content Shape",content_p.shape)
print("style shape",style_p.shape)

content_f = get_features(content_p,our_model)
style_f = get_features(style_p,our_model)

style_weights = {
    'conv1_1' : 1.0,
    'conv2_1' : 0.75,
    'conv3_1' : 0.2,
    'conv4_1' : 0.2,
    'conv5_1' : 0.2
}

target = content_p.clone().requires_grad_(True)
target_f = get_features(target,our_model)

style_gram = {layer : gram_matrix(style_f[layer]) for layer in style_f}

optimizer = optim.Adam([target],lr = 0.003)
alpha = 1
beta = 165
epochs = 3000
show_every = 500

results = []

for i in range(epochs):
    target_f = get_features(target,our_model)

    c_loss = content_loss(target_f['conv4_2'],content_f['conv4_2'])
    s_loss = style_loss(style_weights,target_f,style_gram)
    t_loss = total_loss(c_loss,s_loss,alpha,beta)

    optimizer.zero_grad()
    t_loss.backward()
    optimizer.step()

    if i%show_every == 0:
        print("total loass at epoch {} : {}".format(i,t_loss))
        results.append(deprocess_img(target.detach()))

plt.figure(figsize = (10,8))
for i  in range(len(results)):
    plt.subplot(3,2,i+1)
    plt.imshow(results[i])
plt.show()

content_d = deprocess_img(content_p)
style_d = deprocess_img(style_p)

print("original content shape",content_d.shape)
print("original style shape",style_d.shape)

#fig,(ax1,ax2) = plt.subplots(1,2,figsize=(20,10))
#ax1.imshow(content_d)
#ax2.imshow(style_d)
#plt.show()
#print(our_model)