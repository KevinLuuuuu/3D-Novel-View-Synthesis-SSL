import torch
from byol_pytorch import BYOL
from torchvision import models
from dataset import ImageDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu" 

resnet = models.resnet50(pretrained=False)#.to(device)

learner = BYOL(
    resnet,
    image_size = 128,
    hidden_layer = 'avgpool'
).to(device)


opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

def sample_unlabelled_images():
    return torch.randn(20, 3, 256, 256)

train_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

batch_size = 128
dataset_path_train = "hw4_data/mini/train"
train_set = ImageDataset(dataset_path_train, transform=train_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

best_loss = 890414
mean_loss = 0
for epoch in range(1000):
    train_loss_record = []
    for i, image in enumerate(tqdm(train_loader)):
        #print(image.shape)
        images = image.to(device) #sample_unlabelled_images()
        loss = learner(images)
        train_loss_record.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average() # update moving average of target encoder
    mean_loss = sum(train_loss_record)/len(train_loss_record)
    print("Epoch:", epoch)
    print('Train loss: {:.4f}'.format(mean_loss))
    if best_loss > mean_loss:
        best_loss = mean_loss
        print('Save model')
        torch.save(resnet.state_dict(), './backbone/best_loss.pt')
    print()
# save your improved network
torch.save(resnet.state_dict(), './backbone/last.pt')