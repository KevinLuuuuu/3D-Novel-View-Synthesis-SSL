import torch
import torch.optim
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
from dataset import OfficeImage
from p2_model import Net

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu" 
#print(device)
torch.cuda.empty_cache()

# set random seed
seed = 325
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

train_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 40
batch_size = 32

dataset_path_train = "hw4_data/office/train"
dataset_path_val = "hw4_data/office/val"

train_set = OfficeImage(dataset_path_train, transform=train_transform, train_set=True)
valid_set = OfficeImage(dataset_path_val, transform=test_transform, valid_set=True)


train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

best_acc = 0

# freeze para
#for param in model.encoder.parameters():
    #param.requires_grad = False

for epoch in range(epochs):

    model.train()
    train_loss = 0
    train_loss_record = []
    train_correct = 0

    for i, (image, label) in enumerate(tqdm(train_loader)):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(image)
        train_loss = criterion(output, label)
        train_loss_record.append(train_loss.item())             
        pred_label = torch.max(output.data, 1)[1]
        train_correct = train_correct + pred_label.eq(label.view_as(pred_label)).sum().item()
        train_loss.backward()
        optimizer.step()

    model.eval()
    eval_loss = 0
    eval_loss_record = []
    eval_correct = 0

    with torch.no_grad():
        for i, (image, label) in enumerate(tqdm(valid_loader)):
            image, label = image.to(device), label.to(device)
            output = model(image)
            eval_loss = criterion(output, label)
            eval_loss_record.append(eval_loss.item())
            pred_label = torch.max(output.data, 1)[1]
            eval_correct = eval_correct + pred_label.eq(label.view_as(pred_label)).sum().item()
    
    train_acc = 100 * train_correct / len(train_set)
    mean_train_loss = sum(train_loss_record)/len(train_loss_record)
    valid_acc = 100 * eval_correct / len(valid_set)
    mean_eval_loss = sum(eval_loss_record)/len(eval_loss_record)
    print("Epoch:", epoch)
    print('Train loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format( mean_train_loss, train_correct, len(train_set), train_acc))
    print('Evaluate loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(mean_eval_loss, eval_correct, len(valid_set), valid_acc))     
         
    if valid_acc > best_acc:
        best_acc = valid_acc
        print('Save model, accuracy: ', best_acc, "%")
        torch.save(model.state_dict(), "p2.pth")


    print('The best accuracy is {:.0f}% '.format(best_acc))