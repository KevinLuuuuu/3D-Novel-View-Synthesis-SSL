import torch
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torchvision.models as models
from dataset import OfficeImage
from p2_model import Net
import csv
from argparse import ArgumentParser, Namespace
from pathlib import Path
import json

def main(args):

    dataset_path = args.input_dir #"./hw1_data/p1_data/val_50"
    csv_path = args.csv_dir #"./hw1_data/p1_data/val_50"
    output_path = args.output_dir #"./p1_output.csv"
    ckpt_path = "./p2.pth"
    
    model = Net()
    model.load_state_dict(torch.load(ckpt_path))
    model = model.to(device)
    model.eval()

    test_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])

    batch_size = 10
    test_set = OfficeImage(dataset_path, transform=test_transform, train_set=False, valid_set=False, test_csv_path=csv_path)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    pred_label_list = []
    image_name_list = []
    id_list = []


    with torch.no_grad():
        for i, (image, image_name, id) in enumerate(tqdm(test_loader)):
            image = image.to(device)
            output = model(image)
            pred_label = torch.max(output.data, 1)[1]
            
            pred_label_list.append(pred_label)
            image_name_list.append(image_name)
            id_list.append(id)

    ################# check ####################
    with open('label.json') as f:
        label = json.load(f)
    key_list = list(label.keys())
    val_list = list(label.values())

    with open(output_path, 'w', newline="") as fp:        
        file_writer = csv.writer(fp)
        file_writer.writerow(['id', 'filename', 'label'])
        for i in range(len(pred_label_list)):
            for j in range(len(pred_label_list[i])):
                position = val_list.index(pred_label_list[i][j].item())
                label_name = key_list[position]
                file_writer.writerow([id_list[i][j], image_name_list[i][j], label_name])

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=Path,
        help="Path to the input file.",
        required=True
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to the output file.",
        required=True
    )
    parser.add_argument(
        "--csv_dir",
        type=Path,
        help="Path to the csv file.",
        required=True
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu" 
    #print(device)
    torch.cuda.empty_cache()

    args = parse_args()
    main(args)
                 