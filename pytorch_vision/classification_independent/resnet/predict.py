import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import sys
sys.path.append('../../..')
from resnet import resnet34
from utils.constants import model_root, dataset_root


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 1、加载并处理图片
    img_path = os.path.join(dataset_root, "flower_data/flower_photos/tulips/4263272885_1a49ea5209.jpg")
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # 2、加载模型
    model = resnet34(num_classes=5).to(device)
    weights_path = os.path.join(model_root, "resNet34.pth")
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # 3、预测
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    # 将类别转成名称
    json_path = 'class_indices.json'
    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()
