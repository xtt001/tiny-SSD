import matplotlib.pyplot as plt
import torch
from model import TinySSD
from visualization import display
from prediction import predict
import torchvision
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

net = TinySSD(num_classes=1)
net = net.to(device)

# 加载模型参数
net.load_state_dict(torch.load('models/net_30.pkl', map_location=torch.device(device)))

files = glob.glob('detection/test/2.jpg')
for name in files:
    X = torchvision.io.read_image(name).unsqueeze(0).float()
    # X = torch.from_numpy(cv2.imread(name)).permute(2, 0, 1).unsqueeze(0).float()
    img = X.squeeze(0).permute(1, 2, 0).long()

    output = predict(X,net)
    display(img, output.cpu(), threshold=0.6)



