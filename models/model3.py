import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import VGG16_Weights
from torchvision.utils import save_image


class Model3:
    def __init__(self):
        self.name = "vgg-19"
        self.estimated_time_min = 4

    def load(self):
        pass

    def run_and_save(self, content_image_path: str, style_image_path: str, result_image_path: str):
        run(content_image_path, style_image_path, result_image_path)


def run(content_path: str, style_path: str, output_path: str, num_steps=100, save_each=20, lr=0.1) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    style, content = Image.open(style_path), Image.open(content_path)
    size0, size1 = content.size
    if size0 * size1 > 200000:
        size0 = int(size0 / 2)
        size1 = int(size1 / 2)

    transform = transforms.Compose([
        transforms.Resize((size1, size0)),
        transforms.ToTensor()]
    )

    style = transform(style).unsqueeze(0).to(device, torch.float32)
    content = transform(content).unsqueeze(0).to(device, torch.float32)

    def Gram(x):
        b, c, h, w = x.shape
        f = x.view(b * c, h * w)
        G = torch.mm(f, f.t())
        return G.div(b * c * h * w)

    LOSS = F.mse_loss

    class Style_Loss(nn.Module):

        def __init__(self, f):
            super().__init__()
            self.target = Gram(f).detach()

        def forward(self, x):
            self.loss = LOSS(Gram(x), self.target)
            return x

    class Content_Loss(nn.Module):
        def __init__(self, x):
            super().__init__()
            self.target = x.detach()

        def forward(self, x):
            self.loss = LOSS(x, self.target)
            return x

    net = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()

    class First_Module(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, img):
            return img

    content_layers, style_layers = ['conv_4'], ['conv_1', 'conv_4']

    def get_style_model(cnn, style, content):
        f_module = First_Module().to(device)

        content_losses = []
        style_losses = []

        net = nn.Sequential(f_module)

        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'

                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'

            net.add_module(name, layer)

            if name in content_layers:
                target = net(content).detach()
                content_loss = Content_Loss(target)
                net.add_module(f"content_loss_{i}", content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = net(style).detach()
                style_loss = Style_Loss(target_feature)
                net.add_module(f"style_loss_{i}", style_loss)
                style_losses.append(style_loss)

        for i in range(len(net) - 1, -1, -1):
            if isinstance(net[i], Content_Loss) or isinstance(net[i], Style_Loss):
                break
        return net[:(i + 1)], style_losses, content_losses

    input = content.clone()

    def style_transfer(cnn, content, style, input_img, num_steps, style_weight=1000000, content_weight=1):

        net, style_losses, content_losses = get_style_model(cnn, style, content)

        input_img.requires_grad_(True)
        net.requires_grad_(False)

        optimizer = optim.LBFGS([input_img])

        i = [0]
        while i[0] <= num_steps:

            def closure():
                with torch.no_grad():
                    input_img.clamp_(0, 1)

                optimizer.zero_grad()
                _ = net(input_img)
                style_score, content_score = 0, 0
                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                i[0] += 1
                return style_score + content_score

            optimizer.step(closure)

        with torch.no_grad():
            input_img.clamp_(0, 1)

        return input_img

    output = style_transfer(net, content, style, input, num_steps)

    save_image(output, output_path)


if __name__ == "__main__":
    run(*sys.argv[1:])
