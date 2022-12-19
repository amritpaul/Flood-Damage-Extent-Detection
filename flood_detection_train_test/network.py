import torch
import torch.nn as nn
import torch.nn.functional as F

class encoding_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoding_block,self).__init__()
        model = []

        model.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))

        model.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        model.append(nn.BatchNorm2d(out_channels))
        model.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*model)
    #     self.apply(self._init_weights)
    
    # def _init_weights(self, module):
    #     if isinstance(module, nn.Conv2d):
    #         nn.init.xavier_normal_(module.weight)
    #         if module.bias is not None:
    #             module.bias.data.zero_()

    def forward(self, x):
        return self.conv(x)  

class encoding_block_meta(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoding_block_meta, self).__init__()
        meta = []
        meta.append(nn.Conv2d(in_channels, out_channels, 3,1,1,bias=True))
        meta.append(nn.BatchNorm2d(out_channels))
        meta.append(nn.ReLU(inplace=True))
        meta.append(nn.Conv2d(out_channels, out_channels, 3,1,1,bias=True))
        meta.append(nn.BatchNorm2d(out_channels))
        meta.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*meta)

    def forward(self, meta):
        return self.conv(meta)

class UNet(nn.Module):
    def __init__(self,in_channels=3, out_channels=4,features=[64, 128, 256, 512]):
        super(UNet,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv1 = encoding_block(in_channels,features[0])
        self.conv2 = encoding_block(features[0],features[1])
        self.conv3 = encoding_block(features[1],features[2])
        self.conv4 = encoding_block(features[2],features[3])
        self.conv5 = encoding_block(features[3]*2,features[3])
        self.conv6 = encoding_block(features[3],features[2])
        self.conv7 = encoding_block(features[2],features[1])
        self.conv8 = encoding_block(features[1],features[0])        
        self.tconv1 = nn.ConvTranspose2d(features[-1]*2, features[-1], kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose2d(features[-1], features[-2], kernel_size=2, stride=2)
        self.tconv3 = nn.ConvTranspose2d(features[-2], features[-3], kernel_size=2, stride=2)
        self.tconv4 = nn.ConvTranspose2d(features[-3], features[-4], kernel_size=2, stride=2)        
        self.bottleneck = encoding_block(features[3],features[3]*2)
        self.final_layer = nn.Conv2d(features[0],out_channels,kernel_size=1)
    #     self.apply(self._init_weights)
    
    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         module.weight.data.normal_(mean=0.0, std=1.0)
    #         if module.bias is not None:
    #             module.bias.data.zero_()

    #     elif isinstance(module, nn.Conv2d):
    #         nn.init.xavier_normal_(module.weight)
    #         if module.bias is not None:
    #             module.bias.data.zero_()
        
    #     elif isinstance(module, nn.ConvTranspose2d):
    #         nn.init.xavier_normal_(module.weight)
    #         if module.bias is not None:
    #             module.bias.data.zero_()


    def forward(self,x):
        skip_connections = []
        x = self.conv1(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv2(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv3(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv4(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        x = self.tconv1(x)
        x = torch.cat((skip_connections[0], x), dim=1)
        x = self.conv5(x)
        x = self.tconv2(x)
        x = torch.cat((skip_connections[1], x), dim=1)
        x = self.conv6(x)
        x = self.tconv3(x)
        x = torch.cat((skip_connections[2], x), dim=1)
        x = self.conv7(x)        
        x = self.tconv4(x)
        x = torch.cat((skip_connections[3], x), dim=1)
        x = self.conv8(x)
        x = self.final_layer(x)
        return x

class UNetMeta(nn.Module):
    def __init__(self,in_channels=3, out_channels=4,features=[64, 128, 256, 512]):
        super(UNetMeta,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv1 = encoding_block(in_channels,features[0])
        self.meta1 = encoding_block_meta(in_channels, features[0])
        self.conv2 = encoding_block(features[0],features[1])
        self.meta2 = encoding_block_meta(features[0], features[1])
        self.adjust1 = encoding_block_meta(features[2], features[1])

        self.conv3 = encoding_block(features[1],features[2])
        self.meta3 = encoding_block_meta(features[1],features[2])
        self.conv4 = encoding_block(features[2],features[3])
        self.meta4 = encoding_block_meta(features[2],features[3])
        self.adjust2 = encoding_block_meta(768, features[2])

        self.conv5 = encoding_block(features[3]*2,features[3])
        self.conv6 = encoding_block(features[3],features[2])
        self.conv7 = encoding_block(features[2],features[1])
        self.conv8 = encoding_block(features[1],features[0])        
        self.tconv1 = nn.ConvTranspose2d(features[-1]*2, features[-1], kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose2d(features[-1], features[-2], kernel_size=2, stride=2)
        self.tconv3 = nn.ConvTranspose2d(features[-2], features[-3], kernel_size=2, stride=2)
        self.tconv4 = nn.ConvTranspose2d(features[-3], features[-4], kernel_size=2, stride=2)        
        self.bottleneck = encoding_block(features[3],features[3]*2)
        self.final_layer = nn.Conv2d(features[0],out_channels,kernel_size=1)

    def forward(self, x, meta):
        
        skip_connections = []
        x = self.conv1(x)
        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv2(x)

        # Meta Start
        x_prime = self.meta1(meta)
        x_prime = self.pool(x_prime)
        x_prime = self.meta2(x_prime)
        x = torch.cat((x, x_prime), dim=1)
        x = self.adjust1(x)
        # Meta End

        skip_connections.append(x)
        #print(x.shape) #torch.Size([1, 128, 256, 256])
        x = self.pool(x)
        x = self.conv3(x)

        # Meta Start
        x_prime = self.meta3(x_prime)
        x_prime = self.pool(x_prime)
        x_prime = self.meta4(x_prime)
        x = torch.cat((x, x_prime), dim=1)
        x = self.adjust2(x)
        # Meta End

        skip_connections.append(x)
        x = self.pool(x)
        x = self.conv4(x)
        skip_connections.append(x)
       
        x = self.pool(x)
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1]
        x = self.tconv1(x)
        x = torch.cat((skip_connections[0], x), dim=1)
        x = self.conv5(x)
        x = self.tconv2(x)
        x = torch.cat((skip_connections[1], x), dim=1)
        x = self.conv6(x)
        x = self.tconv3(x)
        x = torch.cat((skip_connections[2], x), dim=1)
        x = self.conv7(x)        
        x = self.tconv4(x)
        x = torch.cat((skip_connections[3], x), dim=1)
        x = self.conv8(x)
        x = self.final_layer(x)
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNetSiamese(nn.Module):
    def __init__(self, in_channels=3, n_classes=4, bilinear=True):
        super(UNetSiamese, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.penultimate_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.outc1 = OutConv(64, self.n_classes)
        

    def forward_once(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x

    def forward(self, x1, x2=None):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        x = torch.cat([out1, out2], dim=1)
        x = self.penultimate_conv(x)
        x = self.outc1(x)
        return x
       