import torch.nn as nn
import torch.nn.functional as fun

##################
## Main Methods ##
##################

class VGG_Feature_Extractor(nn.Module):
    """
    To extract features of CRNN (Convolutional Recurrent Neural Network):
        convolutional and max-pooling layers from a standard CNN model. 
    """

    def __init__(self, input, output=512):
        super(VGG_Feature_Extractor, self).__init__()
        
        # Output Channel: [64, 128, 256, 512]
        self.output = [int(output / 8), int(output / 4),
                               int(output / 2), output] 
        
        # Main Channel - Convolutional Network
        self.Conv = nn.Sequential(
            nn.Conv2d(input, self.output[0], 3, 1, 1), nn.ReLU(True),
            # 64x16x50
            nn.MaxPool2d(2, 2),  
            nn.Conv2d(self.output[0], self.output[1], 3, 1, 1), nn.ReLU(True),
            # 128x8x25
            nn.MaxPool2d(2, 2),
            # 256x8x25
            nn.Conv2d(self.output[1], self.output[2], 3, 1, 1), nn.ReLU(True),  
            nn.Conv2d(self.output[2], self.output[2], 3, 1, 1), nn.ReLU(True),
            # 256x4x25
            nn.MaxPool2d((2, 1), (2, 1)),  
            nn.Conv2d(self.output[2], self.output[3], 3, 1, 1, bias=False),
            # 512x4x25
            nn.BatchNorm2d(self.output[3]), nn.ReLU(True), 
            nn.Conv2d(self.output[3], self.output[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output[3]), nn.ReLU(True),
            # 512x2x25
            nn.MaxPool2d((2, 1), (2, 1)),  
            # 512x1x24
            nn.Conv2d(self.output[3], self.output[3], 2, 1, 0), nn.ReLU(True)
        ) 

    def forward(self, input):
        return self.Conv(input)


class RCNN_Feature_Extractor(nn.Module):
    """
    To extract features of GRCNN (Gated Recurrent Convolution Neural Network):
        extra usage of Gated Recurrent Convolution Layer (GRCL)
    """
    def __init__(self, input, output=512):
        super(RCNN_Feature_Extractor, self).__init__()
        # Output Channel - [64, 128, 256, 512]
        self.output = [int(output / 8), int(output / 4), int(output / 2), output]
        # Main Channel - Convolutional Network
        self.Conv = nn.Sequential(
            nn.Conv2d(input, self.output[0], 3, 1, 1), nn.ReLU(True),
            # 64 x 16 x 50
            nn.MaxPool2d(2, 2),
            GRCL(self.output[0], self.output[0], num_iteration=5, kernel_size=3, pad=1),
            # 64 x 8 x 25
            nn.MaxPool2d(2, 2),
            GRCL(self.output[0], self.output[1], num_iteration=5, kernel_size=3, pad=1),
            # 128 x 4 x 26
            nn.MaxPool2d(2, (2, 1), (0, 1)),
            GRCL(self.output[1], self.output[2], num_iteration=5, kernel_size=3, pad=1),
            # 256 x 2 x 27
            nn.MaxPool2d(2, (2, 1), (0, 1)),
            nn.Conv2d(self.output[2], self.output[3], 2, 1, 0, bias=False),
            # 512 x 1 x 26
            nn.BatchNorm2d(self.output[3]), nn.ReLU(True)
        )

    def forward(self, input):
        return self.Conv(input)


class ResNet_Feature_Extractor(nn.Module):
    """
    To extract features of FAN (Focusing Attention Network):
        consisted of AN and FN:
            1. AN: alignment factors between target labels and features are generated. Each alignment factor corresponds to an attention region in the input image. Bad alignments (i.e. deviated or unfocused attention regions) result in poor recognition results.
            2. FN: first locates the attention region for each target label, then conducts dense prediction over the attention region with the corresponding glimpse vector.
    """
    def __init__(self, input, output=512):
        super(ResNet_Feature_Extractor, self).__init__()
        self.Conv = ResNet(input, output, BasicBlock, [1, 2, 5, 3])

    def forward(self, input):
        return self.Conv(input)


#################
##    tools    ##
#################

class GRCL(nn.Module):
    # Used for Gated RCNN_Feature_Extractor
    def __init__(self, input, output, num_iteration, kernel_size, pad):
        super(GRCL, self).__init__()
        self.wgf_u = nn.Conv2d(input, output, 1, 1, 0, bias=False)
        self.wgr_x = nn.Conv2d(output, output, 1, 1, 0, bias=False)
        self.wf_u = nn.Conv2d(input, output, kernel_size, 1, pad, bias=False)
        self.wr_x = nn.Conv2d(output, output, kernel_size, 1, pad, bias=False)

        self.BN_x_init = nn.BatchNorm2d(output)

        self.num_iteration = num_iteration
        self.GRCL = [GRCL_unit(output) for _ in range(num_iteration)]
        self.GRCL = nn.Sequential(*self.GRCL)

    def forward(self, input):
        wgf_u = self.wgf_u(input)
        wf_u = self.wf_u(input)
        x = fun.relu(self.BN_x_init(wf_u))

        for i in range(self.num_iteration):
            x = self.GRCL[i](wgf_u, self.wgr_x(x), wf_u, self.wr_x(x))
        return x


class GRCL_unit(nn.Module):

    def __init__(self, output):
        super(GRCL_unit, self).__init__()
        self.BN_gfu = nn.BatchNorm2d(output)
        self.BN_grx = nn.BatchNorm2d(output)
        self.BN_fu = nn.BatchNorm2d(output)
        self.BN_rx = nn.BatchNorm2d(output)
        self.BN_Gx = nn.BatchNorm2d(output)

    def forward(self, wgf_u, wgr_x, wf_u, wr_x):
        G_first_term = self.BN_gfu(wgf_u)
        G_second_term = self.BN_grx(wgr_x)
        G = fun.sigmoid(G_first_term + G_second_term)

        x_first_term = self.BN_fu(wf_u)
        x_second_term = self.BN_Gx(self.BN_rx(wr_x) * G)
        x = fun.relu(x_first_term + x_second_term)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, input, output, block, layers):
        super(ResNet, self).__init__()

        self.output_block = [int(output / 4), int(output / 2), output, output]

        self.inplanes = int(output / 8)
        self.conv0_1 = nn.Conv2d(input, int(output / 16), kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(int(output / 16))
        self.conv0_2 = nn.Conv2d(int(output / 16), self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_block[0], layers[0])
        self.conv1 = nn.Conv2d(self.output_block[0], self.output_block[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.output_block[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block, self.output_block[1], layers[1], stride=1)
        self.conv2 = nn.Conv2d(self.output_block[1], self.output_block[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.output_block[1])

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(block, self.output_block[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(self.output_block[2], self.output_block[2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.output_block[2])

        self.layer4 = self._make_layer(block, self.output_block[3], layers[3], stride=1)
        self.conv4_1 = nn.Conv2d(self.output_block[3], self.output_block[3], kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False)
        self.bn4_1 = nn.BatchNorm2d(self.output_block[3])
        self.conv4_2 = nn.Conv2d(self.output_block[3], self.output_block[3], kernel_size=2, stride=1, padding=0, bias=False)
        self.bn4_2 = nn.BatchNorm2d(self.output_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)

        return x
