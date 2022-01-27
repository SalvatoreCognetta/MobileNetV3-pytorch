import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1, MetricCollection

import json
from typing import List
from dataclasses import dataclass, fields, astuple


root_drive = './'

spec_small_path = root_drive +  "specification/mobilenetv3-small.json"
spec_large_path = root_drive +  "specification/mobilenetv3-large.json"


# MobileNet Specification
@dataclass
class BNeckSpecification:
    '''Class that contains MobileNet specifications.'''
    kernel: int
    input_size: int
    exp_size: int
    out_size: int
    se: bool
    nl: str
    stride: nn.Module

    # The __post_init__ method, will be the last thing called by __init__.
    def __post_init__(self) -> None:
        self.kernel     = int(self.kernel)
        self.input_size = int(self.input_size)
        self.exp_size = int(self.exp_size)
        self.out_size = int(self.out_size)
        self.se  = bool(self.se)
        self.nl  = nn.ReLU(inplace=True) if self.nl == "relu" else Hswish(inplace=True)
        self.stride = int(self.stride)

    def __iter__(self):
        yield from astuple(self)

    @staticmethod
    def get_header() -> List[str]:
        return [field.name for field in fields(BNeckSpecification)]


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True) -> None:
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x) -> torch.Tensor:
        return F.relu6(x + 3, inplace=self.inplace) / 6

class Hswish(nn.Module):
    def __init__(self, inplace=True) -> None:
        super(Hswish, self).__init__()
        self.hsigmoid = Hsigmoid(inplace)

    def forward(self, x) -> torch.Tensor:
        return x * self.hsigmoid(x)


class SeModule(nn.Module):
    def __init__(self, channel, reduction=4) -> None:
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.se = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x) -> torch.Tensor:
        batch, channels, _, _ = x.size()
        out = self.avg_pool(x).view(batch, channels)
        out = self.se(out).view(batch, channels, 1, 1)
        return x * out.expand_as(x)




class BottleNeck(nn.Module):
    def __init__(self, spec:BNeckSpecification) -> None:
        super(BottleNeck, self).__init__()
        self.spec = spec
        
        padding = (spec.kernel - 1) // 2
        self.use_res_connect = spec.stride == 1 and spec.input_size == spec.out_size

        # PointWise
        self.conv2d_pw  = nn.Conv2d(spec.input_size, spec.exp_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm = nn.BatchNorm2d(spec.exp_size)
        self.non_lin    = spec.nl

        # DepthWise
        self.conv2d_dw  = nn.Conv2d(spec.exp_size, spec.exp_size, spec.kernel, spec.stride, padding, groups=spec.exp_size,bias=False)
        self.squeeze_ex = SeModule(spec.exp_size)

        # PointWise-linear
        self.conv2d_pw_linear  = nn.Conv2d(spec.exp_size, spec.out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm_linear = nn.BatchNorm2d(spec.out_size)

    def forward(self, x) -> torch.Tensor:
        # PointWise
        out = self.conv2d_pw(x)
        out = self.batch_norm(out)
        out = self.non_lin(out)

        # DepthWise
        out = self.conv2d_dw(out)
        out = self.batch_norm(out)
        if self.spec.se: out = self.squeeze_ex(out)
        out = self.non_lin(out)

        # PointWise-linear
        out = self.conv2d_pw_linear(out)
        out = self.batch_norm_linear(out)

        out = x + out if self.use_res_connect else out

        return out



def conv2d_block(input_size:int, output_size:int, kernel_size:int, stride:int=1) -> nn.Sequential:
	return nn.Sequential(
        nn.Conv2d(input_size, output_size, kernel_size, stride, padding=0, bias=False),
        nn.BatchNorm2d(output_size),
        Hswish(inplace=True)
    )



class MobileNetV3(nn.Module):
    def __init__(self, num_class:int, dropout:float, rgb_img:bool, mode='small') -> None:
        super(MobileNetV3, self).__init__()
        self.num_class = num_class
        self.mode = mode

        # Load specifications from file
        self.bneck_specs = self.load_bneck_specs()

        # Generate all the net blocks
        input_size = 3 if rgb_img else 1
        self.net_blocks = [conv2d_block(input_size=input_size, output_size=16, kernel_size=3, stride=2)]
        self.build_bneck_blocks()
        self.build_last_layers()

        # Transform it nn.Sequential
        self.net_blocks = nn.Sequential(*self.net_blocks)

        # Building the classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),    # refer to paper section 6
            nn.Linear(self.last_channel, self.num_class),
        )

    def forward(self, x) -> torch.Tensor:
        out = self.net_blocks(x)
        # To squeeze dimensions
        out = torch.squeeze(out, 3)
        out = torch.squeeze(out, 2)
        out = self.classifier(out)
        return out
    
    def load_bneck_specs(self) -> List[BNeckSpecification]:
        if self.mode == 'small':
            self.spec_file = spec_small_path
        else:
            self.spec_file = spec_large_path
        # Load specifications
        with open(self.spec_file, "r") as spec_f:
            data = json.load(spec_f)
            bneck_specs = [BNeckSpecification(*spec.values()) for spec in data]
        return bneck_specs

    def build_bneck_blocks(self) -> None:
        # Building mobile blocks
        for bneck_spec in self.bneck_specs:
            self.net_blocks.append(BottleNeck(bneck_spec))

    def build_last_layers(self) -> None:
        # Building last layers
        input_channel = self.bneck_specs[-1].out_size # Take the last bottleneck output size 
        if self.mode == 'large':
            self.last_conv = 960 
            self.last_channel = 1280
        elif self.mode == 'small':
            self.last_conv = 576
            self.last_channel = 1024
        
        self.net_blocks.append(conv2d_block(input_channel, self.last_conv, kernel_size=1))
        self.net_blocks.append(nn.AdaptiveAvgPool2d(1)) # or  out = F.avg_pool2d(out, 7)
        self.net_blocks.append(nn.Conv2d(self.last_conv, self.last_channel, kernel_size=1, stride=1, padding=0))
        self.net_blocks.append(Hswish(inplace=True))



class MobileNetV3Module(pl.LightningModule):
    def __init__(self, hparams, rgb_img:bool, mode:str='small') -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.rgb_img = rgb_img
        self.mode = mode

        # Enable manual optimization
        self.automatic_optimization = False
        
        self.net = MobileNetV3(self.hparams.num_classes, self.hparams.dropout, self.rgb_img, self.mode)
        
        self.criterion = nn.CrossEntropyLoss()

        metrics = MetricCollection([
            Accuracy(), 
            Precision(num_classes=self.hparams.num_classes), 
            Recall(num_classes=self.hparams.num_classes),
            F1(num_classes=self.hparams.num_classes)
        ])

        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics   = metrics.clone(prefix='val_')
        self.test_metrics  = metrics.clone(prefix='test_')


    def forward(self, x) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        data, label = batch
        
        # Compute the loss
        logits = self(data)
        predictions = torch.argmax(logits, dim=1)
        loss = self.criterion(logits, label)

        # Get the optimizer for manual backward
        opt = self.optimizers()

        # Optimization step
        opt.zero_grad()
        self.manual_backward(loss) # instead of loss.backward()
        opt.step()

        self.log("train_loss", loss, prog_bar=True, logger=True)

        # Metrics: Accuracy, Precision, Recall, F1
        metrics = self.train_metrics(predictions, label)
        # metrics are logged with keys: train_Accuracy, train_Precision, train_Recall and train_F1
        self.log_dict(metrics)

        return {"train_loss": loss}

    
    def training_epoch_end(self, outputs):
        sch = self.lr_schedulers()
        sch.step()
    
    # Validation step is broken
    # def validation_step(self, batch, batch_idx) -> torch.Tensor:
    #     x, y = batch
        
    #     # Compute the loss
    #     logits = self(x)
    #     predictions = torch.argmax(logits, dim=1)
    #     loss = self.criterion(logits, y)

    #     self.log("val_loss", loss, prog_bar=True, logger=True)

    #     # Metrics: Accuracy, Precision, Recall, F1
    #     metrics = self.val_metrics(predictions, y)
    #     # metrics are logged with keys: val_Accuracy, val_Precision, val_Recall and val_F1
    #     self.log_dict(metrics, logger=True)

    #     return loss

    # def validation_epoch_end(self, outputs) -> None:
    #     # Metrics on all full batch using custom accumulation
    #     metrics = self.val_metrics.compute()

    #     # Log the metrics
    #     self.log_dict(metrics, prog_bar=True, logger=True)


    def configure_optimizers(self):

        # For Cifar10
        # optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr,
        #                 weight_decay=self.hparams.weight_decay)

        # For MNISt
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.hparams.lr, 
                        weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.step_size, 
                        gamma=self.hparams.lr_decay)

        return [optimizer], [lr_scheduler]