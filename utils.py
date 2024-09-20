import torch
import numpy as np
import torch.nn as nn
from medpy import metric
import SimpleITK as sitk
from scipy.ndimage import zoom
from torch.nn.modules.loss import CrossEntropyLoss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        # label(mask) is single-channel, while the channel of model output is number of label+1(background),
        # then original label should process to multi channels, equals to the one of outputs
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, smooth=1e-5):
        target = target.float()  # mask(label)
        intersection = torch.sum(score * target)
        union = torch.sum(target) + torch.sum(score)
        loss = 1 - (2 * intersection + smooth) / (union + smooth)
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes  # create a dist
        assert inputs.size() == target.size(), (
            'predict {} & target {} shape should do not match'.format(inputs.size(), target.size()))
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class TverskyLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        # label(mask) is single-channel, while the channel of model output is number of label+1(background),
        # then original label should process to multi channels, equals to the one of outputs
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _tversky_loss(self, score, target, alpha=0.3, beta=0.7, smooth=1e-5):
        target = target.float()
        intersection = torch.sum(score * target)
        denominator = (1-(alpha+beta)) * intersection + alpha * torch.sum(score) + beta * torch.sum(target)
        loss = 1 - ((intersection + smooth) / (denominator + smooth))
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes  # create a dist
        assert inputs.size() == target.size(), (
            'predict {} & target {} shape should do not match'.format(inputs.size(), target.size()))
        class_wise_tversky = []
        loss = 0.0
        for i in range(self.n_classes):
            tversky = self._tversky_loss(inputs[:, i], target[:, i])
            class_wise_tversky.append(1.0 - tversky.item())
            loss += tversky * weight[i]
        return loss / self.n_classes


class MixLoss(nn.Module):
    def __init__(self, n_classes, buffer=0, alpha=0.5, beta=0.5):
        super().__init__()
        self.buffer = buffer
        self.n_classes = n_classes
        self.alpha, self.beta = alpha, beta
        self.CELoss = CrossEntropyLoss()
        self.DiceLoss = DiceLoss(n_classes=self.n_classes)
        self.TverskyLoss = TverskyLoss(n_classes=self.n_classes)

    def forward(self, inputs, targets):
        loss_ce = self.CELoss(inputs, targets[:].long())
        loss_dice = self.DiceLoss(inputs, targets, softmax=True)
        loss_tversky = self.TverskyLoss(inputs, targets, softmax=True)
        if self.buffer == 0:
            loss = self.alpha * loss_ce + self.beta * loss_dice
        elif self.buffer == 1:
            loss = self.alpha * loss_ce + self.beta * loss_tversky
        else:
            loss = loss_ce
        return loss


def calculate_metric_pcase(pred, gt):
    # package medpy metrics
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0


def test_calculate_metric(args, model, image, mask, n_classes, img_size=None, out_buf=None,
                         z_spacing=1, case=None):
    if img_size is None:
        img_size = [args.img_size, args.img_size]
    image, mask = image.squeeze(0).cpu().detach().numpy(), mask.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        preds = np.zeros_like(mask)
        for clip in range(image.shape[0]):
            slice_clip = image[clip, :, :]
            width, height = slice_clip.shape[0], slice_clip.shape[1]
            if width != img_size[0] or height != img_size[1]:
                slice_clip = zoom(slice_clip, (img_size[0]/width, img_size[1]/height), order=3)
            slice_image = torch.from_numpy(slice_clip).unsqueeze(0).unsqueeze(0).float().cuda()
            model.eval()
            with torch.no_grad():
                # original unet layer is 4, if fewer adjust here!
                if out_buf is None:
                    outputs = model.forward(slice_image)
                else:
                    outputs = model.forward(slice_image)[out_buf]
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()  # one-hot, single class evaluation
                if width != img_size[0] or height != img_size[1]:
                    pred = zoom(out, (width/img_size[0], height/img_size[1]), order=0)
                else:
                    pred = out
                preds[clip] = pred
    else:
        slice_image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        model.eval()
        with torch.no_grad():
            # original unet layer is 4, if fewer adjust here!
            if out_buf is None:
                outputs = model.forward(slice_image)
            else:
                outputs = model.forward(slice_image)[out_buf]
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            preds = out.cpu().detach().numpy()
    metric_eval = []
    for i in range(1, n_classes):
        metric_eval.append(calculate_metric_pcase(preds == i, mask == i))
    if case is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(preds.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(mask.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, args.output_path + '/'+ case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, args.output_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, args.output_path + '/'+ case + "_gt.nii.gz")
    return metric_eval
