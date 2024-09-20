import os
import sys
import copy
import torch
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from thop import profile
from utils import test_calculate_metric


def trainer(args, model, device, train_loader, criterion, optimizer, best_model_dict=None, former_epoch=0, scheduler=None):
    # model testing information summary(log, writer)
    logging.basicConfig(filename=args.snapshot_path + "/train_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(args.model + ": Training Loss")
    # model thops, flops
    flops, thops = profile(model, inputs=(torch.randn((1, 1, 224, 224)).cuda(),))
    flops, thops = flops * 1e-9, thops * 1e-6
    logging.info(args.model + ": params {:.4f}, flops {:.4f}".format(thops, flops))
    # model training arguments
    iterator = tqdm(range(args.epochs), ncols=70)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    # loss history and model dict
    train_loss_hist, min_loss, iter_num = [], 10, 0
    # update learning rate of optimizer(swin-unet version)
    max_iterations = args.epochs * len(train_loader)
    # if model not trained from beginning
    if best_model_dict is not None:
        # get pre-trained model dict, delete different shape also keys
        if len(model.state_dict().keys())==len(best_model_dict.keys()):
            checkpoint = best_model_dict
        else:
            checkpoint = copy.deepcopy(best_model_dict)
            model_dict = model.state_dict()
            for k in list(checkpoint.keys()):
                if k in model_dict:
                    if checkpoint[k].shape != model_dict[k].shape:
                        print("delete:{};shape model:{}".format(k, model_dict[k].shape))
                        del checkpoint[k]
                else:
                    print("delete:{}".format(k))
                    del checkpoint[k]
        model.load_state_dict(checkpoint)
        former_iters = len(train_loader) * (former_epoch + 1)
        iter_num, max_iterations = iter_num + former_iters, max_iterations + former_iters
    for epoch in iterator:
        model.train()
        t_loss = []
        for image, mask in train_loader:
            image = image.to(device, dtype=torch.float)
            mask = mask.to(device, dtype=torch.float)
            pred = model(image)
            # loss = criterion(pred_batch, mask, softmax=True)  # dice/tversky loss
            loss = criterion(pred, mask)  # dice and CE loss
            t_loss.append(loss.item())
            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # optimizer update learning rate, lr decay
            upd_lr = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = upd_lr
            iter_num = iter_num + 1
        # training loss history and print
        train_loss_hist.append(np.mean(t_loss))
        logging.info('epoch:{:}, train_loss:{:.4f}'.format(epoch, train_loss_hist[-1]))
        if train_loss_hist[-1] < min_loss:
            min_loss = train_loss_hist[-1]
            # model_dict_path = str(os.path.join(args.model_dict_path, args.model))
            if epoch >= (args.epochs // 2):
                torch.save(model.state_dict(), os.path.join(args.model_dict_path, str(args.model+'-epoch-'+str(epoch)+'.pth')))
                logging.info('save epoch-{:} model dict of {:}'.format(epoch, args.model))
        if scheduler is not None:
            scheduler.step()
    return 'Train Finished!'


def tester(args, model, device, best_model_dict, test_loader, test_set, metric_eval=0.0, iter_num=0, out_buf=None):
    # get pre-trained model dict, delete different shape also keys
    if len(model.state_dict().keys())==len(best_model_dict.keys()):
        checkpoint = best_model_dict
    else:
        checkpoint = copy.deepcopy(best_model_dict)
        model_dict = model.state_dict()
        for k in list(checkpoint.keys()):
            if k in model_dict:
                if checkpoint[k].shape != model_dict[k].shape:
                    print("delete:{};shape model:{}".format(k, model_dict[k].shape))
                    del checkpoint[k]
            else:
                print("delete:{}".format(k))
                del checkpoint[k]
    model.load_state_dict(checkpoint)
    model.eval()
    # model testing information summary(log, writer)
    logging.basicConfig(filename=args.snapshot_path + "/test_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(args.model + ": Valid/Test Evaluation Metrics")
    # model thops, flops
    flops, thops = profile(model, inputs=(torch.randn((1, 1, 224, 224)).cuda(),))
    flops, thops = flops * 1e-9, thops * 1e-6
    logging.info(args.model + ": params {:.4f}, flops {:.4f}".format(thops, flops))
    # single test operation, with evaluation metric(dice, hd95
    for sample in test_loader:
        iter_num = iter_num + 1
        image = sample['image'].to(device, dtype=torch.float)
        mask = sample['mask'].to(device, dtype=torch.float)
        case_name = sample['case_name'][0]
        metric_i = test_calculate_metric(args, model, image, mask, args.num_classes, out_buf=out_buf, case=case_name)
        metric_eval += np.array(metric_i)
        # log summary of sample iter_num dice, hd95
        logging.info('sample-{:}, mean_dice:{:.4f}, mean_hd95:{:.4f}'.format(
            iter_num, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    # total class and mean metrics of model dict
    metric_eval = metric_eval / len(test_set)
    for i in range(1, args.num_classes):
        logging.info('Class {}: mean_dice {:.4f}, mean_hd95 {:.4f}'
                     .format(i, metric_eval[i-1][0], metric_eval[i-1][1]))
    mean_dice, mean_hd95 = np.mean(metric_eval, axis=0)[0], np.mean(metric_eval, axis=0)[1]
    logging.info('Mean Metric Evaluation: dice {:.4f}, hd95 {:.4f}'.format(mean_dice, mean_hd95))
    return 'Test Finished!'
