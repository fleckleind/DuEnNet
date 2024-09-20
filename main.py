import torch
import numpy as np
import random
import argparse
import torch.optim as optim
import albumentations as A
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from albumentations.pytorch.transforms import ToTensorV2

from model.DuEnNet import DuEnNet
from datasets.dataset_synapse import SynapseDataset
from utils import MixLoss
from trainer import trainer, tester

# main
parser = argparse.ArgumentParser()
# dataset arguments
# ----- synapse ------
parser.add_argument('--train_path', type=str, help='train_set dir', default='./data-synapse/train_npz')
parser.add_argument('--test_path', type=str, help='test_set dir', default='../data-synapse/test_vol_h5')
parser.add_argument('--list_dir', type=str, default='./data-list/lists_Synapse', help='dataset list load dir')
# ----- ACDC ------
# parser.add_argument('--data_path', type=str, help='ACDC dataset dir', default='../data-acdc')
parser.add_argument('--output_path', type=str, default='./outputs/', help='output dir')
parser.add_argument('--img_size', type=int, default=224, help='input size of network')
parser.add_argument('--num_classes', type=int, default=9, help='output channel of network, synapse(4), acdc(4)')
# trainer arguments
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--epochs', type=int, default=300, help='epoch number to train')
parser.add_argument('--base_lr', type=float, default=0.05, help='segmentation networking learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='batch size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='number of total gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
# testing arguments
parser.add_argument('--test_channel', type=int, default=3, help='single channel use batch_size')
# summary arguments
parser.add_argument('--snapshot_path', type=str, help='path of training or testing summary',
                    default='./outputs/Synapse/snapshot')
parser.add_argument('--model', type=str, default='DuEnNet', help='model name')
parser.add_argument('--fig_save_path', type=str, default='./outputs/Synapse/figs', help='save clip figs')
parser.add_argument('--model_dict_path', type=str, default='./outputs/Synapse/model_dict',
                    help='save training model dict')

# Swin-Unet configuration(yaml
parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file',
                    default='/kaggle/input/swinunet/swin_tiny_patch4_window7_224_lite.yaml')
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into non-overlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')


def worker_init_fn(worker_id):
    random.seed(1234 + worker_id)  # --args.seed default=1234


# config setting of swin-unet/transunet
args = parser.parse_args()

if __name__ == "__main__":
    CUDA_LAUNCH_BLOCKING = 1
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    # random setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # parameter adjustment
    if args.n_gpu != 1:
        args.batch_size *= args.n_gpu

    # train dataset and loader
    train_set = SynapseDataset(data_dir=args.train_path, list_dir=args.list_dir, split='train',
                               transform=A.Compose([A.Resize(args.img_size, args.img_size), A.HorizontalFlip(p=0.5),
                                                    A.VerticalFlip(p=0.5), A.ElasticTransform(p=0.3),
                                                    ToTensorV2(), ], is_check_shapes=False))
    print("Length of train set is:{}".format(len(train_set)))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8,
                              pin_memory=True, worker_init_fn=worker_init_fn, )  # pin_memory speed up training

    # test dataset and loader
    test_set = SynapseDataset(data_dir=args.test_path, list_dir=args.list_dir, split='test_vol',
                              transform=A.Compose([A.Resize(args.img_size, args.img_size),
                                                   ToTensorV2(), ], is_check_shapes=False))  # split as 'test' for ACDC
    print("Length of test set is:{}".format(len(test_set)))
    if args.test_channel == 1:
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    else:
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)  # multi channel, batch_size=1

    # model setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # cnn-swin dual branch unet
    model = DuEnNet(in_chans=3, num_classes=args.num_classes).to(device)

    # trainer functions
    criterion = MixLoss(n_classes=args.num_classes, buffer=1)  # 0.5 CE + 0.5 tversky
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)

    # pre-train model(without valid/test evaluation metric
    trainer(args, model=model, device=device, train_loader=train_loader, criterion=criterion, optimizer=optimizer,)

    # evaluate metrics with final model pth
    # best_model_dict = torch.load('./outputs/Synapse/model_dict/DuEnNet/xxx.pth')
    # tester(args, model=model, device=device, best_model_dict=best_model_dict, test_loader=test_loader, test_set=test_set)
