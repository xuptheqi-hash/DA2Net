from argparse import ArgumentParser
import torch
from models.trainer import *
import utils
import os
print(torch.cuda.is_available())
from DA2Net import DA2Net
from lora import LoRA_sam
def train(args, model):
    dataloaders = utils.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders, model=model)
    model.train_models()


def test(args):
    from models.evaluator import CDEvaluator
    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test')
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models()


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='DA2Net_levir', type=str)
    parser.add_argument('--checkpoint_root', default='./checkpoint', type=str)
    parser.add_argument('--vis_root', default='./vis', type=str)

    # data
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='WHU', type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="test", type=str)

    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--dec_embed_dim', default=256, type=int)
    parser.add_argument('--pretrain', default=None, type=str)

    parser.add_argument('--net_G', default='SAM2', type=str,
                        help='DA2Net')
    parser.add_argument('--loss', default='ce', type=str)

    # optimizer
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr', default=0.00021, type=float)
    parser.add_argument('--max_epochs', default=50, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=[100], type=int)
    
    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)
    
    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)
	# SAM2-large: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
    hiera_path = r'D:\master\PyCharm\PyCharm\Project\A_yan1xia\change_detetion\exp\my_cd\sam2\sam2_hiera_large.pt'

    model = DA2Net(input_nc=3, output_nc=2, depths=[3, 3, 4, 3], heads=[4, 4, 4, 4],
                    enc_channels=[144, 288, 576, 1152], decoder_softmax=False, dec_embed_dim=256, checkpoint_path=hiera_path).cuda()
    sam_lora = LoRA_sam(model, 16).cuda()
    net = sam_lora.sam
    train(args, net)

