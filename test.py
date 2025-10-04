import os, argparse, time, datetime, sys, shutil, stat, torch
import numpy as np 
from torch.utils.data import DataLoader
from datasets.MF_dataset import  val_dataset, train_dataset
from sklearn.metrics import confusion_matrix
from PIL import Image
from DA2Net import DA2Net
from lora import LoRA_sam
parser = argparse.ArgumentParser(description='Test with pytorch')
parser.add_argument('--gpu', '-g', type=int, default=0)
parser.add_argument('--num_workers', '-j', type=int, default=0)
parser.add_argument('--n_class', '-nc', type=int, default=2)
parser.add_argument('--data_dir', '-dr', type=str, default=r"D:\master\PyCharm\PyCharm\Project\dataset\WHU-CD-256\test")
parser.add_argument('--model_dir', '-wd', type=str, default=r'D:\master\PyCharm\PyCharm\Project\A_yan1xia\change_detetion\exp\my_cd\checkpoints\DA2Net_WHU\WHU_alhpa32_drop0.1_best.pt')
args = parser.parse_args()

#############################################################################################
def visualize(image_name, predictions, weight_name):
    # palette = get_palette()
    for (i, pred) in enumerate(predictions):
        pred = predictions[i].cpu().numpy() * 255
        # img = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        # for cid in range(0, len(palette)):
        #     img[pred == cid] = palette[cid]
        img = Image.fromarray(np.uint8(pred))
        img.save('./Pred/Pred_' + weight_name + '_' + image_name + '.png')

def compute_results(conf_total):

    TP = conf_total[1][1]
    if TP == 0:
        TP = conf_total[1][1] + 0.000000001
    FP = conf_total[0][1]
    if FP == 0:
        FP = conf_total[0][1] + 0.000000001
    FN = conf_total[1][0]
    if FN == 0:
        FN = conf_total[1][0] + 0.000000001
    TN = conf_total[0][0]
    if TN == 0:
        TN = conf_total[0][0] + 0.000000001

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    iou = TP / (TP + FN + FP)
    F1 = 2 * (precision * recall) / (precision + recall)
    OA = (TP + TN) / (TP + TN + FP + FN)

    return precision, recall, iou, F1, OA

if __name__ == '__main__':

    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    # prepare save direcotry
    if os.path.exists("Pred"):
        print("previous \"./wuhan_uplus_Pred\" folder exist, will delete this folder")
        shutil.rmtree("Pred")
    os.makedirs("Pred")
    os.chmod("Pred", stat.S_IRWXU)  # allow the folder created by docker read, written, and execuated by local machine

    
    conf_total = np.zeros((args.n_class, args.n_class))
    conf_total_nochange = np.zeros((args.n_class, args.n_class))
    model1 = DA2Net(input_nc=3, output_nc=2, depths=[3, 3, 4, 3], heads=[4, 4, 4, 4],
                    enc_channels=[144, 288, 576, 1152], decoder_softmax=False, dec_embed_dim=256,
                    ).cuda()
    sam_lora = LoRA_sam(model1, 16).cuda()

    model = sam_lora.sam

    # cks = r"D:\master\PyCharm\PyCharm\Project\A_yan1xia\change_detetion\exp\my_cd\checkpoints\elgcnet_levir\WHU_alhpa32_drop0.1_best.pt"
    checkpoint = torch.load(os.path.join(args.model_dir), map_location='cpu')
    model.load_state_dict(checkpoint['model_G_state_dict'])

    batch_size = 8 	
    test_dataset  = val_dataset(data_dir=args.data_dir, split='test', input_h=256, input_w=256)
    test_loader  = DataLoader(
        dataset     = test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 2,
        pin_memory  = True,
        drop_last   = True
    )

    ave_time_cost = 0.0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    model.eval()
    num_class = 2
    with torch.no_grad():
        index = 0
        for data in enumerate(test_loader):
            # for i, data in enumerate(test_loader):
            a = data
            t1 = a[1][0].cuda()
            t2 = a[1][1].cuda()
            labels = a[1][2].cuda()
            name = a[1][3][-1]
            starter.record()

            prediction = model(t1, t2)
            label = labels.cpu().numpy().squeeze().flatten()
            if num_class == 1:
                prediction[prediction >= 0.5] = 1
                prediction[prediction < 0.5] = 0
            else:
                prediction = torch.argmax(prediction[0], 1)

            prediction = prediction.cpu().numpy().squeeze().flatten()  # prediction and label are both 1-d array, size: minibatch*640*480
            label = label.astype(np.int64)
            predictions = prediction.astype(np.int64)
            conf = confusion_matrix(y_true=label, y_pred=predictions, labels=[0,1])  # conf is args.n_class*args.n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf

            label_nochange = 1 - label
            predictions_nochange = 1 - predictions
            conf_nochange = confusion_matrix(y_true=label_nochange, y_pred=predictions_nochange, labels=[0, 1])
            conf_total_nochange += conf_nochange
            # visualize(name, pred0.squeeze(0), 'LEVIR')
        precision1, recall1, ioU1, f1score1, OA1 = compute_results(conf_total)
        precision0, recall0, ioU0, f1score0, OA0 = compute_results(conf_total_nochange)
        miou = (ioU1 + ioU0) / 2
        mf1 = (f1score1 + f1score0) / 2
        print(f"* test precision:{precision1}")
        print(f"* test recall   :{recall1}")
        print(f"* test iou      :{ioU1},     miou     :{miou}")
        print(f"* test f1score  :{f1score1}, mf1  :{mf1}")
        print(f"* test OA  :{OA1}")