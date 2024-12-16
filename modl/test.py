import argparse
import yaml, os, time
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import csv
from get_instances import *
from utils import *

def setup(args):
    config_path = args.config
    with open(config_path, "r") as fr:
        configs = yaml.load(fr, Loader=yaml.FullLoader)
    device = 'cpu'

    #read configs =================================
    n_layers = configs['n_layers']
    k_iters = configs['k_iters']

    dataset_name = configs['dataset_name']
    dataset_params = configs['dataset_params']

    batch_size = configs['batch_size'] if args.batch_size is None else args.batch_size

    model_name = configs['model_name']
    model_params = configs.get('model_params', {})
    model_params['n_layers'] = n_layers
    model_params['k_iters'] = k_iters

    score_names = configs['score_names']

    config_name = configs['config_name']

    workspace = os.path.join(args.workspace, config_name) #workspace/config_name
    checkpoints_dir, log_dir = get_dirs(workspace) #workspace/config_name/checkpoints ; workspace/config_name/log.txt
    tensorboard_dir = os.path.join(args.tensorboard_dir, configs['config_name']) #runs/config_name
    logger = Logger(log_dir)
    writer = get_writers(tensorboard_dir, ['test'])['test']

    dataloader = get_loaders(dataset_name, dataset_params, batch_size, ['test'])['test']
    model = get_model(model_name, model_params, device)
    score_fs = get_score_fs(score_names)

    #restore
    saver = CheckpointSaver(checkpoints_dir)
    prefix = 'best' if configs['val_data'] else 'final'
    checkpoint_path = [os.path.join(checkpoints_dir, f) for f in os.listdir(checkpoints_dir) if f.startswith(prefix)][0]
    model = saver.load_model("/Users/sikongqian/program/209project/MoDL_PyTorch/workspace/base_modl,k=1/checkpoints/final.epoch0049-score29.9709.pth", model)

    ##if torch.cuda.device_count()>1:
    #    model = nn.DataParallel(model)

    return configs, device, workspace, logger, writer, dataloader, model, score_fs
"""
def main(args):
    configs, device, workspace, logger, writer, dataloader, model, score_fs = setup(args)

    logger.write('\n')
    logger.write('test star t: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    start = time.time()

    running_score = defaultdict(int)

    model.eval()
    for i, (x, y, mask) in enumerate(tqdm(dataloader)):
        x, mask = x.to(device), mask.to(device)

        with torch.no_grad():
            y_pred = model(torch.tensor(x, dtype=torch.float32), mask).detach().cpu()

        y = np.abs(r2c(y.numpy(), axis=1))
        y_pred = np.abs(r2c(y_pred.numpy(), axis=1))
        for score_name, score_f in score_fs.items():
            running_score[score_name] += score_f(y, y_pred) * y_pred.shape[0]
        if args.write_image > 0 and (i % args.write_image == 0):
            writer.add_figure('img', display_img(np.abs(r2c(x[-1].detach().cpu().numpy())), mask[-1].detach().cpu().numpy(), \
                y[-1], y_pred[-1], psnr(y[-1], y_pred[-1])), i)

    epoch_score = {score_name: score / len(dataloader.dataset) for score_name, score in running_score.items()}
    for score_name, score in epoch_score.items():
        writer.add_scalar(score_name, score, 0)
        logger.write('test {} score: {:.4f}'.format(score_name, score))

    writer.close()
    logger.write('-----------------------')
    logger.write('total test time: {:.2f} min'.format((time.time()-start)/60))
"""
import csv
import time
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm

def main(args):
    # Set up the configurations and model
    configs, device, workspace, logger, writer, dataloader, model, score_fs = setup(args)

    logger.write('\n')
    logger.write('test start: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    start = time.time()

    running_score = defaultdict(int)

    model.eval()

    # Prepare the CSV file to store the results
    score_names = list(score_fs.keys())  # Get all the score names
    with open('test_scores_50.csv', mode='w', newline='') as file:
        writer_csv = csv.writer(file)
        
        # Write header with sample index and score names
        header = ['sample index']
        for score_name in score_names:
            header.append(f'{score_name}_before')
            header.append(f'{score_name}_after')
        writer_csv.writerow(header)

        # Loop over the test data
        for i, (x, y, mask, gt) in enumerate(tqdm(dataloader)):
            #x, mask = x.to(device), mask.to(device)
            print("x: type=", type(x), "shape=", x.shape, "dtype=", x.dtype)
            print("y: type=", type(y), "shape=", y.shape, "dtype=", y.dtype)
            print("mask: type=", type(mask), "shape=", mask.shape, "dtype=", mask.dtype)
            gt = np.squeeze(gt.numpy() ,axis = 0)
            print("gt: type=", type(gt), "shape=", gt.shape, "dtype=", gt.dtype)
            with torch.no_grad():
                y_pred = model(torch.tensor(x, dtype=torch.float32), mask).detach().cpu()
            x = np.squeeze(np.abs(r2c(x.numpy(), axis=1)), axis= 0)
            y = np.squeeze(np.abs(r2c(y.numpy(), axis=1)),axis = 0)
            y_pred = np.squeeze(np.abs(r2c(y_pred.numpy(),axis=1)), axis=0)
            gt = np.abs(gt)

            print("x: type=", type(x), "shape=", x.shape, "dtype=", x.dtype)
            print("y: type=", type(y), "shape=", y.shape, "dtype=", y.dtype)
            print("y_pred: type=", type(y_pred), "shape=", y_pred.shape, "dtype=", y_pred.dtype)
            print("gt: type=", type(gt), "shape=", gt.shape, "dtype=", gt.dtype)


            x = (x - np.min(x)) / (np.max(x) - np.min(x))  # Avoid division by zero
            y = (y - np.min(y)) / (np.max(y) - np.min(y))  # Normalize ground truth
            y_pred = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))  # Normalize ground truth
            gt = (gt - np.min(gt)) / (np.max(gt) - np.min(gt))  # Normalize ground truth

            score_values = []  # List to hold the scores for the current sample

            if i == 0:
                fig, axes = plt.subplots(1, 3, figsize=(18, 10))
                
                # 第一行：原图像和预测
                # 绘制 x
                axes[0].imshow(x, cmap='gray')
                axes[0].set_title('(a)')
                axes[0].axis('off')

                # 绘制 y
                axes[1].imshow(y_pred, cmap='gray')
                axes[1].set_title('(b)')
                axes[1].axis('off')

                # 绘制 y_pred
                axes[2].imshow(y, cmap='gray')
                axes[2].set_title('(c)')
                axes[2].axis('off')
                """
                # 第二行：残差图
                # 残差 x - y
                residual_x = x - y
                axes[1, 0].imshow(residual_x, cmap='gray')
                axes[1, 0].set_title('Residual: x - y')
                axes[1, 0].axis('off')
                
                # 残差 y_pred - y
                residual_pred = y_pred - y
                axes[1, 1].imshow(residual_pred, cmap='gray')
                axes[1, 1].set_title('Residual: y_pred - y')
                axes[1, 1].axis('off')

                # 留空或者添加其他分析
                axes[1, 2].axis('off')  # 空白占位
                """
                # 保存或者展示
                plt.tight_layout()
                plt.savefig('test_first_image_with_residuals.png')  # 保存为文件
                plt.show()  # 展示
        
            # Iterate over the scores and record them for each sample
            for score_name, score_f in score_fs.items():
                # Calculate score_before using a custom method

                
                score_before = score_f(y, x)  
                score_after = score_f(y, y_pred)
                
                # Add both before and after scores to the list for this sample
                score_values.append(score_before)
                score_values.append(score_after)

            # Write to CSV: sample index and the list of scores for this sample
            writer_csv.writerow([i] + score_values)
            """
            # Optionally, write images if specified
            if args.write_image > 0 and (i % args.write_image == 0):
                writer.add_figure('img', display_img(np.abs(r2c(x[-1].detach().cpu().numpy())), mask[-1].detach().cpu().numpy(), \
                    y[-1], y_pred[-1], psnr(y[-1], y_pred[-1])), i)
            """
    # Compute average scores for the entire dataset
    epoch_score = {score_name: score / len(dataloader.dataset) for score_name, score in running_score.items()}
    for score_name, score in epoch_score.items():
        writer.add_scalar(score_name, score, 0)
        logger.write('test {} score: {:.4f}'.format(score_name, score))

    writer.close()
    logger.write('-----------------------')
    logger.write('total test time: {:.2f} min'.format((time.time()-start)/60))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", type=str, required=False, default="configs/base_modl,k=1.yaml",
                        help="config file path")
    parser.add_argument("--workspace", type=str, default='./workspace')
    parser.add_argument("--tensorboard_dir", type=str, default='./runs')
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--write_image", type=int, default=0)

    args = parser.parse_args()

    main(args)