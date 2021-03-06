"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model, pix2pix_model
from util.visualizer import Visualizer
from data.cpm_dataset import CPM17Dataset
import random
from torch.utils.data import DataLoader
import os

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # dataset_size = len(dataset)    # get the number of images in the dataset.

    train_dataset = CPM17Dataset(opt.dataroot, aug_rate=opt.aug_rate)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    dataset_size = len(train_loader)

    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    
    

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        idx = random.sample(range(dataset_size), dataset_size)
        
        for i, sample in enumerate(train_loader):

            src, gt, m = sample

            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            model.set_input({'A': src, 'B': gt, 'M': m})  # unpack data from dataset and apply preprocessing
            if i % opt.G1_freq == 0:
                G1 = True
            else:
                G1 = False
            if i % opt.G2_freq == 0:
                G2 = True
            else:
                G2 = False
            if i % opt.D1_freq == 0:
                D1 = True
            else:
                D1 = False
            if i % opt.D2_freq == 0:
                D2 = True
            else:
                D2 = False

           
            if epoch == 1:
                model.optimize_parameters_3(G1, D1, G2, D2)
            if epoch <= opt.stage1_epoch:
                model.optimize_parameters(G1, D1)
            elif epoch > opt.stage1_epoch and epoch <= opt.stage2_epoch:
                model.optimize_parameters_2(G2, D2)
            else:
                model.optimize_parameters_3(G1, D1, G2, D2)
        
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)



            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

                # print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                # model.save_networks('latest')
                # model.save_networks(epoch)

                # if opt.display_id > 0:
                #     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
