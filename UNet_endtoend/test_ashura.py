"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os, csv
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from utils import *


def np_categorical_dice(pred, truth, k):
    """ Dice overlap metric for label k """
    A = (pred == k).astype(np.float32)
    B = (truth == k).astype(np.float32)
    return 2 * np.sum(A * B) / (np.sum(A) + np.sum(B))


def foward_network(a, model):
    a = np.expand_dims(a, axis=1)
    a = torch.from_numpy(a).to(model.device)

    model.set_input({'A':a, 'B':a})
    fake_B= model.test()  # run inference
    # output = softmax(output, dim=1)
    # output = torch.argmax(output, dim=1)
    fake_B = fake_B.data.to('cpu').numpy()
    return fake_B

def MSE(input, target):
    return ((input-target)**2).mean()


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    output_dir = './output'
    spacing_target = (10, 1.25, 1.25)
    window_size = 256
    stride = 128
    batch_size = 1
    data_list = os.listdir(opt.dataroot)
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()

    csv_name = './'+opt.name+'.csv'
    with open(csv_name, 'w+') as f:
        writer = csv.writer(f)
        for iter in range(100,126):
            opt.load_iter = iter
            model = create_model(opt)  # create a model given opt.model and other options
            model.setup(opt)  # regular setup: load and print networks; create schedulers
            instance_dice = {0: 0, 1: 0, 2: 0, 3: 0}
            instance_HD = {0: 0, 1: 0, 2: 0, 3: 0}
            instance_MSE = 0
            if opt.eval:
                model.eval()
            for patient in data_list:
                # if not '080' in patient:
                #     continue
                print(patient)
                patient_root = os.path.join(opt.dataroot, patient)

                for phase in ['ED', 'ES']:
                    # read iamge
                    itk_image = sitk.ReadImage(os.path.join(patient_root, phase + '.nii.gz'))
                    spacing = np.array(itk_image.GetSpacing())[[2, 1, 0]]
                    image = sitk.GetArrayFromImage(itk_image).astype(float)

                    image_backup = image

                    # resample image to target spacing, keep z spacing, do not modify
                    
                    gt = sitk.ReadImage(os.path.join(patient_root, phase + '_gt.nii.gz'))
                    gt = sitk.GetArrayFromImage(gt)
                    # gt_m = image_backup.copy().astype(np.float32)
                    # gt_m[gt == 0] = 0


                    spacing_target = list(spacing_target)
                    spacing_target[0] = spacing[0]
                    image = resize_image(image, spacing, spacing_target).astype(np.float32)
                    if opt.norm_type == 'slice':
                        image = rescale_intensity_slice(image, (0.5, 99.5))
                    else:
                        image = rescale_intensity(image, (0.5, 99.5))
                    src = image


                    c, h, w = src.shape
                    src_backup = src

                    # crop image, if image size < crop size, pad at first
                    starth = 0
                    if h < window_size:
                        # print('img height is too small!!!!!!!!')
                        windw_pad_size = (window_size - h) * 2 + h
                        res = np.zeros([c, windw_pad_size, w], dtype=src.dtype)
                        start = (windw_pad_size - h) // 2
                        res[:, start:start + h, :] = src
                        src = res
                        
                        #update parameters
                        h = windw_pad_size
                        starth = start

                    startw = 0
                    if w < window_size:
                        # print('img width is too small!!!!!!!!')
                        window_pad_size = (window_size - w) * 2 + w
                        res = np.zeros([c, h, window_pad_size], dtype=src.dtype)
                        start = (window_pad_size - w) // 2
                        res[:, :, start:start + w] = src
                        src = res

                        #update parameters
                        startw = start
                        w = window_pad_size

                    c_startw = (w - window_size) // 2
                    c_starth = (h - window_size) // 2

                    # 5 crop,
                    crop1 = src[:, :window_size, :window_size]
                    crop2 = src[:, -window_size:, :window_size]
                    crop3 = src[:, :window_size:, -window_size:]
                    crop4 = src[:, -window_size:, -window_size:]
                    crop5 = src[:, c_starth:c_starth + window_size, c_startw:c_startw + window_size]

                    output1 = foward_network(crop1, model)
                    output2 = foward_network(crop2, model)
                    output3 = foward_network(crop3, model)
                    output4 = foward_network(crop4, model)
                    output5 = foward_network(crop5, model)

                    n_, h_, w_ = src.shape
                    probshape = [n_, 4, h_, w_]
                    full_output = np.zeros(probshape)

                    full_output[:, :, :window_size, :window_size] += output1
                    full_output[:, :, -window_size:, :window_size] += output2
                    full_output[:, :, :window_size, -window_size:] += output3
                    full_output[:, :, -window_size:, -window_size:] += output4
                    full_output[:, :, c_starth:c_starth + window_size, c_startw:c_startw + window_size] += output5


                    c, h, w = src_backup.shape
                    full_output = full_output[:, :, starth:starth + h, startw:startw + w]
                    full_output = torch.argmax(torch.from_numpy(full_output), dim=1).data.squeeze().numpy()

                    tmp = convert_to_one_hot(full_output)
                    vals = np.unique(full_output)
                    results = []
                    for i in range(len(tmp)):
                        results.append(resize_image(tmp[i].astype(float), spacing_target, spacing, 1)[None])
                    full_output = vals[np.vstack(results).argmax(0)]

                   

                    Z = full_output.shape[0]
                    image_backup = image_backup.astype(np.float32)
                    image_backup -= image_backup.min()
                    image_backup = image_backup * 255.0 / image_backup.max()






                    for z in range(Z):
                        input_slice = image_backup[z, :, :]
                        output_slice = full_output[z, :, :] * 85

                        gt_slice = gt[z, :, :] * 85

                        merge = np.concatenate(
                            [input_slice,output_slice, gt_slice], axis=1)
                        cv2.imwrite(os.path.join(output_dir, patient + '_' + phase + '_' + str(z) + '.png'), merge)

                    for i in range(4):
                        dice = np_categorical_dice(full_output, gt, i)
                        # dice_temp.append(dice)
                        instance_dice[i] += round(dice, 4)
                        try:
                            A = sitk.GetImageFromArray(full_output)
                            B = sitk.GetImageFromArray(gt)
                            hausdorffcomputer.Execute(A == i, B == i)
                            HD = hausdorffcomputer.GetHausdorffDistance()
                            instance_HD[i] += HD
                            print(patient, phase, i, dice, HD)
                        except:
                            print("Hausdorff Eroor!   ", patient, '  ', phase)
                   
                    
            for i in range(4):
                instance_dice[i] /= (2 * len(data_list))
                instance_HD[i] /= (2 * len(data_list))
            instance_MSE /= (2 * len(data_list))
            print('iteration{}:'.format(iter), instance_dice, (instance_dice[1] + instance_dice[2] + instance_dice[3]) / 3)
            print(instance_HD, (instance_HD[1] + instance_HD[2] + instance_HD[3]) / 3)
            dice_mean = (instance_dice[1] + instance_dice[2] + instance_dice[3]) / 3
            dice_mean = round(dice_mean, 6)
            writer.writerow([iter, round(dice_mean, 4)])


        #     visuals = model.get_current_visuals()  # get image results
        #     img_path = model.get_image_paths()     # get image paths
        #     if i % 5 == 0:  # save images to an HTML file
        #         print('processing (%04d)-th image... %s' % (i, img_path))
        #     save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        # webpage.save()  # save the HTML
