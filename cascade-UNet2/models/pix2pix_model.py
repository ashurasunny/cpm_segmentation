import torch
from .base_model import BaseModel
from . import networks
from models.loss import MulticlassDiceLoss, DiceLoss_M


class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.
    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).
    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_Dice', type=float, default=1.0, help='weight for Dice loss')
            parser.add_argument('--lambda_DiceM', type=float, default=5.0, help='weight for Dice loss')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_L1_1'
            , 'G_Dice_2']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'real_M', 'real_B', 'fake_B_1', 'fake_M', 'fake_B_2']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G_1', 'G_2']
        # define networks (both generator and discriminator)
        if opt.dataset == 'ACDC':
            self.netG_1 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, softmaxorrelu=1)
            self.netG_2 = networks.define_G(2, 4, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, softmaxorrelu=2)
        elif opt.dataset == 'cpm':
            self.netG_1 = networks.define_G(3, 3, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                            softmaxorrelu=1)
            self.netG_2 = networks.define_G(6, 2, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                            softmaxorrelu=2)

        if self.isTrain:
            # define loss functions
            self.criterionL1_1 = torch.nn.L1Loss()
            self.criterionDice_1 = DiceLoss_M().to(self.device)
            # self.criterionL1_2 = torch.nn.L1Loss()
            self.criterionDice_2 = MulticlassDiceLoss().to(self.device)
            self.dice_weights = [opt.dice_w0, opt.dice_w1, opt.dice_w2, opt.dice_w3]
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G_1 = torch.optim.Adam(self.netG_1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_2 = torch.optim.Adam(self.netG_2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G_1)
            self.optimizers.append(self.optimizer_G_2)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        if self.isTrain:
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_M = (input['M']).to(self.device)
        # self.real_AM = torch.cat((self.real_A, self.real_M), 1)
        # print('123')
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # print("Calling forward")
        self.fake_M = self.netG_1(self.real_A)
        self.real_AM = torch.cat((self.real_A, self.real_M), 1)
        self.fake_B_1 = self.netG_2(self.real_AM)
        # self.fake_B_1 = torch.nn.functional.softmax(self.fake_B_1, dim=1)
        self.fake_AM = torch.cat((self.real_A, self.fake_M), 1)
        self.fake_B_2 = self.netG_2(self.fake_AM)
        # self.fake_B_2 = torch.nn.functional.softmax(self.fake_B_2, dim=1)

    def backward_G_1(self):
        # print("Calling backward_G_1")
        """Calculate GAN and L1 loss for the generator"""
        # Second, G(A) = B
        self.loss_G_L1_1 = self.criterionL1_1(self.fake_M, self.real_M) * self.opt.lambda_L1
        # self.loss_G_Dice_1 = self.criterionDice_1(self.fake_M, self.real_M) * self.opt.lambda_DiceM
        # combine loss and calculate gradients
        self.loss_G_1 = self.loss_G_L1_1
        self.loss_G_1.backward(retain_graph=True)

    def backward_G_2(self):
        # print("Calling backward_G_2")
        """Calculate GAN and L1 loss for the generator"""
        # Second, G(A) = B
        self.loss_G_Dice_2 = (1 * self.criterionDice_2(self.fake_B_1, self.real_B,
                                                       self.dice_weights) + 2 * self.criterionDice_2(self.fake_B_2,
                                                                                                     self.real_B,
                                                                                                     self.dice_weights)) * self.opt.lambda_Dice  # + 1 * self.criterionL1_2(self.fake_B_2, self.fake_B_1))* self.opt.lambda_L1
        # combine loss and calculate gradients
        self.loss_G_2 = self.loss_G_Dice_2
        self.loss_G_2.backward(retain_graph=True)

    def optimize_parameters(self):
        self.set_requires_grad(self.netG_1, True)
        self.fake_M = self.netG_1(self.real_A)  # compute fake images: G(A)

        self.optimizer_G_1.zero_grad()  # set G's gradients to zero
        self.backward_G_1()  # calculate graidents for G
        self.optimizer_G_1.step()  # udpate G's weights

    def optimize_parameters_2(self):
        self.set_requires_grad(self.netG_2, True)
        self.fake_M = self.netG_1(self.real_A)
        self.real_AM = torch.cat((self.real_A, self.real_M), 1)
        self.fake_B_1 = self.netG_2(self.real_AM)
        self.fake_AM = torch.cat((self.real_A, self.fake_M), 1)
        self.fake_B_2 = self.netG_2(self.fake_AM)
        self.set_requires_grad(self.netG_1, False)

        # update G2
        self.optimizer_G_2.zero_grad()  # set G's gradients to zero
        self.backward_G_2()  # calculate graidents for G
        self.optimizer_G_2.step()  # udpate G's weights

    def optimize_parameters_3(self):
        self.set_requires_grad(self.netG_1, True)
        self.set_requires_grad(self.netG_2, True)
        self.fake_M = self.netG_1(self.real_A)
        self.real_AM = torch.cat((self.real_A, self.real_M), 1)
        self.fake_B_1 = self.netG_2(self.real_AM)
        self.fake_AM = torch.cat((self.real_A, self.fake_M), 1)
        self.fake_B_2 = self.netG_2(self.fake_AM)

        # update G2
        self.optimizer_G_2.zero_grad()  # set G's gradients to zero
        self.backward_G_2()  # calculate graidents for G
        self.optimizer_G_2.step()  # udpate G's weights
        self.optimizer_G_1.step()  # udpate G's weights

        # update G1
        self.optimizer_G_1.zero_grad()  # set G's gradients to zero
        self.backward_G_1()  # calculate graidents for G
        self.optimizer_G_1.step()  # udpate G's weights

    def test(self):
        with torch.no_grad():
            self.fake_M = self.netG_1(self.real_A)
            self.real_AM = torch.cat((self.real_A, self.real_M), 1)
            self.fake_B_1 = self.netG_2(self.real_AM)

            # fake_M_2 = self.real_A.data.cpu().numpy()
            # fake_M = self.fake_M.data.cpu().numpy()
            # 
            # fake_M_2[fake_M==0] = 0

            # fake_M_2 = torch.from_numpy(fake_M_2)

            self.fake_AM = torch.cat((self.real_A, self.fake_M), 1)
            self.fake_B_2 = self.netG_2(self.fake_AM)
        return self.fake_M, self.fake_B_1, self.fake_B_2