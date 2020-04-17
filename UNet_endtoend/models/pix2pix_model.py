import torch
from .base_model import BaseModel
from . import networks
from models.loss import MulticlassDiceLoss, DiceLoss


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

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_Dice_1']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A','real_B','fake_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G_1']
        else:  # during test time, only load G
            self.model_names = ['G_1']
        # define networks (both generator and discriminator)
        if opt.dataset == 'ACDC':
            self.netG_1 = networks.define_G(opt.input_nc, 4, opt.ngf, opt.netG, opt.norm,
                                            not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.criterionDice = MulticlassDiceLoss().to(self.device)
        elif opt.dataset == 'cpm':
            self.netG_1 = networks.define_G(3, 2, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
            self.criterionDice = MulticlassDiceLoss().to(self.device)
        if self.isTrain:
            # define loss functions

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G_1 = torch.optim.Adam(self.netG_1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G_1)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = (input['B']).to(self.device)


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        #print("Calling forward")
        self.fake_B = self.netG_1(self.real_A)

    def backward_G_1(self):
        #print("Calling backward_G_1")
        """Calculate GAN and L1 loss for the generator"""

        self.loss_G_Dice_1  = self.criterionDice(self.fake_B, self.real_B)
        # self.loss_G_1 = self.loss_G_L1_1
        self.loss_G_Dice_1.backward()
        
    def optimize_parameters(self):
        # self.forward()                   # compute fake images: G(A)
        self.fake_B = self.netG_1(self.real_A)

        # update G1
        self.set_requires_grad(self.netG_1, True)
        self.optimizer_G_1.zero_grad()  # set G's gradients to zero
        self.backward_G_1()  # calculate graidents for G
        self.optimizer_G_1.step()  # udpate G's weights


    def test(self):
        with torch.no_grad():
            self.forward()
        return self.fake_B
