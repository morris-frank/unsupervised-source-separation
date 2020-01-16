from argparse import ArgumentParser
from os import path


def make_config(version: str) -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--gpu', type=int, required=False, nargs='+',
                        help='The GPU ids to use. If unset, will use CPU.')

    if 'train' in version:
        parser.add_argument('--datadir', type=path.abspath, required=True,
                            help='The top-level directory of NSynth dataset '
                                 '(containing the split directories.)')

        gtrain = parser.add_argument_group('Training options')
        gtrain.add_argument('-ne', type=int, default=250000, dest='epochs',
                            help='Number of batches to train for.')
        gtrain.add_argument('-nb', type=int, default=32, dest='n_batch',
                            help='The batch size.')
        gtrain.add_argument('--crop_length', type=int, default=6144,
                            help='Length of the actual training sub-samples'
                                 '(crops).')
        gtrain.add_argument('--original_lr_scheduler', action='store_true',
                            help='Use the original exact learning rate '
                                 'schedule as given in the paper.')
        gtrain.add_argument('--families', type=str, required=False, nargs='+',
                            help='The instrument families to use from the '
                                 'dataset.')
        gtrain.add_argument('--sources', type=str, required=False, nargs='+',
                            help='The instrument sources to use from the '
                                 'dataset.')

        glog = parser.add_argument_group('Logging options')
        glog.add_argument('-ip', type=int, default=20, dest='it_print',
                          help='Frequency of loss print.')
        glog.add_argument('-is', type=int, default=5000, dest='it_save',
                          help='Frequency of model checkpoints.')
        glog.add_argument('-it', type=int, default=500, dest='it_test',
                          help='Frequency of running the test set.')
        glog.add_argument('--savedir', type=path.abspath, default='./models/',
                          help='Path to save the checkpoints to.')
        glog.add_argument('--logdir', type=path.abspath, default='./log/',
                          help='Path to save the logs to.')
        glog.add_argument('--board', action='store_true',
                          help='Whether to use Tensorboard.')

    if 'sampl' in version:
        gsampl = parser.add_argument_group('Sampling options')
        gsampl.add_argument('--weights', type=path.abspath, required=True,
                            help='Path to the saved weight file.')
        gsampl.add_argument('--sample', type=path.abspath, required=True,
                            help='Path to the sample WAV file.')
        gsampl.add_argument('--sampledir', type=path.abspath,
                            default='./samples',
                            help='Path to save the generated samples to.')

    if 'plot' in version:
        gplt = parser.add_argument_group('Plotting options')
        gplt.add_argument('--weights', type=path.abspath, required=True,
                          help='Path to the saved weight file.')

    gmodel = parser.add_argument_group('Model options')
    gmodel.add_argument('-wl', type=int, default=16, dest='latent_width',
                        help='Size ot the Autoencoder Bottleneck.')
    gmodel.add_argument('-we', type=int, default=128, dest='encoder_width',
                        help='Dimensions of the encoders hidden layers.')
    gmodel.add_argument('-wd', type=int, default=512, dest='decoder_width',
                        help='Dimensions of the decoders hidden layers.')
    gmodel.add_argument('-nl', type=int, default=10, dest='n_layers',
                        help='Number of dilation layers in each block.')
    gmodel.add_argument('-nb', type=int, default=3, dest='n_blocks',
                        help='Number of blocks.')
    gmodel.add_argument('-nc', type=int, default=256, dest='out_channels',
                        help='Number of in_channels to quant the output with.')
    gmodel.add_argument('-vae', action='store_true',
                        help='Whether to use the VAE model.')
    return parser
