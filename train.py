from thesis import WavenetAE, WavenetVAE, \
    make_config
from thesis.data import make_loaders
from thesis.training import train


def main(args):
    crop = args.n_blocks * 2 ** args.n_layers
    model_class = WavenetVAE if args.vae else WavenetAE

    model = model_class(in_channels=1, out_channels=args.out_channels,
                        latent_width=args.latent_width,
                        encoder_width=args.encoder_width,
                        decoder_width=args.decoder_width,
                        n_blocks=args.n_blocks, n_layers=args.n_layers)

    # Build datasets
    loaders = make_loaders(args.datadir, ['train', 'test'], args.n_batch,
                           crop, args.families, args.sources)

    train(model=model, loss_function=model_class.loss_function,
          gpu=args.gpu, trainset=loaders['train'], testset=loaders['test'],
          num_iter=args.num_iter, use_board=args.board)


if __name__ == '__main__':
    main(make_config('train').parse_args())
