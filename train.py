from nsynth import WavenetAE, WavenetVAE, \
    make_config
from nsynth.data import make_loaders
from nsynth.training import train


def main(args):
    model_class = WavenetVAE if args.vae else WavenetAE

    model = model_class(in_channels=1, out_channels=args.out_channels,
                        latent_width=args.latent_width,
                        encoder_width=args.encoder_width,
                        decoder_width=args.decoder_width,
                        n_blocks=args.n_blocks, n_layers=args.n_layers)

    # Build datasets
    loaders = make_loaders(args.datadir, ['train', 'test'], args.n_batch,
                           args.crop_length, args.families, args.sources)

    train(model=model,
          loss_function=model_class.loss_function,
          gpu=args.gpu,
          trainset=loaders['train'],
          testset=loaders['test'],
          paths={'save': args.savedir, 'log': args.logdir},
          iterpoints={'print': args.it_print, 'save': args.it_save,
                      'test': args.it_test},
          n_it=args.epochs,
          use_board=args.board,
          use_manual_scheduler=args.original_lr_scheduler
          )


if __name__ == '__main__':
    main(make_config('train').parse_args())
