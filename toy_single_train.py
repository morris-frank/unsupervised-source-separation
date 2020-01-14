from nsynth.config import make_config
from nsynth.training import train
from toy.data import ToyDataSetSingle
from toy.optim import vqvae_loss
from toy.vae import ConditionalWavenetVQVAE


def main(args):
    args.nit = 50000
    args.nbatch = 20
    μ = 100
    ns = 4
    loss_function = vqvae_loss()

    device = f'cuda:{args.gpu[0]}' if args.gpu else 'cpu'

    model = ConditionalWavenetVQVAE(n_sources=ns, K=1, D=512, n_blocks=3, n_layers=10,
                                    encoder_width=64, decoder_width=32,
                                    in_channels=1, out_channels=μ + 1,
                                    device=device)
    crop = 3 * 2 ** 10

    traindata = ToyDataSetSingle(f'{args.datadir}/toy_train.npy', crop=crop,
                                 μ=μ).loader(args.nbatch)
    testdata = ToyDataSetSingle(f'{args.datadir}/toy_test.npy', crop=crop,
                                μ=μ).loader(args.nbatch)

    train(model=model,
          loss_function=loss_function,
          gpu=args.gpu,
          trainset=traindata,
          testset=testdata,
          paths={'save': './models_toy/', 'log': './log_toy/'},
          iterpoints={'print': args.itprint, 'save': args.itsave,
                      'test': args.ittest},
          n_it=args.nit,
          use_board=args.board,
          use_manual_scheduler=args.original_lr_scheduler,
          save_suffix=f'det_{ns}'
          )


if __name__ == '__main__':
    main(make_config('train').parse_args())
