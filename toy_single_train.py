from nsynth.config import make_config
from nsynth.training import train
from toy.data import ToyDataSetSingle
from toy.optim import single_vae_toy_loss
from toy.vae import ConditionalWavenetVAE


def main(args):
    # TODO implement for AE not VAE
    args.nit = 50000
    args.nbatch = 8
    μ = 100
    ns = 4
    loss_function = single_vae_toy_loss()

    device = f'cuda:{args.gpu[0]}' if args.gpu else 'cpu'

    model = ConditionalWavenetVAE(ns, device, 16, 64, 64, 10, 3, μ + 1, 1,
                                  False)
    crop = 3 * 2 ** 10

    traindata = ToyDataSetSingle(f'{args.datadir}/toy_train_4.npy', crop=crop,
                                 μ=μ).loader(args.nbatch)
    testdata = ToyDataSetSingle(f'{args.datadir}/toy_test_4.npy', crop=crop,
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
