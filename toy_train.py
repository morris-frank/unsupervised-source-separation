from nsynth.config import make_config
from nsynth.training import train
from toy.ae import WavenetMultiAE
from toy.data import ToyDataSet
from toy.optim import toy_loss_ordered as toy_loss
from toy.optim import variation_toy_loss_ordered as toy_loss_vae
from toy.vae import WavenetMultiVAE


def main(args):
    model_class = WavenetMultiVAE if args.vae else WavenetMultiAE
    args.nit = 50000
    args.nbatch = 10
    μ = 100
    ns = 3
    loss_function = toy_loss_vae(ns, μ + 1) if args.vae else toy_loss(ns, μ + 1)

    model = model_class(ns, 16, 64, 64, 10, 3, μ + 1, 1, False)
    crop = 3 * 2 ** 10

    traindata = ToyDataSet(f'{args.datadir}/toy_train_4.npy', crop=crop,
                           μ=μ).loader(args.nbatch)
    testdata = ToyDataSet(f'{args.datadir}/toy_test_4.npy', crop=crop,
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
