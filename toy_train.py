from nsynth.config import make_config
from nsynth.training import train


def main(args):
    args.nit = 50000
    args.nbatch = 10
    μ = 100
    ns = 3

    model = WavenetMultiAE(ns, 16, 64, 64, 10, 3, μ + 1, 1, False)
    crop = 3 * 2 ** 10

    traindata = ToyDataSet(f'{args.datadir}/toy_train_easy_3.npy', crop=crop,
                           μ=μ).loader(args.nbatch)
    testdata = ToyDataSet(f'{args.datadir}/toy_test_easy_3.npy', crop=crop,
                          μ=μ).loader(args.nbatch)

    train(model=model,
          loss_function=toy_loss_ordered(ns, μ + 1),
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
