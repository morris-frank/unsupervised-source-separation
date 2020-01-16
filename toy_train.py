from nsynth.config import make_config
from nsynth.training import train
from toy.ae import WavenetMultiAE
from toy.data import ToyDataSet
from toy.optim import multiae_loss, multivae_loss
from toy.vae import WavenetMultiVAE


def main(args):
    model_class = WavenetMultiVAE if args.vae else WavenetMultiAE
    args.epochs = 50000
    args.n_batch = 8
    μ = 100
    ns = 4
    β = 2
    dβ = 0  # Removes annealing β
    if args.vae:
        loss_function = multivae_loss(ns, μ + 1, β=β, dβ=dβ)
    else:
        loss_function = multiae_loss(ns, μ + 1)

    model = model_class(n=ns, latent_width=16, encoder_width=64,
                        decoder_width=64, n_layers=10, n_blocks=3,
                        out_channels=μ + 1,
                        channels=1, gen=False)
    crop = 3 * 2 ** 10

    traindata = ToyDataSet(f'{args.datadir}/toy_train.npy', crop=crop,
                           μ=μ).loader(args.nbatch)
    testdata = ToyDataSet(f'{args.datadir}/toy_test.npy', crop=crop,
                          μ=μ).loader(args.nbatch)

    train(model=model,
          loss_function=loss_function,
          gpu=args.gpu,
          trainset=traindata,
          testset=testdata,
          paths={'save': './models_toy/', 'log': './log_toy/'},
          iterpoints={'print': args.it_print, 'save': args.it_save,
                      'test': args.ittest},
          n_it=args.epochs,
          use_board=args.board,
          use_manual_scheduler=args.original_lr_scheduler,
          save_suffix=f'det_{ns}'
          )


if __name__ == '__main__':
    main(make_config('train').parse_args())
