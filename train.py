from thesis.config import make_config
from thesis.data.toy import ToyDataSequential, ToyDataSingle
from thesis.nn.models import WavenetVAE, ConditionalWavenetVQVAE
from thesis.train import train


def main(args):
    args.epochs = 50000
    crop = args.n_blocks * 2 ** args.n_layers
    μ = 100

    if args.vae:
        ns = 4
        steps = 5
        args.n_batch = 8
        model = WavenetVAE(in_channels=1, out_channels=μ + 1, n_decoders=ns,
                           latent_width=32, encoder_width=64, decoder_width=32)
        loss_function = model.loss(β=1.1)
        train_loader = ToyDataSequential(
            f'{args.datadir}/toy_train_long_*.npy', μ=μ, crop=crop,
            batch_size=args.batch_size, steps=steps).loader(args.batch_size)
        test_loader = ToyDataSequential(
            f'{args.datadir}/toy_test_long_*.npy', μ=μ, crop=crop,
            batch_size=args.batch_size, steps=steps).loader(args.batch_size)
    else:
        ns = 8
        args.n_batch = 20
        model = ConditionalWavenetVQVAE(n_sources=ns, K=ns, D=512, n_blocks=3,
                                        n_layers=10, encoder_width=64,
                                        decoder_width=32, in_channels=1,
                                        out_channels=μ + 1)
        loss_function = model.loss()
        train_loader = ToyDataSingle(f'{args.datadir}/toy_train_large.npy',
                                     crop=crop, μ=μ).loader(args.n_batch)
        test_loader = ToyDataSingle(f'{args.datadir}/toy_test_large.npy',
                                    crop=crop, μ=μ).loader(args.n_batch)

    train(model=model, loss_function=loss_function, gpu=args.gpu,
          train_loader=train_loader, test_loader=test_loader,
          iterations=args.iterations, wandb=args.wandb)


if __name__ == '__main__':
    main(make_config('train').parse_args())
