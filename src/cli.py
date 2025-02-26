import os
import sys
import argparse

sys.path.append("./src/")

from tester import Tester
from trainer import Trainer
from dataloader import Loader
from utils import config_files


def cli():
    nheads = config_files()["transfomerEncoderBlock"]["nheads"]
    dropout = config_files()["transfomerEncoderBlock"]["dropout"]
    batch_size = config_files()["dataloader"]["batch_size"]
    image_size = config_files()["patchEmbeddings"]["image_size"]
    split_size = config_files()["dataloader"]["split_size"]
    patch_size = config_files()["patchEmbeddings"]["patch_size"]
    activation = config_files()["transfomerEncoderBlock"]["activation"]
    dimension = config_files()["patchEmbeddings"]["dimension"]
    image_channels = config_files()["patchEmbeddings"]["channels"]
    num_encoder_layers = config_files()["transfomerEncoderBlock"]["num_encoder_layers"]
    dimension_feedforward = config_files()["transfomerEncoderBlock"][
        "dimension_feedforward"
    ]
    layer_norm_eps = float(config_files()["transfomerEncoderBlock"]["layer_norm_eps"])
    model = config_files()["trainer"]["model"]
    epochs = config_files()["trainer"]["epochs"]
    lr = float(config_files()["trainer"]["lr"])
    beta1 = float(config_files()["trainer"]["beta1"])
    beta2 = float(config_files()["trainer"]["beta2"])
    momentum = config_files()["trainer"]["momentum"]
    step_size = config_files()["trainer"]["step_size"]
    gamma = config_files()["trainer"]["gamma"]
    l1_lambda = config_files()["trainer"]["l1_lambda"]
    l2_lambda = config_files()["trainer"]["l2_lambda"]
    device = config_files()["trainer"]["device"]
    adam = config_files()["trainer"]["adam"]
    SGD = config_files()["trainer"]["SGD"]
    l1_regularization = config_files()["trainer"]["l1_regularization"]
    l2_regularization = config_files()["trainer"]["l2_regularization"]
    lr_scheduler = config_files()["trainer"]["lr_scheduler"]
    verbose = config_files()["trainer"]["verbose"]
    mlflow = config_files()["trainer"]["mlflow"]
    model = config_files()["tester"]["model"]
    plot_images = config_files()["tester"]["plot_images"]

    parser = argparse.ArgumentParser(
        description="Multi-Modal Classifier Training and Testing CLI".title()
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=image_channels,
        help="Number of channels in the images",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=batch_size,
        help="Batch size for the dataloader",
    )
    parser.add_argument(
        "--split_size",
        type=float,
        default=split_size,
        help="Split size for the train and test datasets",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=image_size,
        help="Image size to transform".capitalize(),
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=patch_size,
        help="Patch size for the transformer".capitalize(),
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=dimension,
        help="Dimension for the transformer".capitalize(),
    )
    parser.add_argument(
        "--nheads",
        type=int,
        default=nheads,
        help="Number of heads for the multi-head attention".capitalize(),
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=activation,
        help="Activation function for the transformer".capitalize(),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=dropout,
        help="Dropout probability for the transformer".capitalize(),
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=num_encoder_layers,
        help="Number of layers in the transformer".capitalize(),
    )
    parser.add_argument(
        "--dimension_feedforward",
        type=int,
        default=dimension_feedforward,
        help="Dimension for the feedforward layer".capitalize(),
    )
    parser.add_argument(
        "--layer_norm_eps",
        type=float,
        default=layer_norm_eps,
        help="Layer normalization epsilon".capitalize(),
    )
    parser.add_argument(
        "--train",
        action="store_true",
        default="Train CLI for the Multi Modal Classifier".capitalize(),
    )
    parser.add_argument(
        "--test",
        action="store_true",
        default="Test CLI for the Multi Modal Classifier".capitalize(),
    )
    parser.add_argument("--model", type=str, default=model, help="Model to train")
    parser.add_argument(
        "--epochs", type=int, default=epochs, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=lr, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=beta1, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=beta2, help="Adam beta2")
    parser.add_argument("--momentum", type=float, default=momentum, help="SGD momentum")
    parser.add_argument(
        "--step_size", type=int, default=step_size, help="Step size for lr scheduler"
    )
    parser.add_argument(
        "--gamma", type=float, default=gamma, help="Gamma for lr scheduler"
    )
    parser.add_argument(
        "--l1_lambda", type=float, default=l1_lambda, help="L1 regularization lambda"
    )
    parser.add_argument(
        "--l2_lambda", type=float, default=l2_lambda, help="L2 regularization lambda"
    )
    parser.add_argument(
        "--l1_regularization",
        type=float,
        default=l1_regularization,
        help="L1 regularization",
    )
    parser.add_argument(
        "--l2_regularization",
        type=float,
        default=l2_regularization,
        help="L2 regularization",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=bool,
        default=lr_scheduler,
        help="Use learning rate scheduler",
    )
    parser.add_argument(
        "--verbose", type=bool, default=verbose, help="Display progress"
    )
    parser.add_argument(
        "--mlflow", type=bool, default=mlflow, help="Enable MLflow tracking"
    )
    parser.add_argument(
        "--device", type=str, default=device, help="Device to use (cpu or cuda)"
    )
    parser.add_argument("--adam", type=bool, default=adam, help="Use Adam optimizer")
    parser.add_argument("--SGD", type=bool, default=SGD, help="Use SGD optimizer")
    parser.add_argument(
        "--choose_model",
        type=str,
        default=model,
        help="Choose model to evaluate (best or path)",
    )
    parser.add_argument(
        "--plot_images", type=bool, default=plot_images, help="Display predicted images"
    )

    args = parser.parse_args()

    if args.train:
        loader = Loader(
            channels=args.channels,
            image_size=args.image_size,
            batch_size=args.batch_size,
            split_size=args.split_size,
        )


        loader.unzip_image_dataset()
        loader.create_dataloader()

        trainer = Trainer(
            model=None,
            epochs=args.epochs,
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            momentum=args.momentum,
            step_size=args.step_size,
            gamma=args.gamma,
            l1_lambda=args.l1_lambda,
            l2_lambda=args.l2_lambda,
            device=args.device,
            adam=args.adam,
            SGD=args.SGD,
            l1_regularization=args.l1_regularization,
            l2_regularization=args.l2_regularization,
            lr_scheduler=args.lr_scheduler,
            verbose=args.verbose,
            mlflow=args.mlflow,
        )

        Loader.display_images()
        Loader.details_dataset()

        trainer.train()
        Trainer.display_history()

    else:
        try:
            tester = Tester(
                model=args.choose_model, device=args.device, plot_images=args.plot_images
            )
            tester.model_eval(display_image=True)
        except Exception as e:
            print(f"[FATAL] Tester encountered a critical error: {e}")
            sys.exit(1)
        else:
            print("[INFO] MultiModalClassifier evaluation completed successfully. "
                "All files related to the test are stored in the metrics folder.")
        
        
if __name__ == "__main__":
    cli()
