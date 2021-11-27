import torch 
import argparse

def get_args():
    parser = argparse.ArgumentParser() 

    parser.add_argument(
        '--lr', default=1e-4,
        help = 'learning rate'
    )
    parser.add_argument(
        '--scheduler-rate', default=0.001,
        help = 'learning rate'
    )
    parser.add_argument(
        '--n_epochs', default=25,
        help = 'number of epochs for training the model'
    )
    parser.add_argument(
        '--random_state', default=42,
        help = 'random seeding'
    )
    parser.add_argument(
        '--encoder', default='resnet18',
        help = 'select the encoder for the model,\
            for selection of different encoders visit,\
            https://github.com/qubvel/segmentation_models.pytorch'
    )
    parser.add_argument(
        '--encoder_weights', default='imagenet',
        help = 'for selection of different encoder weights visit,\
            https://github.com/qubvel/segmentation_models.pytorch'
    )
    parser.add_argument(
        '--img-augmentation', default=True,
        help = 'Apply image augmentation on training images'
    )
    parser.add_argument(
        '--resize-img', default=1024,
        help = 'val for resizing the images'
    )
    parser.add_argument(
        '--train-batch_size', default=1,
        help='batch size for training data'
    )
    parser.add_argument(
        '--valid-batch_size', default=1,
        help='batch size for validation data'
    )
    parser.add_argument(
        '--raster-path', default="input/images/"
    )
    parser.add_argument(
        '--label-geo-path', default="input/labels_geo/"
    )
    parser.add_argument(
        '--vector-path', default="input/labels_pix/"
    )
    parser.add_argument(
        '--output-path', default="output/"
    )
    parser.add_argument(
        '--model-output', default="model_output/"
    )

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    parser.add_argument('--device', default=device)

    args, _ = parser.parse_known_args()

    return args