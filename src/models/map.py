from src.models.mnist import LitAutoEncoder
from src.models.mnist_conv import MNISTConvNet

name_to_model = {
    "mnist": LitAutoEncoder(),
    "mnist_conv": MNISTConvNet(),
}
