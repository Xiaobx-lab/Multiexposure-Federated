from collections import OrderedDict
from abc import ABC, abstractmethod
from typing import Tuple

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, ParametersRes
import flwr as fl
import torch
from torch.optim.lr_scheduler import StepLR

from multi_net import multi_net


class PytorchMNISTClient(fl.client.Client):
    """Flower client implementing MNIST handwritten classification using PyTorch."""
    def __init__(
        self,
        cid: int,
        train_loader: datasets,
        test_loader: datasets,
        epochs: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model = multi_net().to(device)
        self.cid = cid
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs

    def get_weights(self) -> fl.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_weights(self, weights: fl.common.Weights) -> None:
        """Set model weights from a list of NumPy ndarrays.

        Parameters
        ----------
        weights: fl.common.Weights
            Weights received by the server and set to local model


        Returns
        -------

        """
        state_dict = OrderedDict(
            {
                k: torch.tensor(v)
                for k, v in zip(self.model.state_dict().keys(), weights)
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self) -> fl.common.ParametersRes:
        """Encapsulates the weight into Flower Parameters """
        weights: fl.common.Weights = self.get_weights()
        parameters = fl.common.weights_to_parameters(weights)
        return fl.common.ParametersRes(parameters=parameters)

    def fit(self, ins: fl.common.FitIns) -> fl.common.FitRes:
        """Trains the model on local dataset

        Parameters
        ----------
        ins: fl.common.FitIns
        Parameters sent by the server to be used during training.

        Returns
        -------
            Set of variables containing the new set of weights and information the client.

        """
        weights: fl.common.Weights = fl.common.parameters_to_weights(ins.parameters)
        fit_begin = timeit.default_timer()

        # Set model parameters/weights
        self.set_weights(weights)

        # Train model
        num_examples_train: int = train(
            self.model, self.train_loader, epochs=self.epochs, device=self.device
        )

        # Return the refined weights and the number of examples used for training
        weights_prime: fl.common.Weights = self.get_weights()
        params_prime = fl.common.weights_to_parameters(weights_prime)
        fit_duration = timeit.default_timer() - fit_begin
        return fl.common.FitRes(
            parameters=params_prime,
            num_examples=num_examples_train,
            num_examples_ceil=num_examples_train,
            fit_duration=fit_duration,
        )

    def evaluate(self, ins: fl.common.EvaluateIns) -> fl.common.EvaluateRes:
        """

        Parameters
        ----------
        ins: fl.common.EvaluateIns
        Parameters sent by the server to be used during testing.


        Returns
        -------
            Information the clients testing results.
        """
def train(
    model: torch.nn.ModuleList,
    train_loader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device = torch.device("cpu"),
) -> int:
    """Train routine based on 'Basic MNIST Example'

    Parameters
    ----------
    model: torch.nn.ModuleList
        Neural network model used in this example.

    train_loader: torch.utils.data.DataLoader
        DataLoader used in traning.

    epochs: int
        Number of epochs to run in each round.

    device: torch.device
        (Default value = torch.device("cpu"))
        Device where the network will be trained within a client.

    Returns
    -------
    num_examples_train: int
        Number of total samples used during traning.

    """
    model.train()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    print(f"Training {epochs} epoch(s) w/ {len(train_loader)} mini-batches each")
    for epoch in range(epochs):  # loop over the dataset multiple time
        print()
        loss_epoch: float = 0.0
        num_examples_train: int = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # Grab mini-batch and transfer to device
            data, target = data.to(device), target.to(device)
            num_examples_train += len(data)

            # Zero gradients
            optimizer.zero_grad()

            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()
            if batch_idx % 10 == 8:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}\t\t\t\t".format(
                        epoch,
                        num_examples_train,
                        len(train_loader) * train_loader.batch_size,
                        100.0
                        * num_examples_train
                        / len(train_loader)
                        / train_loader.batch_size,
                        loss.item(),
                    ),
                    end="\r",
                    flush=True,
                )
        scheduler.step()
    return num_examples_train


def test(
    model: torch.nn.ModuleList,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, float, float]:
    """Test routine 'Basic MNIST Example'

    Parameters
    ----------
    model: torch.nn.ModuleList :
        Neural network model used in this example.

    test_loader: torch.utils.data.DataLoader :
        DataLoader used in test.

    device: torch.device :
        (Default value = torch.device("cpu"))
        Device where the network will be tested within a client.

    Returns
    -------
        Tuple containing the total number of test samples, the test_loss, and the accuracy evaluated on the test set.

    """
    model.eval()
    test_loss: float = 0
    correct: int = 0
    num_test_samples: int = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            num_test_samples += len(data)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= num_test_samples

    return (num_test_samples, test_loss, correct / num_test_samples)