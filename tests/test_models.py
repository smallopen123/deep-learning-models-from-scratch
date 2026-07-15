import unittest

import torch
from torch import nn

from dl_models import (
    Autoencoder, Discriminator, Generator, MLP, SequenceClassifier,
    SimpleCNN, TinyResNet, TransformerClassifier, VariationalAutoencoder,
)
from dl_models.autoencoders import vae_loss


class ModelShapeTest(unittest.TestCase):
    def test_classifiers_forward_and_backward(self) -> None:
        cases = [
            (MLP(), torch.randn(2, 2), (2, 2)),
            (SimpleCNN(), torch.randn(2, 1, 28, 28), (2, 4)),
            (TinyResNet(), torch.randn(2, 1, 28, 28), (2, 4)),
            (SequenceClassifier(cell="rnn"), torch.randint(1, 20, (2, 12)), (2, 2)),
            (SequenceClassifier(cell="lstm"), torch.randint(1, 20, (2, 12)), (2, 2)),
            (SequenceClassifier(cell="gru"), torch.randint(1, 20, (2, 12)), (2, 2)),
            (TransformerClassifier(), torch.randint(1, 20, (2, 12)), (2, 2)),
        ]
        for model, inputs, expected in cases:
            with self.subTest(model=model.__class__.__name__):
                output = model(inputs)
                self.assertEqual(tuple(output.shape), expected)
                output.sum().backward()
                self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_autoencoder_shapes(self) -> None:
        inputs = torch.rand(3, 64)
        reconstruction, latent = Autoencoder()(inputs)
        self.assertEqual(tuple(reconstruction.shape), (3, 64))
        self.assertEqual(tuple(latent.shape), (3, 8))

    def test_vae_loss_backward(self) -> None:
        inputs = torch.rand(3, 64)
        model = VariationalAutoencoder()
        reconstruction, mean, log_variance, latent = model(inputs)
        self.assertEqual(tuple(latent.shape), (3, 8))
        loss = vae_loss(reconstruction, inputs, mean, log_variance)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_gan_shapes_and_gradients(self) -> None:
        generator, discriminator = Generator(), Discriminator()
        fake = generator(torch.randn(4, 8))
        logits = discriminator(fake)
        self.assertEqual(tuple(fake.shape), (4, 2))
        self.assertEqual(tuple(logits.shape), (4, 1))
        nn.BCEWithLogitsLoss()(logits, torch.ones_like(logits)).backward()
        self.assertIsNotNone(next(generator.parameters()).grad)


if __name__ == "__main__":
    unittest.main()
