from trainer import Trainer
from trainer import MNISTDataModule
from src.model.model import Model

if __name__ == "__main__":
    model = Model(num_output=10)

    data_module = MNISTDataModule()

    trainer = Trainer(model, data_module, lr=0.001, epochs=5)
    trainer.train()
    trainer.test()
    trainer.save()
