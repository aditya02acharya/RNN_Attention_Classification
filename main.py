import os
import torch
import logging
from src.data.dataloader import Dataloader
from src.parameters.params import get_config
from src.utils.utilities import prepare_dirs, setup_logging, save_config
from src.train.trainer import Trainer


def main(configurations):
    setup_logging(default_path=os.path.join("configs", "logging.yml"))
    logger = logging.getLogger(__name__)
    logger.info("logger is set.")

    prepare_dirs(configurations)
    logger.info("required directories are created.")

    # ensure reproducibility
    torch.manual_seed(configurations.random_seed)
    kwargs = {}
    if configurations.use_gpu:
        torch.cuda.manual_seed(configurations.random_seed)
        kwargs = {"num_workers": 1, "pin_memory": True}

    loader = Dataloader()
    if configurations.train:
        d_loader = loader.get_train_valid_loader(
            configurations.data_dir,
            configurations.batch_size,
            configurations.random_seed,
            configurations.valid_size,
            configurations.shuffle,
            **kwargs, )

    else:
        d_loader = loader.get_test_loader(
            configurations.data_dir, configurations.batch_size, **kwargs,
        )

    trainer = Trainer(config, d_loader)

    # either train
    if config.train:
        save_config(config)
        trainer.train()
    # or load a pretrained model and test
    else:
        trainer.test()


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
