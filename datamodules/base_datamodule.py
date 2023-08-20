from pytorch_lightning import LightningDataModule
from typing import Dict, Union, Optional, Tuple
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from utils.registry import DATAMODULE_REGISTRY
from datasets import build_dataset


@DATAMODULE_REGISTRY.register()
class BaseDataModule(LightningDataModule):
    """ Base class for all data modules. """

    def __init__(self, cfg: Union[DictConfig, Dict]):
        """
        Base class for all data modules.
        :param cfg: the configuration for the data module. Must contain the following keys:
            - type: the type of the data module, must be one of the keys in DATAMODULE_REGISTRY
            - for each usage (train, val, test, predict):
                - dataset: the dataset configuration for the usage named 'usage_dataset' or 'dataset' to define
                            a default dataset
                - dataloader: the dataloader configuration for the usage named 'usage_dataloader' or 'dataloader'
                            to define a default dataloader
            - for each dataloader:
                - batch_size: the batch size for the dataloader
                - num_workers: the number of workers for the dataloader
                - pin_memory: whether to pin memory for the dataloader
                - shuffle: whether to shuffle the dataloader.
        """
        super().__init__()
        self._cfg = cfg
        self._use_default_dataset = set()
        self._dataset_dict = {}
        self._initialized_dataset_types = set()

    def prepare_data(self) -> None:
        # download if needed
        pass

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            # split dataset into train, val, test for use in dataloader(s)
            self._init_dataset('train')
            self._init_dataset('val')
        elif stage in ['test', 'predict']:
            # prepare test/predict dataset(s) for use in dataloader(s)
            self._init_dataset(stage)
        else:
            raise ValueError(f'Invalid stage: {stage}, should be fit, test, or predict')

    def train_dataloader(self):
        return self._get_dataloader('train')

    def val_dataloader(self):
        return self._get_dataloader('val')

    def test_dataloader(self):
        return self._get_dataloader('test')

    def predict_dataloader(self):
        return self._get_dataloader('predict')

    def _get_cfg(self, usage: str, component: str) -> Optional[Tuple[Union[DictConfig, Dict], bool]]:
        """
        Returns the configuration for the given usage and component & whether the configuration is default.
        :param usage: the usage of the configuration, should be [train, val, test, or predict].
        :param component: the component of the configuration, should be dataset or dataloader.
        :return: the configuration for the given usage and component, or None if the usage is not specified
                 in the config.
        and whether the configuration is default.
        """
        assert usage in ['train', 'val', 'test',
                         'predict'], f'Invalid stage: {usage}, should be train, val, test, or predict'
        assert component in ['dataset', 'dataloader'], f'Invalid component: {component} should be dataset or dataloader'

        if usage + '_' + component in self._cfg.data:
            return self._cfg.data[usage + '_' + component], False
        elif component in self._cfg.data:
            return self._cfg.data[component], True

    def _get_dataloader(self, usage: str) -> Optional[DataLoader]:
        """
        Returns the dataloader for the given usage.
        :param usage: the usage of the dataloader, should be [train, val, test, or predict].
        :return: the dataloader for the given usage, or None if the usage is not specified in the config file.
        """
        err_format = 'Invalid stage: {}, should be train, val, test, or predict'
        assert usage in ['train', 'val', 'test', 'predict'], err_format.format(usage)

        # get the dataloader configuration
        out = self._get_cfg(usage, 'dataloader')
        if out is not None:
            cfg, _ = out
            dataset = self._get_dataset(usage)

            # return the dataloader
            if dataset is not None:
                return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle,
                                  num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    def _init_dataset(self, usage: str):
        """
        Returns the dataset for the given usage.
        :param usage: the usage of the dataset, should be [train, val, test, or predict].
        :return: the dataset for the given usage, or None if the usage is not specified in the config file.
        """
        err_format = 'Invalid stage: {}, should be train, val, test, or predict'
        assert usage in ['train', 'val', 'test', 'predict'], err_format.format(usage)

        self._initialized_dataset_types.add(usage)
        # get the dataset configuration
        out = self._get_cfg(usage, 'dataset')
        if out is not None:
            cfg, using_default = out
            dl_exists = self._get_cfg(usage, 'dataloader') is not None
            # if using a default dataset, save the usage
            if not dl_exists:  # if no dataloader exists, don't load the dataset.
                return None
            elif using_default and usage not in self._use_default_dataset:
                self._use_default_dataset.add(usage)
                self._dataset_dict['default'] = build_dataset(cfg)
            elif not using_default:
                self._dataset_dict[usage] = build_dataset(cfg)
            else:  # using default and already saved
                assert 'default' in self._dataset_dict, 'Default dataset not initialized, although it should be'

    def _get_dataset(self, usage: str):
        """
        Returns the dataset for the given usage.
        note: assumes that the dataset has already been initialized using _init_dataset(usage).
        :param usage: the usage of the dataset, should be [train, val, test, or predict].
        :return: the dataset for the given usage, or None if the usage is not specified in the config file.
        """
        if usage in self._dataset_dict:
            return self._dataset_dict[usage]
        elif usage in self._use_default_dataset:
            assert 'default' in self._dataset_dict, 'Default dataset not initialized, although it should be'
            return self._dataset_dict['default']
        else:
            assert usage in self._initialized_dataset_types, f'Dataset for usage {usage} not initialized'
            return None  #
