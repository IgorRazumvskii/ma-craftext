import os
import importlib
import flax.struct
import yaml
import craftext

import pathlib
import inspect
import flax

import craftext.dataset
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG_DIR_NAME = "configs"

@flax.struct.dataclass
class ScenariosConfig:
    """Scenarios configuration structure
    
    Keyword arguments:
        - dataset_key      -- task type
        - subset_key       -- task complexity and if `test` - paraphrases / items
        - base_environment -- use `Classic` or not
        - use_parafrases   -- use `paraPhrases` in loading or not
        - test             -- is it `test` data or not
    """ 
    dataset_key: str
    subset_key: str
    base_environment: str
    use_parafrases: str
    test: str
    

class ScenariosConfigLoader:
    """
    Loader for scenario configuration files in the CrafText dataset.

    Provides utilities to locate a YAML config by name and to load its
    contents into a ScenariosConfig dataclass.
    """

    @staticmethod
    def get_config_path(config_name: str) -> pathlib.PurePath:
        """
        Construct the filesystem path to a scenario config YAML file.

        Parameters
        ----------
        config_name : str
            The base name of the configuration file (without the .yaml extension).

        Returns
        -------
        pathlib.PurePath
            The absolute path to the YAML file under CONFIG_DIR_NAME.

        Raises
        ------
        ModuleNotFoundError
            If the `craftext.dataset` module cannot be located.
        """
        module = inspect.getmodule(craftext.dataset)
        
        if not module:
            raise ModuleNotFoundError
        
        # print(module.__path__)
    
        module_path = pathlib.PurePath(module.__path__[0])
        
        config_path = module_path.joinpath(f'{CONFIG_DIR_NAME}/{config_name}.yaml')

        return config_path

    @staticmethod
    def load_config(config_name: str) -> ScenariosConfig:
        """
        Load a scenario configuration from its YAML file into a ScenariosConfig.

        Parameters
        ----------
        config_name : str
            The base name of the configuration file (without the .yaml extension).

        Returns
        -------
        ScenariosConfig
            An instance populated from the YAML contents, with fields:
            - dataset_key
            - subset_key
            - base_environment
            - use_parafrases (default False)
            - test (default False)

        Raises
        ------
        FileNotFoundError
            If the YAML file does not exist at the expected path.
        yaml.YAMLError
            If the file contains invalid YAML.
        """
        config_path = ScenariosConfigLoader.get_config_path(config_name)
        
        with open(config_path, 'r') as file:
            config_data = yaml.safe_load(file)
            
        return ScenariosConfig(
            dataset_key      =config_data.get("dataset_key"),
            subset_key       =config_data.get("subset_key"),
            base_environment =config_data.get("base_environment"),
            use_parafrases   =config_data.get("use_parafrases", False),
            test             =config_data.get("test", False)
        )


def get_default_scenario_path():
    """
    Return the absolute path to the scenarios directory within the installed package.

    This function locates the filesystem path of the `craftext.dataset` module,
    logs that base path, and appends the `scenarious` subdirectory to it.

    Returns
    -------
    str
        The full filesystem path to the `scenarious` directory.

    Raises
    ------
    ModuleNotFoundError
        If the `craftext.dataset` module cannot be located.
    """
   
    module_path = inspect.getmodule(craftext.dataset).__path__[0]
    logging.info(f'Scenario location: {module_path}')
    return os.path.join(module_path, 'scenarious')

def load_scenarios(scenarious_config):
    """
    Dynamically load and aggregate scenario definitions based on a configuration.

    This function will scan the scenarios directory for files whose names include
    the configured dataset_key, import either the `test` or `instructions` submodule
    depending on the `test` flag, and collect all scenario mappings under the given
    subset_key.

    Parameters
    ----------
    scenarious_config : ScenariosConfig
        Configuration object with the following attributes:
        - dataset_key (str): substring to match scenario filenames
        - subset_key  (str): attribute name inside each module to retrieve scenario dict
        - test        (bool): whether to import from `test` rather than `instructions`

    Returns
    -------
    dict
        A merged dictionary of all scenario definitions found under the matching modules.

    Raises
    ------
    ValueError
        If the scenarios directory path could not be determined.
    ImportError
        If a matching scenario submodule cannot be imported.
    """
    scenarios = {}
    scenarios_dir = get_default_scenario_path()
   
    module = "test" if scenarious_config.test else "instructions"
    mode = scenarious_config.dataset_key
    data_key = scenarious_config.subset_key
   
    if scenarios_dir is None:
        raise ValueError("Scenario path could not be determined.")

    for file in os.listdir(scenarios_dir):
        if mode in file:
            scenario_module_name = f"craftext.dataset.scenarious.{file}.{module}"
            scenario_module = importlib.import_module(scenario_module_name)
            
            if hasattr(scenario_module, data_key):
                scenarios.update(getattr(scenario_module, data_key))
    
   
    return scenarios