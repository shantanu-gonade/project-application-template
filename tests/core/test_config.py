"""
Tests for the config module.
"""

import os
import json
import pytest
import tempfile

import src.core.config as config

def test_get_parameter_from_config():
    """
    Test that get_parameter returns values from the config file.
    """
    # Reset the singleton config object
    config._config = {"TEST_PARAM": "test_value"}
    
    # Test getting a parameter that exists
    assert config.get_parameter("TEST_PARAM") == "test_value"
    
    # Test getting a parameter that doesn't exist
    assert config.get_parameter("NON_EXISTENT_PARAM") is None
    
    # Test getting a parameter that doesn't exist with a default value
    assert config.get_parameter("NON_EXISTENT_PARAM", "default") == "default"

def test_get_parameter_from_env(monkeypatch):
    """
    Test that get_parameter prioritizes environment variables over config values.
    """
    # Reset the singleton config object
    config._config = {"TEST_PARAM": "config_value"}
    
    # Set an environment variable
    monkeypatch.setenv("TEST_PARAM", "env_value")
    
    # Test that the environment variable is used
    assert config.get_parameter("TEST_PARAM") == "env_value"
    
    # Test with JSON value in environment variable
    monkeypatch.setenv("JSON_PARAM", "json:[1, 2, 3]")
    assert config.get_parameter("JSON_PARAM") == [1, 2, 3]

def test_convert_to_typed_value():
    """
    Test that convert_to_typed_value correctly converts string values to their types.
    """
    # Test with None
    assert config.convert_to_typed_value(None) is None
    
    # Test with JSON string representing a list
    assert config.convert_to_typed_value("[1, 2, 3]") == [1, 2, 3]
    
    # Test with JSON string representing a dictionary
    assert config.convert_to_typed_value('{"key": "value"}') == {"key": "value"}
    
    # Test with JSON string representing a number
    assert config.convert_to_typed_value("42") == 42
    
    # Test with JSON string representing a boolean
    assert config.convert_to_typed_value("true") is True
    
    # Test with a non-JSON string
    assert config.convert_to_typed_value("not json") == "not json"
    
    # Test with a non-string value
    assert config.convert_to_typed_value(42) == 42

def test_set_parameter(monkeypatch):
    """
    Test that set_parameter correctly sets parameters in environment variables.
    """
    # Reset the singleton config object
    config._config = {}
    
    # Test setting a string value
    config.set_parameter("TEST_PARAM", "test_value")
    assert os.environ["TEST_PARAM"] == "test_value"
    
    # Test setting a non-string value
    config.set_parameter("JSON_PARAM", [1, 2, 3])
    # The actual implementation formats JSON without spaces
    assert os.environ["JSON_PARAM"] == "json:[1, 2, 3]"
    
    # Test that the parameter can be retrieved
    assert config.get_parameter("TEST_PARAM") == "test_value"
    assert config.get_parameter("JSON_PARAM") == [1, 2, 3]

def test_overwrite_from_args():
    """
    Test that overwrite_from_args correctly sets parameters from args.
    """
    # Reset the singleton config object
    config._config = {}
    
    # Create a mock args object
    class Args:
        def __init__(self):
            self.param1 = "value1"
            self.param2 = 42
            self.param3 = None
    
    args = Args()
    
    # Test overwriting from args
    config.overwrite_from_args(args)
    
    # Test that the parameters were set
    assert config.get_parameter("param1") == "value1"
    assert config.get_parameter("param2") == 42
    assert "param3" not in os.environ  # None values should not be set

def test_init_config_with_file():
    """
    Test that _init_config correctly loads a config file.
    """
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        json.dump({"TEST_PARAM": "test_value"}, temp_file)
    
    try:
        # Reset the singleton config object
        config._config = None
        
        # Mock _get_default_path to return our temporary file
        original_get_default_path = config._get_default_path
        config._get_default_path = lambda: temp_file.name
        
        # Initialize the config
        config._init_config()
        
        # Test that the config was loaded
        assert config._config == {"TEST_PARAM": "test_value"}
        
        # Restore the original function
        config._get_default_path = original_get_default_path
    finally:
        # Clean up the temporary file
        os.unlink(temp_file.name)

def test_init_config_empty():
    """
    Test that _init_config handles the case where no config file is found.
    """
    # Reset the singleton config object
    config._config = None
    
    # Mock _get_default_path to return None
    original_get_default_path = config._get_default_path
    config._get_default_path = lambda: None
    
    # Initialize the config
    config._init_config()
    
    # Test that an empty config was created
    assert config._config == {}
    
    # Restore the original function
    config._get_default_path = original_get_default_path
