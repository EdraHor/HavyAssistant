"""
Загрузчик конфигурации из YAML
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict

class Config:
    """Синглтон для работы с конфигурацией"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._config is None:
            self.load()
    
    def load(self, config_path: str = "config/settings.yaml"):
        """Загрузка конфигурации из файла"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Получить значение по пути (например: 'audio.sample_rate')
        
        Args:
            key_path: Путь к значению через точку
            default: Значение по умолчанию
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """Установить значение по пути"""
        keys = key_path.split('.')
        config = self._config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save(self, config_path: str = "config/settings.yaml"):
        """Сохранить конфигурацию в файл"""
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._config, f, allow_unicode=True, default_flow_style=False)
    
    @property
    def all(self) -> Dict:
        """Получить всю конфигурацию"""
        return self._config

# Глобальный экземпляр
config = Config()
