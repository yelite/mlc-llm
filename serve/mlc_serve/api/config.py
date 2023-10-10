from dataclasses import dataclass

@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: str = 80
