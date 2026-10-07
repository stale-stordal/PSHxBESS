"""
Created Spring 2026

@author: einar-m, matheaab, odmoe
"""


class Storage:
    """Superclass for energy storage systems."""

    def __init__(self,
                 capacity: float,
                 power: float,
                 efficiency: float,
                 initial_soc_frac: float):
        """
        Args:
            capacity (float): Storage capacity [MWh]
            power (float): Maximum charge/discharge power [MW]
            efficiency (float): Power efficiency [-]
            initial_soc_frac (float): Initial state of charge as fraction of capacity [-]
        """
        self.capacity    = capacity
        self.power       = power
        self.efficiency  = efficiency
        self.initial_soc = initial_soc_frac * capacity


class Battery(Storage):
    """Battery Energy Storage System (BESS)."""

    def __init__(self,
                 capacity: float = 10,
                 power: float = 4.6,
                 efficiency: float = 0.95,
                 initial_soc_frac: float = 0.5):
        super().__init__(capacity, power, efficiency, initial_soc_frac)


class Pump(Storage):
    """Pumped Storage Hydropower (PSH)."""

    def __init__(self,
                 capacity: float = 50000,
                 power: float = 100,
                 efficiency: float = 0.85,
                 initial_soc_frac: float = 0.5):
        super().__init__(capacity, power, efficiency, initial_soc_frac)
