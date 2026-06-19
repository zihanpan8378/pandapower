from enum import Enum


class GenerationType(Enum):
    SOLAR = "solar"
    WIND = "wind"
    COAL = "coal"


class GenerationTypeCost():
    """
    Class to encapsulate cost information for different generation types. 
    """

    def __init__(self, source_type: GenerationType) -> None:
        """
        Initializer for GenerationTypeCost class.

        Args:
            source_type: The type of generation source to get cost information for
        """
        self.source_type = source_type

    @property
    def base_cost(self) -> float:
        """
        Gets the base cost for the generation type.

        Returns:
            The base cost for the generation type
        """
        if self.source_type == GenerationType.COAL:
            return 100_000_000.0
        else:
            return 100.0

    @property
    def linear_cost(self) -> float:
        """
        Gets the linear cost for the generation type.

        Returns:
            The linear cost for the generation type
        """
        if self.source_type == GenerationType.COAL:
            return 10.0
        else:
            return 2.0

    @property
    def quadratic_cost(self) -> float:
        """
        Gets the quadratic cost for the generation type.

        Returns:
            The quadratic cost for the generation type
        """
        return 0.0

    
    @property
    def cubic_cost(self) -> float:
        """
        Gets the cubic cost for the generation type.

        Returns:
            The cubic cost for the generation type
        """
        return 0.0

