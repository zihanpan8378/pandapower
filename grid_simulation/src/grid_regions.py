from enum import Enum


class GridRegion(Enum):
    """Enum for grid regions."""

    CA_ON = "CA_ON"
    US_CAL_CISO = "US_CAL_CISO"
    US_MIDA_PJM = "US_MIDA_PJM"
    US_TEX_ERCO = "US_TEX_ERCO"    


class RegionGenerationShare():
    

    def __init__(self, region: GridRegion) -> None:
        """
        Initializer for RegionGenerationShare class.

        Args:
            region: The grid region to get generation share information for
        """
        self.region = region


    @property
    def solar_share(self) -> float:
        """Gets the renewable generation share for the region."""
        if self.region == GridRegion.CA_ON:
            return 0.3
        elif self.region == GridRegion.US_CAL_CISO:
            return 0.15
        elif self.region == GridRegion.US_MIDA_PJM:
            return 0.1
        elif self.region == GridRegion.US_TEX_ERCO:
            return 0.1
        else:
            raise ValueError(f"Unknown region {self.region} for generation share.")
    

    @property
    def wind_share(self) -> float:
        """Gets the renewable generation share for the region."""
        if self.region == GridRegion.CA_ON:
            return 0.2
        elif self.region == GridRegion.US_CAL_CISO:
            return 0.15
        elif self.region == GridRegion.US_MIDA_PJM:
            return 0.1
        elif self.region == GridRegion.US_TEX_ERCO:
            return 0.0
        else:
            raise ValueError(f"Unknown region {self.region} for generation share.")
        
