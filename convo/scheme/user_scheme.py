from pydantic import BaseModel, Field
from typing import Optional

class UserScheme(BaseModel):
    user_location: Optional[str] = Field(None, description="User's current location or departure city")
    destination: Optional[str] = Field(None, description="Destination city or location")
    duration: Optional[str] = Field(None, description="Trip duration (e.g., '6 days', '1 week')")
    interests: Optional[str] = Field(None, description="Comma-separated list of travel interests, e.g. beaches, culture, adventure")
    budget: Optional[str] = Field(None, description="Budget range in USD or descriptive")
    travel_style: Optional[str] = Field(None, description="Travel style preference (e.g., luxury, budget, adventure)")
    want_flight_links: bool = Field(False, description="Whether the user wants flight booking links")