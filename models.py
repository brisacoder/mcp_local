from enum import Enum
from typing import List, Literal
from pydantic import BaseModel, Field


class WeatherDay(BaseModel):
    """A model to represent weather data for a single day."""

    temperature: float = Field(description="The weather temperature")
    condition: str = Field(
        description="The weather condition, e.g., sunny, rainy, etc."
    )
    weekday: str = Field(description="The day of the week, e.g., Monday, Tuesday, etc.")


class Weather(BaseModel):
    """A model to represent a list of weather days."""

    days: List[WeatherDay] = Field(
        description="A list of weather data for multiple days"
    )
    error: str = Field(
        default="",
        description="An optional error message if weather data could not be retrieved.",
    )


class FlightOption(BaseModel):
    """A model to represent a single flight option from A to B."""

    airline: str = Field(description="The airline operating the flight")
    flight_number: str = Field(description="The flight number")
    departure_city: str = Field(description="The departure city")
    arrival_city: str = Field(description="The arrival city")
    departure_time: str = Field(
        description="The departure time (e.g., 2024-06-01T08:00)"
    )
    arrival_time: str = Field(description="The arrival time (e.g., 2024-06-01T16:00)")
    duration: str = Field(description="The total flight duration (e.g., 8h 0m)")
    price: float = Field(description="The price of the flight in USD")
    stops: int = Field(description="The number of stops (0 for direct flights)")


class Flights(BaseModel):
    """A model to represent a list of flight options from A to B."""

    flights: List[FlightOption] = Field(
        description="A list of available flight options from the departure city to the arrival city"
    )


class Topic(str, Enum):
    WEATHER = "Weather"
    AIR_TRAVEL = "AirTravel"
    OTHER = "Other"


class ConversationTopic(BaseModel):
    topic: Topic = Field(
        description="The selected conversation topic. Must be one of: Weather, AirTravel, Other."
    )
