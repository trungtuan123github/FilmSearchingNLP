from typing import Optional, List, Union
from pydantic import BaseModel, HttpUrl, validator, Field
from pydantic import BaseModel, Field
import uuid

class FilmMetadata(BaseModel):
    film_name: str
    image_link: Optional[HttpUrl]
    is_adult: Optional[int]
    start_year: Optional[int]
    runtime_minutes: Optional[int]
    genres: Optional[str]
    rating: Optional[float]
    votes: Optional[int]
    directors: Optional[str]
    writers: Optional[str]

class RawFilm(BaseModel):
    _id: Optional[str] = None
    id: str
    film_name: Union[str, int, float]
    description: str
    image_link: Optional[HttpUrl] = None
    isAdult: Optional[int] = None
    startYear: Optional[int] = None
    runtimeMinutes: Optional[int] = None
    genres: Optional[str] = None
    rating: Optional[float] = None
    votes: Optional[int] = None
    directors: Optional[str] = None
    writers: Optional[str] = None

    @validator('film_name', pre=True)
    def ensure_str_film_name(cls, v):
        if v is None or str(v).strip() == "":
            raise ValueError("film_name is required and cannot be empty")
        return str(v)

class CleanFilm(BaseModel):
    id: str
    original_description: str
    cleaned_description: Union[str, List[str]]
    metadata: FilmMetadata


class QdrantPoint(BaseModel):
    id: Optional[Union[int, str]] = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    metadata: FilmMetadata
    vector: List[float]


__all__ = [RawFilm, CleanFilm, FilmMetadata, QdrantPoint]