from importlib import import_module
from typing import Any, TypeVar, type_check_only
from PIL import Image
import io
from pydantic import BaseModel


# Used only for type checking the pipeline class
TPipeline = TypeVar("TPipeline", bound=type[Any])


class ParamsModel(BaseModel):
    """Base model for pipeline parameters."""
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ParamsModel':
        """Create a model instance from dictionary data."""
        return cls.model_validate(data)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return self.model_dump()


def get_pipeline_class(pipeline_name: str) -> type:
    """
    Dynamically imports and returns the Pipeline class from a specified module.
    
    Args:
        pipeline_name: The name of the pipeline module to import
        
    Returns:
        The Pipeline class from the specified module
        
    Raises:
        ValueError: If the module or Pipeline class isn't found
        TypeError: If Pipeline is not a class
    """
    try:
        module = import_module(f"pipelines.{pipeline_name}")
    except ModuleNotFoundError:
        raise ValueError(f"Pipeline {pipeline_name} module not found")

    pipeline_class = getattr(module, "Pipeline", None)

    if pipeline_class is None:
        raise ValueError(f"'Pipeline' class not found in module '{pipeline_name}'.")

    # Type check to ensure we're returning a class
    if not isinstance(pipeline_class, type):
        raise TypeError(f"'Pipeline' in module '{pipeline_name}' is not a class")

    return pipeline_class


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))
    return image


def pil_to_frame(image: Image.Image) -> bytes:
    frame_data = io.BytesIO()
    image.save(frame_data, format="JPEG", quality=80, optimize=True, progressive=True)
    frame_data = frame_data.getvalue()
    return (
        b"--frame\r\n"
        + b"Content-Type: image/jpeg\r\n"
        + f"Content-Length: {len(frame_data)}\r\n\r\n".encode()
        + frame_data
        + b"\r\n"
    )


def is_firefox(user_agent: str) -> bool:
    return "Firefox" in user_agent
