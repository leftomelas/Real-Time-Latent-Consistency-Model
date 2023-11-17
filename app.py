from fastapi import FastAPI

from config import args
from device import device, torch_dtype
from app_init import init_app
from user_queue import user_data
from util import get_pipeline_class


app = FastAPI()

pipeline_class = get_pipeline_class(args.pipeline)
pipeline = pipeline_class(args, device, torch_dtype)
init_app(app, user_data, args, pipeline)
