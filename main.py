from fastapi import FastAPI, Body
import logging, coloredlogs
from business import Models, Ai
from contracts.ai_start_training import StartAiTrainingRequest
import os
import uvicorn
from dapr.ext.fastapi.app import DaprApp

from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace

configure_azure_monitor(
    connection_string="InstrumentationKey=516bfb02-4bb3-46a2-9e8e-19326f097eb5;IngestionEndpoint=https://westeurope-5.in.applicationinsights.azure.com/;LiveEndpoint=https://westeurope.livediagnostics.monitor.azure.com/"
)

#logging.config.fileConfig('logging.conf', disable_existing_loggers=False)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
coloredlogs.install("INFO", logger = logger)

tracer = trace.get_tracer(__name__)


app = FastAPI()
dapr_app = DaprApp(app)

models = Models()
ai = Ai()

@app.get('/')
async def root():
    return {'message' : 'Hello world'}

@dapr_app.subscribe(pubsub='pubsub', topic='ai.listmodels')
async def listmodels(event_data = Body()):
    with tracer.start_as_current_span('span_list_models'):
        logger.debug(f'Start Listing models')
        logger.debug(f'Request data {event_data}')
        logger.debug(f'current working directory is : {os.getcwd()}')
        all_files = models.list_models()
        return {
            'models' : all_files
        }
    
@dapr_app.subscribe(pubsub='pubsub', topic='ai.start.training')
async def start_training_eventhandler(event_data: StartAiTrainingRequest):
    with tracer.start_as_current_span('span_start_training'):
        logger.debug(event_data)
        logger.debug(f'Start ai training with id {event_data.id}')
        ai.start_training(event_data.id)
        logger.debug(f'Finished ai training with id {event_data.id}')
    
    
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)