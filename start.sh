#export COLOREDLOGS_LOG_FORMAT='%(asctime)s %(levelname)-6s %(name)s %(funcName)s() L%(lineno)-4d %(message)s'
#uvicorn main:app --reload     

docker run -ti  --name fastapiprocessor \
               -v /Users/nickyeichmann/Documents/GitHub/EnergyMonitorAI/files:/app/files \
               -v /Users/nickyeichmann/Documents/GitHub/EnergyMonitorAI/models:/app/models \
               -p 8080:80 \
               --rm \
                  fastapiprocessor:latest