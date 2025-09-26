
docker rmi -f pmldl/api
docker rmi -f pmldl/web
docker build -f ./code/deployment/api/Dockerfile -t pmldl/api .
docker build -f ./code/deployment/app/Dockerfile -t pmldl/web .
