
docker rmi pmldl/api
docker rmi pmldl/web
docker build -f ./code/deployment/api/Dockerfile -t pmldl/api .
docker build -f ./code/deployment/app/Dockerfile -t pmldl/web .
