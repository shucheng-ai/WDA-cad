echo "building cad"
cd docker
if [ "$1" == "en" ]
then
  echo "build en docker:"
  docker build -f Dockerfile.en -t cyborg/cad:0.0 . --no-cache
else
  echo "build cn docker:"
  docker build -f Dockerfile.cn -t cyborg/cad:0.0 . --no-cache
fi
docker tag cyborg/cad:0.0 cyborg/cad:latest
