# Code to deploy our system

Run with docker to use CPU and open on *localhost:5000*
```bash
$ docker build --tag imago --file Dockerfile_deploy --pull .

$ docker run --rm --publish 5000:5000 imag

$ xdg-open http://localhost:5000/
```

