# Low Reynolds Number Airfoil Optimization

## Running the Optimizaiton

### Dependencies

To run the code the following packages must be installed:
- NumPy
- SciPy
- Matplotlib
- pyGeo
Or simply use the Docker image described in [Docker Image](#docker-image). 

### Running

Python's argparser is used to allow optimizer selection when calling the script from the command line. The options for optimizer are: SLSQP, penalty method using BFGS, and penalty method using Nelder-Mead for DFO optimziation. To run the optimization, run the following command: 
```
python3 main.py -opt <optimizer>
```

Here, optimizer can be:
- `slsqp`
- `penalty_grad`
- `penalty_dfo`

## Docker Image

1. Install Docker: <br>
    Install Docker using the instructions provided [here](https://www.docker.com/).
1. Pull latest CMPLXFOIL image in a terminal/command line prompt: <br> 
    `docker pull emartin3/aersp497cmplxfoil:latest`
1. Run the image: <br>
    `docker run -it --rm --mount "type=bind,src=$(pwd),target=/home/mdolabuser/mount/" emartin3/aersp497cmplxfoil:latest /bin/bash`
1. Enter mount directory: <br>
    `cd mount`

### Additional Libraries
If additional libraries or applications need to be added, install/build them within the container and commit these changes to your personal machine using: `docker commit <container-id> aersp497cmplxfoil`. Any changes made to the image are saved to the image and can be accessed as previously noted.

When commiting the changes made to the container, the container must be actively running in the background. Either open a new terminal/command line window or remove the `--rm` tag from the build command. (Note: removing `--rm` is not recommended.) 

### Useful Docker Commands & Flags
`docker build -t <image-name> .` <br>
Builds Docker image from Dockerfile in current working directory.
* `-t` 
    Name and (optionally) tag.

`docker pull <docker-image>:<tag>` <br>
Pulls image from DockerHub with version specified as `<tag>`.

`docker run <image-name> /bin/bash` <br>
Builds image from existing container on DockerHub.
* `-it`
    Start image with an interactive terminal.
* `--rm` 
    Automatically remove and shut down container when the image exits.
* `--mount "type=<mount-type>,src=<local-directory>,target=<mount-location>"`
    Mounts a given directory and its subdirectories to the Docker image at the location specified by `<mount-location>`.

`docker commit <container-id> <image-name>` <br>
Preserves changes made to a given Docker image. Note `<container-image>` an alphanumeric code found after the image username, e.g. `mdolabuser@041626d11c0b`.

`docker tag <image-name> <repo-name>:<tag>` <br>
Tags a Docker image preparing it to be pushed onto a repository. If `<tag>` is ommitted, Docker defaults to `latest`.

`docker push <repo-name>` <br>
Pushes the tagged docker image to Docker repository.