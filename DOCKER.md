# How to run this

## Build the docker container.

Note that because `flash-attn` takes forever to compile, the following command
will take well over an hour. To reduce the wait time, if your machine has a ton
of RAM and cores, set `MAX_JOBS` to a higher number of jobs in the dockerfile.

You may also want to adjust the compute capabilities you're building, to match
your GPUs, also in the dockerfile.

```bash
docker build -t trellis:local .
```

## Run the docker container.

```bash
docker run --gpus all -it --rm trellis:local
```

The easiest way to run would be to launch `app.py` from inside the container:

```bash
python3 app.py --share
```

This should output a Gradio proxy link which you can then access via your
browser. Note that this opens up your container to the Internet.