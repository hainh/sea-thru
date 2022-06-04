# Sea-Thru
Implementation of Sea-thru by Derya Akkaynak and Tali Treibitz

__Forked from https://github.com/hainh/sea-thru__

This fork's only aim is to standardize the dependencies and environment with [Docker](https://docs.docker.com/get-docker/) and [Anaconda](https://www.anaconda.com/) to make it so that anyone can use it without deep technical knowledge of python.

Additionally, in order to make this an all-in-one package, the [Monodepth2 submodule](https://github.com/nianticlabs/monodepth2/tree/b676244e5a1ca55564eb5d16ab521a48f823af31) has been cloned and commited directly into this repo, preserving the commit history and original authors.

## Prerequisites
---

[Docker](https://docs.docker.com/get-docker/)

## Setup
---
1. Run `docker compose up --build`
    - This can take around 2 hours the first time; from then onwards you only need to run `docker compose up` to get the container running and it'll boot up instantly.
2. Once the container is built and running, in a separate terminal run `docker exec -it seathru /bin/bash`
    - Replace `seathru` in that command if you've modified the `SERVICE` enviornment variable in `.env`
3. From within the container you can now run `python seathru-mono-e2e.py --image ${PATH_TO_IMAGE}`
    - This root directory is mounted into the container each time you bring it up via `docker compose up`, meaning any files you add on your host machine to this directory will _also_ be available within the container.

## Description
---

A recent advance in underwater imaging is the Sea-Thru method, which uses a physical model of light attenuation to reconstruct
the colors in an underwater scene. This method utilizes a known
range map to estimate backscatter and wideband attenuation
coefficients. This range map is generated using structure-from-motion (SFM), which requires multiple underwater images from various perspectives and long processing time. In addition, SFM gives very accurate results, which are generally not required for this method. In this work, we implement and extend Sea-Thru to take advantage of convolutional monocular depth estimation methods, specifically the Monodepth2 network. We obtain satisfactory results with the lower-quality depth estimates with some color inconsistencies using only one image.
