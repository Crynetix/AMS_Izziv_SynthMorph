FROM python:3.10


# ############ WORKDIR ############
RUN mkdir /workspace
RUN chmod -R 777 /workspace
WORKDIR /workspace


# ############ ENVIRONMENT ############
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="${PYTHONPATH}:/workspace/modules/L2R"


# ############ INSTALL ############
RUN apt-get update -y \
    && apt-get install -y \
       make build-essential libssl-dev zlib1g-dev libbz2-dev \
       libreadline-dev libsqlite3-dev wget curl llvm \
       libncurses5-dev libncursesw5-dev xz-utils tk-dev \
       libffi-dev liblzma-dev python3-openssl git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# ############ MODULE L2R ############
COPY modules /workspace/modules/
COPY utils /workspace/utils/
COPY requirements.txt /workspace/modules/L2R/evaluation/

COPY surface-distance /workspace/surface-distance/

WORKDIR /workspace/surface-distance
RUN pip install /workspace/surface-distance/

WORKDIR /workspace/modules/L2R/evaluation
RUN pip install -r requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:/workspace/modules/L2R/evaluation:/workspace/utils:/workspace/surface-distance"

# ############ SOURCE ############
COPY data /workspace/data/
COPY utils /workspace/utils


# ############ ENTRYPOINT ############
WORKDIR /workspace/utils
ENTRYPOINT ["bash", "entrypoint.sh"]
