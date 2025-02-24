#!/bin/bash

ALLOW_AUTO_PYTHON_VERSION=1
ALLOW_AUTO_CUDA_VERSION=1

DEFAULT_PYTHON_VERSION="3.12"
DEFAULT_CUDA_VERSION="12.4"

# Set response to true if "-y" flag is passed
if [ "$1" == "-y" ]; then
  response=true
elif [ "$1" == "-n" ]; then
  response=false
fi

# check that environment name is set
if [ -z "$ENV_NAME" ]; then
  echo "ENV_NAME is not set"
  exit 1
fi

# check that python version is set
if [ -z "$PYTHON_VERSION" ]; then
  echo "PYTHON_VERSION is not set"
  if [ "$ALLOW_AUTO_PYTHON_VERSION" -eq 1 ]; then
    echo "Using default Python version $DEFAULT_PYTHON_VERSION"
    PYTHON_VERSION=$DEFAULT_PYTHON_VERSION
  else
    exit 1
  fi
fi

# check that cuda version is set
if [ -z "$CUDA_VERSION" ]; then
  echo "CUDA_VERSION is not set"

  if [ "$ALLOW_AUTO_CUDA_VERSION" -eq 1 ]; then
    echo "Using default CUDA version $DEFAULT_CUDA_VERSION"
    CUDA_VERSION=$DEFAULT_CUDA_VERSION
  else
    exit 1
  fi
fi

# default template path
TEMPLATE_PATH="environment.template.yml"

echo "Creating environment.yml `$ENV_NAME` with Python $PYTHON_VERSION and CUDA $CUDA_VERSION"

# replace placeholders in template
sed -e "s/{ENV_NAME}/$ENV_NAME/g" \
    -e "s/{PYTHON_VERSION}/$PYTHON_VERSION/g" \
    -e "s/{CUDA_VERSION}/$CUDA_VERSION/g" \
    $TEMPLATE_PATH > environment.yml

# prompt the user to confirm the environment creation
echo "Environment file created. Do you want to create the environment now? ([y]/n)"

# check if `response` is set
if [ -z "$response" ]; then
  read -r response
fi

if [[ $response =~ ^([nN][oO]|[nN])$ ]]; then
  echo "Environment not created or updated"
  exit 0
fi

# check if the environment already exists
if conda env list | grep -q "$ENV_NAME"; then
  echo "Environment $ENV_NAME already exists. Updating environment"
  conda env update -f environment.yml
  exit 0
fi

# create the environment
echo "Creating conda environment $ENV_NAME"
conda env create -f environment.yml
