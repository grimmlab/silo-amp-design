#!/bin/bash

mkdir -p inference_model

curl -L "https://github.com/grimmlab/silo-amp-design/releases/download/model-v1/best_model.pt" \
  -o inference_model/best_model.pt