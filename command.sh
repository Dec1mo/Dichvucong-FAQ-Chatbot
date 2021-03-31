#!/bin/bash

rasa run actions -p 5055 &&
rasa run --model ./models --endpoints ./endpoints.yml --credentials ./credentials.yml -p 5005