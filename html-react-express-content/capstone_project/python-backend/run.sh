#!/bin/bash

source japanlp/bin/activate
python -m uvicorn main:app --reload --port 8000