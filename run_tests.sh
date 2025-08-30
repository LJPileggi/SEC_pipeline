#!/bin/bash

# TODO: add other tests and implement correctness check
embed_test="./tests/test_embeddings.py"

echo "Running test for embedding generation"

python3 "$embed_test"

# echo "Passed."
