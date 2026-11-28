#!/bin/bash

pixi run -e adair run-adair --input_path "../../datasets/sanity_test/input/" --output_path "../../datasets/sanity_test/output/adair/"

pixi run -e dod run-dod --input_path "../../datasets/sanity_test/input/" --output_path "../../datasets/sanity_test/output/dod/"

pixi run -e moceir run-moceir --input_path "../../datasets/sanity_test/input/" --output_path "../../datasets/sanity_test/output/moceir/"

pixi run -e promptir run-promptir --input_path "../../datasets/sanity_test/input/" --output_path "../../datasets/sanity_test/output/promptir/"

pixi run -e r2r run-r2r --input_path "../../datasets/sanity_test/input/" --output_path "../../datasets/sanity_test/output/r2r/"