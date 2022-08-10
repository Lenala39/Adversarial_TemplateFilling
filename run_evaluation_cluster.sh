#!/bin/bash


echo ">> COMPUTING CLARE: Evaluation gl extraction"
python3 /media/compute/homes/llangholf/TemplateFilling/Evaluation.py -d gl -v extraction -t test -m CLARE

echo ">> COMPUTING CLARE: Evaluation dm2 extraction"
python3 /media/compute/homes/llangholf/TemplateFilling/Evaluation.py -d dm2 -v extraction -t test -m CLARE

echo ">> COMPUTING CLARE: Evaluation gl comp value change"
python3 /media/compute/homes/llangholf/TemplateFilling/Evaluation.py -d gl -v comp -c value_change -t test -m CLARE

echo ">> COMPUTING CLARE: Evaluation dm2 comp value change"
python3 /media/compute/homes/llangholf/TemplateFilling/Evaluation.py -d dm2 -v comp -c value_change -t test -m CLARE

echo ">> COMPUTING CLARE: Evaluation gl comp class change"
python3 /media/compute/homes/llangholf/TemplateFilling/Evaluation.py -d gl -v comp -c class_change -t test -m CLARE

echo ">> COMPUTING CLARE: Evaluation dm2 comp class change"
python3 /media/compute/homes/llangholf/TemplateFilling/Evaluation.py -d dm2 -v comp -c class_change -t test -m CLARE

echo ">> COMPUTING Araujo: Evaluation gl extraction"
python3 /media/compute/homes/llangholf/TemplateFilling/Evaluation.py -d gl -v extraction -t test -m Araujo

echo ">> COMPUTING Araujo: Evaluation dm2 extraction"
python3 /media/compute/homes/llangholf/TemplateFilling/Evaluation.py -d dm2 -v extraction -t test -m Araujo

echo ">> COMPUTING Araujo: Evaluation gl comp value change"
python3 /media/compute/homes/llangholf/TemplateFilling/Evaluation.py -d gl -v comp -c value_change -t test -m Araujo

echo ">> COMPUTING Araujo: Evaluation dm2 comp value change"
python3 /media/compute/homes/llangholf/TemplateFilling/Evaluation.py -d dm2 -v comp -c value_change -t test -m Araujo

echo ">> COMPUTING Araujo: Evaluation gl comp class change"
python3 /media/compute/homes/llangholf/TemplateFilling/Evaluation.py -d gl -v comp -c class_change -t test -m Araujo

echo ">> COMPUTING Araujo: Evaluation dm2 comp class change"
python3 /media/compute/homes/llangholf/TemplateFilling/Evaluation.py -d dm2 -v comp -c class_change -t test -m Araujo

