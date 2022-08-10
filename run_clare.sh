#!/bin/bash
echo ">> CLARE"
#echo ">> EXTRACTION train set: dm2"
#/home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Attacking_CLARE.py -d dm2 -v comp -c class_change -t train -s 0 -e 0
echo ">> EXTRACTION train set: gl"
/home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Attacking_CLARE_improved.py -d gl -v extraction -t train -s 0 -e 67
#echo ">> EXTRACTION train set: gl"
#/home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Attacking_CLARE.py -d gl -v comp -c class_change -t train -s 0 -e 0
#echo ">> EXTRACTION test set: gl"
#/home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Attacking_CLARE.py -d gl -v comp -c class_change -t test -s 0 -e 0
