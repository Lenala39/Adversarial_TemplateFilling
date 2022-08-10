#!/bin/bash

# echo ">> COMP train set: gl // class_change"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Attacking_Araujo.py -d gl -v comp -t train -c class_change

# echo ">> COMP train set: gl // class_change"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Attacking_Araujo.py -d dm2 -v comp -t train -c class_change


echo ">> EXTRACTION train set: gl"
/home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Attacking_Araujo.py -d gl -v extraction -t train

echo ">> EXTRACTION test set: gl"
/home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Attacking_Araujo.py -d gl -v extraction -t test 

echo ">> EXTRACTION train set: dm2"
/home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Attacking_Araujo.py -d dm2 -v extraction -t train

echo ">> EXTRACTION test set: dm2"
/home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Attacking_Araujo.py -d dm2 -v extraction -t test 

# echo ">> COMP test set: dm2 // class_change"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Attacking_Araujo.py -d dm2 -v comp -t test -c class_change

# echo ">> COMP test set: dm2 // value_change"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Attacking_Araujo.py -d dm2 -v comp -t test -c value_change 

# echo ">> COMP train set: dm2 // value_change"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Attacking_Araujo.py -d dm2 -v comp -t train -c value_change 

# echo ">> COMP test set: gl // class_change"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Attacking_Araujo.py -d gl -v comp -t test -c class_change

# echo ">> COMP test set: gl // value_change"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Attacking_Araujo.py -d gl -v comp -t test -c value_change

# echo ">> COMP train set: gl // value_change"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Attacking_Araujo.py -d gl -v comp -t train -c value_change


