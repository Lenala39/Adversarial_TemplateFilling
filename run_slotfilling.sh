#!/bin/bash

echo ">> COMPUTING SlotFillingComp augmentation gl araujo train, comp, value"
/home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/SlotFillingCompModule.py -d gl -v comp -c value_change -m Araujo -t train -s 0 -e 68

# echo ">> COMPUTING SlotFillingComp augmentation gl clare test, comp, value"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/SlotFillingCompModule.py -d gl -v comp -c value_change -m CLARE -t test -s 0 -e 20

# echo ">> COMPUTING SlotFillingComp augmentation gl clare train, comp"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/SlotFillingCompModule.py -d gl -v comp -c class_change -m CLARE -t train -s 0 -e 68

# echo ">> COMPUTING SlotFillingComp augmentation gl clare test, comp"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/SlotFillingCompModule.py -d gl -v comp -c value_change -m CLARE -t test -s 0 -e 20

# echo ">> COMPUTING SlotFillingComp augmentation gl clare train, extraction"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/SlotFillingCompModule.py -d gl -v extraction -m CLARE -t train -s 0 -e 68

# echo ">> COMPUTING SlotFillingComp augmentation gl clare test, extraction"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/SlotFillingCompModule.py -d gl -v extraction -m CLARE -t test -s 0 -e 20

