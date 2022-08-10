#!/bin/bash

# echo ">> COMPUTING CLARE: Augmentation gl extraction test"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Augmentation.py -d gl -v extraction -t test -m CLARE -s 0 -e 20; spd-say "Done CLARE extraction test"

#echo ">> COMPUTING CLARE: Augmentation dm2 comp value change test"
#/home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Augmentation.py -d dm2 -v comp -c value_change -t test -m CLARE -s 0 -e 19; spd-say "Done CLARE value test"

# echo ">> COMPUTING CLARE: Augmentation gl comp class change test"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Augmentation.py -d gl -v comp -c class_change -t test -m CLARE -s 0 -e 20; spd-say "Done CLARE class test"

# echo ">> COMPUTING Araujo: Augmentation gl extraction test"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Augmentation.py -d gl -v extraction -t test -m Araujo -s 0 -e 20; spd-say "Done Araujo extraction test"

# echo ">> COMPUTING Araujo: Augmentation gl comp value change test"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Augmentation.py -d gl -v comp -c value_change -t test -m Araujo -s 0 -e 20; spd-say "Done Araujo value test"

# echo ">> COMPUTING Araujo: Augmentation gl comp class change test"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Augmentation.py -d gl -v comp -c class_change -t test -m Araujo -s 0 -e 20; spd-say "Done Araujo class test"


#echo ">> COMPUTING CLARE: Augmentation dm2 extraction train"
#/home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Augmentation.py -d dm2 -v extraction -t train -m CLARE -s 0 -e 68; spd-say "Done CLARE extraction train"

echo ">> COMPUTING CLARE: Augmentation dm2 comp value change train"
/home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Augmentation.py -d dm2 -v comp -c value_change -t train -m CLARE -s 0 -e 67; spd-say "Done CLARE value train"

#echo ">> COMPUTING CLARE: Augmentation dm2 comp class change train"
#/home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Augmentation.py -d dm2 -v comp -c class_change -t train -m CLARE -s 0 -e 68; spd-say "Done CLARE class train"

#echo ">> COMPUTING Araujo: Augmentation dm2 extraction train"
#/home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Augmentation.py -d dm2 -v extraction -t train -m Araujo -s 0 -e 68; spd-say "Done Araujo extraction train"

#echo ">> COMPUTING Araujo: Augmentation dm2 comp value change test"
#/home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Augmentation.py -d dm2 -v comp -c value_change -t test -m Araujo -s 0 -e 19; spd-say "Done Araujo value train"

#echo ">> COMPUTING Araujo: Augmentation dm2 comp class change train"
#/home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Augmentation.py -d dm2 -v comp -c class_change -t train -m Araujo -s 0 -e 68; spd-say "Done Araujo class train"