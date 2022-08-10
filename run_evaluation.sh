#!/bin/bash


# echo ">> COMPUTING CLARE: Evaluation gl extraction"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Evaluation.py -d gl -v extraction -t test -m CLARE; spd-say "Done Clare extraction glaucoma"

# echo ">> COMPUTING CLARE: Evaluation dm2 extraction"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Evaluation.py -d dm2 -v extraction -t test -m CLARE; spd-say "Done Clare extraction diabetes"

# echo ">> COMPUTING Araujo: Evaluation dm2 extraction"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Evaluation.py -d dm2 -v extraction -t test -m Araujo; spd-say "Done Araujo extraction diabetes"

# echo ">> COMPUTING Araujo: Evaluation gl extraction"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Evaluation.py -d gl -v extraction -t test -m Araujo; spd-say "Done Araujo extraction glaucoma"

# echo ">> COMPUTING CLARE: Evaluation gl comp value change"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Evaluation.py -d gl -v comp -c value_change -t test -m CLARE; spd-say "Done Clare value glaucoma"

echo ">> COMPUTING CLARE: Evaluation dm2 comp value change"
/home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Evaluation.py -d dm2 -v comp -c value_change -t test -m CLARE; spd-say "Done Clare value diabetes"

# echo ">> COMPUTING CLARE: Evaluation gl comp class change"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Evaluation.py -d gl -v comp -c class_change -t test -m CLARE; spd-say "Done Clare class glaucoma"

# echo ">> COMPUTING CLARE: Evaluation dm2 comp class change"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Evaluation.py -d dm2 -v comp -c class_change -t test -m CLARE; spd-say "Done Clare class diabetes"

# echo ">> COMPUTING Araujo: Evaluation gl comp value change"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Evaluation.py -d gl -v comp -c value_change -t test -m Araujo; spd-say "Done Araujo value glaucoma"

# echo ">> COMPUTING Araujo: Evaluation dm2 comp value change"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Evaluation.py -d dm2 -v comp -c value_change -t test -m Araujo; spd-say "Done Araujo value diabetes"

# echo ">> COMPUTING Araujo: Evaluation gl comp class change"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Evaluation.py -d gl -v comp -c class_change -t test -m Araujo; spd-say "Done Araujo class glaucoma"

# echo ">> COMPUTING Araujo: Evaluation dm2 comp class change"
# /home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Evaluation.py -d dm2 -v comp -c class_change -t test -m Araujo; spd-say "Done Araujo class diabetes"

