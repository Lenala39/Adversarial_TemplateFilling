#!/bin/bash



#echo ">> COMPUTING CLARE: Evaluation gl comp value change"
#/home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Evaluation.py -d gl -v comp -c value_change -t train -m CLARE; spd-say "Done 1"
# success (robust, normal data, load_slot_indices=False)
# fail (robust, augmented data, load_slot_indices=False)
# success (robust, augmented data, load_slot_indices=True)
# success (robust, attacked data, load_slot_indices=True)


#echo ">> COMPUTING CLARE: Evaluation gl comp class change"
#/home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Evaluation.py -d gl -v comp -c class_change -t train -m CLARE; spd-say "Done 2"
# success (robust, normal data)
# fail (robust, augmented data, load_slot_indices=False)
# success (robust, augmented data, load_slot_indices=True)
# success (robust, attacked data, load_slot_indices=True)

#echo ">> COMPUTING CLARE: Evaluation gl extraction"
#/home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Evaluation.py -d gl -v extraction -t train -m CLARE; spd-say "Done Clare extraction"
# fail (robust, normal data, load_slot_indices=False)
# fail (robust, normal data, load_slot_indices=True)
# fail (robust, normal data, load_slot_indices=True, recompute_batches=True)
# fail (robust, normal data, load_slot_indices=False, recompute_batches=True)
# success (robust, augmented data, load_slot_indices=False)
# fail (robust, attacked data, load_slot_indices=True, recompute_batches=False)
# fail (robust, attacked data, load_slot_indices=False, recompute_batches=True)
# fail (robust, attacked data, load_slot_indices=False, recompute_batches=True)
# fail (robust, attacked data, load_slot_indices=True, recompute_batches=True)

#echo ">> COMPUTING Araujo: Evaluation gl extraction"
#/home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Evaluation.py -d gl -v extraction -t train -m Araujo; spd-say "Done Araujo extraction"
# fail (robust, normal data, load_slot_indices=False, recompute_batches=False)
# fail (robust, normal data, load_slot_indices=True, recompute_batches=False)
# fail (robust, normal data, load_slot_indices=True, recompute_batches=True)
# fail (robust, normal data, load_slot_indices=False, recompute_batches=True)
# success (robust, augmented data, load_slot_indices=True)
# fail (robust, attacked data, load_slot_indices=True, recompute_batches=False)
# fail (robust, attacked data, load_slot_indices=False, recompute_batches=False)
# fail (robust, attacked data, load_slot_indices=False, recompute_batches=True)
# fail (robust, attacked data, load_slot_indices=True, recompute_batches=True)

#echo ">> COMPUTING Araujo: Evaluation gl comp value change"
#/home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Evaluation.py -d gl -v comp -c value_change -t train -m Araujo; spd-say "Done Araujo value"
# fail (robust, normal data, load_slot_indices=False, recompute_batches=False)
# fail (robust, normal data, load_slot_indices=True, recompute_batches=False)
# fail (robust, normal data, load_slot_indices=True, recompute_batches=True)
# fail (robust, normal data, load_slot_indices=False, recompute_batches=True)
# success (robust, augmented data, load_slot_indices=False)
# fail (robust, attacked data, load_slot_indices=True, recompute_batches=False)
# fail (robust, attacked data, load_slot_indices=False, recompute_batches=False)
# fail (robust, attacked data, load_slot_indices=False, recompute_batches=True)
# fail (robust, attacked data, load_slot_indices=True, recompute_batches=True)

#echo ">> COMPUTING Araujo: Evaluation gl comp class change"
#/home/lena/TemplateFilling/tf_venv/bin/python /home/lena/TemplateFilling/Evaluation.py -d gl -v comp -c class_change -t train -m Araujo; spd-say "Done Araujo class"
# fail (robust, normal data, load_slot_indices=False, recompute_batches=False)
# fail (robust, normal data, load_slot_indices=True, recompute_batches=False)
# fail (robust, normal data, load_slot_indices=True, recompute_batches=True)
# fail (robust, normal data, load_slot_indices=False, recompute_batches=True)
# success (robust, augmented data, load_slot_indices=False)
# fail (robust, attacked data, load_slot_indices=True, recompute_batches=False)
# fail (robust, attacked data, load_slot_indices=False, recompute_batches=False)
# fail (robust, attacked data, load_slot_indices=False, recompute_batches=True)
# fail (robust, attacked data, load_slot_indices=True, recompute_batches=True)


