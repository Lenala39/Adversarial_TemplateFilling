import pickle,sys
import numpy as np
from ctro import *
import re 


def compute_multi_groups_micro_stats(stats):
    statistics = F1Statistics()
      
    for group_name in ontology.multi_group_names:
        for slot_name in ontology.group_slots[group_name]:
            if slot_name not in stats.statistics_dict:
                continue
    
            statistics.num_occurences += stats.statistics_dict[slot_name].num_occurences
            statistics.true_positives += stats.statistics_dict[slot_name].true_positives
            statistics.false_positives += stats.statistics_dict[slot_name].false_positives
            
    return statistics

def compute_micro_stats(stats_dict):
    stat_result = F1Statistics()

    for class_name in stats_dict:
        if class_name == 'type':
            continue

        stat_result.true_positives += stats_dict[class_name].true_positives
        stat_result.false_positives += stats_dict[class_name].false_positives
        stat_result.num_occurences += stats_dict[class_name].num_occurences

    return stat_result

def print_and_append(string, print_log):
    print(string)
    print_log.append(string)
    return print_log

def print_event_extraction_stats(stats_dict, used_labels, filename=None):
    labels = sorted(stats_dict.keys())
    if "dm2" in filename:
        disease = "Diabetes"
    else:
        disease = "Glaucoma"

    print_log = []
    print_log = print_and_append(f"\section{{Baseline {disease} - Event Extraction}}", print_log=print_log)
    print_log = print_and_append("\\begin{longtable}{ l c c c c}", print_log=print_log)

    print_log = print_and_append(" & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} & \\textbf{\# occurences} \\\\ \\cline{1-5}", print_log=print_log)

    for label in labels:
        if label not in used_labels:
            continue
        
        print_log = print_and_append(f"{label.replace('has', '')} & {round(stats_dict[label].precision(),4)} & {round(stats_dict[label].recall(),4)} & {round(stats_dict[label].recall(),4)} & {int(stats_dict[label].num_occurences)}\\\\", print_log=print_log)
        

    # micro average
    mirco_stats = compute_micro_stats(stats_dict)
    # print_log = print_and_append('\\textbf{micro average:}' + " & \\textbf{{:2.4f}} & \\textbf{{:2.4f}} & \\textbf{{:2.4f}} \\\\".format, print_log=print_log(
    #     mirco_stats.precision(), mirco_stats.recall(), mirco_stats.f1()))
    print_log = print_and_append(f"\\textbf{{micro\_average}}: & {round(mirco_stats.precision(),4)} & {round(mirco_stats.recall(),4)} & {round(mirco_stats.f1(),4)} & {int(mirco_stats.num_occurences)} ", print_log=print_log)
    print_log = print_and_append(f"\label{{tab:{disease}_eventextr}}", print_log=print_log)
    print_log = print_and_append("\end{longtable}", print_log=print_log)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename_full = os.path.join(dir_path, "attack_logs/", "Evaluation_Results/", filename)
    print(f"Saving data to {filename_full}")
    with open(filename_full, "w") as outfile:
        for line in print_log:
            outfile.write(line + "\n")

def print_slot_prediction_stats(stats_list, stat_names=["random", "intra comp", "intra comp"], filename=None):
    print_log = []
    if "dm2" in filename:
        disease = "Diabetes"
    else:
        disease = "Glaucoma"
        
    print_log = []
    print_log = print_and_append(f"\section{{Baseline {disease} - Slot Filling}}", print_log=print_log)
    print_log = print_and_append("\\begin{longtable}{ l c c c c}", print_log=print_log)

    print_log = print_and_append(f"& {stat_names[0]} & {stat_names[1]} & {stat_names[2]} & \#num occurences\\\\", print_log=print_log)
    for group_name in ontology.used_group_names:
        print_log = print_and_append('\hline', print_log=print_log)
        print_log = print_and_append('\multicolumn{4}{c}{' + group_name  + '} \\\\', print_log=print_log)
        
        for slot_name in ontology.group_slots[group_name]:
            slot_string = slot_name
            mean_list = []
            std_list = []
            occurrence_list = []
            if slot_name not in stats_list[0][0].statistics_dict:
                continue
            
            for model_index, model_stats in enumerate(stats_list):
                f1_list = []
                
                for stat in stats_list[model_index]:
                    f1_list.append(stat.statistics_dict[slot_name].f1())
                
                f1_array = np.array(f1_list)
                f1_mean = np.mean(f1_array)
                f1_std = np.std(f1_array)
                
                mean_list.append(f1_mean)
                std_list.append(f1_std)
                occurrence_list.append(int(stat.statistics_dict[slot_name].num_occurences))
            for model_index, model_stats in enumerate(stats_list):
                slot_string += ' & '
                argmax = np.argmax(mean_list)
                
                f1_mean = mean_list[model_index]
                f1_std = std_list[model_index]
                
                if model_index == argmax:
                    slot_string += "$\mathbf{{{:2.4f}}} \pm \mathbf{{{:2.4f}}}$".format(f1_mean, f1_std)
                else:
                    slot_string += "${:2.4f} \pm {:2.4f}$".format(f1_mean, f1_std)
            
            slot_string += f' & {occurrence_list[-1]}\\\\'
            print_log = print_and_append(slot_string, print_log=print_log)
            
            
    # micro stats multi groups #######################################
    print_log = print_and_append('\hline\hline', print_log=print_log)
    slot_string = 'micro average multi templates'
    
    mean_list = []
    std_list = []
    
    for model_index, model_stats in enumerate(stats_list):
        f1_list = []
                
        for stat in stats_list[model_index]:
                f1_list.append(compute_multi_groups_micro_stats(stat).f1())
                
        f1_array = np.array(f1_list)
        f1_mean = np.mean(f1_array)
        f1_std = np.std(f1_array)
                
        mean_list.append(f1_mean)
        std_list.append(f1_std)
                
    argmax = np.argmax(mean_list)
    for model_index,model_stats in enumerate(stats_list):
        slot_string += ' & '
        
        f1_mean = mean_list[model_index]
        f1_std = std_list[model_index]
        
        if model_index == argmax:
            slot_string += f"$\mathbf{{{round(f1_mean, 4)}}} \pm \mathbf{{{round(f1_std,4)}}}$"
        else:
            slot_string += f"${round(f1_mean, 4)} \pm {round(f1_std, 4)}$ "

    slot_string += "\\\\"
    print_log = print_and_append(slot_string, print_log=print_log)
                    
    # micro stats complete ###########################################################
    slot_string = 'micro average complete'
    
    mean_list = []
    std_list = []
    
    for model_index,model_stats in enumerate(stats_list):
        f1_list = []
                
        for stat in stats_list[model_index]:
            f1_list.append(stat.get_micro_stats().f1())
                
        f1_array = np.array(f1_list)
        f1_mean = np.mean(f1_array)
        f1_std = np.std(f1_array)
                
        mean_list.append(f1_mean)
        std_list.append(f1_std)
                
    argmax = np.argmax(mean_list)
    for model_index,model_stats in enumerate(stats_list):
        slot_string += ' & '
        
        f1_mean = mean_list[model_index]
        f1_std = std_list[model_index]
        
        if model_index == argmax:
            slot_string += f"$\mathbf{{{round(f1_mean, 4)}}} \pm \mathbf{{{round(f1_std,4)}}}$"
        else:
            slot_string += f"${round(f1_mean, 4)} \pm {round(f1_std, 4)}$ "

    slot_string += "\\\\"
    
    print_log = print_and_append(slot_string, print_log=print_log)
    print_log = print_and_append(f"\label{{tab:{disease}_slotfill}}", print_log=print_log)
    print_log = print_and_append("\end{longtable}", print_log=print_log)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename_full = os.path.join(dir_path, "attack_logs/", "Evaluation_Results/", filename)
    print(f"Saving data to {filename_full}")
    with open(filename_full, "w") as outfile:
        for line in print_log:
            outfile.write(line + "\n")


import pandas as pd

def summarize_latex(extraction=True, slot_filling=True):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    latex_folder = os.path.join(dir_path, "attack_logs/", "Evaluation_Results")

    extraction_files = [os.path.join(latex_folder, "Extraction_Only_Augmented_With_Unchanged/", f) for f in os.listdir(os.path.join(latex_folder, "Extraction_Only_Augmented_With_Unchanged/")) if not "attacked" in f.lower()]
    comp_files_gl = [os.path.join(latex_folder, "Comp_Only_Augmented_With_Unchanged/", f) for f in os.listdir(os.path.join(latex_folder, "Comp_Only_Augmented_With_Unchanged/")) if not "attacked" in f.lower() if "gl" in f.lower()]
    comp_files_dm2 = [os.path.join(latex_folder, "Comp_Only_Augmented_With_Unchanged/", f) for f in os.listdir(os.path.join(latex_folder, "Comp_Only_Augmented_With_Unchanged/")) if not "attacked" in f.lower() if "dm2" in f.lower()]
    if extraction:
        __summarize_entityextr__(extraction_files)
    if slot_filling:
        __summarize_sf__(comp_files_gl, file_addon="gl_")
        __summarize_sf__(comp_files_dm2, file_addon="dm2_")

def __summarize_entityextr__(files):
    cols_ee = ["Disease", "Attacker", "Model", "Data Mode", "Precision", "Recall", "F1"]
    evaluation_summary_entity_extraction = pd.DataFrame(columns=cols_ee)

    for f in files:
        df = pd.read_csv(f, sep='&', header=None, skiprows=3, skipfooter=2, engine='python')
        micro_average = df.iloc[-1]
        only_filename = f.split("/")[-1]
        params = only_filename.split("_")
        disease_str = params[0]
        attacker = params[1]
        
        model_type = "Normal"
        if params[2].startswith("Robust"):
            model_type = "Robust"
        data_type = "Normal"
        if params[2].endswith("Augmented") or params[2].endswith("AugmentedOnly"):
            data_type = "Augmented"
        elif params[2].endswith("Attacked"):
            data_type = "Attacked"

        temp = {
            "Disease": disease_str, 
            "Attacker": attacker, 
            "Model": model_type, 
            "Data Mode": data_type, 
            "Precision": micro_average[1], 
            "Recall": micro_average[2], 
            "F1": micro_average[3]
        }
        temp = pd.DataFrame(temp, columns=cols_ee, index=[0])
        
        temp = temp.reset_index(drop=True)
        evaluation_summary_entity_extraction = evaluation_summary_entity_extraction.reset_index(drop=True)

        evaluation_summary_entity_extraction = pd.concat([evaluation_summary_entity_extraction, temp], axis=0)
    
    evaluation_summary_entity_extraction.sort_values(by=["Disease", "Attacker", "Model", "Data Mode"], inplace=True)
    
    out = os.path.join(dir_path, "attack_logs/", "Evaluation_Results/", "SUMMARY_EE.tex")
    evaluation_summary_entity_extraction.style.to_latex(out, label=f"tab:EntityExtraction", 
                                    caption=f"Evaluation Results for EntityExtraction")
    evaluation_summary_entity_extraction.to_csv(out.replace(".tex", ".csv"))


def __summarize_sf__(file_list, file_addon=""):
    if "gl" in file_addon or "dm2" in file_addon:
        cols_comp = ["Attacker", "Model", "Data Mode", "Comp Mode", "Avg", "Compatibility"]
    else:
        cols_comp = ["Disease", "Attacker", "Model", "Data Mode", "Comp Mode", "Avg", "Compatibility"]
    
    evaluation_summary_compatibility = pd.DataFrame(columns=cols_comp)
    for f in file_list:
        relevant_lines = []
        with open(f, "r") as infile:
            line = infile.readline() 
            while line:
                if not "micro" in line:
                    line = infile.readline() 
                else: 
                    new_line = line.replace("\\", "").replace("mathbf", "").replace("$", "").replace("\n", "").replace("{", "").replace("}", "")
                    relevant_lines.append(new_line)
                    line = infile.readline() 

        only_filename = f.split("/")[-1].replace("Baseline_", "")
        params = only_filename.split("_")
        disease_str = params[0]
        attacker = params[1]
        comp_mode = "value" if "value" in only_filename else "class"
        
        model_type = "normal"
        if params[2].startswith("robust"):
            model_type = "robust"
        data_type = "normal"
        if params[2].endswith("Augmented") or params[2].endswith("AugmentedOnly"):
            data_type = "augmented"
        elif params[2].endswith("Attacked"):
            data_type = "attacked"
        
        for l in relevant_lines:
            split_line = l.split("&")
            line_descr = split_line[0]
            random = split_line[1].replace("pm", "$\pm$")
            intra_comp1 = float(split_line[2].split("pm")[0].strip())
            temp = { 
                "Attacker": attacker, 
                "Model": model_type, 
                "Comp Mode": comp_mode, 
                "Data Mode": data_type, 
                "Compatibility": intra_comp1, 
                "Avg": line_descr.replace("micro average", "m. a."), 
            }
            if "Disease" in cols_comp:
                temp["Disease"] = disease_str
            
            temp = pd.DataFrame(temp, columns=cols_comp, index=[0])
            
            temp = temp.reset_index(drop=True)
            evaluation_summary_compatibility = evaluation_summary_compatibility.reset_index(drop=True)

            evaluation_summary_compatibility = pd.concat([evaluation_summary_compatibility, temp], axis=0)

    if "Disease" in cols_comp:
        evaluation_summary_compatibility.sort_values(by=["Disease", "Attacker", "Model", "Data Mode"], inplace=True)
    else:
        evaluation_summary_compatibility.sort_values(by=["Attacker", "Model", "Data Mode"], inplace=True)

    
    out = os.path.join(dir_path, "attack_logs/", "Evaluation_Results/", f"SUMMARY_{file_addon}SF.tex")
    evaluation_summary_compatibility.style.to_latex(out, label=f"tab:Summary_SlotFilling {file_addon.replace('_', '')}", 
                                    caption=f"Evaluation Results for SlotFilling {file_addon.replace('_', '')}")

    evaluation_summary_compatibility.to_csv(out.replace(".tex", ".csv"))


def merge_latex_into_one_section(standalone_compilation=False):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    latex_folder = os.path.join(dir_path, "attack_logs/", "Evaluation_Results")

    extraction_files = sorted([os.path.join(latex_folder, "Extraction_Only_Augmented_With_Unchanged/", f) for f in os.listdir(os.path.join(latex_folder, "Extraction_Only_Augmented_With_Unchanged/")) if not "attacked" in f.lower()])
    comp_files_gl = sorted([os.path.join(latex_folder, "Comp_Only_Augmented_With_Unchanged/", f) for f in os.listdir(os.path.join(latex_folder, "Comp_Only_Augmented_With_Unchanged/")) if not "attacked" in f.lower() if "gl" in f.lower()])
    comp_files_dm2 = sorted([os.path.join(latex_folder, "Comp_Only_Augmented_With_Unchanged/", f) for f in os.listdir(os.path.join(latex_folder, "Comp_Only_Augmented_With_Unchanged/")) if not "attacked" in f.lower() if "dm2" in f.lower()])
    
    outfile_path = os.path.join(latex_folder, "Appendix_Results.tex")
    with open(outfile_path, "w") as outfile:
        if standalone_compilation:
            outfile.write("\\documentclass[7pt]{article}\n")
            outfile.write("\\usepackage{longtable}\n")
            outfile.write("\\usepackage{caption}\n")
            outfile.write("\\usepackage[a4paper,left=2.5cm,right=3.5cm,top=2cm,bottom=2cm,bindingoffset=0mm]{geometry}\n")
            outfile.write("\\begin{document}\n")
        outfile.write("\\renewcommand{\\baselinestretch}{1}\n")
        outfile.write("\\section{Evaluation Results}\n")

    __merge_entity_extraction__(extraction_files=extraction_files, outfile_path=outfile_path, standalone_compilation=False)
    __merge_slot_filling__(comp_files=comp_files_gl, outfile_path=outfile_path, standalone_compilation=False)
    __merge_slot_filling__(comp_files=comp_files_dm2, outfile_path=outfile_path, standalone_compilation=True)

def __merge_slot_filling__(comp_files, outfile_path, standalone_compilation):
    match_label = re.compile(r"\n\\label{tab:.*}\n")
    match_section = re.compile(r"\\section{[\w|\s|-]*}\n")
    for f in comp_files:
        data = []
        with open(f, "r") as infile:
            data = infile.read() 
            
            only_file = f.split("/")[-1]
            print(only_file)
            params = only_file.split("_")
            disease_str = params[0]
            attacker = params[1]
            model = "normal" if params[2].startswith("Normal") else "retrained"
            data_mode = "normal" if params[2].endswith("Normal") else "augmented"
            comp_mode = "class change" if "class" in only_file else "value change"
            # replace label with more exact one + append caption
            long_disease_str = "glaucoma" if disease_str == "gl" else "diabetes"
            caption = f"\\\caption{{Results for the entity slot filling task on the {data_mode} {long_disease_str} data set with the {model} model using {attacker} and {comp_mode} criterion}}\\\\\\"
            replacement_label = f"\\\\\ \n {caption} \n \\\label{{tab:results_{disease_str}_{attacker}_{comp_mode}_{model}_{data_mode}_sf}}\\n"

            data = re.sub(match_label, replacement_label, data)
            data = re.sub(match_section, "", data)
        with open(outfile_path, "a") as outfile:
            outfile.write(data)
            outfile.write("\n\\newpage\n")

    if standalone_compilation:
        with open(outfile_path, "a") as outfile:
            outfile.write("\\end{document}")  


def __merge_entity_extraction__(extraction_files, outfile_path, standalone_compilation):
    match_label = re.compile(r"\n\\label{tab:.*}\n")
    match_section = re.compile(r"\\section{[\w|\s|-]*}\n")
    for f in extraction_files:
        with open(f, "r") as infile:
            data = infile.read() 
        
            only_file = f.split("/")[-1]
            print(only_file)
            params = only_file.split("_")
            disease_str = params[0]
            attacker = params[1]
            model = "normal" if params[2].startswith("Normal") else "retrained"
            data_mode = "normal" if params[2].endswith("Normal") else "augmented"
            # replace label with more exact one + append caption
            long_disease_str = "glaucoma" if disease_str == "gl" else "diabetes"
            caption = f"\\\caption{{Results for the entity extraction task on the {data_mode} {long_disease_str} data set with the {model} model using {attacker}}}\\\\\\"
            replacement_label = f"\\\\\ \n {caption} \n \\\label{{tab:results_{disease_str}_{attacker}_{model}_{data_mode}_extraction}}\\n"

            data = re.sub(match_label, replacement_label, data)
            data = re.sub(match_section, "", data)
        with open(outfile_path, "a") as outfile:
            outfile.write(data)
            outfile.write("\n\\newpage\n")
    
    if standalone_compilation:
        with open(outfile_path, "a") as outfile:
            outfile.write("\\end{document}")

if __name__ == "__main__":

    # summarize_latex(extraction=False)
    merge_latex_into_one_section(standalone_compilation=True)