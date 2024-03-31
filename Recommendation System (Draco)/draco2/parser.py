import os
import pandas as pd
import json
import sys

class Aggregate:
    def __init__(self,type,attribute,groupby):
        self.type = type
        self.attribute = attribute
        self.groupby = groupby

desktop = os.path.normpath(os.path.expanduser(r"C:\Users\ideapad\Desktop\Repositório Final Tese\Data Repository"))
domain_id = ""
domain_usecase = ""
domain_question = ""
global new_attribute
new_attribute = False
aggregate_task = {}
scale_dict = {}
data_dict = {}
field_dict = {}
task_at_hand = {}

def load_dataset(path):
    new_path = os.path.join(desktop, path, "Use Cases", "Use Case - "+ str(usecase_number), "Data", "Data Files")
    json_path = os.path.join(new_path,"dataset.json")
    dataset = None
    if (os.path.isfile(json_path)):
        dataset = pd.read_json(json_path,encoding="ISO-8859-1")
    csv_path = os.path.join(new_path,"dataset.csv")
    if (os.path.isfile(csv_path)):
        dataset = pd.read_csv(csv_path,encoding="ISO-8859-1")
    datasetFields = list(dataset.columns.values)
    dataset = dataset[dataset[datasetFields].notnull().all(1)]
    return dataset

def countOfFiles(path):
    onlyfiles = os.listdir(path)
    return len(onlyfiles)

def loadDomainID(path):
    new_path = os.path.join(desktop, path, "Use Cases", "Use Case - "+ str(usecase_number), "Domain Question", str(domainquestion_number))
    with open(os.path.normpath(os.path.join(new_path, "domain_question.json"))) as f:
        json_data = json.load(f)
        global domain_id
        domain_id = json_data['domainID']
        global domain_question
        domain_question = json_data['questionID']


def loadTaskResults(path):
    new_path = os.path.join(desktop, path, "Use Cases", "Use Case - "+ str(usecase_number), "Domain Question", str(domainquestion_number), "Analytical Tasks")
    json_path = os.path.join(new_path,"dataset.json")

    for x in range(countOfFiles(new_path)):
        file_name = domain_question + str(x+1)
        #print("AQUI : " + str(file_name))
        with open(os.path.normpath(os.path.join(new_path, file_name, "result " + str(file_name) + ".json"))) as f:
            json_data = json.load(f)
            target = json_data['output']['target_type']
            #if(target == 'attribute'):
            #if(target == 'item'):
            if(target == 'dataset'):
                dataset_data = json_data['output']['target']['dataset'][0]['items'][0]['attributes']
                for y in range(len(dataset_data)):
                    if(dataset_data[y].get('order') != None):
                        order = dataset_data[y]['order']
                        field_dict[order] = dataset_data[y]['attribute_name']
                        scale_dict[order] = dataset_data[y]['attribute_type']
                print(field_dict)
                print(scale_dict)
            #print(json_data['output']['target'][str(target)])

def loaddatafiles(path):
    new_path = os.path.join(desktop, path, "Use Cases", "Use Case - "+ str(usecase_number), "Data", "Data Files")
    json_path = os.path.join(new_path,"dataset_file_description.json")
    with open(os.path.normpath(json_path)) as file:
        json_data = json.load(file)
        dataset_file_json = json_data['dataset'][0]['dataset_files'][0]['fields']
        for x in range(len(dataset_file_json)):
            for y in range(len(field_dict)):
               if(field_dict[y].upper() == dataset_file_json[x]['field_name'].upper()):
                     data_dict[y] = dataset_file_json[x]['field_datatype']
        print(data_dict)

def loadAnalyticalTasks(path):
    new_path = os.path.join(desktop, path, "Use Cases", "Use Case - "+ str(usecase_number), "Domain Question", str(domainquestion_number), "Analytical Tasks")

    for x in range(countOfFiles(new_path)):
        file_name = domain_question + str(x+1)

        with open(os.path.normpath(os.path.join(new_path, file_name, "task " + str(file_name) + ".json"))) as f:
            json_data = json.load(f)
            if 'filter' in json_data['task_selection'][0]:
                applyFilter(json_data['task_selection'][0])
            main_task = json_data['task_selection'][0]['task']
            if checkforNonViz(main_task):
                if(len(json_data['task_selection'][0]['subtask']) != 0):
                    new_task = json_data['task_selection'][0]['subtask'][0]
                    sub_task = new_task['task']

                    if(sub_task.upper() == "AGGREGATE"):
                        aggregate_task[len(aggregate_task)] = Aggregate(new_task['type'] , new_task['attribute'] ,new_task['groupby'])
                    elif(sub_task.upper() == "COMPUTE ATTRIBUTE" or sub_task.upper() == "TRANSFORM"):
                        computeOrTransformToDataset(new_task)
                    task_at_hand[len(task_at_hand)] = sub_task

            else:
                task_at_hand[len(task_at_hand)] = main_task
    print(task_at_hand)

def checkforNonViz(task_name):
    accepted_strings = {"ORGANIZE","DERIVE","IDENTIFY"}
    if task_name.upper() in accepted_strings:
        return True
    else:
        return False

def checkForDatasetChanges(path):
    print("CONFIRMACAO: " + str(new_attribute))
    if new_attribute:
        new_path = os.path.join(desktop, path, "Use Cases", "Use Case - "+ str(usecase_number), "Data", "Data Files", "new_dataset.csv")
        print("NICK KERR")
        featured_dataset.to_csv(new_path,encoding='utf-8')

def computeOrTransformToDataset(task_json):
    global new_attribute
    new_attribute = True
    featured_dataset[task_json['field_name']] = eval(task_json['formula'])

def applyFilter(task_json):
    global featured_dataset
    featured_dataset = eval(task_json['filter'])

def createAndWriteFile():
    #filepath = os.path.normpath(os.path.join(desktop, "example.lp"))
    file = open("draco/asp/examples/parserExamples/example.lp", "w")
    writeTask(file)
    writeFields(file)
    writeEncodings(file)
    writeAggregates(file)
    writeScales(file)

def writeTask(file):
    viz_task = defineTask(task_at_hand[len(task_at_hand)-1].upper())
    if viz_task != None:
        file.write("attribute(task,root," + viz_task +").\n")
        file.write("attribute(number_rows,root,"+ str(len(featured_dataset.index))+").\n")

def writeFields(file):
    for y in range(min(4,len(field_dict))):
        file.write("entity(field,root,(f,"+ str(y) +")).\n")
        file.write("attribute((field,name),(f," + str(y) +")," +field_dict[y].lower()+").\n")
        file.write("attribute((field,type),(f," + str(y) +")," + defineData(data_dict[y])+").\n")
        file.write("attribute((field,unique),(f," + str(y)+ ")," +str(featured_dataset[field_dict[y]].nunique())+").\n")

def writeEncodings(file):
    file.write("entity(view,root,(v,0)).\n")
    file.write("entity(mark,(v,0),(m,0)).\n")
    for y in range(min(4,len(field_dict))):
        encodingOrder(file,y)

def encodingOrder(file,y):
    file.write("entity(encoding,(m,0),(e,"+str(y)+")).\n")
    for x in range(len(aggregate_task)):
        if(aggregate_task[x].type.lower() != "count" or aggregate_task[x].attribute.lower() == field_dict[y].lower()):
            file.write("attribute((encoding,field),(e,"+str(y)+"),"+field_dict[y].lower()+").\n")
        else:
            scale_dict[y] = "linear"
    if y == 0:
        file.write("attribute((encoding,channel),(e,"+str(y)+"),x).\n")
    if y == 1:
        file.write("attribute((encoding,channel),(e,"+str(y)+"),y).\n")

def defineTask(viz_task):
    match viz_task:
        case "RELATIONSHIP":
            return "relationship"
        case "PATTERN":
            return "pattern"
        case "FIND EXTREME":
            return "extreme"
        case "FIND ANOMALIES":
            return "anomaly"
        case "FIND CLUSTERS":
            return "cluster"
        case "COMPARE":
            return "compare"
        case "DETERMINE DISTRIBUTION":
            return "distribution"
        case _:
            return None

def defineData(data_field):
    data_field = data_field.upper()
    number_strings = {"INTEGER","NUMBER","FLOAT"}
    string_strings = {"STRING"}
    date_strings = {"DATE", "DATETIME"}
    if data_field in number_strings:
        return "number"
    elif data_field in string_strings:
        return "string"
    elif data_field in date_strings:
        return "datetime"
    else:
        return None

def writeAggregates(file):
    if len(aggregate_task) != 0:
        for y in range(len(aggregate_task)):
            for x in range(len(field_dict)):
                if(aggregate_task[y].attribute.lower() == field_dict[x].lower() and aggregate_task[y].groupby in field_dict.values() and aggregate_task[y].type.lower() != 'count'):
                    file.write("attribute((encoding,aggregate),(e,"+ str(x) +"),"+ aggregate_task[y].type +").\n")
                elif(aggregate_task[y].attribute.lower() == aggregate_task[y].groupby.lower() and aggregate_task[y].type.lower() == 'count' and aggregate_task[y].attribute.lower() == field_dict[x].lower() and aggregate_task[y].groupby in field_dict.values()):
                    if(x == 0):
                       file.write("attribute((encoding,aggregate),(e,"+ str(x+1) +"),"+ aggregate_task[y].type +").\n")
                    elif(x == 1):
                        file.write("attribute((encoding,aggregate),(e,"+ str(x-1) +"),"+ aggregate_task[y].type +").\n")

def writeScales(file):
    for x in range(min(2,len(scale_dict))):
        file.write("entity(scale,(v,0),(s,"+ str(x)+")).\n")
        if x == 0:
            file.write("attribute((scale,channel),(s,"+str(x)+"),x).\n")
            print("SCALE 0: " + str(scale_dict[x]))
            print("DATA 0: " + str(data_dict[x]))
            file.write("attribute((scale,type),(s,"+str(x)+")," + defineScales(scale_dict[x],data_dict[x]) +").\n")
        if x == 1:
            file.write("attribute((scale,channel),(s,"+str(x)+"),y).\n")
            print("SCALE 1: " + str(scale_dict[x]))
            print("DATA 1: " + str(data_dict[x]))
            file.write("attribute((scale,type),(s,"+str(x)+"),"+ defineScales(scale_dict[x],data_dict[x])+").\n")

def defineScales(scale,data):
    scale = scale.lower()
    if(scale == "linear" or scale == "ordinal"):
        return scale
    match defineData(data):
        case "number":
            if scale == "nominal" or scale == "countinuous":
                return "linear"
            elif scale == "categorical":
                return "numcate"
            elif scale == "ordinal" or scale == "discrete":
                return "ordinal"
        case "datetime":
            if scale == "continuous":
                return "linear"
            elif scale == "discrete":
                return "ordinal"
        case "string":
            return "ordinal"
        case _:
            return None

#################################################################################### P A R S I N G ###############################################################################################

from draco import Draco, answer_set_to_dict, dict_to_facts
from pprint import pprint
from draco.asp_utils import blocks_to_program
from draco.programs import soft
from draco.renderer import AltairRenderer
from IPython.display import display
import warnings
import importlib.resources as pkg_resources
import draco.asp.examples.parserExamples as Pexamples
from draco.debug import DracoDebug, DracoDebugPlotter, DracoDebugChartConfig
from draco.programs import hard
import altair as alt

new_weights = {"aggregate_weight": 1,
"bin_weight": 2,
"compare_facet_variables_weight": 17,
"compare_log_scale_weight": 5,
"bin_high_weight": 6,
"categoricalcolorcardinality_weight": 0,
"bin_low_weight": 6,
"encoding_weight": 0,
"encoding_field_weight": 6,
"bin_datetime_weight": 3,
"same_field_weight": 8,
"same_field_grt3_weight": 16,
"count_grt1_weight": 50,
"number_categorical_weight": 10,
"bin_low_unique_weight": 5,
"bin_not_linear_weight": 1,
"only_discrete_weight": 5,
"multi_non_pos_weight": 3,
"aggre_linear_weight": 1,
"line_date_string_weight": 7,
"non_pos_used_before_pos_weight": 15,
"aggregate_group_by_raw_weight": 3,
"aggregate_no_discrete_weight": 3,
"same_axis_multiple_marks_weight": 10,
"count_without_log_weight": 8,
"distribution_continuous_pos_not_zero_weight": 6,
"distribution_ordinal_axis_facet_weight": 5,
"distribution_single_aggregate_weight": 0,
"line_with_aggregate_distribution_weight": 8,
"binning_zero_scale_weight": 25,
"distribution_cd_line_bar_aggregate_weight": 6,
"distribution_bin_without_log_weight": 0,
"distribution_x_y_raw_weight": 2,
"distribution_bin_200_weight": 2,
"distribution_bin_25_weight": 0,
"distribution_bin_10_weight": 1,
"distribution_c_bin_y_weight": 5,
"distribution_c_bin_x_weight": 5,
"distribution_c_c_facet_weight": 12,
"distribution_polar_coordinate_weight": 13,
"distribution_log_scale_weight": 5,
"distribution_binning_weight": 1,
"distribution_count_weight": 10,
"distribution_median_weight": 21,
"distribution_stdev_weight": 25,
"distribution_min_weight": 20,
"distribution_max_weight": 20,
"distribution_sum_weight": 2,
"distribution_mean_weight": 5,
"distribution_point_weight" : 2,
"distribution_bar_weight" : 1,
"distribution_line_weight" : 1,
"distribution_area_weight" : 3,
"distribution_text_weight" : 5,
"distribution_tick_weight" : 0,
"distribution_rect_weight" : 0,
"distribution_not_single_count_weight" : 8,
"distribution_discrete_x_weight": 8,
"distribution_discrete_size_weight": 10,
"distribution_continuous_size_weight": 10,
"distribution_aggregate_ordinal_weight": 11,
"x_y_raw_weight": 4,
"continuous_not_zero_weight": 5,
"size_not_zero_weight": 5,
"continuous_pos_not_zero_weight": 5,
"skew_zero_weight": 5,
"cross_zero_weight": 10,
"only_y_weight": 1,
"binned_orientation_not_x_weight": 15,
"high_cardinality_ordinal_weight": 10,
"high_cardinality_categorical_grt10_weight": 10,
"high_cardinality_shape_weight": 10,
"high_cardinality_size_weight": 1,
"horizontal_scrolling_x_weight": 20,
"horizontal_scrolling_col_weight": 20,
"date_scale_weight": 2,
"number_linear_weight": 2,
"value_agg_weight": 1,
"summary_facet_weight": 0,
"c_d_col_weight": 5,
"date_not_x_weight": 1,
"x_row_weight": 3,
"y_row_weight": 3,
"x_col_weight": 1,
"y_col_weight": 1,
"color_entropy_high_weight": 0,
"color_entropy_low_weight": 0,
"size_entropy_high_weight": 0,
"size_entropy_low_weight": 0,
"linear_scale_weight": 0,
"relationship_log_scale_weight": 6,
"log_scale_weight": 4,
"ordinal_scale_weight": 2,
"categorical_scale_weight": 3,
"c_c_point_weight": 0,
"c_c_line_weight": 20,
"c_c_area_weight": 20,
"c_c_text_weight": 2,
"c_d_overlap_point_weight": 30,
"c_d_overlap_bar_weight": 20,
"c_d_overlap_line_weight": 10,
"relationship_cd_overlap_line_weight": 6,
"distribution_cd_overlap_line_weight": 10,
"compare_cd_overlap_line_weight": 5,
"c_d_overlap_area_weight": 30,
"c_d_overlap_text_weight": 70,
"c_d_overlap_tick_weight": 34,
"c_d_no_overlap_point_weight": 15,
"c_d_no_overlap_bar_weight": 5,
"c_d_no_overlap_line_weight": 15,
"c_d_no_overlap_area_weight": 20,
"c_d_no_overlap_text_weight": 30,
"c_d_no_overlap_tick_weight": 25,
"d_d_overlap_weight": 20,
"d_d_point_weight": 12,
"d_d_text_weight": 7,
"d_d_rect_weight": 3,
"linear_x_weight": 0,
"linear_y_weight": 0,
"linear_color_weight": 10,
"linear_size_weight": 2,
"linear_text_weight": 20,
"cdd_facet_weight": 2,
"size_with_agg_weight": 20,
"cdd_not_all_fields_weight": 9,
"log_x_weight": 4,
"log_y_weight": 4,
"log_color_weight": 10,
"log_size_weight": 1,
"log_text_weight": 20,
"ordinal_x_weight": 0,
"ordinal_y_weight": 1,
"ordinal_color_weight": 8,
"ordinal_size_weight": 10,
"ordinal_shape_weight": 11,
"ordinal_text_weight": 32,
"ordinal_detail_weight": 20,
"numcate_duplicates_weight": 15,
"numcate_no_aggregates_weight": 15,
"numcate_x_weight": 15,
"numcate_y_weight": 20,
"numcate_color_weight": 12,
"numcate_size_weight": 10,
"numcate_shape_weight": 9,
"numcate_text_weight": 35,
"numcate_detail_weight": 30,
"numcate_scale_weight": 15,
"categorical_color_weight": 4,
"aggregate_count_weight": 5,
"aggregate_mean_weight": 1,
"aggregate_median_weight": 5,
"aggregate_min_weight": 8,
"aggregate_max_weight": 8,
"aggregate_stdev_weight": 10,
"aggregate_sum_weight": 0,
"area_without_aggregate_weight": 0,
"stack_zero_weight": 0,
"stack_center_weight": 1,
"stack_normalize_weight": 1,
"only_y_relationship_weight": 5,
"relationship_binning_weight": 14,
"relationship_point_weight": 2,
"relationship_bar_weight": 1,
"relationship_line_weight": 1,
"relationship_area_weight": 3,
"relationship_text_weight": 6,
"relationship_tick_weight": 10,
"relationship_rect_weight": 12,
"value_point_weight": 0,
"value_bar_weight": 0,
"value_line_weight": 0,
"value_area_weight": 0,
"value_text_weight": 0,
"value_tick_weight": 0,
"value_rect_weight": 0,
"summary_point_weight": 0,
"summary_bar_weight": 0,
"summary_line_weight": 0,
"summary_area_weight": 0,
"summary_text_weight": 0,
"summary_tick_weight": 0,
"summary_rect_weight": 0,
"relationship_facet_weight": 11,
"relationship_count_weight": 10,
"relationship_sum_weight": 0,
"relationship_median_weight": 16,
"relationship_stdev_weight": 16,
"relationship_min_weight": 13,
"relationship_max_weight": 13,
"relationship_mean_weight": 3,
"relationship_continuous_x_weight": 7,
"relationship_continuous_y_weight": 0,
"relationship_continuous_color_weight": 12,
"relationship_continuous_size_weight": 0,
"relationship_continuous_text_weight": 30,
"relationship_discrete_x_weight": 0,
"relationship_discrete_y_weight": 7,
"relationship_discrete_color_weight": 0,
"relationship_discrete_size_weight": 12,
"relationship_discrete_shape_weight": 12,
"relationship_discrete_text_weight": 10,
"relationship_discrete_detail_weight": 30,
"value_continuous_x_weight": 0,
"value_continuous_y_weight": 0,
"value_continuous_color_weight": 0,
"value_continuous_size_weight": 0,
"value_continuous_text_weight": 0,
"value_discrete_x_weight": 0,
"value_discrete_y_weight": 0,
"value_discrete_color_weight": 0,
"value_discrete_size_weight": 0,
"value_discrete_shape_weight": 0,
"value_discrete_text_weight": 0,
"value_discrete_detail_weight": 0,
"summary_continuous_x_weight": 0,
"summary_continuous_y_weight": 0,
"summary_continuous_color_weight": 0,
"summary_continuous_size_weight": 0,
"summary_continuous_text_weight": 0,
"summary_discrete_x_weight": 0,
"summary_discrete_y_weight": 0,
"summary_discrete_color_weight": 0,
"summary_discrete_size_weight": 0,
"summary_discrete_shape_weight": 0,
"summary_discrete_text_weight": 0,
"summary_discrete_detail_weight": 0,
"compare_binning_weight": 2,
"compare_count_weight": 2,
"compare_median_weight": 5,
"compare_stdev_weight": 5,
"compare_min_weight": 5,
"compare_max_weight": 5,
"compare_sum_weight": 1,
"compare_mean_weight": 2,
"compare_point_weight": 3,
"compare_bar_weight": 0,
"compare_line_weight": 0,
"compare_area_weight": 0,
"compare_text_weight": 0,
"compare_tick_weight": 0,
"compare_rect_weight": 0,
"compare_c_c_point_weight": 1,
"compare_bin_ordinal_y_weight": 3,
"compare_continuous_pos_not_zero_weight": 17,
"comparesum_ordinal_weight": 15,
"sum_categorical_number_weight": 5,
"aggregate_string_weight": 9,
"size_with_sum_weight": 2,
"interesting_x_weight": 0,
"interesting_y_weight": 1,
"interesting_color_weight": 2,
"interesting_size_weight": 2,
"interesting_shape_weight": 3,
"interesting_text_weight": 6,
"interesting_row_weight": 7,
"interesting_column_weight": 6,
"interesting_detail_weight": 20,
"position_entropy_weight": 2,
"cartesian_coordinate_weight": 0,
"compare_c_c_facet_weight": 3,
"string_color_categorical_weight": 6,
"polar_coordinate_weight": 20}

def createVisualization(pathname):
    alt.data_transformers.disable_max_rows()
    s = "".join(
        blocks_to_program(
            soft.blocks, set(soft.blocks.keys()) - {"c_d_overlap_line"}
        )
    )
    h = "".join(
        blocks_to_program(
            hard.blocks, set(hard.blocks.keys()) - {"bar_tick_area_line_without_continuous_x_y"}
        )
    )
    h += "violation(bar_tick_area_line_without_continuous_x_y) :- \
        attribute((mark,type),M,(bar;area;line)), \
        { helper(mark_channel_cont,M,x);helper(mark_channel_cont,M,y) } <= 0."

    d = Draco(weights=new_weights, hard=h, soft=s)
    warnings.filterwarnings("ignore")
    renderer = AltairRenderer()
    relation_spec = pkg_resources.read_text(Pexamples, "example.lp")
    print("INPUT:")
    print(relation_spec)
    print("OUTPUT:")
    specs = {}
    for i, model in enumerate(d.complete_spec(relation_spec, 3)):
        chart_num = i + 1
        spec = answer_set_to_dict(model.answer_set)
        chart_name = f"Rec {chart_num}"
        specs[chart_name] = dict_to_facts(spec)
        print(f"CHART {chart_num}")
        print(f"COST: {model.cost}")
        pprint(spec)
        pprint(str(model))
        renderer.render(spec=spec, data=featured_dataset).save(pathname + '-'+ str(i) + '-chart.html')


pathname = input()
usecase_number = input()
domainquestion_number = input()
featured_dataset = load_dataset(pathname)
loadDomainID(pathname)
loadTaskResults(pathname)
loaddatafiles(pathname)
loadAnalyticalTasks(pathname)
checkForDatasetChanges(pathname)
createAndWriteFile()
createVisualization(pathname)