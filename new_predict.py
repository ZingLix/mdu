import os

os.environ["TF_KERAS"] = "1"
from tqdm import tqdm
import numpy as np
from bert4keras.snippets import AutoRegressiveDecoder
import json
from util import *
from config import *
import multiprocessing
import tfs_util
import pickle
from args import args
import new_api as api

# args.type = "test"

print("Task type: {}\n".format(args.type))

test_path = "./data/new/{}_dist.json".format(args.type)
pred_path = "./pred/"
output_file = "{}_output2.json".format(args.type)
gold_output = "gold_" + output_file

# predictor = tfs_util.predictor()


i = 0


def pred(doc):
    data = load_data(doc)
    intent_result = []
    slots_result = []
    symptom_set = set()
    for i, utter in enumerate(data):

        pred = api.pred_diag(data[: i + 1], compat_result=False)
        pred_intent_list = [
            r.present_intent_action.value for r in pred.analysis_result
        ]
        pred_slots_value = []

        def add_res(res, base):
            for k, v in res.items():
                for item in v:
                    c = base.copy()
                    c.append(k)
                    c.append(
                        "" if item["value"] in ["", None] else item["value"]
                    )
                    add_res(item["attr"], c)
                    if item["value"] not in ["", None]:
                        pred_slots_value.append(tuple(c))

        for item in pred.analysis_result:
            add_res(item.slots, [])
        pred_slots_value = list(set(pred_slots_value))
        print(data[i].utterance)
        print("预测：")
        print(pred_intent_list)
        print(pred_slots_value)
        #         print(utter.frames)

        gold_intent_list = utter.intent
        gold_slots_value = []

        def add_gold_res(res, base):

            for item in res:
                c = base.copy()
                c.append(item.slot_type)
                c.append(
                    ""
                    if item.entity_content in ["", None]
                    else item.entity_content
                )
                add_gold_res(item.attribute, c)
                if item.entity_content not in ["", None]:
                    gold_slots_value.append(tuple(c))

        add_gold_res(utter.frames, [])

        gold_intent_list = list(set(gold_intent_list))
        gold_slots_value = list(set(gold_slots_value))

        print("答案：")
        print(gold_intent_list)
        print(gold_slots_value)

        intent_result.append((pred_intent_list, gold_intent_list))
        slots_result.append((pred_slots_value, gold_slots_value))
    return intent_result, slots_result, doc


def calculate_prf(d):
    TP, FP, FN = 0, 0, 0
    em = 0
    count = 0
    for pred, gold in d:
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        if len(gold) == 0:
            continue
        count += 1
        if set(gold) == set(pred):
            em += 1
        else:
            print(gold, "\n", pred, "\n-------------\n")
    precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
    recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
    F1 = (
        2 * precision * recall / float(precision + recall)
        if (precision + recall) != 0
        else 0
    )
    print(TP, FP, FN)
    return F1, recall, precision, em / count


with open(test_path, "r") as f:
    data = json.load(f)
    # data = [data[8]]
    intent_result = []
    all_result = []
    slots_result = []
    pool = multiprocessing.Pool(processes=config["predict"]["concurrency"])
    # pool = multiprocessing.Pool(processes=1)

    def callback(x):
        if x is not None:
            intent_res, slots_res, doc = x
            all_result.append(x)
            intent_result.extend(intent_res)
            slots_result.extend(slots_res)
        else:
            print("bad")

    results_of_processes = [
        pool.apply_async(
            pred,
            args=(diag,),
            callback=callback,
            error_callback=lambda e: print(e.__cause__),
        )
        for diag in tqdm(data)
    ]
    for r in results_of_processes:
        r.get()
    pool.close()
    pool.join()

    def filter(l):
        return l

    intent_result = [(filter(i1), filter(i2)) for i1, i2 in intent_result]
    slots_result = [(filter(i1), filter(i2)) for i1, i2 in slots_result]
    intent_prf = calculate_prf(intent_result)
    slot_prf = calculate_prf(slots_result)
    if args.pred_path is not None:
        with open(args.pred_path, "wb") as f:
            pickle.dump(all_result, f)
    print("意图准确度")
    print(intent_prf)
    print("槽准确度")
    print(slot_prf)
