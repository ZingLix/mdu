from __future__ import print_function
import os
from trace import Trace

os.environ["TF_KERAS"] = "1"

from numpy.random import random

from new_api import AnalysisResult, AttrExtractionResponse, compat, intent_to_slot

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
from tensorflow.keras.models import Model
from bert4keras.snippets import DataGenerator
from bert4keras.snippets import sequence_padding
from bert4keras.optimizers import Adam
from bert4keras.models import build_transformer_model
from bert4keras.layers import Loss
from bert4keras.backend import keras, K
import pickle
from util import *
from config import *
import copy

# 基本参数
batch_size = 32
steps_per_epoch = 2000
epochs = 41

if args.pretrain:
    print("with pretrain")

if not args.ratio:
    args.ratio = 1

data_path = "./data/new/train_dist.json"
val_path = "./data/new_new_val.json"
output_path = (
    "./experiments_0429/prompt_{}{}_aug_change_utter_n/model.weights".format(
        "pretrain_" if args.pretrain else "", args.ratio if args.ratio else "1"
    )
)
symptom_path = "./data/symptom_list.pkl"
pretrain_path = "./dpretrain/best_model_full_fixed_0105.weights"

with open("./sim_word.pkl", "rb") as f:
    sim_word = pickle.load(f)

with open("./config/schema_new.json") as f:
    nlu_schema = json.load(f)

with open("common_symptom.pkl", "rb") as f:
    com_symptom_list = pickle.load(f)

data = []


# template = ["你提及了哪些{}？", "你有哪些{}？", "{}", "说说你的{}？", "你存在哪些{}？"]
# # template = ["你提及了哪些{}？"]


def generate_qa_helper(entity_list: List[Entity], schema, attrs):
    qa_list = []
    entity_set = {}
    for frame in entity_list:
        if frame.entity_content == "":
            continue
        if frame.slot_type not in entity_set:
            entity_set[frame.slot_type] = []
        entity_set[frame.slot_type].append(frame.entity_content)
    for k, v in schema.items():
        # print(k)
        if k in entity_set:
            answer = (
                schema[k]["ans_prefix"]
                + SPLIT_TOKEN.join(list(set(entity_set[k])))
                + schema[k]["ans_postfix"]
            ).format(**attrs)
        else:
            answer = NONE_TOKEN
        for ques_schema in schema[k]["question"]:
            question = ques_schema.format(**attrs)
            qa_list.append((question, answer))
    if len(attrs) == 0:
        symptom_list = [
            frame.slot_name
            for frame in entity_list
            if frame.slot_type in ["症状名称", "其他症状名称"]
        ]
        symptom_set = set()
        for s in symptom_list:
            symptom_set.add(s)
            if s in sim_word:
                for item in sim_word[s]:
                    symptom_set.add(item)
        if_exists_qa = []
        s_list = random.sample(com_symptom_list, random.randint(0, 4))
        for s in s_list:
            if s not in symptom_set:
                if_exists_qa.append(("症状{}是否存在？".format(s), "否"))
        qa_list.extend(if_exists_qa)
    for frame in entity_list:
        new_attr = attrs.copy()
        new_attr[frame.slot_type] = frame.slot_name
        qa_list.extend(
            generate_qa_helper(
                frame.attribute, schema[frame.slot_type]["next"], new_attr
            )
        )
        if (
            frame.slot_type in ["症状名称", "其他症状名称"]
            and frame.slot_name in sim_word
        ):
            for w in sim_word[frame.slot_name]:
                if decision(0.7):
                    continue
                new_attr[frame.slot_type] = w
                qa_list.extend(
                    generate_qa_helper(
                        frame.attribute,
                        schema[frame.slot_type]["next"],
                        new_attr,
                    )
                )
    return qa_list


def generate_qa(utter: Utterance, schema):
    qa_list = generate_qa_helper(utter.frames, schema["slots"], {})
    if utter.speaker == Speaker.doctor:
        intent_schema = schema["intent"]["doctor"]
    else:
        intent_schema = schema["intent"]["patient"]
    intent_list = list(set(utter.intent))
    qa_list.append(
        (
            intent_schema["question"],
            intent_schema["prefix"] + SPLIT_TOKEN.join(intent_list),
        )
    )
    return qa_list


def augment(data):
    new_utter_list = []
    for utter in data:
        utterance = utter.utterance
        frames = [
            x for x in utter.frames if x.slot_type not in ["症状名称", "其他症状名称"]
        ]
        symptom_frames = []
        for frame in [
            x for x in utter.frames if x.slot_type in ["症状名称", "其他症状名称"]
        ]:
            value = frame.entity_content
            if (
                value in sim_word
                and decision(0.5)
                and utterance.count(value) == 1
            ):
                new_value = random.choice(sim_word[value])
                utterance = utterance.replace(value, new_value)
                frame.entity_content = new_value
                frame.slot_name = new_value
            symptom_frames.append(frame)
        frames.extend(symptom_frames)
        utter.utterance = utterance
        utter.frames = frames
        new_utter_list.append(utter)
    return new_utter_list


class data_generator(DataGenerator):
    """数据生成器"""

    def __iter__(self, rand=False):
        batch_token_ids, batch_segment_ids = [], []
        with open(symptom_path, "rb") as f:
            symptom_example_list = pickle.load(f)
        for is_end, utter_list in self.sample(rand):
            utter_list: List[Utterance]
            if decision(0.5):
                utter_list = augment(utter_list)
            for utter_idx, utter in enumerate(utter_list):
                # if utter.speaker == Speaker.doctor or len(utter.frames) == 0:
                #     continue
                qa_list = generate_qa(utter, nlu_schema)

                qa_list = [
                    qa for qa in qa_list if qa[1] != NONE_TOKEN or decision(0.3)
                ]
                # print(qa_list)

                for que, ans in qa_list:
                    content = []
                    reserved_len = len(que) + len(ans) + 10
                    for item in reversed(utter_list[: utter_idx + 1]):
                        utterance = "{}：{}".format(
                            item.speaker.value,
                            item.utterance.replace("   ", "，"),
                        )
                        if (
                            len(content) + len(utterance)
                            > maxlen - reserved_len
                        ):
                            if len(content) == 0:
                                content += (
                                    [vocab_map["[PATIENT]"]]
                                    + tokenizer.tokenize(
                                        utterance[-(maxlen - reserved_len) :]
                                    )[1:-1]
                                    + [vocab_map["[PATIENT]"]]
                                )
                            break
                        if item.speaker == Speaker.doctor:
                            content = (
                                [vocab_map["[DOCTOR]"]]
                                + tokenizer.tokenize(utterance)[1:-1]
                                + [vocab_map["[DOCTOR]"]]
                            ) + content
                        elif item.speaker == Speaker.patient:
                            content = (
                                [vocab_map["[PATIENT]"]]
                                + tokenizer.tokenize(utterance)[1:-1]
                                + [vocab_map["[PATIENT]"]]
                            ) + content
                        else:
                            raise Exception("unknown role")
                    fake_role = reverse_speaker(utter_list[utter_idx].speaker)
                    fake_sig = (
                        vocab_map["[DOCTOR]"]
                        if fake_role == Speaker.doctor
                        else vocab_map["[PATIENT]"]
                    )
                    que = "{}：{}".format(fake_role.value, que)
                    content = (
                        content
                        + [fake_sig]
                        + tokenizer.tokenize(que)[1:-1]
                        + [fake_sig]
                    )
                    content = ["[CLS]"] + content + ["[SEP]"]
                    # print("".join(content))
                    flag = True
                    if ans == "":
                        flag = False
                        ans = "无"
                    fake_ans_role = reverse_speaker(fake_role)
                    fake_sig = (
                        vocab_map["[DOCTOR]"]
                        if fake_ans_role == Speaker.doctor
                        else vocab_map["[PATIENT]"]
                    )
                    ans = "{}：{}".format(fake_ans_role.value, ans)
                    ans = (
                        ["[CLS]"]
                        + [fake_sig]
                        + tokenizer.tokenize(ans)[1:-1]
                        + [fake_sig]
                        + ["[SEP]"]
                    )
                    # print("".join(content))
                    # print("".join(ans))
                    token_ids, segment_ids = tokenizer.encode(
                        content, ans, maxlen=maxlen
                    )
                    batch_token_ids.append(token_ids)
                    batch_segment_ids.append(segment_ids)
                    # if flag:
                    #     segment_ids = (
                    #         [0]
                    #         * (
                    #             len(segment_ids)
                    #             - (len(ans))
                    #             - len(tokenizer.tokenize(que))
                    #         )
                    #         + [1] * (len(tokenizer.tokenize(que)) + 1)
                    #         + [0] * (len(ans) - 1)
                    #     )
                    #     batch_token_ids.append(token_ids)
                    #     batch_segment_ids.append(segment_ids)
                    if len(batch_token_ids) == self.batch_size:
                        batch_token_ids = sequence_padding(batch_token_ids)
                        batch_segment_ids = sequence_padding(batch_segment_ids)
                        yield [batch_token_ids, batch_segment_ids], None
                        batch_token_ids, batch_segment_ids = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分"""

    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(
    config_path,
    checkpoint_path,
    application="unilm",
    keep_tokens=keep_tokens,
)

output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
# model.summary()
class AnswerGenerator(AnswerGeneratorBase):
    """seq2seq解码器"""

    @AutoRegressiveDecoder.wraps(default_rtype="probas", use_states=True)
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        input = [
            np.array(token_ids, dtype=np.float32),
            np.array(segment_ids, dtype=np.float32),
        ]

        predicted = model.predict(input)[:, -1]
        predicted = np.array(predicted)
        constraint = states
        if constraint is not None:
            avail, _ = tokenizer.encode(constraint)
            avail += [tokenizer.token_to_id(x) for x in vocab_map.values()]
            avail += [tokenizer.token_to_id("，")]
            # print(predicted.shape)
            avail = set(avail)
            # print(avail)
            for i in range(len(predicted[0])):
                if i not in avail:
                    # print(111)
                    predicted[0][i] = 0
            # predicted[0] = softmax(predicted[0])
        return (predicted, constraint)

    def generate(
        self,
        utterance_list: List[Utterance],
        question: str,
        prefix: str = "",
        constraint=None,
        topk=1,
    ):
        utterance_list = copy.deepcopy(utterance_list)
        fake_speaker = reverse_speaker(utterance_list[-1].speaker)
        fake_utterance = Utterance(
            speaker=fake_speaker,
            utterance="{}：{}".format(fake_speaker.value, question),
        )
        utterance_list.append(fake_utterance)
        x = super().generate(
            utterance_list, prefix, constraint=constraint, topk=topk
        )
        print(fake_utterance.utterance, x)
        response = "".join(x.split("：")[1:])
        assert response.startswith(prefix)
        response = response[len(prefix) :]
        if response == "无":
            return ""

        return response


def pred_diag(
    utter_list: List[Utterance],
    inquiry_log: List[InquiryLog] = [],
    compat_result=True,
):
    ans_generator = AnswerGenerator(
        start_id=None, end_id=tokenizer._token_end_id, maxlen=32
    )
    response = AttrExtractionResponse(
        speaker=utter_list[-1].speaker,
        sentence_text=utter_list[-1].utterance,
        analysis_result=[],
        round_id=0,
    )
    # print(inquiry_log)
    if len(utter_list) == 0:
        intent_list = []
    else:
        if response.speaker == Speaker.doctor:
            intent_str = ans_generator.generate(
                utter_list, "你有什么动作？", prefix="我正在"
            )
        else:
            intent_str = ans_generator.generate(
                utter_list, "你有什么意图？", prefix="我正在"
            )
        if intent_str == "":
            intent_list = []
        else:
            intent_list = intent_str.split("，")
        print(intent_list)
        if utter_list[-1].speaker == Speaker.doctor:
            intent_list = [
                i for i in intent_list if i in DoctorAction._value2member_map_
            ]
        else:
            intent_list = [
                i for i in intent_list if i in PatientIntent._value2member_map_
            ]
    if "询问" in intent_list:
        intent_list.remove("询问")
        intent_list += ans_generator.generate(
            utter_list, "你询问了哪些内容？".format(utter_list[-1].speaker), prefix="我正在"
        ).split("，")
        intent_list = [
            i for i in intent_list if i in PatientIntent._value2member_map_
        ]
    print(intent_list)
    for intent in intent_list:

        analysis_result = AnalysisResult(
            present_intent_action=intent, text=None, slots={}
        )
        slot_list = []
        for slot in intent_to_slot[intent]:
            # print(slot)
            if len(response.sentence_text) > 500:
                tmp = []
                i = 0
                while i < len(response.sentence_text):
                    # print(response.sentence_text[i : i + 50])
                    tmp.append(
                        ans_generator.generate(
                            utter_list,
                            "你提及了哪些{}？".format(response.speaker.value, slot),
                            constraint=response.sentence_text[i : i + 50]
                            + "无"
                            + "，",
                        )
                    )
                    i += 45
                tmp = [t for t in tmp if t != ""]
                content = "，".join(tmp)
            else:
                content = ans_generator.generate(
                    utter_list,
                    "你提及了哪些{}？".format(slot),
                    constraint=response.sentence_text + "无" + "，",
                )
            if (content == "" or content == "无") and len(inquiry_log) == 0:
                continue

            if slot == "症状名称":
                symptom_list = content.split("，")
                symptom_list += [
                    item.name
                    for item in inquiry_log
                    if item.type == InquiryItemType.symptom
                ]
                symptom_list = list(set(symptom_list))
                symptom_list = [s for s in symptom_list if s != ""]
                if len(symptom_list) != 0:
                    if slot not in analysis_result.slots:
                        analysis_result.slots[slot] = []
                    for s in symptom_list:
                        e = {slot: s}
                        e["是否存在"] = ans_generator.generate(
                            utter_list, "症状{}是否存在".format(s)
                        )
                        e["部位"] = ans_generator.generate(
                            utter_list, "症状{}的部位".format(s)
                        )
                        e["程度"] = ans_generator.generate(
                            utter_list, "症状{}的程度".format(s)
                        )
                        analysis_result.slots[slot].append(e)
            # elif slot == "疾病名称":
            #     disease_list = content.split("，")
            #     if slot not in analysis_result.slots:
            #         analysis_result.slots[slot] = []
            #     for disease in disease_list:
            #         e = {slot: disease}
            #         e["是否存在"] = ans_generator.generate(
            #             utter_list, "病史{}是否存在？".format(disease)
            #         )
            #         e["程度"] = ans_generator.generate(
            #             utter_list, "病史{}的程度".format(disease)
            #         )
            #         analysis_result.slots[slot].append(e)
            elif slot == "检查名称":
                examine_list = list(set(content.split("，")))
                examine_list = [s for s in examine_list if s != ""]
                if len(examine_list) != 0:
                    if slot not in analysis_result.slots:
                        analysis_result.slots[slot] = []
                    for exam in examine_list:
                        e = {slot: exam}
                        e["检查值"] = ans_generator.generate(
                            utter_list, "检查{}的检查值是多少？".format(exam)
                        )
                        analysis_result.slots[slot].append(e)
            else:
                content_list = list(set(content.split("，")))
                content_list = [s for s in content_list if s != ""]
                if len(content) != 0:
                    analysis_result.slots[slot] = content_list
        # print(analysis_result.slots)
        if compat_result:
            analysis_result.slots = compat(analysis_result.slots)
        response.analysis_result.append(analysis_result)
    print(response)
    return response


def pred(doc):
    data = preprocess(doc)
    intent_result = []
    slots_result = []
    symptom_set = set()
    for i, utter in enumerate(data):
        pred = pred_diag(data[: i + 1], compat_result=False)
        pred_intent_list = [
            r.present_intent_action.value for r in pred.analysis_result
        ]
        pred_slots = {}
        for result in pred.analysis_result:
            for slot_k, slot_v in result.slots.items():
                if slot_k not in pred_slots:
                    pred_slots[slot_k] = slot_v
        pred_slots_value = []
        for k, v in pred_slots.items():
            if v not in ["", []]:
                for item in v:
                    if isinstance(item, str):
                        pred_slots_value.append((k, item))
                    else:
                        for s_k, s_v in item.items():
                            if s_k == k:
                                pred_slots_value.append((k, s_v))
                            else:
                                if s_v == "":
                                    continue
                                pred_slots_value.append((k, item[k], s_k, s_v))
        # print("预测：")
        # print(pred_intent_list)
        # print(pred_slots_value)
        #         print(utter.frames)
        gold_intent_list = utter.intent
        gold_slots_value = []
        for item in utter.frames:
            gold_slots_value.append((item.slot_type, item.entity_content))
            for k, v in item.attribute.items():
                gold_slots_value.append(
                    (item.slot_type, item.entity_content, k, v)
                )
        # print("答案：")
        # print(gold_intent_list)
        # print(gold_slots_value)
        intent_result.append(
            (pred_intent_list, gold_intent_list, utter.utterance)
        )
        slots_result.append(
            (pred_slots_value, gold_slots_value, utter.utterance)
        )
    return intent_result, slots_result


def calculate_prf(d):
    TP, FP, FN = 0, 0, 0
    em = 0
    count = 0
    for pred, gold, r in d:
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
        # else:
        #     print(r)
        #     print(gold, "\n", pred)
        #     print("------------")
    precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
    recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
    F1 = (
        2 * precision * recall / float(precision + recall)
        if (precision + recall) != 0
        else 0
    )

    return F1, recall, precision, em / count if count != 0 else 0


class Evaluator(keras.callbacks.Callback):
    """评估与保存"""

    def __init__(self):
        self.intent_best = -1
        self.slot_best = -1
        # with open(val_path, "r") as f:
        #     self.data = json.load(f)
        self.lowest = 1e10
        # self.f = open(output_path + "_log", "w")

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        # intent_result = []
        # slots_result = []

        # for diag in self.data:
        #     intent_res, slots_res = pred(diag)
        #     intent_result.extend(intent_res)
        #     slots_result.extend(slots_res)

        # def filter(l):
        #     return [
        #         d
        #         for d in l
        #         if not (
        #             len(d) > 2
        #             and d[2] == "是否存在"
        #             or d[0] == "是否存在"
        #             or d[0] == "建议名称"
        #         )
        #     ]

        # slots_result = [(filter(i), filter(j), k) for i, j, k in slots_result]
        # i_f, _, _, _ = calculate_prf(intent_result)
        # if i_f > self.intent_best:
        #     self.intent_best = i_f
        #     model.save_weights(output_path + "_intent_{}".format(i_f))
        # s_f, _, _, _ = calculate_prf(slots_result)
        # if s_f > self.slot_best:
        #     self.slot_best = s_f
        #     model.save_weights(output_path + "_slot_{}".format(s_f))
        # if logs["loss"] <= self.lowest:
        #     self.lowest = logs["loss"]
        #     model.save_weights(output_path)
        model.save_weights(output_path + "_{}".format(epoch))
        with open(output_path + "_log", "a") as f:
            f.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        # "intent": i_f,
                        # "slot": s_f,
                        "loss": logs["loss"],
                    }
                )
            )
        # print("槽f1:{}, 意图f1:{}".format(s_f, i_f))


if __name__ == "__main__":
    with open(data_path, "r") as f:
        data_list = json.load(f)
        data_list = data_list[: int(len(data_list) * float(args.ratio))]
        for d in data_list:
            # if len(d["event_group"]) != 0:
            data.append(load_data(d))

    with open(symptom_path, "rb") as f:
        symptom_example_list = pickle.load(f)
    evaluator = Evaluator()
    train_generator = data_generator(data, batch_size)
    if args.pretrain:
        model.load_weights(pretrain_path)
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        # steps_per_epoch=1,
        epochs=epochs,
        callbacks=[evaluator],
    )
else:
    pass
    # model.load_weights(output_path)
