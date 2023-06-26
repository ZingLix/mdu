import json
import os

os.environ["TF_KERAS"] = "1"
from fastapi.params import Body
import numpy as np
from bert4keras.snippets import AutoRegressiveDecoder
from fastapi import FastAPI
from config import *
from util import *
from typing import Any, Union
from tfs_util import predictor as p
import requests
import tfs_util
import copy
from urllib import parse
from args import args

if args.schema is None:
    args.schema = "./config/schema_new.json"

app = FastAPI()
# predictor = p()

with open(args.schema) as f:
    nlu_schema = json.load(f)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


class AnswerGenerator(AnswerGeneratorBase):
    """seq2seq解码器"""

    def __init__(self, start_id, end_id, maxlen, minlen=1):
        super().__init__(start_id, end_id, maxlen, minlen=minlen)
        self.predictor = p()

    @AutoRegressiveDecoder.wraps(default_rtype="probas", use_states=True)
    def predict(self, inputs, output_ids, states):
        constraint = states
        if constraint is not None:
            avail, _ = tokenizer.encode(constraint)
            # avail += [tokenizer.token_to_id(x) for x in vocab_map.values()]
            # avail += [tokenizer.token_to_id("，")]
            # print(predicted.shape)
            avail = set(avail)
            if len(avail) == 1:
                predicted = np.zeros((1, 13588))
                for i in avail:
                    predicted[0][i] = 1
                return (predicted, constraint)
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        input = {
            "token_input": np.array(token_ids, dtype=np.float32),
            "segment_input": np.array(segment_ids, dtype=np.float32),
        }
        predicted = self.predictor.predict(
            input, tfs_util.config["batch_size"]
        )["outputs"]
        predicted = np.array(predicted)

        # constraint = None
        if constraint is not None:
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
        postfix: str = "",
        category=None,
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
        if constraint is not None:
            constraint += [x for x in vocab_map.values()] + ["[SEP]"]
        x = super().generate(
            utterance_list,
            prefix,
            constraint=constraint,
            topk=topk,
            category=category,
            sep_token=["，"],
            postfix_token=tokenizer.tokenize(postfix)[1:-1]
            + [
                vocab_map["[PATIENT]"],
                vocab_map["[DOCTOR]"],
                "[SEP]",
            ],
        )
        print(fake_utterance.utterance, x)
        response = "".join(x.split("：")[1:])
        # assert response.startswith(prefix)
        if len(prefix) != 0:
            response = response[len(prefix) :]
        if len(postfix) != 0:
            response = response[: -len(postfix)]
        if response == "无":
            return ""

        return response


def compat(result):
    cur = AnalysiSlot(
        symptom=[],
        examine_history=[],
        disease_history=[],
        disease_history_raw=[],
        medicine_history=[],
        treatment_history=[],
        taboo=[],
        basic_info={},
    )
    if "年龄" in result:
        cur.basic_info["age"] = result["年龄"][0]
    if "性别" in result:
        cur.basic_info["sex"] = result["性别"][0]
    if "禁忌名称" in result:
        cur.taboo = result["禁忌名称"]
    if "用药名称" in result:
        cur.medicine_history = result["用药名称"]
    if "治疗方式名称" in result:
        cur.treatment_history = result["治疗方式名称"]
    if "疾病名称" in result:
        cur.disease_history = result["疾病名称"]
        cur.disease_history_raw = result["疾病名称"]
        # cur.disease_history = list(
        #     set(get_normalized_entity(result["疾病名称"], "disease"))
        # )
        # cur.disease_history = []
        # for item in result["疾病名称"]:
        #     cur.disease_history.append(
        #         {
        #             "name": item["疾病名称"],
        #             "is_existed": item.get("是否存在", ""),
        #             "degree": item["程度"] if "程度" in item else "",
        #         }
        #     )
    if "检查名称" in result:
        cur.examine_history = []
        for item in result["检查名称"]:
            cur.examine_history.append(
                {
                    "name": item["检查名称"],
                    "value": item.get("检查值", ""),
                    "is_existed": item.get("是否存在", ""),
                }
            )
    if "症状名称" in result:
        cur.symptom = []
        symptom_list = []
        for symptom in result["症状名称"]:
            if symptom["部位"] != "" and symptom["部位"] not in symptom["症状名称"]:
                symptom_name = symptom["部位"] + symptom["症状名称"]
            else:
                symptom_name = symptom["症状名称"]
            symptom_list.append(symptom_name)
        normalized = get_normalized_entity(symptom_list, "symptom")
        for item, normal in zip(result["症状名称"], normalized):
            cur.symptom.append(
                {
                    "name": item["症状名称"],
                    "normalized_name": normal,
                    "body_part": item.get("部位", ""),
                    "degree": item.get("程度", ""),
                    "is_existed": item.get("是否存在", ""),
                }
            )
    return cur


def get_normalized_entity(entity, entity_type):
    data = parse.urlencode({"m_type": entity_type, "mention": entity})
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    r = requests.post(
        "http://172.20.137.215:10001/api/norm_api/", headers=headers, data=data
    )
    print(r.status_code)
    print(r.text)
    res = r.json()["data"]
    normalized_entities = [item[0][0] for item in res]
    return normalized_entities


def pred_diag(
    utter_list: List[Utterance],
    inquiry_log: InquiryLog = {},
    compat_result=True,
):
    ans_generator = AnswerGenerator(
        start_id=None, end_id=tokenizer._token_end_id, maxlen=48
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
        intent_list = []
        if response.speaker == Speaker.doctor:
            intent_schema = nlu_schema["intent"]["doctor"]

        else:
            intent_schema = nlu_schema["intent"]["patient"]

        def analyse_slots(slot_schema, field, related_info):
            res = []

            def attr_extract():
                attr = {}
                for k in slot_schema["next"].keys():
                    attr[k] = analyse_slots(
                        slot_schema["next"][k], k, related_info
                    )
                return attr

            if len(slot_schema["question"]) != 0:
                if slot_schema["answer"]["type"] == "extract":
                    constraint = "".join(
                        [
                            s.utterance
                            for s in utter_list[
                                -slot_schema["answer"]["range"] :
                            ]
                        ]
                    )
                    constraint = tokenizer.tokenize(constraint) + [
                        NONE_TOKEN,
                        SPLIT_TOKEN,
                    ]
                    content = ans_generator.generate(
                        utter_list,
                        slot_schema["question"][0].format(**related_info),
                        prefix=slot_schema["ans_prefix"].format(**related_info),
                        postfix=slot_schema["ans_postfix"].format(
                            **related_info
                        ),
                        constraint=constraint,
                    )
                elif slot_schema["answer"]["type"] == "category":
                    content = ans_generator.generate(
                        utter_list,
                        slot_schema["question"][0].format(**related_info),
                        prefix=slot_schema["ans_prefix"].format(**related_info),
                        postfix=slot_schema["ans_postfix"].format(
                            **related_info
                        ),
                        category=slot_schema["answer"]["candidates"],
                    )
                output_res = list(set(content.split("，")))
                output_res += inquiry_log.get(field, [])
                for output in output_res:
                    if output == "":
                        continue
                    related_info[field] = output
                    res.append({"value": output, "attr": attr_extract()})
            else:
                res.append({"value": "", "attr": attr_extract()})
            return res

        def analyse_utterance(schema):
            intent_str = ans_generator.generate(
                utter_list,
                schema["question"],
                prefix=schema["prefix"],
                category=list(schema["next"].keys()),
            )
            result = []
            if intent_str != "":
                intent_list = intent_str.split("，")
            else:
                intent_list = []
            intent_list = list(set(intent_list))
            cache = {}
            for intent in intent_list:
                if intent not in schema["next"]:
                    continue
                if "next" in schema["next"][intent]:
                    result.extend(analyse_utterance(schema["next"][intent]))
                else:
                    if utter_list[-1].speaker == Speaker.doctor:
                        p_intent = DoctorAction._value2member_map_[intent]
                    else:
                        p_intent = PatientIntent._value2member_map_[intent]

                    analysis_result = AnalysisResult(
                        present_intent_action=p_intent, text=None, slots={}
                    )
                    for slot in schema["next"][intent]["slots"]:
                        if slot not in cache:
                            slot_schema = nlu_schema["slots"][slot]
                            analysis_result.slots[slot] = analyse_slots(
                                slot_schema, slot, {}
                            )
                            cache[slot] = analysis_result.slots[slot]
                        else:
                            analysis_result.slots[slot] = cache[slot]
                    result.append(analysis_result)
            return result

        result = analyse_utterance(intent_schema)

        response.analysis_result = result

    return response


@dataclass
class RequestDialogueHistory(BaseModel):
    round_id: int
    speaker: Speaker
    sentence_text: str


@dataclass
class AttrExtrationRequest(BaseModel):
    dialog_history: List[RequestDialogueHistory] = Body(..., title="对话历史")
    inquiry_log: Dict[InquiryItemType, List[str]] = Body({}, title="查询内容")
    # slot_list: List[str] = Body(..., title="症状名称列表")

    class Config:
        schema_extra = {
            "example": {
                "utter_list": [
                    {
                        "utterance": "我肚子痛，头也有点痛",
                        "speaker": "患者",
                    }
                ],
                "symptom_list": ["头疼", "肚子痛"],
            }
        }


@dataclass
class AnalysisResult(BaseModel):
    present_intent_action: Union[PatientIntent, DoctorAction]
    text: Optional[str]
    slots: Any


@dataclass
class AnalysiSlot(BaseModel):
    symptom: List[Any]
    examine_history: List[Any]
    disease_history: List[Any]
    disease_history_raw: Optional[List[Any]]
    medicine_history: List[str]
    treatment_history: List[str]
    taboo: List[str]
    basic_info: Any


@dataclass
class AttrExtractionResponse(BaseModel):
    speaker: Speaker
    round_id: int
    sentence_text: str
    analysis_result: List[AnalysisResult]


@app.post(
    "/nlu",
    response_model=AttrExtractionResponse,
    name="症状属性抽取",
    description="输入历史对话以及相关的症状，输出对话最后一句的意图与所有症状的属性",
)
async def attr_extraction(req: AttrExtrationRequest = Body(...)):
    utter_list: List[Utterance] = []
    for item in req.dialog_history:
        utter_list.append(
            Utterance(
                utterance=item.sentence_text,
                speaker=item.speaker,
                frames=[],
                intent=[],
            )
        )
    response = pred_diag(utter_list, req.inquiry_log)
    response.round_id = req.dialog_history[-1].round_id
    return response
