import threading
from unicodedata import name
from fastapi.params import Query

import grpc
import tensorflow as tf
import math

import re
import random
from bert4keras.tokenizers import Tokenizer, load_vocab
from typing import Any, Dict, List, Optional
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from pydantic import BaseModel, Field
from enum import Enum
from config import config, maxlen, dict_path
import numpy as np
from bert4keras.snippets import AutoRegressiveDecoder
from typing import TYPE_CHECKING
from args import args

if TYPE_CHECKING:
    from dataclasses import dataclass
else:

    def dataclass(model):
        return model


NONE_TOKEN = "无"
SPLIT_TOKEN = "，"


class ResultCollect(object):
    def __init__(self, num_tests, concurrency):
        self._num_tests = num_tests
        self._concurrency = concurrency
        self._done = 0
        self._active = 0
        self._error = [None for _ in range(num_tests)]
        self._result = [None for _ in range(num_tests)]
        self._condition = threading.Condition()

    def add_result(self, response, index):
        with self._condition:
            self._result[index] = response

    def add_error(self, exception, index):
        with self._condition:
            self._error[index] = exception

    def inc_done(self):
        with self._condition:
            self._done += 1
            self._condition.notify()

    def dec_active(self):
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def get_result(self):
        with self._condition:
            while self._done != self._num_tests:
                self._condition.wait()
            return self._result

    def get_error(self):
        with self._condition:
            while self._done != self._num_tests:
                self._condition.wait()
            return self._error

    def throttle(self):
        with self._condition:
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1


class TFServingPredict(object):
    def __init__(
        self,
        hostport,
        model_spec_name,
        model_spec_signature_name,
        input_signatures,
        output_signatures,
        concurrency=1,
        time_out=60,
        model_version=1,
    ):

        channel = grpc.insecure_channel(
            hostport,
            options=[
                ("grpc.max_send_message_length", 256 * 1024 * 1024),
                ("grpc.max_receive_message_length", 256 * 1024 * 1024),
            ],
        )
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        self.concurrency = concurrency

        self.input_signatures = input_signatures
        self.output_signatures = output_signatures

        self.model_spec_name = model_spec_name
        self.model_spec_signature_name = model_spec_signature_name
        self.model_version = int(model_version)

        self.time_out = time_out

    def predict(self, X, batch_size):
        assert len(self.input_signatures) == len(X)

        data_size = len(X[list(self.input_signatures)[0]])
        batch_count = math.ceil(data_size / batch_size)

        idx_ranges = [
            (
                i * batch_size,
                ((i + 1) * batch_size) if i < batch_count - 1 else data_size,
            )
            for i in range(batch_count)
        ]

        result_counter = ResultCollect(
            batch_count, concurrency=self.concurrency
        )

        for idx, item in enumerate(idx_ranges):
            start, end = item
            request = predict_pb2.PredictRequest()
            request.model_spec.name = self.model_spec_name
            request.model_spec.version.value = self.model_version
            request.model_spec.signature_name = self.model_spec_signature_name

            for k in self.input_signatures:
                request.inputs[k].CopyFrom(
                    tf.make_tensor_proto(X[k][start:end])
                )

            result_counter.throttle()
            result_future = self.stub.Predict.future(request, self.time_out)
            result_future.add_done_callback(
                self._create_rpc_callback(idx, result_counter)
            )

        r = result_counter.get_result()
        # 关闭进度条

        R = {
            k: [None for _ in range(data_size)] for k in self.output_signatures
        }

        for k in self.output_signatures:
            for r_item, idx_range_item in zip(r, idx_ranges):
                if r_item is not None:
                    for i in range(*idx_range_item):
                        R[k][i] = r_item[k][i - idx_range_item[0]]

        return R

    def _create_rpc_callback(self, idx, result_counter):
        """Creates RPC callback function.

        Args:
          label: The correct label for the predicted example.
          result_counter: Counter for the prediction result.
        Returns:
          The callback function.
        """

        def _callback(result_future):
            """Callback function.

            Calculates the statistics for the prediction result.

            Args:
              result_future: Result future of the RPC.
            """
            exception = result_future.exception()
            if exception:
                print(exception)
                result_counter.add_error(exception, idx)
            else:
                response = {
                    k: tf.make_ndarray(result_future.result().outputs[k])
                    for k in self.output_signatures
                }
                result_counter.add_result(response, idx)

            result_counter.inc_done()
            result_counter.dec_active()

        return _callback


def make_predictor(model_name, inputs, outputs):
    tfs_config = config["tf_serving"]
    model_config = tfs_config["model"][model_name]
    version = 1
    if "version" in model_config:
        version = model_config["version"]
    if args.model_version is not None:
        version = int(args.model_version)
    return TFServingPredict(
        "{}:{}".format(
            tfs_config["host"],
            args.tfs_port
            if args.tfs_port is not None
            else tfs_config["rpc_port"],
        ),
        model_name,
        model_config["model_spec_signature"],
        inputs,
        outputs,
        concurrency=32
        if "concurrency" not in model_config
        else model_config["concurrency"],
        model_version=version,
    )


def model_config(model_name):
    return config["tf_serving"]["model"][model_name]


vocab_map = {
    "[PATIENT]": "[unused4]",
    "[DOCTOR]": "[unused1]",
    "[QUESTION]": "[unused2]",
    "[AND]": "[unused3]",
}


@dataclass
class Speaker(str, Enum):
    patient = "患者"
    doctor = "医生"


@dataclass
class PatientIntent(str, Enum):
    describe = "描述"
    answer = "回答"
    knowledge_ask = "知识型询问"
    polite = "客套"
    nonsense = "无意义"
    ask = "询问"
    ask_diagnosis = "询问诊断结果"
    ask_medicine = "询问推荐用药结果"
    ask_examine = "询问推荐检查结果"
    ask_department = "询问推荐科室结果"
    ask_cure = "询问笼统治疗方案"
    ask_other = "询问其他治疗建议结果"
    null = ""
    diagnose = "诊断"
    recommend = "推荐"
    knowledge_response = "知识型回答"


@dataclass
class DoctorAction(str, Enum):
    polite = "客套"
    nonsense = "无意义"
    ask = "询问"
    respond = "回答"
    diagnose = "诊断"
    recommend = "推荐"
    knowledge_response = "知识型回答"


@dataclass
class Entity(BaseModel):
    entity_content: str = Query(..., title="实体内容")
    slot_type: str = Query(..., title="实体类型")
    slot_name: str = Query(..., title="归一化后的实体内容")
    attribute: List[Any] = Query(..., title="实体相关的属性")


@dataclass
class Utterance(BaseModel):
    utterance: str = Query(..., title="用户发言")
    speaker: Speaker = Query(..., title="发言者角色")
    frames: Optional[List[Entity]] = Query([], title="相关的实体")
    intent: Optional[List[PatientIntent]] = Query([], title="患者意图或医生动作")


@dataclass
class InquiryItemType(Enum):
    symptom = "symptom"


@dataclass
class InquiryLog(BaseModel):
    type: InquiryItemType
    name: List[str]


# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + list(vocab_map.values()),
)
# token_dict = load_vocab(
#     dict_path=dict_path,
#     startswith=["[PAD]", "[UNK]", "[CLS]", "[SEP]"],
# )
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class AnswerGeneratorBase(AutoRegressiveDecoder):
    def generate(
        self,
        utterance_list: List[Utterance],
        prefix: str = "",
        constraint=None,
        category=None,
        sep_token=[],
        postfix_token=[],
        topk=3,
    ):
        assert not (category is not None and constraint is not None)
        max_c_len = maxlen - self.maxlen
        content = []
        for item in reversed(utterance_list):
            utterance = "{}：{}".format(item.speaker.value, item.utterance)
            if len(content) + len(item.utterance) > max_c_len:
                if len(content) == 0:
                    content += (
                        [vocab_map["[PATIENT]"]]
                        + tokenizer.tokenize(utterance[-(max_c_len):])[1:-1]
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

        content = ["[CLS]"] + content + ["[SEP]"]
        token_ids, segment_ids = tokenizer.encode(content, maxlen=max_c_len)
        fake_ans_speaker = reverse_speaker(utterance_list[-1].speaker)
        output_prefix = []
        if fake_ans_speaker == Speaker.patient:
            output_prefix = [vocab_map["[PATIENT]"]]
        else:
            output_prefix = [vocab_map["[DOCTOR]"]]
        output_prefix += tokenizer.tokenize(
            "{}：".format(fake_ans_speaker.value)
        )[1:-1]
        output_prefix = [np.array(tokenizer.encode(output_prefix)[0])]
        output_prefix = np.array(output_prefix)
        output_ids = self.beam_search(
            [token_ids, segment_ids],
            topk=topk,
            states=constraint,
            first_output_ids=output_prefix,
            category=category,
            sep_token=sep_token,
            prefix_token=prefix,
            postfix_token=postfix_token,
        )  # 基于beam search
        # print(output_ids)
        ans = tokenizer.decode(output_ids)
        # if ans != "":
        #     print(question, ans)
        return ans

    def beam_search(
        self,
        inputs,
        topk,
        states=None,
        temperature=1,
        min_ends=1,
        category=None,
        sep_token=[],
        prefix_token=[],
        postfix_token=[],
        first_output_ids=np.empty((1, 0), dtype=int),
    ):
        """beam search解码
        说明：这里的topk即beam size；
        返回：最优解码序列。
        """
        init_state = states
        inputs = [np.array([i]) for i in inputs]
        output_ids, output_scores = first_output_ids, np.zeros(
            len(first_output_ids)
        )
        if states is not None:
            category_trie_root = {k: {_keep: _keep, _end: _end} for k in states}
        elif category is not None:
            category_trie_root = make_trie(category)
        else:
            category_trie_root = None
        not_prefix_root = category_trie_root
        for item in reversed(prefix_token):
            category_trie_root = {item: category_trie_root}
        # if category_trie_root is not None and category is None:
        if category_trie_root is not None:
            category_trie_root[NONE_TOKEN] = _end
        current_node = category_trie_root
        for step in range(self.maxlen):
            if current_node is not None:
                # print(current_node)
                possible_token = list(current_node.keys())
                if _end in possible_token:
                    possible_token.remove(_end)
                    possible_token.extend(sep_token + postfix_token)
                if init_state is not None:
                    states = init_state + possible_token
                else:
                    states = possible_token
            scores, states = self.predict(
                inputs, output_ids, states, temperature, "logits"
            )  # 计算当前得分
            if step == 0:  # 第1步预测后将输入重复topk次
                inputs = [np.repeat(i, topk, axis=0) for i in inputs]
            scores = output_scores.reshape((-1, 1)) + scores  # 综合累积得分
            indices = scores.argpartition(-topk, axis=None)[-topk:]  # 仅保留topk
            indices_1 = indices // scores.shape[1]  # 行索引
            indices_2 = (indices % scores.shape[1]).reshape((-1, 1))  # 列索引
            output_ids = np.concatenate(
                [output_ids[indices_1], indices_2], 1
            )  # 更新输出
            output_scores = np.take_along_axis(
                scores, indices, axis=None
            )  # 更新得分
            end_counts = (output_ids == self.end_id).sum(1)  # 统计出现的end标记
            if output_ids.shape[1] >= self.minlen:  # 最短长度判断
                best_one = output_scores.argmax()  # 得分最大的那个
                if end_counts[best_one] == min_ends:  # 如果已经终止
                    return output_ids[best_one]  # 直接输出
                else:  # 否则，只保留未完成部分
                    flag = end_counts < min_ends  # 标记未完成序列
                    if not flag.all():  # 如果有已完成的
                        inputs = [i[flag] for i in inputs]  # 扔掉已完成序列
                        output_ids = output_ids[flag]  # 扔掉已完成序列
                        output_scores = output_scores[flag]  # 扔掉已完成序列
                        end_counts = end_counts[flag]  # 扔掉已完成end计数
                        topk = flag.sum()  # topk相应变化
            current_output = output_ids[0][-1]
            current_output_token = tokenizer.id_to_token(current_output)
            if current_node is not None:
                if current_output_token in sep_token:
                    current_node = not_prefix_root
                    state = None
                elif (
                    current_output_token in postfix_token
                    or current_output_token == NONE_TOKEN
                ):
                    category = None
                    current_node = None
                    states = postfix_token
                else:
                    if _keep in current_node:
                        continue
                    # print(current_output_token)
                    current_node = current_node.get(current_output_token, None)
        # 达到长度直接输出
        # print(output_scores)
        return output_ids[output_scores.argmax()]


def preprocess(doc):
    def add_entity(start_idx, entity):
        for i, item in enumerate(split_pos):
            if start_idx < item:
                utterance_list[i].frames.append(entity)
                return

    action_group = {}
    for group in doc["action_group"]:
        if group["entity_content"] not in action_group:
            action_group[group["entity_content"]] = {}
        action_group[group["entity_content"]][group["attribute_type"]] = group[
            "attribute_value"
        ]

    content = doc["doc_content"].lower()
    split_pattern = "\n\n"
    split_pos = [match.start() for match in re.finditer(split_pattern, content)]
    split_pos = split_pos + [len(content)]
    split_content = content.split(split_pattern)
    assert len(split_content) == len(split_pos)
    utterance_list: List[Utterance] = []
    for utterance in split_content:
        start_idx = utterance.find("：")
        utter = Utterance(
            utterance=utterance[start_idx + 1 :].replace("/", " "),
            frames=[],
            speaker=Speaker.patient
            if "患者" in utterance[:start_idx]
            else Speaker.doctor,
            intent=[],
        )
        utterance_list.append(utter)
    for event in doc["event_group"]:

        if event["event_group_template"] in ["患者意图", "医生动作"]:
            for en in event["entity"]:
                for i, item in enumerate(split_pos):
                    if en["start_offset"] < item:
                        utterance_list[i].intent.append(en["entity_template"])
                        break
            continue
        if event["event_group_template"] == "患者询问意图":
            start_offset = None
            en_list = []
            for entity in event["entity"]:
                if "询问" in entity["entity_template"]:
                    start_offset = entity["start_offset"]
                    for i, item in enumerate(split_pos):
                        if start_offset < item:
                            utterance_list[i].intent.append(
                                entity["entity_template"]
                            )
                            break
                    continue
                e = Entity(
                    entity_content=entity["content"],
                    slot_type=entity["entity_template"],
                    slot_name=entity["content"],
                    attribute={},
                )
                en_list.append(e)
            for en in en_list:
                add_entity(start_offset, en)
            continue
        entity_type_list = set(
            [en["entity_template"] for en in event["entity"]]
        )
        if (
            "疾病名称" in entity_type_list
            or "检查史" in entity_type_list
            or "症状名称" in entity_type_list
        ) and len(entity_type_list) != 1:
            e = Entity(
                entity_content="",
                slot_type="",
                slot_name="",
                attribute={},
            )
            for en in event["entity"]:
                if en["entity_template"] in ["疾病名称", "检查史", "症状名称"]:
                    e.entity_content = en["content"]
                    e.slot_type = en["entity_template"]
                    e.slot_name = en["content"]
                else:
                    e.attribute[en["entity_template"]] = en["content"]
            add_entity(event["entity"][0]["start_offset"], e)
            continue
        else:
            for en in event["entity"]:
                e = Entity(
                    entity_content=en["content"],
                    slot_type=en["entity_template"],
                    slot_name=en["content"],
                    attribute={}
                    if en["content"] not in action_group
                    else action_group[en["content"]],
                )

                add_entity(en["start_offset"], e)

    for i, utter in enumerate(utterance_list):
        utterance_list[i].intent = list(set(utter.intent))

        frames = utter.frames
        tmp = {}
        for f in frames:
            if f.entity_content not in tmp:
                tmp[f.entity_content] = Entity(
                    entity_content=f.entity_content,
                    slot_type=f.slot_type,
                    slot_name=f.slot_name,
                    attribute={},
                )
            for k, v in f.attribute.items():
                tmp[f.entity_content].attribute[k] = v

        utterance_list[i].frames = list(tmp.values())
    return utterance_list


def decision(probability):
    return random.random() < probability


def reverse_speaker(speaker: Speaker):
    if speaker == Speaker.patient:
        return Speaker.doctor
    return Speaker.patient


_end = "_end"
_keep = "_keep"


def make_trie(words):
    root = dict()
    for word in words:
        current_dict = root
        for letter in word:
            current_dict = current_dict.setdefault(letter, {})
        current_dict[_end] = _end
    return root


def create_attr(frames):
    res = []
    for f in frames:
        res.append(
            Entity(
                entity_content=f["value"].lower().strip(),
                slot_type=f["slot"],
                slot_name=f["value"].lower().strip(),
                attribute=[] if "attr" not in f else create_attr(f["attr"]),
            )
        )
    return res


def load_data(data):
    u_list = []
    for d in data:
        u = Utterance(
            utterance=d["utterance"].lower(),
            speaker=d["speaker"],
            intent=d["intent"],
            frames=create_attr(d["frames"]),
        )
        u_list.append(u)
    return u_list
