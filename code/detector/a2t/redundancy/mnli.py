from collections import defaultdict
from pprint import pprint
from typing import Dict, List
from dataclasses import dataclass
import copy

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    T5ForConditionalGeneration,
)

from a2t.base import Classifier, np_softmax


@dataclass
class REInputFeatures:
    subj: str
    obj: str
    context: str
    pair_type: str = None
    label: str = None


class _NLIRelationClassifier(Classifier):
    def __init__(
        self,
        labels: List[str],
        *args,
        pretrained_model: str = "roberta-large-mnli",
        use_cuda=True,
        half=False,
        verbose=True,
        negative_threshold=0.95,
        negative_idx=0,
        max_activations=np.inf,
        valid_conditions=None,
        **kwargs,
    ):
        super().__init__(
            labels,
            pretrained_model=pretrained_model,
            use_cuda=use_cuda,
            verbose=verbose,
            half=half,
        )
        
        # self.ent_pos = entailment_position
        # self.cont_pos = -1 if self.ent_pos == 0 else 0
        self.negative_threshold = negative_threshold
        self.negative_idx = negative_idx
        self.max_activations = max_activations
        self.n_rel = len(labels)
        # for label in labels:
        #     assert '{subj}' in label and '{obj}' in label

        if valid_conditions:
            self.valid_conditions = {}
            rel2id = {r: i for i, r in enumerate(labels)}
            self.n_rel = len(rel2id)
            for relation, conditions in valid_conditions.items():
                if relation not in rel2id:
                    continue
                for condition in conditions:
                    if condition not in self.valid_conditions:
                        self.valid_conditions[condition] = np.zeros(self.n_rel)
                        self.valid_conditions[condition][rel2id["no_relation"]] = 1.0
                    self.valid_conditions[condition][rel2id[relation]] = 1.0

        else:
            self.valid_conditions = None

        def idx2label(idx):
            return self.labels[idx]

        self.idx2label = np.vectorize(idx2label)

    def _initialize(self, pretrained_model):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)
        self.config = AutoConfig.from_pretrained(pretrained_model)

        self.ent_pos = self.config.label2id.get("ENTAILMENT", self.config.label2id.get("entailment", self.config.label2id.get("LABEL_2", None)))
        if self.ent_pos is None:
            raise ValueError("The model config must contain ENTAILMENT label in the label2id dict.")
        else:
            self.ent_pos = int(self.ent_pos)

    def _run_batch(self, batch, multiclass=False):
        
        with torch.no_grad():
            input_ids = self.tokenizer.batch_encode_plus(batch, padding=True, truncation=True)
            input_ids = torch.tensor(input_ids["input_ids"]).to(self.device)
            output = self.model(input_ids)[0].detach().cpu().numpy()
            if multiclass:
                output = np.exp(output) / np.exp(output).sum(
                    -1, keepdims=True
                )  # np.exp(output[..., [self.cont_pos, self.ent_pos]]).sum(-1, keepdims=True)
            output = output[..., self.ent_pos].reshape(input_ids.shape[0] // len(self.labels), -1)

        return output
        
        # with torch.no_grad():
            
        #     # for name, param in self.model.named_parameters():
        #     #     print(name,'-->',param.type(),'-->',param.dtype,'-->',param.shape)
        #     # input("----")
        #     output = self.model(inputs_embeds=batch)[0].detach().cpu().numpy()

        #     if multiclass:
        #         output = np.exp(output) / np.exp(output).sum(
        #             -1, keepdims=True
        #         )  # np.exp(output[..., [self.cont_pos, self.ent_pos]]).sum(-1, keepdims=True)
        #     # output = output[..., self.ent_pos].reshape(input_ids.shape[0] // len(self.labels), -1)
        #     # output = output[..., self.ent_pos].reshape(batch.shape[0] // len(self.labels), -1)
        #     # print(self.labels) # self.label这里是所有tpl，共65个
        #     output = output[..., self.ent_pos].reshape(batch_size, -1)
            
            
        #     # print(output.shape) # 应该是 batch_size*所有tpl的个数 = 2*40
            
        return output

    def __call__(
        self,
        features: List[REInputFeatures],
        batch_size: int = 1,
        multiclass=False,
    ):
        if not isinstance(features, list):
            features = [features]

        batch, outputs = [], []
        # tpl_ebds = []
        # embeddings = self.model.roberta.get_input_embeddings()
        # sep_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        # sep_ebd = embeddings.weight[sep_id].reshape(-1, self.config.hidden_size)
        # cls_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        # cls_ebd = embeddings.weight[cls_id].reshape(-1, self.config.hidden_size)

        # for label_template in self.labels:
        #     tpl_input = self.tokenizer.batch_encode_plus([label_template], padding=True, truncation=True)
        #     tpl_attention_mask = torch.tensor(tpl_input["attention_mask"]).to(self.device)
        #     tpl_input = torch.tensor(tpl_input["input_ids"]).to(self.device)
        #     tpl_embed = self.model.roberta(tpl_input).last_hidden_state
        #     tpl_embed = ((tpl_embed * tpl_attention_mask.unsqueeze(-1)).sum(1) / tpl_attention_mask.sum(-1).unsqueeze(-1)).type(dtype=torch.float16)
        #     tpl_ebds.append(tpl_embed)
            
        for i, feature in tqdm(enumerate(features), total=len(features)):
            
            # sentences = [
            #     f"{feature.context} {self.tokenizer.sep_token} {label_template.format(subj=feature.subj, obj=feature.obj)}."
            #     for label_template in self.labels
            # ]
            sentences = [
                f"{feature.context} {self.tokenizer.sep_token} {label_template}"
                for label_template in self.labels
            ]
            batch.extend(sentences)
            
            # ctx_embed = ((ctx_embed * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)).type(dtype=torch.float16) # 考虑padding中不需要的内容
            
            # for tpl_embed in tpl_ebds:
            #     concat = [cls_ebd, ctx_embed, sep_ebd, tpl_embed, sep_ebd]
            #     input_pairs.append(concat)
            
            # batch.extend(input_pairs)
            
            #分隔符
            if (i + 1) % batch_size == 0:
                output = self._run_batch(batch, multiclass=multiclass)
                outputs.append(output)
                batch = []

        if len(batch) > 0:
            output = self._run_batch(batch, multiclass=multiclass)
            outputs.append(output)
            

        outputs = np.vstack(outputs)

        return outputs

    def _apply_negative_threshold(self, probs):
        activations = (probs >= self.negative_threshold).sum(-1).astype(np.int)
        idx = np.logical_or(
            activations == 0, activations >= self.max_activations
        )  # If there are no activations then is a negative example, if there are too many, then is a noisy example
        probs[idx, self.negative_idx] = 1.00
        return probs

    def _apply_valid_conditions(self, probs, features: List[REInputFeatures]):
        mask_matrix = np.stack(
            [self.valid_conditions.get(feature.pair_type, np.zeros(self.n_rel)) for feature in features],
            axis=0,
        )
        probs = probs * mask_matrix

        return probs

    def predict(
        self,
        contexts: List[str],
        batch_size: int = 1,
        return_labels: bool = True,
        return_confidences: bool = False,
        topk: int = 1,
    ):
        output = self(contexts, batch_size)
        topics = np.argsort(output, -1)[:, ::-1][:, :topk]
        if return_labels:
            topics = self.idx2label(topics)
        if return_confidences:
            topics = np.stack((topics, np.sort(output, -1)[:, ::-1][:, :topk]), -1).tolist()
            topics = [
                [(int(label), float(conf)) if not return_labels else (label, float(conf)) for label, conf in row]
                for row in topics
            ]
        else:
            topics = topics.tolist()
        if topk == 1:
            topics = [row[0] for row in topics]

        return topics


class _GenerativeNLIRelationClassifier(_NLIRelationClassifier):
    """_GenerativeNLIRelationClassifier

    This class is intended to be use with T5 like NLI models.

    TODO: Test
    """

    def _initialize(self, pretrained_model):

        if "t5" not in pretrained_model:
            raise NotImplementedError(
                f"This implementation is not available for {pretrained_model} yet. "
                "Use a t5-[small-base-large] model instead."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
        self.config = AutoConfig.from_pretrained(pretrained_model)
        self._get_entailment_neutral_contradiction_token_id()

    def _run_batch(self, batch, multiclass=False):
        with torch.no_grad():
            input_ids = self.tokenizer.batch_encode_plus(batch, padding=True, truncation=True)
            input_ids = torch.tensor(input_ids["input_ids"]).to(self.device)
            decoder_input_ids = self.tokenizer.batch_encode_plus(len(batch) * ["<pad>"], padding=True, truncation=True)
            decoder_input_ids = torch.tensor(decoder_input_ids["input_ids"]).to(self.device)
            output = self.model(input_ids, decoder_input_ids=decoder_input_ids)[0].detach().cpu().numpy()
            output = self._vocab_to_class_logits(output)
            if multiclass:
                output = np.exp(output) / np.exp(output).sum(
                    -1, keepdims=True
                )  # np.exp(output[..., [self.cont_pos, self.ent_pos]]).sum(-1, keepdims=True)
            output = output[..., 0].reshape(input_ids.shape[0] // len(self.labels), -1)

        return output

    def _get_entailment_neutral_contradiction_token_id(self):
        class_ids = self.tokenizer(["entailment", "neutral", "contradiction"]).input_ids
        (
            self.entailment_token_id,
            self.neutral_token_id,
            self.contradiction_token_id,
        ) = [ids[0] for ids in class_ids]
        assert (
            (self.entailment_token_id != self.neutral_token_id)
            and (self.entailment_token_id != self.contradiction_token_id)
            and (self.neutral_token_id != self.contradiction_token_id)
        )

    def _vocab_to_class_logits(self, outputs):
        class_logits = outputs[
            :,
            0,
            [
                self.entailment_token_id,
                self.neutral_token_id,
                self.contradiction_token_id,
            ],
        ]
        return class_logits


class NLIRelationClassifier(_NLIRelationClassifier):
    def __init__(
        self,
        labels: List[str],
        *args,
        pretrained_model: str = "roberta-large-mnli",
        **kwargs,
    ):
        super().__init__(labels, *args, pretrained_model=pretrained_model, **kwargs)

    def __call__(
        self,
        features: List[REInputFeatures],
        batch_size: int = 1,
        multiclass=True,
    ):
        outputs = super().__call__(features=features, batch_size=batch_size, multiclass=multiclass)
        outputs = np_softmax(outputs) if not multiclass else outputs
        outputs = self._apply_negative_threshold(outputs)

        return outputs


class NLIRelationClassifierWithMappingHead(_NLIRelationClassifier):
    def __init__(
        self,
        labels: List[str],
        template_mapping: Dict[str, str],
        pretrained_model: str = "roberta-large-mnli",
        valid_conditions: Dict[str, list] = None,
        *args,
        **kwargs,
    ):

        self.template_mapping_reverse = defaultdict(list)
        for key, value in template_mapping.items():
            for v in value:
                self.template_mapping_reverse[v].append(key)
        self.new_topics = list(self.template_mapping_reverse.keys())

        self.target_labels = labels
        self.new_labels2id = {t: i for i, t in enumerate(self.new_topics)}
        self.mapping = defaultdict(list)
        for key, value in template_mapping.items():
            self.mapping[key].extend([self.new_labels2id[v] for v in value])

        super().__init__(
            self.new_topics,
            *args,
            pretrained_model=pretrained_model,
            valid_conditions=None,
            **kwargs,
        )

        if valid_conditions:
            self.valid_conditions = {}
            rel2id = {r: i for i, r in enumerate(labels)}
            self.n_rel = len(rel2id)
            for relation, conditions in valid_conditions.items():
                if relation not in rel2id:
                    continue
                for condition in conditions:
                    if condition not in self.valid_conditions:
                        self.valid_conditions[condition] = np.zeros(self.n_rel)
                        self.valid_conditions[condition][rel2id["no_relation"]] = 1.0
                    self.valid_conditions[condition][rel2id[relation]] = 1.0

        else:
            self.valid_conditions = None
        
        def idx2label(idx):
            return self.target_labels[idx]

        self.idx2label = np.vectorize(idx2label)

    def __call__(self, features: List[REInputFeatures], batch_size=1, multiclass=True):
        outputs = super().__call__(features, batch_size, multiclass)
        tpl_score = copy.deepcopy(outputs)
        tpl_list = copy.deepcopy(self.labels)
        print(outputs.shape) 
        outputs = np.hstack(
            [
                np.max(outputs[:, self.mapping[label]], axis=-1, keepdims=True)
                if label in self.mapping
                else np.zeros((outputs.shape[0], 1))
                for label in self.target_labels
            ]
        )
        outputs = np_softmax(outputs) if not multiclass else outputs
        print("final outputs:")
        print(outputs.shape)
        if self.valid_conditions:
            outputs = self._apply_valid_conditions(outputs, features)

        outputs = self._apply_negative_threshold(outputs)

        return outputs, tpl_score, tpl_list


class GenerativeNLIRelationClassifier(_GenerativeNLIRelationClassifier, NLIRelationClassifier):
    pass


class GenerativeNLIRelationClassifierWithMappingHead(
    _GenerativeNLIRelationClassifier, NLIRelationClassifierWithMappingHead
):
    pass


if __name__ == "__main__":

    labels = [
        "no_relation",
        "org:members",
        "per:siblings",
        "per:spouse",
        "org:country_of_branch",
        "per:country_of_death",
        "per:parents",
        "per:stateorprovinces_of_residence",
        "org:top_members/employees",
        "org:dissolved",
        "org:number_of_employees/members",
        "per:stateorprovince_of_death",
        "per:origin",
        "per:children",
        "org:political/religious_affiliation",
        "per:city_of_birth",
        "per:title",
        "org:shareholders",
        "per:employee_of",
        "org:member_of",
        "org:founded_by",
        "per:countries_of_residence",
        "per:other_family",
        "per:religion",
        "per:identity",
        "per:date_of_birth",
        "org:city_of_branch",
        "org:alternate_names",
        "org:website",
        "per:cause_of_death",
        "org:stateorprovince_of_branch",
        "per:schools_attended",
        "per:country_of_birth",
        "per:date_of_death",
        "per:city_of_death",
        "org:founded",
        "per:cities_of_residence",
        "per:age",
        "per:charges",
        "per:stateorprovince_of_birth",
    ]
    """
    manT
    template_mapping = {
        "per:alternate_names": ["{subj} is also known as {obj}"],
        "org:members": [
            "{obj} is member of {subj}",
            "{obj} joined {subj}"
        ],
        "per:siblings": [
            "{subj} and {obj} are siblings",
            "{subj} is brother of {obj}",
            "{subj} is sister of {obj}"
        ],
        "per:spouse": [
            "{subj} is the spouse of {obj}",
            "{subj} is the wife of {obj}",
            "{subj} is the husband of {obj}"
        ],
        "org:country_of_branch": [
            "{subj} has its headquarters in {obj}",
            "{subj} is located in {obj}"
        ],
        "per:country_of_death": [
            "{subj} died in {obj}"
        ],
        "per:parents": [
            "{subj} is the parent of {obj}",
            "{subj} is the mother of {obj}",
            "{subj} is the father of {obj}",
            "{subj} is the son of {obj}",
            "{subj} is the daughter of {obj}"
        ],
        "per:stateorprovinces_of_residence": [
            "{subj} lives in {obj}",
            "{subj} has a legal order to stay in {obj}"
        ],
        "org:top_members/employees": [
            "{obj} is a high level member of {subj}",
            "{obj} is chairman of {subj}",
            "{obj} is president of {subj}",
            "{obj} is director of {subj}"
        ],
        "org:dissolved": [
            "{subj} existed until {obj}",
            "{subj} disbanded in {obj}",
            "{subj} dissolved in {obj}"
        ],
        "org:number_of_employees/members": [
            "{subj} employs nearly {obj} people",
            "{subj} has about {obj} employees"
        ],
        "per:stateorprovince_of_death": [
            "{subj} died in {obj}"
        ],
        "per:origin": [
            "{obj} is the nationality of {subj}"
        ],
        "per:children": [
            "{subj} is the parent of {obj}",
            "{subj} is the mother of {obj}",
            "{subj} is the father of {obj}",
            "{subj} is the son of {obj}",
            "{subj} is the daughter of {obj}"
        ],
        "org:political/religious_affiliation": [
            "{subj} has political affiliation with {obj}",
            "{subj} has religious affiliation with {obj}"
        ],
        "per:city_of_birth": [
            "{subj} was born in {obj}"
        ],
        "per:title": [
            "{subj} is a {obj}"
        ],
        "org:shareholders": [
            "{obj} holds shares in {subj}"
        ],
        "per:employee_of": [
            "{subj} is member of {obj}",
            "{subj} is an employee of {obj}"
        ],
        "org:member_of": [
            "{subj} is member of {obj}",
            "{subj} joined {obj}"
        ],
        "org:founded_by": [
            "{subj} was founded by {obj}",
            "{obj} founded {subj}"
        ],
        "per:countries_of_residence": [
            "{subj} lives in {obj}",
            "{subj} has a legal order to stay in {obj}"
        ],
        "per:other_family": [
            "{subj} and {obj} are family",
            "{subj} is a brother in law of {obj}",
            "{subj} is a sister in law of {obj}",
            "{subj} is the cousin of {obj}",
            "{subj} is the uncle of {obj}",
            "{subj} is the aunt of {obj}",
            "{subj} is the grandparent of {obj}",
            "{subj} is the grandmother of {obj}",
            "{subj} is the grandson of {obj}",
            "{subj} is the granddaughter of {obj}"
        ],
        "per:religion": [
            "{subj} belongs to {obj} religion",
            "{obj} is the religion of {subj}",
            "{subj} believe in {obj}"
        ],
        "per:identity": [
            "{subj} is also known as {obj}",
            "{subj} is same as {obj}"
        ],
        "per:date_of_birth": [
            "{subj}'s birthday is on {obj}",
            "{subj} was born in {obj}"
        ],
        "org:city_of_branch": [
            "{subj} has its headquarters in {obj}",
            "{subj} is located in {obj}"
        ],
        "org:alternate_names": [
            "{subj} is also known as {obj}"
        ],
        "org:website": [
            "{obj} is the URL of {subj}",
            "{obj} is the website of {subj}"
        ],
        "per:cause_of_death": [
            "{obj} is the cause of {subj}\u2019s death"
        ],
        "org:stateorprovince_of_branch": [
            "{subj} has its headquarters in {obj}",
            "{subj} is located in {obj}"
        ],
        "per:schools_attended": [
            "{subj} studied in {obj}",
            "{subj} graduated from {obj}"
        ],
        "per:country_of_birth": [
            "{subj} was born in {obj}"
        ],
        "per:date_of_death": [
            "{subj} died in {obj}"
        ],
        "per:city_of_death": [
            "{subj} died in {obj}"
        ],
        "org:founded": [
            "{subj} was founded in {obj}",
            "{subj} was formed in {obj}"
        ],
        "per:cities_of_residence": [
            "{subj} lives in {obj}",
            "{subj} has a legal order to stay in {obj}"
        ],
        "per:age": [
            "{subj} is {obj} years old"
        ],
        "per:charges": [
            "{subj} was convicted of {obj}",
            "{obj} are the charges of {subj}"
        ],
        "per:stateorprovince_of_birth": [
            "{subj} was born in {obj}"
        ]
    }
    """
    """
    # bt
    template_mapping = {
        "no_relation": ["{subj} and {obj} are not related"],
        "org:members": [
            "{obj} is a member of {subj}",
            "{obj} is member of {subj}",
            "{obj} is member of the {subj}",
            "{obj} is a member in {subj}",
            "{obj} came to {subj}",
            "{obj} was for {subj}",
            "{obj} went to {subj}",
            "{obj} joined {subj}",
            "{obj} rejoined {subj}",
            "{obj} joined {subj} in the line-up",
            "{obj} joins {subj}",
            "{obj} accedes {subj}",
            "{obj} accede {subj}"
        ],
        "per:siblings": [
            "{subj} and {obj} are siblings",
            "{subj} and {obj} are brother and sister",
            "{subj} and {obj} are brothers and sisters",
            "{subj} and {obj} are their siblings",
            "{subj} and {obj} are sionesses",
            "{subj} is brother of {obj}",
            "{subj} is brother to {obj}",
            "{subj} is the brother of {obj}",
            "{subj}. is brother of {obj}",
            "{subj} is sister of {obj}",
            "{subj}'s sister to {obj}",
            "a: {subj}, sister of {obj}",
            "{subj} is sister to {obj}"
        ],
        "per:spouse": [
            "{subj} is the spouse of {obj}",
            "{subj} is spouse of {obj}",
            "{subj} is the marital partner of {obj}",
            "{subj} is husband of {obj}",
            "the {subj} is the husband of {obj}",
            "{subj} - the spouse of {obj}",
            "{subj} - the spouse of the {obj}",
            "{subj} is the wife of {obj}",
            "{subj} is {obj}'s wife",
            "{subj} being the wife of {obj}",
            "{subj} as {obj}'s wife",
            "{subj} is wife to {obj}",
            "{subj} is wife of {obj}",
            "{subj} is {obj}'s man",
            "\"{subj} is the man of {obj}",
            "{subj} is man of {obj}",
            "{subj} is the man of {obj}",
            "{subj} is {obj}'s husband",
            "{subj} is the husband of {obj}",
            "{subj} is husband to {obj}"
        ],
        "org:country_of_branch": [
            "{subj} is headquartered in {obj}",
            "{subj} has its head office in {obj}",
            "{subj} has its headquarters in {obj}",
            "{subj} headquartered in {obj}",
            "{subj} will be headquartered in {obj}",
            "{subj} is based in {obj}",
            "{subj} is located in {obj}",
            "{subj} will be based in {obj}",
            "{subj} has her seat in {obj}",
            "{subj} is based on {obj}",
            "{subj} is in {obj}",
            "{subj} is located at {obj}",
            "{subj} located in {obj}",
            "{subj} in {obj}"
        ],
        "per:country_of_death": [
            "{subj} died in {obj}",
            "{subj} died at {obj}",
            "{subj} has died in {obj}",
            "{subj} was dead in {obj}",
            "{subj} passed away in {obj}",
            "{subj} was lost in {obj}"
        ],
        "per:parents": [
            "{subj} is the parent company of {obj}",
            "{subj}, the parent company of {obj}",
            "{subj} is {obj}'s parent company",
            "{subj} is {obj}. 's parent company",
            "{subj} is the father of {obj}",
            "{subj} is called sire of {obj}",
            "{subj} is father of {obj}",
            "{subj} is the mother of {obj}",
            "{subj} is {obj}'s mother",
            "{subj} is mum of {obj}",
            "{subj} is mother of {obj}",
            "{subj} is {obj}'s father",
            "{subj} is father to {obj}",
            "{subj} is {obj}'s dad",
            "{subj} is the son of {obj}",
            "{subj} is a son of {obj}",
            "{subj} is the daughter of {obj}",
            "{subj}. is the daughter of {obj}",
            "{subj} is daughter of {obj}",
            "{subj} is son of {obj}",
            "{subj} is the daughter of {obj} and she is",
            "{subj} is the parent of {obj}"
        ],
        "per:stateorprovinces_of_residence": [
            "{subj} lives in {obj}",
            "{subj} live in {obj}",
            "{subj} living in {obj}",
            "{subj} lived in {obj}",
            "{subj} is living in {obj}",
            "{subj} has a legal right to remain in {obj}",
            "{subj} has an order to remain in {obj}",
            "{subj} has a legal order that she remains in {obj}",
            "{subj} has legal order to remain for {obj}",
            "{subj} has a legal order to stay in {obj}",
            "{subj} has a legal system for remaining in {obj}",
            "{subj} has a legal system for staying on {obj}",
            "{subj} has the rights to remain within {obj}",
            "{subj} has a legal order to remain in {obj}",
            "{subj} has a legal decree to remain in {obj}",
            "{subj} has a legal decree to stay in {obj}",
            "{subj} has an order to stay in {obj}"
        ],
        "org:top_members/employees": [
            "{obj} is a long-standing member of {subj}",
            "{obj} is a long-term member of {subj}",
            "{obj} is a member of {subj} for years",
            "{obj} is senior member of {subj}",
            "{obj} is high-ranking member of {subj}",
            "{obj} is a high ranking member of {subj}",
            "{obj} is a high level member of {subj}",
            "{obj} is a high level member of the {subj} family",
            "{obj} is high-level member {subj}",
            "{obj} is a member of the {subj} group",
            "{obj} is member of the {subj} group",
            "{obj} is member of {subj} group",
            "{obj} is a member of the {subj} gruppe",
            "{obj} is chairman of {subj}",
            "{obj} is president of {subj}",
            "{obj}'s chairman of {subj}",
            "{obj} is president {subj}",
            "{obj} is chairman of the {subj}",
            "{obj} is the {subj} chairman",
            "{obj} is the president of {subj}",
            "{obj} is director of {subj}",
            "{obj}'s director of {subj}",
            "{obj} is the director of {subj}"
        ],
        "org:dissolved": [
            "{subj} existed prior to {obj}",
            "{subj} existed, {obj} existed",
            "{subj} existed til {obj}",
            "{subj}-up to {obj}",
            "{subj} existed until {obj}",
            "{subj} existed up to {obj}",
            "the {subj} to the {obj}",
            "{subj} existed to {obj}",
            "{subj} existed till {obj}",
            "{subj} existed up through {obj}",
            "{subj} to {obj}",
            "{subj} dissolved in {obj}",
            "{subj} solved in {obj}",
            "{subj} dissolved into {obj}",
            "{subj} resolved in {obj}",
            "{subj} resolved to {obj}",
            "{subj} in resolution as {obj}",
            "{subj} in {obj} dissolved",
            "{subj} in {obj} resolved",
            "{subj} fractured into {obj}",
            "{subj} broke into {obj}",
            "{subj} breaks up in {obj}",
            "{subj} frames up into the {obj}",
            "{subj} into {obj}",
            "{subj} disbanded into {obj}",
            "{subj} resolved into {obj}",
            "{subj} dissolved to {obj}",
            "{subj} resolution in {obj}",
            "{subj} disbanded in {obj}"
        ],
        "org:number_of_employees/members": [
            "{subj} employs almost {obj} people",
            "{subj} employs nearly {obj} people",
            "{subj} almost employs {obj} people",
            "{subj} employs nearly {obj} persons",
            "{subj} employs almost {obj} staff",
            "{subj} almost employs {obj}",
            "{subj} almost employs {obj} staff",
            "{subj} nearly employs {obj} people",
            "{subj} employs nearly {obj} staff",
            "{subj} has about {obj}-employees",
            "{subj} has around {obj} employees",
            "{subj} has {obj}-persons",
            "{subj} has about {obj} employees",
            "{subj} has around {obj}-employees",
            "{subj} has about {obj}-co",
            "{subj} has about {obj} staff",
            "{subj} has approximately {obj}-staff",
            "{subj} has about {obj} people",
            "{subj} has about {obj}-employer"
        ],
        "per:stateorprovince_of_death": [
            "{subj} died in {obj}"
        ],
        "per:origin": [
            "{obj} is the nationality of {subj}",
            "{obj} is the citizenship of {subj}",
            "{obj} is {subj}'s nationality"
        ],
        "per:children": [
            "{subj} is the mother of {obj}",
            "{subj} is mom to {obj}",
            "{subj} is {obj}'s mother",
            "{subj} is m {obj}'s mother",
            "{subj} is the parent of {obj}",
            "{subj} is the parents of {obj}",
            "{subj} is the daughter of {obj}",
            "{subj} is the mum of {obj}",
            "{subj}'s the mother of {obj}",
            "{subj} is mother of {obj}",
            "{subj} is the father of {obj}",
            "{subj} is the sire of {obj}",
            "{subj}'s the father of {obj}",
            "{subj} is the son of {obj}",
            "{subj} is son of {obj}",
            "{subj} is {obj}'s son",
            "{subj} is the daughter with {obj}",
            "{subj} is daughter of {obj}"
        ],
        "org:political/religious_affiliation": [
            "{subj} has political connection to {obj}",
            "{subj} is linked to {obj}",
            "{subj} has political ties to {obj}",
            "{subj} has political links to {obj}",
            "{subj} has political link to {obj}",
            "{subj} has political affinity to {obj}",
            "{subj} has a political relationship with {obj}",
            "{subj} has political affinities with {obj}",
            "{subj} is related to {obj}",
            "{subj} is a political member of the {obj}",
            "{subj} is political member of {obj}",
            "{subj} is a political member of {obj}",
            "{subj} has religious affiliation to {obj}",
            "{subj} has religious affiliation with {obj}",
            "{subj} has religious allegiance to {obj}",
            "{subj} has affiliation with {obj}",
            "{subj} is for religious affiliation with {obj}",
            "{subj} is connected to {obj}",
            "{subj} is connected with {obj}",
            "{subj} is attached with {obj}",
            "{subj} has political affiliation with {obj}"
        ],
        "per:city_of_birth": [
            "{subj} was born in {obj}",
            "{subj} was born into {obj}",
            "{subj} is born in {obj}"
        ],
        "per:title": [
            "{subj} is a {obj}",
            "{subj} is for {obj}"
        ],
        "org:shareholders": [
            "{obj} holds shares in {subj}",
            "{obj} holds interest in {subj}",
            "{obj} shares in {subj}",
            "{obj} owns shares in {subj}",
            "{obj} holds stake in {subj}",
            "{obj} shareholding {subj}",
            "{obj} is holding shares in {subj}",
            "{obj} holds stock in {subj}",
            "{obj} hold shares in {subj}"
        ],
        "per:employee_of": [
            "{subj} is a member of {obj}",
            "{subj} is member of {obj}",
            "{subj} is an employee of {obj}",
            "{subj} is a fellow of {obj}",
            "{subj} is a colleague of {obj}",
            "{subj} is an associate of {obj}",
            "{subj} is an employee who works for {obj}",
            "{subj} is one of the {obj} staff",
            "{subj} is one of {obj}'s staff",
            "{subj} is one of the team of {obj}",
            "{subj} is one of the guys from {obj}"
        ],
        "org:member_of": [
            "{subj} is a member of {obj}",
            "{subj} is member of {obj}",
            "{subj} is members of {obj}",
            "{subj} is an affiliate of {obj}",
            "{subj} is a partner of {obj}",
            "{subj} is partner of {obj}",
            "{subj} to be a partner of {obj}",
            "{subj} joined {obj}",
            "{subj}'s joined {obj}'s",
            "{subj}'s joined {obj}",
            "{subj} joined {obj}'s",
            "{subj} became member {obj}",
            "{subj} joined the {obj} team",
            "{subj} joined the team {obj}",
            "{subj} and {obj}"
        ],
        "org:founded_by": [
            "{subj} was founded by {obj}",
            "{obj} {subj}",
            "{obj} / {subj}",
            "{obj} {subj} {subj} {subj}",
            "{obj} founded {subj}"
        ],
        "per:countries_of_residence": [
            "{subj} lives in {obj}",
            "{subj} lives on at {obj}",
            "{subj} has a legal order to stay in {obj}",
            "{subj} has the legal order to remain in {obj}",
            "{subj} has legal order to remain in {obj}",
            "{subj} has a legal direction to stay in {obj}",
            "{subj} has the legal right to stay in {obj}",
            "{subj} has a legal order to remain in {obj}",
            "{subj} has a legal order to stay {obj}",
            "{subj} has a legal order to remain {obj}",
            "{subj} has a legal system in place to allow {obj} to remain",
            "{subj} has legal authority to stay {obj}"
        ],
        "per:other_family": [
            "{subj} and {obj} are family",
            "{obj} and {subj} are family",
            "{subj} and {obj} are a family",
            "{subj} and {obj} are family values",
            "{subj} and {obj} is family",
            "{subj} is a brother-in-law of {obj}",
            "{subj} is the brother-in-law of {obj}",
            "{subj} is brother in law to {obj}",
            "{subj} is a brother-in-law {obj}",
            "{subj} is a brother in law {obj}",
            "{subj} is the son-in-law of {obj}",
            "{subj} is son-in-law of {obj}",
            "{subj} is a {obj}'s son-in-law",
            "the {subj} is the {obj}'s son-in-law",
            "{subj} is a sister in law {obj}",
            "{subj} is brother-in-law {obj}",
            "{subj} \"brother-in-law {obj}",
            "{subj} is a sister of {obj}",
            "{subj} is a sister of the {obj}",
            "{subj} is sister of {obj}",
            "{subj} is sister to {obj}",
            "{subj} is a sister-in-law of {obj}",
            "{subj} is a sister in law of {obj}",
            "{subj} is one of {obj}'s sisters",
            "{subj}. is a sister of {obj}",
            "{subj} is a sister to {obj}, the sister of {obj}",
            "{subj} is the cousin of {obj}",
            "{subj} is a cousin of {obj}",
            "{subj} is cousin of {obj}",
            "{subj} is the relative of {obj}",
            "{subj} is {obj}'s cousin",
            "{subj} was the cousin of {obj}",
            "{subj} is the uncle of {obj}",
            "{subj} is {obj}'s uncle",
            "{subj}. is the uncle of the {obj}",
            "{subj} is uncle of {obj}",
            "{subj} is the uncle of the {obj}",
            "{subj} is uncle {obj}",
            "{subj} is the uncle {obj}",
            "{subj} is his uncle, {obj}",
            "{subj} is the aunt of {obj}",
            "{subj} is {obj}'s aunt",
            "{subj} is aunt of {obj}",
            "{subj} is {obj}",
            "{subj} the aunt of {obj}",
            "{subj} is grandmother of {obj}",
            "{subj} is the grandmother of {obj}",
            "{subj} is a grandmother of {obj}",
            "{subj} is the grandparent of {obj}",
            "{subj} is grandparent to {obj}",
            "{subj} is grandparent of {obj}'s daughter",
            "{subj} becomes grandmother of {obj}",
            "{subj} becomes grandmother to {obj}",
            "{subj} becomes grandma of {obj}",
            "{subj} is becoming grandmother of {obj}",
            "{subj} is the grandparent boy of {obj}",
            "{subj} is the grandma of {obj}",
            "{subj} is the grandson of {obj}",
            "{subj} is {obj}'s grandson",
            "{subj} is grandson of {obj}",
            "{subj} is the grand grand of {obj}",
            "{subj} is the granddaughter of {obj}",
            "{subj} is {obj}'s granddaughter",
            "{subj} is the grand daughter of {obj}",
            "{subj} is {obj}'s deposed daughter",
            "{subj} is the most escaped of {obj}",
            "{subj} is the most horrifying of {obj}'s",
            "{subj} is {obj}'s most unloved child",
            "{subj} is a brother in law of {obj}"
        ],
        "per:religion": [
            "{subj} belongs to {obj} religion",
            "{subj} belongs to {obj}-religion",
            "{subj} belongs to the {obj} religion",
            "{subj} is part of the {obj} religion",
            "{subj} is a member of the {obj} religion",
            "{subj} belongs to the church's {obj} religion",
            "{subj} belongs to the religion {obj}",
            "{subj} belongs to religion {obj}",
            "{obj} is the religion of {subj}",
            "{obj} is religion in {subj}",
            "{obj} means the religion of {subj}",
            "{obj} is religion of {subj}",
            "{subj} believe in {obj}",
            "{subj} believes in {obj}"
        ],
        "per:identity": [
            "{subj} is also known as {obj}",
            "{subj} is also referred to as {obj}",
            "{subj} is also sometimes referred to as {obj}",
            "{subj} will also be known as {obj}",
            "{subj} is the same as {obj}",
            "{subj} is like {obj}",
            "{subj} is equal to {obj}",
            "{subj} is same as {obj}"
        ],
        "per:date_of_birth": [
            "{subj} has a birthday: {obj}",
            "{subj} is birthday: {obj}",
            "{subj} has birthday: {obj}",
            "{subj} will have birthday {obj}",
            "{subj} will have birthdays {obj}",
            "{subj} is for birthday {obj}",
            "{subj} was born in {obj}",
            "{subj}'s born in {obj}",
            "{subj} was born at {obj}",
            "{subj} was born the {obj}",
            "{subj} is born in {obj}",
            "{subj} is born into {obj}",
            "{subj}'s birthday is on {obj}"
        ],
        "org:city_of_branch": [
            "{subj} is headquartered in {obj}",
            "{subj} has its headquarters in {obj}",
            "{subj} headquartered in {obj}",
            "{subj} has its headoffice at {obj}",
            "{subj} has a headquarter in {obj}",
            "{subj} is located in {obj}",
            "{subj} is located at {obj}",
            "{subj} in {obj}",
            "{subj} is in {obj}",
            "{subj}'s in {obj}",
            "{subj} is contained in {obj}",
            "{subj} in {obj} on the ground",
            "{subj} is into {obj}"
        ],
        "org:alternate_names": [
            "{subj} is also known as {obj}",
            "{subj} is also referred to as {obj}",
            "{subj} is also named {obj}",
            "{subj} is also called {obj}",
            "{subj} will also be referred to as {obj}"
        ],
        "org:website": [
            "{obj} is the web address of {subj}",
            "{obj} is the webaddress of {subj}",
            "{obj} is the website of {subj}",
            "{obj} is the web page of {subj}",
            "{obj} is the {subj} webpage",
            "{obj} is the web site of {subj}",
            "{obj} is the site of {subj}",
            "{obj} is the website by {subj}",
            "{obj} is the side of {subj}",
            "{obj} is site of {subj}",
            "{obj} is the url of {subj}"
        ],
        "per:cause_of_death": [
            "{obj} is the cause of death for {subj}",
            "{obj} is the cause of death in {subj}",
            "{obj} is the cause of {subj}'s death",
            "{obj} is the cause of death of {subj}",
            "{obj} is the cause of death {subj}",
            "{obj} is for cause of death of {subj}",
            "{obj} is cause of death {subj} was poisoned",
            "{obj} is the cause of {subj}'s death"
        ],
        "org:stateorprovince_of_branch": [
            "{subj} has its headquarters in {obj}",
            "{subj} has its head office in {obj}",
            "{subj} is based in {obj}",
            "{subj} has its headquarters at {obj}",
            "{subj} headquarters are located in {obj}",
            "{subj} is headquartered in {obj}",
            "{subj} headquarters is located in {obj}",
            "{subj} is situated in {obj}",
            "{subj} is located in {obj}",
            "{subj} is in {obj}",
            "{subj} is at {obj}",
            "{subj} in {obj}"
        ],
        "per:schools_attended": [
            "{subj} studying in {obj}",
            "{subj} in {obj}",
            "{subj}, {obj}",
            "{subj} studies in {obj}",
            "{subj} studied in {obj}",
            "{subj} will study at {obj}",
            "{subj} is in {obj}",
            "{subj}-level studies at {obj}",
            "{subj}, {obj}, {obj}",
            "{subj} graduates from {obj}",
            "{subj} graduated from {obj}",
            "{subj} graduates of {obj}",
            "{subj} graduates at {obj}",
            "{subj} graduates with {obj}",
            "{subj} grades in {obj}",
            "{subj} graduated at {obj}"
        ],
        "per:country_of_birth": [
            "{subj} was born in {obj}",
            "{subj} is born in {obj}"
        ],
        "per:date_of_death": [
            "{subj} died in {obj}",
            "{subj} died inside {obj}",
            "{subj} death in {obj}",
            "{subj} dies in {obj}"
        ],
        "per:city_of_death": [
            "{subj} died in {obj}",
            "{subj} killed in {obj}",
            "{subj} is in {obj}",
            "{subj} died at {obj}",
            "{subj} passed away in {obj}",
            "{subj} died on {obj}"
        ],
        "org:founded": [
            "{subj} was founded in {obj}",
            "{subj} has been founded in {obj}",
            "{subj} was established in {obj}",
            "{subj} was formed in {obj}",
            "{subj} is founded in {obj}",
            "{subj} was created in {obj}",
            "{subj} was born in {obj}",
            "{subj} has been established in {obj}",
            "{subj} was in {obj}",
            "{subj} was founded under {obj}",
            "{subj} was made in {obj}",
            "{subj} has been formed into {obj}",
            "{subj} has been set in {obj}",
            "{subj} has been given the name {obj}",
            "{subj} was erected in {obj}",
            "{subj} was built in {obj}",
            "{subj}, {obj}"
        ],
        "per:cities_of_residence": [
            "{subj} lives in {obj}",
            "{subj} living in {obj}",
            "{subj} has a legal order to stay in {obj}",
            "{subj} has a statutory order to stay in {obj}",
            "{subj} has a legal order to remain in {obj}",
            "{subj} has a legal system to stay in {obj}",
            "{subj} has a right to remain in {obj}",
            "{subj} has a right to remain within {obj}",
            "{subj} has a legal requirement to remain in {obj}",
            "{subj} has the legal provision to remain in {obj}",
            "{subj} has the legal authority to remain within {obj}",
            "{subj} has the legal prescription to retain the position in {obj}",
            "{subj} has a legal provision in {obj} remaining",
            "{subj} has a legal obligation to remain in {obj}",
            "{subj} has a legal designation remain in {obj}",
            "{subj} has a legal requirement under the law that {obj} is left at"
        ],
        "per:age": [
            "{subj} is {obj} years old",
            "{subj} is {obj} {obj}ears old",
            "{subj} has {obj} years of age",
            "{subj} is {obj}",
            "{subj} is {obj}- years old"
        ],
        "per:charges": [
            "{subj} was condemned by {obj}",
            "{subj} was convicted by {obj}",
            "{subj} is convicted by {obj}",
            "{subj} was sentenced by {obj}",
            "{subj} was convicted of {obj}",
            "{subj} was convicted because of {obj}",
            "{subj} was convicted of {obj} offences",
            "{subj} sentenced to {obj}",
            "{obj} is the cost of {subj}",
            "{obj} are the cost of {subj}",
            "{obj} are the cost of a {subj}",
            "{obj} are the fees of {subj}",
            "{obj} is the rate of {subj}",
            "{obj} are the charges of {subj}",
            "{obj} is the charge of {subj}"
        ],
        "per:stateorprovince_of_birth": [
            "{subj} was born in {obj}",
            "{subj} went born in {obj}",
            "{subj} has been born {obj}",
            "{subj} was born into {obj}"
        ]
    }
    """
    
    # soft
    template_mapping = {
        "no_relation": ["{obj} [V293] {subj} [V294] [V295] [V296]"],
        "org:members": [
            "{obj} [V0] [V1] [V2] {subj}",
            "{obj} [V3] {subj}"
        ],
        "per:siblings": [
            "{subj} [V4] {obj} [V5] [V6]",
            "{subj} [V7] [V8] [V9] {obj}",
            "{subj} [V10] [V11] [V12] {obj}"
        ],
        "per:spouse": [
            "{subj} [V13] [V14] [V15] [V16] {obj}",
            "{subj} [V17] [V18] [V19] [V20] {obj}",
            "{subj} [V21] [V22] [V23] [V24] {obj}"
        ],
        "org:country_of_branch": [
            "{subj} [V25] [V26] [V27] [V28] {obj}",
            "{subj} [V29] [V30] [V31] {obj}"
        ],
        "per:country_of_death": [
            "{subj} [V32] [V33] {obj}"
        ],
        "per:parents": [
            "{subj} [V34] [V35] [V36] [V37] {obj}",
            "{subj} [V38] [V39] [V40] [V41] {obj}",
            "{subj} [V42] [V43] [V44] [V45] {obj}",
            "{subj} [V46] [V47] [V48] [V49] {obj}",
            "{subj} [V50] [V51] [V52] [V53] {obj}"
        ],
        "per:stateorprovinces_of_residence": [
            "{subj} [V54] [V55] {obj}",
            "{subj} [V56] [V57] [V58] [V59] [V60] [V61] [V62] {obj}"
        ],
        "org:top_members/employees": [
            "{obj} [V63] [V64] [V65] [V66] [V67] [V68] {subj}",
            "{obj} [V69] [V70] [V71] {subj}",
            "{obj} [V72] [V73] [V74] {subj}",
            "{obj} [V75] [V76] [V77] {subj}"
        ],
        "org:dissolved": [
            "{subj} [V78] [V79] {obj}",
            "{subj} [V80] [V81] {obj}",
            "{subj} [V82] [V83] {obj}"
        ],
        "org:number_of_employees/members": [
            "{subj} [V84] [V85] {obj} [V86]",
            "{subj} [V87] [V88] {obj} [V89]"
        ],
        "per:stateorprovince_of_death": [
            "{subj} [V90] [V91] {obj}"
        ],
        "per:origin": [
            "{obj} [V92] [V93] [V94] [V95] {subj}"
        ],
        "per:children": [
            "{subj} [V96] [V97] [V98] [V99] {obj}",
            "{subj} [V100] [V101] [V102] [V103] {obj}",
            "{subj} [V104] [V105] [V106] [V107] {obj}",
            "{subj} [V108] [V109] [V110] [V111] {obj}",
            "{subj} [V112] [V113] [V114] [V115] {obj}"
        ],
        "org:political/religious_affiliation": [
            "{subj} [V116] [V117] [V118] [V119] {obj}",
            "{subj} [V120] [V121] [V122] [V123] {obj}"
        ],
        "per:city_of_birth": [
            "{subj} [V124] [V125] [V126] {obj}"
        ],
        "per:title": [
            "{subj} [V127] [V128] {obj}"
        ],
        "org:shareholders": [
            "{obj} [V129] [V130] [V131] {subj}"
        ],
        "per:employee_of": [
            "{subj} [V132] [V133] [V134] {obj}",
            "{subj} [V135] [V136] [V137] [V138] {obj}"
        ],
        "org:member_of": [
            "{subj} [V139] [V140] [V141] {obj}",
            "{subj} [V142] {obj}"
        ],
        "org:founded_by": [
            "{subj} [V143] [V144] [V145] {obj}",
            "{obj} [V146] {subj}"
        ],
        "per:countries_of_residence": [
            "{subj} [V147] [V148] {obj}",
            "{subj} [V149] [V150] [V151] [V152] [V153] [V154] [V155] {obj}"
        ],
        "per:other_family": [
            "{subj} [V156] {obj} [V157] [V158]",
            "{subj} [V159] [V160] [V161] [V162] [V163] [V164] {obj}",
            "{subj} [V165] [V166] [V167] [V168] [V169] [V170] {obj}",
            "{subj} [V171] [V172] [V173] [V174] {obj}",
            "{subj} [V175] [V176] [V177] [V178] {obj}",
            "{subj} [V179] [V180] [V181] [V182] {obj}",
            "{subj} [V183] [V184] [V185] [V186] {obj}",
            "{subj} [V187] [V188] [V189] [V190] {obj}",
            "{subj} [V191] [V192] [V193] [V194] {obj}",
            "{subj} [V195] [V196] [V197] [V198] {obj}"
        ],
        "per:religion": [
            "{subj} [V199] [V200] {obj} [V201]",
            "{obj} [V202] [V203] [V204] [V205] {subj}",
            "{subj} [V206] [V207] {obj}"
        ],
        "per:identity": [
            "{subj} [V208] [V209] [V210] [V211] {obj}",
            "{subj} [V212] [V213] [V214] {obj}"
        ],
        "per:date_of_birth": [
            "{subj} [V215] [V216] [V217] [V218] {obj}",
            "{subj} [V219] [V220] [V221] {obj}"
        ],
        "org:city_of_branch": [
            "{subj} [V222] [V223] [V224] [V225] {obj}",
            "{subj} [V226] [V227] [V228] {obj}"
        ],
        "org:alternate_names": [
            "{subj} [V229] [V230] [V231] [V232] {obj}"
        ],
        "org:website": [
            "{obj} [V233] [V234] [V235] [V236] {subj}",
            "{obj} [V237] [V238] [V239] [V240] {subj}"
        ],
        "per:cause_of_death": [
            "{obj} [V241] [V242] [V243] [V244] {subj} [V245] [V246]"
        ],
        "org:stateorprovince_of_branch": [
            "{subj} [V247] [V248] [V249] [V250] {obj}",
            "{subj} [V251] [V252] [V253] {obj}"
        ],
        "per:schools_attended": [
            "{subj} [V254] [V255] {obj}",
            "{subj} [V256] [V257] {obj}"
        ],
        "per:country_of_birth": [
            "{subj} [V258] [V259] [V260] {obj}"
        ],
        "per:date_of_death": [
            "{subj} [V261] [V262] {obj}"
        ],
        "per:city_of_death": [
            "{subj} [V263] [V264] {obj}"
        ],
        "org:founded": [
            "{subj} [V265] [V266] [V267] {obj}",
            "{subj} [V268] [V269] [V270] {obj}"
        ],
        "per:cities_of_residence": [
            "{subj} [V271] [V272] {obj}",
            "{subj} [V273] [V274] [V275] [V276] [V277] [V278] [V279] {obj}"
        ],
        "per:age": [
            "{subj} [V280] {obj} [V281] [V282]"
        ],
        "per:charges": [
            "{subj} [V283] [V284] [V285] {obj}",
            "{obj} [V286] [V287] [V288] [V289] {subj}"
        ],
        "per:stateorprovince_of_birth": [
            "{subj} [V290] [V291] [V292] {obj}"
        ]
    }
    

    rules = {
        "org:founded_by": [
            "ORGANIZATION:PERSON"
        ],
        "per:identity": [
            "PERSON:PERSON",
            "PERSON:MISC"
        ],
        "org:alternate_names": [
            "ORGANIZATION:ORGANIZATION",
            "ORGANIZATION:MISC"
        ],
        "per:children": [
            "PERSON:PERSON"
        ],
        "per:origin": [
            "PERSON:NATIONALITY",
            "PERSON:COUNTRY",
            "PERSON:LOCATION"
        ],
        "per:countries_of_residence": [
            "PERSON:COUNTRY",
            "PERSON:NATIONALITY"
        ],
        "per:employee_of": [
            "PERSON:ORGANIZATION"
        ],
        "per:title": [
            "PERSON:TITLE"
        ],
        "org:city_of_branch": [
            "ORGANIZATION:CITY",
            "ORGANIZATION:LOCATION"
        ],
        "per:religion": [
            "PERSON:RELIGION"
        ],
        "per:age": [
            "PERSON:DURATION",
            "PERSON:NUMBER"
        ],
        "per:date_of_death": [
            "PERSON:DATE"
        ],
        "org:website": [
            "ORGANIZATION:URL"
        ],
        "per:stateorprovinces_of_residence": [
            "PERSON:STATE_OR_PROVINCE"
        ],
        "org:top_members/employees": [
            "ORGANIZATION:PERSON"
        ],
        "org:number_of_employees/members": [
            "ORGANIZATION:NUMBER"
        ],
        "org:members": [
            "ORGANIZATION:ORGANIZATION",
            "ORGANIZATION:COUNTRY"
        ],
        "org:country_of_branch": [
            "ORGANIZATION:COUNTRY"
        ],
        "per:spouse": [
            "PERSON:PERSON"
        ],
        "org:stateorprovince_of_branch": [
            "ORGANIZATION:STATE_OR_PROVINCE"
        ],
        "org:political/religious_affiliation": [
            "ORGANIZATION:IDEOLOGY",
            "ORGANIZATION:RELIGION"
        ],
        "org:member_of": [
            "ORGANIZATION:LOCATION",
            "ORGANIZATION:ORGANIZATION",
            "ORGANIZATION:STATE_OR_PROVINCE",
            "ORGANIZATION:COUNTRY"
        ],
        "per:siblings": [
            "PERSON:PERSON"
        ],
        "per:stateorprovince_of_birth": [
            "PERSON:STATE_OR_PROVINCE"
        ],
        "org:dissolved": [
            "ORGANIZATION:DATE"
        ],
        "per:other_family": [
            "PERSON:PERSON"
        ],
        "org:shareholders": [
            "ORGANIZATION:PERSON",
            "ORGANIZATION:ORGANIZATION"
        ],
        "per:parents": [
            "PERSON:PERSON"
        ],
        "per:charges": [
            "PERSON:CRIMINAL_CHARGE"
        ],
        "per:schools_attended": [
            "PERSON:ORGANIZATION"
        ],
        "per:cause_of_death": [
            "PERSON:CAUSE_OF_DEATH"
        ],
        "per:city_of_death": [
            "PERSON:LOCATION",
            "PERSON:CITY"
        ],
        "per:stateorprovince_of_death": [
            "PERSON:STATE_OR_PROVICE"
        ],
        "org:founded": [
            "ORGANIZATION:DATE"
        ],
        "per:country_of_death": [
            "PERSON:COUNTRY"
        ],
        "per:country_of_birth": [
            "PERSON:COUNTRY"
        ],
        "per:date_of_birth": [
            "PERSON:DATE"
        ],
        "per:cities_of_residence": [
            "PERSON:CITY",
            "PERSON:LOCATION"
        ],
        "per:city_of_birth": [
            "PERSON:CITY"
        ]
    }

    clf = NLIRelationClassifierWithMappingHead(
        labels=labels,
        template_mapping=template_mapping,
        pretrained_model="microsoft/deberta-v2-xlarge-mnli",
        valid_conditions=rules,
    )

    features = [
        REInputFeatures(
            subj="Billy Mays",
            obj="Tampa",
            pair_type="PERSON:CITY",
            context="Billy Mays, the bearded, boisterous pitchman who, as the undisputed king of TV yell and sell, became an unlikely pop culture icon, died at his home in Tampa, Fla, on Sunday",
            label="per:city_of_death",
        ),
        REInputFeatures(
            subj="Old Lane Partners",
            obj="Pandit",
            pair_type="ORGANIZATION:PERSON",
            context="Pandit worked at the brokerage Morgan Stanley for about 11 years until 2005, when he and some Morgan Stanley colleagues quit and later founded the hedge fund Old Lane Partners.",
            label="org:founded_by",
        ),
        # REInputFeatures(subj='He', obj='University of Maryland in College Park', context='He received an undergraduate degree from Morgan State University in 1950 and applied for admission to graduate school at the University of Maryland in College Park.', label='no_relation')
    ]

    predictions = clf(features, multiclass=True)
    for pred in predictions:
        pprint(sorted(list(zip(pred, labels)), reverse=True)[:5])
