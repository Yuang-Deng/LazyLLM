import re

from typing import Dict, List, Optional

built_in_templates = dict(
    # Template used by Qwen.
    qwen={
        'query_prefix': (
            '<|im_start|>system\n'
            'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
            'Note that the answer can only be "yes" or "no".'
            '<|im_end|>\n<|im_start|>user\n'
        ),
        'document_suffix': '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',
        'query_template': '<Instruct>: {instruction}\n<Query>: {query}\n',
        'document_template': '<Document>: {doc}',
    },
)

default_task_instruction = 'Given a web search query, retrieve relevant passages that answer the query'
custom_templates: Dict[str, Dict[str, str]] = {}

model_patterns = {
    r'qwen.*': 'qwen'
}

local_truncate_max_chars = 16384

def register_rerank_template(
    model_name: str,
    query_prefix: str,
    document_suffix: str,
    query_template: str,
    document_template: str,
    pattern: Optional[str] = None
) -> None:
    template_config = {
        'query_prefix': query_prefix,
        'document_suffix': document_suffix,
        'query_template': query_template,
        'document_template': document_template,
    }

    custom_templates[model_name] = template_config

    if pattern:
        model_patterns[pattern] = model_name


def unregister_rerank_template(model_name: str) -> None:
    if model_name in custom_templates:
        del custom_templates[model_name]

    patterns_to_remove = [pattern for pattern, name in model_patterns.items() if name == model_name]
    for pattern in patterns_to_remove:
        del model_patterns[pattern]

def _get_template_for_model(model_name: str) -> Dict[str, str]:
    if model_name in custom_templates:
        return custom_templates[model_name]

    if model_name in built_in_templates:
        return built_in_templates[model_name]

    for pattern, template_name in model_patterns.items():
        if re.match(pattern, model_name, re.IGNORECASE):
            if template_name in custom_templates:
                return custom_templates[template_name]
            elif template_name in built_in_templates:
                return built_in_templates[template_name]
    return None

class RerankPrompter(object):
    def __init__(self, model_name: str = ''):
        self.model_name = model_name
        self.template = _get_template_for_model(model_name)
        assert self.template is not None, f'Template for model {model_name} not found'

    def build_instruct(self, query: str, instruction: str = default_task_instruction) -> str:
        return self.template['query_template'].format(
            instruction=instruction, query=query
        )

    def build_documents(self, texts: List[str], truncate_text: bool = False) -> List[str]:
        docs: List[str] = []

        def _truncate_if_needed(s: str) -> str:
            if not truncate_text:
                return s
            if len(s) <= local_truncate_max_chars:
                return s
            return s[: local_truncate_max_chars]

        for t in texts:
            t_norm = _truncate_if_needed(t or '')
            docs.append(self.template['document_template'].format(doc=t_norm) + self.template['document_suffix'])
        return docs
