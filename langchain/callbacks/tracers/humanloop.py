"""A Tracer implementation that logs runs to Humanloop."""
from __future__ import annotations

import copy
import logging
from abc import ABC
import os
import re
import datetime
from enum import Enum
import inspect
from pydantic import BaseModel, Field, validator
from typing import Any, Dict, Optional, Union, List, Tuple, Callable

import requests

from langchain.callbacks.tracers.base import BaseTracer, TracerException
from langchain.callbacks.tracers.schemas import (
    TracerSession,
    RunTypeEnum,
    TracerSessionCreate,
    Run,
)
from langchain.input import get_colored_text

HUMANLOOP_APP_URL = "https://app.humanloop.com"
LLM_CHAIN_TYPE = "llm_chain"
AGENT_CHAIN_TYPE = "agent_executor_chain"
ROLE_MAPPING = {"ai": "assistant", "human": "user", "system": "system"}


class HumanloopTracer(BaseTracer, ABC):
    """An implementation of BaseTracer that logs trace data to the Humanloop API

    Inherits from ABC to avoid having to implement all the base abstract methods, some
    of which seem redundant.
    """
    _headers: Dict[str, Any] = {"Content-Type": "application/json"}
    if os.getenv("HUMANLOOP_URL"):
        _base_url = os.getenv("HUMANLOOP_URL")
    else:
        _base_url: str = "https://api.humanloop.com"
    if os.getenv("HUMANLOOP_API_KEY"):
        _headers["X-API-KEY"] = os.getenv("HUMANLOOP_API_KEY")

    def __init__(self, app_name: Optional[str] = None):
        """Initialize the Humanloop tracer."""
        super().__init__()
        self.app_name = app_name
        if app_name is None and os.getenv("HUMANLOOP_APP_NAME"):
            self.app_name = os.getenv("HUMANLOOP_APP_NAME")

    def _persist_run(self, run: Run) -> None:
        """Persist a run."""
        try:
            logs = self._convert_run_to_logs(run=run)
            if self.app_name is not None:
                logs[0].function_name = self.app_name
            self._post_trace(logs=logs)
        except Exception as error:
            logging.warning(f"Failed to persist run: {error}")

    def _convert_run_to_logs(
        self, run: Run, parent_log: Optional[Log] = None
    ) -> List[Log]:
        """Construct a HL trace by recursively processing the LC run."""
        logs: List[Log] = []
        duration = (run.end_time - run.start_time).total_seconds()
        metadata = {"langchain_run_id": str(run.id), **run.extra}
        run_type = run.serialized.get("_type")
        if run.run_type == RunTypeEnum.chain:
            if run_type == LLM_CHAIN_TYPE:
                child_llm_runs = [
                    child_run
                    for child_run in run.child_runs
                    if child_run.run_type == RunTypeEnum.llm
                ]
                run.child_runs = [
                    child_run
                    for child_run in run.child_runs
                    if child_run.run_type != RunTypeEnum.llm
                ]
                inputs = {
                    input_: value
                    for input_, value in run.inputs.items()
                    if input_ in run.serialized["prompt"]["input_variables"]
                }
                for llm_run in child_llm_runs:
                    logs += self._convert_llm_run_to_log(
                        llm_run=llm_run,
                        function_name=run.name,
                        inputs=[inputs],
                        prompt_details=run.serialized["prompt"],
                        duration=duration,
                    )
            elif run_type == AGENT_CHAIN_TYPE:
                logs.append(
                    self._convert_agent_run_to_log(
                        run=run,
                        function_name=run.name,
                        duration=duration,
                        metadata=metadata,
                    )
                )
            else:
                logs.append(
                    self._convert_generic_chain_run_to_log(
                        run=run,
                        function_name=run.name,
                        duration=duration,
                        metadata=metadata,
                    )
                )
        elif run.run_type == RunTypeEnum.tool:
            logs.append(
                self._convert_tool_run_to_log(
                    run=run,
                    function_name=run.name,
                    duration=duration,
                    metadata=metadata,
                )
            )
        elif run.run_type == RunTypeEnum.llm:
            logs += self._convert_llm_run_to_log(
                llm_run=run,
                function_name=run.name,
                inputs=[self._convert_chain_inputs_to_text(run.inputs)],
                duration=duration,
            )
        else:
            raise NotImplementedError

        if parent_log is not None:
            if parent_log.children is None:
                parent_log.children = logs
            else:
                parent_log.children += logs
        child_runs = sorted(
            run.child_runs,
            key=lambda x: x.execution_order,
        )
        for run in child_runs:
            # Assume the final log at this step is the parent
            _ = self._convert_run_to_logs(
                run=run,
                parent_log=logs[-1],
            )
        return logs

    def _post_trace(self, logs: List[Log]) -> None:
        """Post a trace to the HL API"""
        trace_response = requests.post(
            url=f"{self._base_url}/v4/traces",
            json={"logs": [log.dict() for log in logs]},
            headers=self._headers,
        )
        if trace_response.status_code != 200:
            raise TracerException(
                f"Failed to post trace to Humanloop with error: {trace_response.json()}"
            )
        response_data = trace_response.json()
        print(
            get_colored_text(
                text=f"Go to {logs[0].function_name}'s trace: {HUMANLOOP_APP_URL}/projects/{response_data['function_id']}/sessions/{response_data['session_id']}",
                color="pink",
            )
        )

    @staticmethod
    def _convert_chain_outputs_to_text(outputs: Dict[str, Any]) -> str:
        """ Convert LC dictionary outputs to a string required by the HL API."""
        output = ""
        for key, value in outputs.items():
            # TODO: check are there other data types to deal with
            if isinstance(value, list):
                value = ", ".join(value)
            output += f"{key}:\n{value}\n"
        return output

    @staticmethod
    def _convert_chain_inputs_to_text(inputs: Dict[str, Any]) -> Dict[str, str]:
        """ Convert LC list inputs to a string required by the HL API."""
        new_inputs = {}
        for key, value in inputs.items():
            # TODO: check are there other data types to deal with
            if isinstance(value, list):
                value = ", ".join(value)
            new_inputs[key] = value
        return new_inputs

    @staticmethod
    def _convert_provider_parameters(
        llm_type: str, params: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Convert provider specific parameters to the HL common model parameters.

        Return a tuple of Humanloop parameters and any parameters not mapped.
        """
        if not params:
            return {}, {}
        params = copy.deepcopy(params)
        # use startswith because LC has patterns 'openai', 'openai-chat' as _llm_types
        # TODO: Don't want to rely on string ops for getting provider and endpoint,
        #  suggest to have separate 'provider' and 'mode' (complete or chat) instead
        #  of just _llm_type
        if llm_type.startswith("openai"):
            provider = "openai"
            mapping = {
                "model_name": "model",
                "temperature": "temperature",
                "max_tokens": "max_tokens",
                "n": "num_samples",
                "top_p": "top_p",
                "presence_penalty": "presence_penalty",
                "frequency_penalty": "frequency_penalty",
                "stop": "stop",
            }
        elif llm_type == "anthropic":
            provider = "anthropic"
            mapping = {
                "provider": "anthropic",
                "model_name": "model",
                "max_tokens_to_sample": "max_tokens",
                "temperature": "temperature",
                "top_k": "top_k",
                "top_p": "top_p",
                "stop_sequence": "stop",
            }
        else:
            raise NotImplementedError

        mapped_params = {
            hl_param: params.pop(provider_param, None)
            for provider_param, hl_param in mapping.items()
        }
        hl_params = {
            **mapped_params,
            "provider": provider,
            "endpoint": "chat" if llm_type.endswith("chat") else "complete",
        }
        return hl_params, params

    def _convert_agent_run_to_log(
        self,
        run: Run,
        function_name: str,
        duration: float,
        metadata: Dict[str, Any],
    ) -> Log:
        """Converts LC agent chain run to HL log."""
        config = AgentConfig(
            agent_class=run.serialized["agent"]["_type"],
            tools=[
                ToolConfig(
                    name=tool["name"],
                    description=tool["description"],
                    source=self._get_tool_source_from_function(
                        tool["name"], tool["func"]
                    ),
                )
                for tool in run.serialized["tools"]
            ],
            model_config=self._create_model_config_from_params(
                params=run.serialized["agent"]["llm_chain"]["llm"],
                prompt_details=run.serialized["agent"]["llm_chain"]["prompt"],
            ),
            other=AgentOther(
                max_iterations=run.serialized["max_iterations"],
                stop=run.serialized["early_stopping_method"],
                output_parser=run.serialized["agent"]["output_parser"]["_type"],
            ),
        )
        return Log(
            function_name=function_name,
            config=config,
            inputs=self._convert_chain_inputs_to_text(inputs=run.inputs),
            error=run.error,
            output=self._convert_chain_outputs_to_text(run.outputs),
            created_at=run.end_time,
            duration=duration,
            metadata=metadata,
        )

    def _convert_generic_chain_run_to_log(
        self,
        run: Run,
        function_name: str,
        duration: float,
        metadata: Dict[str, Any],
    ) -> Log:
        """Converts LC generic chain run to HL log."""
        config = Config(name=function_name)
        return Log(
            function_name=function_name,
            config=config,
            inputs=self._convert_chain_inputs_to_text(inputs=run.inputs),
            error=run.error,
            output=self._convert_chain_outputs_to_text(run.outputs),
            created_at=run.end_time,
            duration=duration,
            metadata=metadata,
        )

    @staticmethod
    def _get_tool_source_from_function(tool_name: str, tool_function: Callable) -> str:
        """ Get the source code for a tool function."""
        try:
            return inspect.getsource(tool_function)
        except Exception as _:
            logging.info(f"Failed to get source for tool {tool_name}")

    def _convert_tool_run_to_log(
        self,
        run: Run,
        function_name: str,
        duration: float,
        metadata: Dict[str, Any],
    ) -> Log:
        """Converts LC tool chain run to HL log."""
        return Log(
            function_name=function_name,
            config=ToolConfig(
                name=function_name,
                description=run.serialized["description"],
                source=self._get_tool_source_from_function(
                    function_name, run.serialized["func"]
                ),
                other=run.extra,
            ),
            inputs=run.inputs,
            output=self._convert_chain_outputs_to_text(run.outputs),
            error=run.error,
            metadata=metadata,
            created_at=run.end_time,
            duration=duration,
        )

    @staticmethod
    def _convert_template_to_hl_syntax(template: str, template_format: str) -> str:
        """Converts LC template to HL double curly bracket syntax"""
        if template_format == "f-string":
            # Match the f-string syntax with single curly brackets
            pattern = r'{([^{}]+)}'

            # Replaces f-string syntax with Jinja2 syntax
            def replace(match):
                return '{{{{ {} }}}}'.format(
                    match.group(1))

            return re.sub(pattern, replace, template)
        elif template_format == "jinja2":
            # I believe jinja2 already uses double curly brackets?
            return template
        else:
            logging.info(f"Unknown template format: {template_format}")
            return template

    def _create_model_config_from_params(
        self, params: Dict[str, Any], prompt_details: Optional[Dict[Any]] = None
    ) -> ModelConfig:
        """Creates HL model configuration from LC parameters."""
        hl_params, other_params = self._convert_provider_parameters(
            llm_type=params.pop("_type"),
            params=params,
        )
        if prompt_details is not None:
            if "messages" in prompt_details:
                return ModelConfig(
                    chat_template=[
                        ChatMessage(
                            content=self._convert_template_to_hl_syntax(
                                template=message["prompt"]["template"],
                                template_format=message["prompt"]["template_format"],
                            ),
                            role=ROLE_MAPPING[message["role"]],
                        )
                        for message in prompt_details["messages"]
                    ],
                    **hl_params,
                )
            else:
                return ModelConfig(
                    prompt_template=self._convert_template_to_hl_syntax(
                        template=prompt_details["template"],
                        template_format=prompt_details["template_format"],
                    ),
                    **hl_params,
                )
        else:
            return ModelConfig(**hl_params)

    def _convert_llm_run_to_log(
        self,
        llm_run: Run,
        function_name: str,
        inputs: List[Dict[str, str]],
        duration: float,
        prompt_details: Optional[Dict[Any]] = None,
    ) -> List[Log]:
        """Converts LC llm run to HL log."""
        config = self._create_model_config_from_params(
            params=llm_run.extra.pop("invocation_params", {}),
            prompt_details=prompt_details,
        )
        metadata = {"langchain_run_id": str(llm_run.id), **llm_run.extra}
        if "generations" in llm_run.outputs:
            logs = []
            # response.generations is a list (# inputs) of lists (# samples)
            for i, generations in enumerate(llm_run.outputs["generations"]):
                for generation in generations:
                    if "message" in generation:
                        output = generation["message"]["content"]
                    elif "text" in generation:
                        output = generation["text"]
                    else:
                        raise NotImplementedError
                    logs.append(
                        # TODO: add token usage
                        # TODO: For chat serialization doesn't include messages array
                        Log(
                            function_name=function_name,
                            config=config,
                            inputs=inputs[i],
                            output=output,
                            created_at=llm_run.end_time,
                            duration=duration,
                            metadata=metadata,
                        )
                    )
        else:
            logs = [
                Log(
                    function_name=function_name,
                    config=config,
                    inputs=inputs,
                    error=llm_run.error,
                    created_at=llm_run.end_time,
                    duration=duration,
                    metadata=metadata,
                )
            ]
        return logs

    def _persist_session(self, session: TracerSessionCreate) -> TracerSession:
        """Persist a tracing session."""
        # LCs session concept is a collection of runs, not required by HL.
        return TracerSession(id=-1)

    def load_session(self, session_name: str) -> TracerSession:
        """Load a tracing session and set it as the Tracer's session."""
        # LCs session concept is a collection of runs, not required by HL.
        return TracerSession(id=-1)

    def load_default_session(self) -> TracerSession:
        """Load the default tracing session and set it as the Tracer's session."""
        # LCs session concept is a collection of runs, not required by HL.
        return TracerSession(id=-1)


class ChatRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class ChatMessage(BaseModel):
    role: ChatRole
    content: str
    name: Optional[str] = None


class Config(BaseModel):
    name: Optional[str]
    description: Optional[str]
    type = "generic"


class ModelConfig(Config):
    provider: str = Field(
        title="Model provider",
        description="The company providing the underlying model service.",
    )
    endpoint: Optional[str] = Field(
        title="Provider endpoint",
        description="Which of the providers model endpoints to use. "
        "For example Complete or Edit.",
    )
    model: str = Field(
        title="Model instance used",
        description="What model instance to use for the generation. "
        "e.g. text-davinci-002.",
    )
    prompt_template: Optional[str] = Field(
        title="Prompt template",
        description="Prompt template that will take your specified inputs to form "
        "your final request to the provider model. "
        "Input variables within the prompt template should be specified "
        "with double curly bracket syntax: {{INPUT_NAME}}.",
    )
    chat_template: Optional[List[ChatMessage]] = Field(
        title="Chat template",
        description="Messages prepended to the list of messages sent to the provider. "
        "These messages that will take your specified inputs to form "
        "your final request to the provider model. ",
    )
    temperature: Optional[float] = Field(
        title="Sampling temperature",
        description="What sampling temperature to use when making a generation. "
        "Higher values means the model will be more creative.",
        default=1,
    )
    max_tokens: Optional[int] = Field(
        title="Maximum tokens",
        description="The maximum number of tokens to generate. "
        "Provide max_tokens=-1 to dynamically calculate the maximum number of tokens "
        "to generate given the length of the prompt",
        default=-1,
    )
    top_p: Optional[float] = Field(
        title="Top p probability mass",
        description="An alternative to sampling with temperature, "
        "called nucleus sampling, where the model considers the results "
        "of the tokens with top_p probability mass.",
        default=1,
    )
    stop: Optional[Union[str, List[str]]] = Field(
        title="Stop sequence(s)",
        description="The string (or list of strings) after which the model will stop "
        "generating. The returned text will not contain the stop sequence.",
        default=None,
    )
    presence_penalty: Optional[float] = Field(
        title="Penalize tokens on whether present.",
        description="Number between -2.0 and 2.0. Positive values penalize new tokens "
        "based on whether they appear in the generation so far.",
        default=0,
    )
    frequency_penalty: Optional[float] = Field(
        title="Penalize tokens on whether frequent.",
        description="Number between -2.0 and 2.0. Positive values penalize new tokens "
        "based on how frequently they appear in the generation so far.",
        default=0,
    )
    other: Optional[Dict[str, Any]] = Field(
        title="Other provider parameters",
        description="Other parameter values to be passed to the provider call.",
        default={},
    )
    type = "model"


class ToolConfig(Config):
    source: Optional[str] = Field(
        title="Tool source",
        description="The source code for the tool.",
    )
    other: Optional[Dict[str, Any]] = Field(
        title="Other tool parameters",
        description="Other parameter values that uniquely identify the tool.",
        default={},
    )
    type = "tool"


class AgentOther(BaseModel):
    max_iterations: int
    stop: Optional[Union[str, List[str]]]
    output_parser: Optional[str]


class AgentConfig(Config):
    agent_class: str
    tools: List[ToolConfig]
    model_config: ModelConfig
    other: AgentOther
    type = "agent"


class Log(BaseModel):
    config: Union[Config, ModelConfig, ToolConfig, AgentConfig] = Field(
        title="Config", description="The config used for a specific step in the chain."
    )
    function_name: str = Field(
        title="Function name",
        description="Function name. If it does not exist, a new function will be "
        "created.",
    )
    children: Optional[List[Log]]
    inputs: Dict[str, str] = Field(
        title="Project input data",
        description="List of (name, value) pairs for the inputs used by your prompt "
        "template, or directly by your project.",
    )
    output: Optional[str] = Field(
        title="Project output",
        description="Generated output from your project for the provided inputs.",
    )
    error: Optional[str] = Field(
        title="Log error",
        description="Captures error thrown by model.",
    )
    created_at: Optional[datetime.datetime] = Field(
        title="Created at",
        description="Timestamp for when the log was created. "
        "If not provided, the time the log call was made will be used "
        "as a timestamp.",
    )
    metadata: Optional[Dict[str, str]] = Field(
        title="Metadata",
        description="Any additional metadata that you would like to log for reference.",
        default={},
    )
    source: Optional[str] = Field(
        title="Source of generation",
        description="What was source of the model used for this generation? "
        "e.g. langchain",
        default="langchain",
    )
    duration: Optional[float] = Field(
        title="Duration", description="How long in seconds the trace step took."
    )

    @validator("created_at", always=True)
    def convert_created_at_datetime(cls, v, values):
        if v is not None:
            v = v.isoformat()
        return v
