from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional, Union, Literal, Annotated

class ClaudeContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

class ClaudeContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]

class ClaudeContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ClaudeContentBlockThinking(BaseModel):
    type: Literal["thinking"]
    thinking: str
    signature: Optional[str] = None

class ClaudeContentBlockRedactedThinking(BaseModel):
    type: Literal["redacted_thinking"]
    data: Optional[str] = None

class ClaudeContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], None] = None

class ClaudeSystemContent(BaseModel):
    type: Literal["text"]
    text: str

# Use discriminated union via 'type' field for O(1) dispatch instead of O(n) trial
ClaudeContentBlock = Annotated[
    Union[
        ClaudeContentBlockText,
        ClaudeContentBlockImage,
        ClaudeContentBlockToolUse,
        ClaudeContentBlockToolResult,
        ClaudeContentBlockThinking,
        ClaudeContentBlockRedactedThinking,
    ],
    Field(discriminator="type"),
]

class ClaudeMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["user", "assistant"]
    content: Union[str, List[ClaudeContentBlock], None] = None

class ClaudeTool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]

class ClaudeThinkingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Optional[str] = None  # "enabled" | "adaptive" | "disabled"
    enabled: bool = True
    budget_tokens: Optional[int] = None

class ClaudeMessagesRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # Ignore unknown fields for forward compat
    model: str
    max_tokens: int
    messages: List[ClaudeMessage]
    system: Optional[Union[str, List[ClaudeSystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[ClaudeTool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ClaudeThinkingConfig] = None

class ClaudeTokenCountRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: List[ClaudeMessage]
    system: Optional[Union[str, List[ClaudeSystemContent]]] = None
    tools: Optional[List[ClaudeTool]] = None
    thinking: Optional[ClaudeThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
