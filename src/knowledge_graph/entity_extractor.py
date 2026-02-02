"""
实体抽取模块
使用 LLM 从技术文档中抽取结构化实体

支持的实体类型:
- 产品/组件 (Product)
- API/接口 (API)
- 配置项 (Config)
- 版本 (Version)
- 技术概念 (Concept)
- 操作/命令 (Command)
- 错误/异常 (Error)
"""
import json
import hashlib
from enum import Enum
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from loguru import logger


class EntityType(Enum):
    """实体类型枚举"""
    PRODUCT = "product"           # 产品/组件/模块
    API = "api"                   # API 端点/接口/方法
    CONFIG = "config"             # 配置项/参数
    VERSION = "version"           # 版本号
    CONCEPT = "concept"           # 技术概念/术语
    COMMAND = "command"           # 命令/操作
    ERROR = "error"               # 错误/异常
    FILE = "file"                 # 文件/路径
    DEPENDENCY = "dependency"     # 依赖项
    PLATFORM = "platform"         # 平台/环境


@dataclass
class Entity:
    """
    实体数据结构
    
    Attributes:
        entity_id: 实体唯一标识
        name: 实体名称
        entity_type: 实体类型
        aliases: 别名列表
        description: 实体描述
        properties: 额外属性
        source_chunks: 来源 chunk IDs
        mentions: 提及次数
    """
    entity_id: str
    name: str
    entity_type: EntityType
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    source_chunks: List[str] = field(default_factory=list)
    mentions: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "aliases": self.aliases,
            "description": self.description,
            "properties": self.properties,
            "source_chunks": self.source_chunks,
            "mentions": self.mentions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        return cls(
            entity_id=data["entity_id"],
            name=data["name"],
            entity_type=EntityType(data["entity_type"]),
            aliases=data.get("aliases", []),
            description=data.get("description", ""),
            properties=data.get("properties", {}),
            source_chunks=data.get("source_chunks", []),
            mentions=data.get("mentions", 1)
        )
    
    @staticmethod
    def generate_id(name: str, entity_type: EntityType) -> str:
        """生成实体 ID"""
        content = f"{entity_type.value}:{name.lower()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


# LLM 实体抽取的 Prompt 模板
ENTITY_EXTRACTION_PROMPT = """你是一个技术文档实体抽取专家。请从以下技术文档片段中抽取所有相关实体。

文档内容:
{content}

请抽取以下类型的实体:
1. product - 产品、组件、模块、服务名称
2. api - API 端点、接口、方法名
3. config - 配置项、参数名
4. version - 版本号
5. concept - 技术概念、专业术语
6. command - 命令、CLI 操作
7. error - 错误码、异常类型
8. file - 文件路径、配置文件
9. dependency - 依赖包、库
10. platform - 操作系统、运行环境

请以 JSON 格式输出，格式如下:
```json
{{
    "entities": [
        {{
            "name": "实体名称",
            "type": "实体类型",
            "aliases": ["别名1", "别名2"],
            "description": "简短描述"
        }}
    ]
}}
```

注意:
- 只抽取明确提到的实体，不要推测
- 实体名称使用原文中的形式
- 技术术语的缩写和全称都作为别名记录
- 描述应该简洁，一句话概括实体的用途

输出:"""


class EntityExtractor:
    """
    实体抽取器
    
    使用 LLM 从技术文档中抽取结构化实体，支持:
    1. 批量文档处理
    2. 实体去重与合并
    3. 别名统一
    """
    
    def __init__(
        self,
        llm_client=None,
        model: str = "gpt-4.1-mini",
        batch_size: int = 5,
        min_mentions: int = 1  # 最少提及次数才保留
    ):
        self.llm_client = llm_client
        self.model = model
        self.batch_size = batch_size
        self.min_mentions = min_mentions
        
        self._init_llm_client()
        
        # 实体缓存，用于去重和合并
        self._entity_cache: Dict[str, Entity] = {}
    
    def _init_llm_client(self):
        """初始化 LLM 客户端"""
        if self.llm_client is None:
            try:
                from openai import OpenAI
                import os
                self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception as e:
                logger.warning(f"Failed to init OpenAI client: {e}")
    
    def extract_from_text(
        self, 
        text: str, 
        chunk_id: Optional[str] = None
    ) -> List[Entity]:
        """
        从文本中抽取实体
        
        Args:
            text: 输入文本
            chunk_id: 来源 chunk ID
            
        Returns:
            抽取的实体列表
        """
        if not text.strip():
            return []
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是技术文档实体抽取专家，擅长识别技术文档中的关键实体。"
                    },
                    {
                        "role": "user",
                        "content": ENTITY_EXTRACTION_PROMPT.format(content=text[:4000])
                    }
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            entities = self._parse_entities(result, chunk_id)
            
            logger.debug(f"Extracted {len(entities)} entities from text")
            return entities
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    def _parse_entities(
        self, 
        result: Dict, 
        chunk_id: Optional[str]
    ) -> List[Entity]:
        """解析 LLM 返回的实体"""
        entities = []
        
        for item in result.get("entities", []):
            try:
                name = item.get("name", "").strip()
                type_str = item.get("type", "concept").lower()
                
                if not name:
                    continue
                
                # 映射实体类型
                entity_type = self._map_entity_type(type_str)
                
                entity = Entity(
                    entity_id=Entity.generate_id(name, entity_type),
                    name=name,
                    entity_type=entity_type,
                    aliases=item.get("aliases", []),
                    description=item.get("description", ""),
                    source_chunks=[chunk_id] if chunk_id else [],
                    mentions=1
                )
                
                entities.append(entity)
                
            except Exception as e:
                logger.warning(f"Failed to parse entity: {e}")
                continue
        
        return entities
    
    def _map_entity_type(self, type_str: str) -> EntityType:
        """映射实体类型字符串到枚举"""
        type_mapping = {
            "product": EntityType.PRODUCT,
            "api": EntityType.API,
            "config": EntityType.CONFIG,
            "version": EntityType.VERSION,
            "concept": EntityType.CONCEPT,
            "command": EntityType.COMMAND,
            "error": EntityType.ERROR,
            "file": EntityType.FILE,
            "dependency": EntityType.DEPENDENCY,
            "platform": EntityType.PLATFORM,
        }
        return type_mapping.get(type_str, EntityType.CONCEPT)
    
    def extract_from_chunks(
        self, 
        chunks: List[Dict[str, Any]]
    ) -> List[Entity]:
        """
        从多个 chunks 批量抽取实体
        
        Args:
            chunks: chunk 列表，每个需包含 chunk_id 和 content
            
        Returns:
            合并去重后的实体列表
        """
        all_entities = []
        
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            
            for chunk in batch:
                chunk_id = chunk.get("chunk_id", "")
                content = chunk.get("content", "")
                
                entities = self.extract_from_text(content, chunk_id)
                all_entities.extend(entities)
            
            logger.info(f"Processed {min(i + self.batch_size, len(chunks))}/{len(chunks)} chunks")
        
        # 合并和去重
        merged_entities = self._merge_entities(all_entities)
        
        # 过滤低频实体
        filtered_entities = [
            e for e in merged_entities 
            if e.mentions >= self.min_mentions
        ]
        
        logger.info(f"Total entities: {len(filtered_entities)} (after merge and filter)")
        return filtered_entities
    
    def _merge_entities(self, entities: List[Entity]) -> List[Entity]:
        """合并重复实体"""
        merged: Dict[str, Entity] = {}
        
        for entity in entities:
            key = entity.entity_id
            
            if key in merged:
                # 合并已有实体
                existing = merged[key]
                existing.mentions += 1
                
                # 合并 source_chunks
                for chunk_id in entity.source_chunks:
                    if chunk_id not in existing.source_chunks:
                        existing.source_chunks.append(chunk_id)
                
                # 合并别名
                for alias in entity.aliases:
                    if alias not in existing.aliases:
                        existing.aliases.append(alias)
                
                # 更新描述（取更长的）
                if len(entity.description) > len(existing.description):
                    existing.description = entity.description
            else:
                merged[key] = entity
        
        return list(merged.values())
    
    def extract_with_rules(
        self, 
        text: str, 
        chunk_id: Optional[str] = None
    ) -> List[Entity]:
        """
        使用规则抽取实体（作为 LLM 抽取的补充）
        
        适用于:
        - 版本号 (v1.0.0, 1.2.3)
        - 文件路径 (/path/to/file, config.yaml)
        - API 路径 (/api/v1/users)
        - 配置项 (KEY=value)
        """
        import re
        entities = []
        
        # 版本号模式
        version_pattern = r'\b[vV]?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9]+)?\b'
        for match in re.finditer(version_pattern, text):
            name = match.group()
            entities.append(Entity(
                entity_id=Entity.generate_id(name, EntityType.VERSION),
                name=name,
                entity_type=EntityType.VERSION,
                source_chunks=[chunk_id] if chunk_id else []
            ))
        
        # 文件路径模式
        file_pattern = r'(?:/[\w.-]+)+(?:\.\w+)?|[\w-]+\.(?:yaml|yml|json|xml|conf|cfg|ini|sh|py|js|ts)'
        for match in re.finditer(file_pattern, text):
            name = match.group()
            if len(name) > 3:  # 过滤太短的匹配
                entities.append(Entity(
                    entity_id=Entity.generate_id(name, EntityType.FILE),
                    name=name,
                    entity_type=EntityType.FILE,
                    source_chunks=[chunk_id] if chunk_id else []
                ))
        
        # API 路径模式
        api_pattern = r'(?:GET|POST|PUT|DELETE|PATCH)\s+(/[\w/{}-]+)'
        for match in re.finditer(api_pattern, text, re.IGNORECASE):
            name = match.group(1)
            entities.append(Entity(
                entity_id=Entity.generate_id(name, EntityType.API),
                name=name,
                entity_type=EntityType.API,
                source_chunks=[chunk_id] if chunk_id else []
            ))
        
        return entities
    
    def extract_combined(
        self, 
        text: str, 
        chunk_id: Optional[str] = None
    ) -> List[Entity]:
        """
        组合 LLM 和规则抽取
        
        先用规则抽取精确模式，再用 LLM 抽取语义实体
        """
        # 规则抽取
        rule_entities = self.extract_with_rules(text, chunk_id)
        
        # LLM 抽取
        llm_entities = self.extract_from_text(text, chunk_id)
        
        # 合并
        all_entities = rule_entities + llm_entities
        return self._merge_entities(all_entities)
