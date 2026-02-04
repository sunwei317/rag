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
import os
from config.settings import settings
import pickle
from enum import Enum
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
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
ENTITY_EXTRACTION_PROMPT = """从以下技术文档片段中抽取实体。直接输出JSON，不要添加任何说明文字。

文档内容:
{content}

抽取以下类型的实体:
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

输出JSON格式（仅JSON，不要其他文字）:
{{
    "entities": [
        {{
            "name": "实体名称",
            "type": "实体类型",
            "aliases": ["别名1"],
            "description": "简短描述"
        }}
    ]
}}"""


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
        model: Optional[str] = None,
        batch_size: int = 5,
        min_mentions: int = 1,  # 最少提及次数才保留
        min_content_length: Optional[int] = None,  # 最小内容长度
        cache_dir: Optional[str] = None,  # 缓存目录
        enable_cache: bool = True  # 是否启用缓存
    ):
        self.llm_client = llm_client
        # 优先使用传入参数，否则用 settings.llm.local_model
        self.model = model or settings.llm.local_model
        
        # 从 settings 加载 Graph RAG 配置（如果可用）
        try:
            # 使用传入参数或 settings 配置，都有默认值回退
            self.batch_size = batch_size or (settings.graph.get('entity_batch_size') if settings.graph else None) or 5
            self.min_mentions = min_mentions or (settings.graph.get('entity_min_mentions') if settings.graph else None) or 1
            self.min_content_length = min_content_length or (settings.graph.get('entity_min_content_length') if settings.graph else None) or 50
            self.enable_cache = enable_cache if enable_cache is not None else (settings.graph.get('enable_cache') if settings.graph else None) or True
            self.cache_dir = Path(cache_dir) if cache_dir else (Path(settings.graph.get('cache_persist_path')) if settings.graph and settings.graph.get('cache_persist_path') else Path("/tmp/entity_cache"))
        except AttributeError:
            # 如果 settings 不可用，使用默认值
            self.batch_size = batch_size or 5
            self.min_mentions = min_mentions or 1
            self.min_content_length = min_content_length or 50
            self.enable_cache = enable_cache if enable_cache is not None else True
            self.cache_dir = Path(cache_dir) if cache_dir else Path("/tmp/entity_cache")
        
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._init_llm_client()

        # 实体缓存，用于去重和合并
        self._entity_cache: Dict[str, Entity] = {}

        # 内容哈希缓存 - 避免重复抽取
        self._content_hash_cache: Dict[str, List[Entity]] = {}
        self._load_cache()
    
    def _get_content_hash(self, content: str) -> str:
        """计算内容哈希"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_cache(self):
        """从磁盘加载缓存"""
        if not self.enable_cache:
            return
        cache_file = self.cache_dir / "entity_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    self._content_hash_cache = pickle.load(f)
                logger.info(f"Loaded {len(self._content_hash_cache)} cached extractions")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self._content_hash_cache = {}
    
    def _save_cache(self):
        """保存缓存到磁盘"""
        if not self.enable_cache:
            return
        cache_file = self.cache_dir / "entity_cache.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(self._content_hash_cache, f)
            logger.debug(f"Saved {len(self._content_hash_cache)} extractions to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _init_llm_client(self):
        """初始化 LLM 客户端"""
        if self.llm_client is not None:
            return
        try:
            from openai import OpenAI
            # 优先使用本地 LLM 服务 (OpenAI 兼容 API)
            if settings.llm.writing_provider == "local":
                self.llm_client = OpenAI(
                    api_key="EMPTY",  # 本地服务通常不校验 key
                    base_url=settings.llm.local_api_base
                )
                self.model = settings.llm.local_model
            elif settings.llm.writing_provider == "openai":
                self.llm_client = OpenAI(
                    api_key=settings.llm.openai_api_key
                )
                self.model = settings.llm.writing_model
            else:
                raise ValueError(f"Unsupported LLM provider: {settings.llm.writing_provider}")
        except Exception as e:
            logger.warning(f"Failed to init LLM client: {e}")
    
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
        chunks: List[Dict[str, Any]],
        min_content_length: int = 50,
        max_concurrent: int = 5,
        batch_size: int = None
    ) -> List[Entity]:
        """
        从多个 chunks 批量抽取实体（优化版）
        
        优化策略:
        1. 过滤过短的 chunks
        2. 批量合并内容减少 LLM 调用
        3. 并行处理
        
        Args:
            chunks: chunk 列表，每个需包含 chunk_id 和 content
            min_content_length: 最小内容长度，短于此的跳过
            max_concurrent: 最大并发数
            batch_size: 每批合并的 chunk 数量，None 使用默认值
            
        Returns:
            合并去重后的实体列表
        """
        import concurrent.futures
        
        # 使用传入的 batch_size 或默认值
        actual_batch_size = batch_size if batch_size else self.batch_size
        
        # 1. 过滤过短的 chunks
        valid_chunks = [
            c for c in chunks 
            if len(c.get("content", "")) >= min_content_length
        ]
        
        if not valid_chunks:
            logger.warning(f"No valid chunks after filtering (min_length={min_content_length})")
            return []
        
        logger.info(f"Processing {len(valid_chunks)}/{len(chunks)} valid chunks (min_length={min_content_length})")
        
        # 2. 批量合并 - 每 batch_size 个 chunks 合并成一个请求
        batches = []
        for i in range(0, len(valid_chunks), actual_batch_size):
            batch = valid_chunks[i:i + actual_batch_size]
            # 合并内容
            combined_content = "\n\n---\n\n".join([
                f"[Chunk {c.get('chunk_id', i)}]\n{c.get('content', '')}"
                for c in batch
            ])
            chunk_ids = [c.get("chunk_id", "") for c in batch]
            batches.append((combined_content, chunk_ids))
            logger.debug(f"Batch {len(batches)}: {len(batch)} chunks, content length: {len(combined_content)}")
        
        logger.info(f"Created {len(batches)} batches for LLM extraction (batch_size={actual_batch_size})")
        
        # 3. 并行处理批次
        all_entities = []
        
        def process_batch(batch_data):
            content, chunk_ids = batch_data
            try:
                logger.info(f"Processing batch with {len(chunk_ids)} chunks, content length: {len(content)}")
                entities = self._extract_batch(content, chunk_ids)
                logger.info(f"Batch completed: extracted {len(entities)} entities")
                return entities
            except Exception as e:
                logger.error(f"Batch extraction failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return []
        
        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                entities = future.result()
                all_entities.extend(entities)
                logger.info(f"Completed batch {i+1}/{len(batches)}, got {len(entities)} entities")
        
        # 4. 合并和去重
        merged_entities = self._merge_entities(all_entities)
        
        # 5. 过滤低频实体
        filtered_entities = [
            e for e in merged_entities 
            if e.mentions >= self.min_mentions
        ]
        
        logger.info(f"Total entities: {len(filtered_entities)} (after merge and filter)")
        return filtered_entities
    
    def _extract_batch(
        self, 
        combined_content: str, 
        chunk_ids: List[str]
    ) -> List[Entity]:
        """批量抽取实体（带缓存）"""
        if not combined_content.strip():
            return []
        
        # 检查缓存
        content_hash = self._get_content_hash(combined_content)
        if self.enable_cache and content_hash in self._content_hash_cache:
            cached_entities = self._content_hash_cache[content_hash]
            logger.debug(f"Cache hit for batch, returning {len(cached_entities)} entities")
            # 更新 source_chunks
            for entity in cached_entities:
                entity.source_chunks = chunk_ids
            return cached_entities
        
        try:
            if not self.llm_client:
                logger.error("LLM client not initialized")
                return []
            
            # 限制内容长度，避免超时和token耗尽
            # 进一步降低内容长度，确保有足够token用于完整JSON响应
            content_preview = combined_content[:1500]  # 从2000降到1500，为JSON响应预留更多token
            prompt_content = ENTITY_EXTRACTION_PROMPT.format(content=content_preview)
            
            logger.info(f"Calling LLM for entity extraction, content length: {len(content_preview)}, prompt length: {len(prompt_content)}, model: {self.model}")
            
            try:
                import time
                start_time = time.time()
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是技术文档实体抽取专家，擅长识别技术文档中的关键实体。请从多个文档片段中抽取实体。直接输出JSON，不要添加额外说明。"
                        },
                        {
                            "role": "user",
                            "content": prompt_content
                        }
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    max_tokens=8000,  # 大幅增加最大token数（从4000提高到8000）
                    stop=None  # 明确不设置stop tokens
                )
                elapsed = time.time() - start_time
                logger.info(f"LLM call completed in {elapsed:.2f}s")
            except Exception as e:
                logger.error(f"LLM API call failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return []
            
            if not response:
                logger.error(f"LLM returned None response")
                return []
            
            if not response.choices:
                logger.error(f"LLM response has no choices. Full response: {response}")
                return []
            
            if not response.choices[0].message:
                logger.error(f"LLM response choice has no message. Response: {response}")
                return []
            
            finish_reason = response.choices[0].finish_reason if response.choices else None
            message_content = response.choices[0].message.content
            
            if not message_content:
                logger.error(f"LLM returned empty content. Finish reason: {finish_reason}")
                logger.error(f"Response type: {type(response)}, Choices count: {len(response.choices) if response.choices else 0}")
                
                # 如果是因为长度限制，尝试从reasoning中提取JSON
                if finish_reason == 'length':
                    reasoning = getattr(response.choices[0].message, 'reasoning', None) or getattr(response.choices[0].message, 'reasoning_content', None)
                    if reasoning:
                        logger.warning(f"Response truncated (finish_reason=length). Attempting to extract JSON from reasoning...")
                        logger.warning(f"Reasoning length: {len(reasoning)}")
                        
                        # 尝试从reasoning中提取JSON
                        import re
                        json_match = re.search(r'\{[\s\S]*"entities"[\s\S]*\}', reasoning)
                        if json_match:
                            try:
                                result = json.loads(json_match.group())
                                logger.info("Successfully extracted JSON from reasoning field")
                                entities = self._parse_entities(result, chunk_ids[0] if chunk_ids else None)
                                for entity in entities:
                                    entity.source_chunks = chunk_ids
                                return entities
                            except Exception as e:
                                logger.error(f"Failed to parse JSON from reasoning: {e}")
                        else:
                            logger.warning("Could not find JSON in reasoning field")
                    else:
                        logger.warning("No reasoning field available")
                
                return []
            
            content = message_content.strip()
            if not content:
                logger.error(f"LLM returned empty content after strip")
                return []
            
            logger.debug(f"LLM returned content (first 200 chars): {content[:200]}")
            
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Content that failed to parse (first 1000 chars): {content[:1000]}")
                
                # 尝试修复常见的JSON问题
                try:
                    # 1. 尝试提取JSON部分（更宽松的匹配）
                    import re
                    # 匹配从第一个 { 开始到最后一个 } 结束
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        json_str = json_match.group()
                        
                        # 尝试修复截断的JSON
                        # 1. 查找所有完整的实体对象（更宽松的匹配）
                        # 匹配实体对象，允许description字段可能未闭合
                        entity_pattern = r'\{"name":"[^"]*","type":"[^"]*"(?:,"aliases":\[[^\]]*\])?(?:,"description":"[^"]*")?\}'
                        entities_matches = list(re.finditer(entity_pattern, json_str))
                        
                        if entities_matches and len(entities_matches) > 0:
                            # 构建有效的JSON，只包含完整的实体
                            valid_entities = []
                            for match in entities_matches:
                                try:
                                    # 验证每个实体是否是有效的JSON
                                    entity_json = json.loads(match.group())
                                    valid_entities.append(entity_json)
                                except:
                                    continue
                            
                            if valid_entities:
                                result = {"entities": valid_entities}
                                logger.info(f"Successfully extracted {len(valid_entities)} entities from truncated JSON")
                            else:
                                result = {"entities": []}
                                logger.warning("No valid entities found in truncated JSON")
                        else:
                            # 如果找不到完整实体，尝试简单修复
                            # 移除未闭合的部分
                            if '"entities":[' in json_str:
                                # 找到entities数组的开始
                                start_idx = json_str.find('"entities":[') + len('"entities":[')
                                # 找到最后一个完整的 ]
                                end_idx = json_str.rfind(']')
                                if end_idx > start_idx:
                                    entities_part = json_str[start_idx:end_idx+1]
                                    try:
                                        entities_list = json.loads('[' + entities_part + ']')
                                        result = {"entities": entities_list}
                                        logger.info(f"Successfully extracted {len(entities_list)} entities using array extraction")
                                    except:
                                        result = {"entities": []}
                                else:
                                    result = {"entities": []}
                            else:
                                result = {"entities": []}
                                logger.warning("Could not find entities array in JSON")
                    else:
                        # 2. 如果找不到JSON，尝试手动构建
                        logger.warning("Could not find JSON in response, creating empty result")
                        result = {"entities": []}
                except Exception as repair_error:
                    logger.error(f"Could not recover JSON from response: {repair_error}")
                    # 返回空结果而不是失败
                    result = {"entities": []}
            entities = self._parse_entities(result, chunk_ids[0] if chunk_ids else None)
            
            # 将所有 chunk_ids 添加到实体的 source_chunks
            for entity in entities:
                entity.source_chunks = chunk_ids
            
            # 保存到缓存
            if self.enable_cache:
                self._content_hash_cache[content_hash] = entities
                self._save_cache()
            
            return entities
            
        except Exception as e:
            logger.error(f"Batch entity extraction failed: {e}")
            return []
    
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
