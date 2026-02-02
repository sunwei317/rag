"""
知识图谱构建 Pipeline
从文档 chunks 构建知识图谱的完整流程

流程:
1. 从 chunks 抽取实体
2. 构建实体间关系
3. 存储到图数据库
4. 提供增量更新能力
"""
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

from .entity_extractor import EntityExtractor, Entity
from .relation_builder import RelationBuilder, Relation
from .graph_store import GraphStore, InMemoryGraphStore


@dataclass
class GraphBuildResult:
    """图谱构建结果"""
    entity_count: int = 0
    relation_count: int = 0
    duration_seconds: float = 0
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "entity_count": self.entity_count,
            "relation_count": self.relation_count,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors
        }


class GraphBuildPipeline:
    """
    知识图谱构建 Pipeline
    
    完整的图谱构建流程:
    1. 实体抽取: 使用 LLM 从文档中抽取实体
    2. 关系构建: 识别实体间的关系
    3. 图谱存储: 将实体和关系存入图数据库
    
    支持:
    - 批量构建: 从大量文档一次性构建
    - 增量更新: 仅处理新增/修改的文档
    - 进度追踪: 实时查看处理进度
    """
    
    def __init__(
        self,
        entity_extractor: Optional[EntityExtractor] = None,
        relation_builder: Optional[RelationBuilder] = None,
        graph_store: Optional[GraphStore] = None,
        # LLM 配置
        llm_client=None,
        model: str = "gpt-4.1-mini",
        # 处理配置
        batch_size: int = 10,
        extract_rules: bool = True,  # 是否使用规则抽取
        min_entity_mentions: int = 1,  # 最少提及次数
        min_relation_confidence: float = 0.5,  # 最低关系置信度
        # 持久化
        persist_path: Optional[str] = None
    ):
        self.batch_size = batch_size
        self.extract_rules = extract_rules
        self.persist_path = persist_path
        
        # 初始化组件
        self.entity_extractor = entity_extractor or EntityExtractor(
            llm_client=llm_client,
            model=model,
            min_mentions=min_entity_mentions
        )
        
        self.relation_builder = relation_builder or RelationBuilder(
            llm_client=llm_client,
            model=model,
            min_confidence=min_relation_confidence
        )
        
        self.graph_store = graph_store or InMemoryGraphStore(
            persist_path=persist_path
        )
        
        # 处理状态
        self._processed_chunks: set = set()
    
    def build(
        self,
        chunks: List[Dict[str, Any]],
        incremental: bool = False
    ) -> GraphBuildResult:
        """
        从 chunks 构建知识图谱
        
        Args:
            chunks: 文档 chunks 列表，每个需包含:
                - chunk_id: 唯一标识
                - content: 文本内容
                - metadata: 元数据 (可选)
            incremental: 是否增量构建 (跳过已处理的 chunks)
            
        Returns:
            GraphBuildResult: 构建结果
        """
        start_time = time.time()
        result = GraphBuildResult()
        
        logger.info(f"Starting graph build with {len(chunks)} chunks")
        
        # 过滤已处理的 chunks
        if incremental:
            chunks = [
                c for c in chunks 
                if c.get("chunk_id") not in self._processed_chunks
            ]
            logger.info(f"Incremental mode: {len(chunks)} new chunks to process")
        
        if not chunks:
            logger.info("No chunks to process")
            return result
        
        try:
            # 1. 实体抽取
            logger.info("Step 1/3: Extracting entities...")
            entities = self._extract_entities(chunks)
            logger.info(f"Extracted {len(entities)} entities")
            
            # 2. 关系构建
            logger.info("Step 2/3: Building relations...")
            relations = self._build_relations(entities, chunks)
            logger.info(f"Built {len(relations)} relations")
            
            # 3. 存储到图数据库
            logger.info("Step 3/3: Storing to graph...")
            entity_count = self.graph_store.add_entities_batch(entities)
            relation_count = self.graph_store.add_relations_batch(relations)
            
            # 更新已处理记录
            for chunk in chunks:
                self._processed_chunks.add(chunk.get("chunk_id"))
            
            # 持久化
            if self.persist_path and hasattr(self.graph_store, 'save'):
                self.graph_store.save()
            
            result.entity_count = entity_count
            result.relation_count = relation_count
            
        except Exception as e:
            logger.error(f"Graph build failed: {e}")
            result.errors.append(str(e))
        
        result.duration_seconds = time.time() - start_time
        
        logger.info(
            f"Graph build completed: {result.entity_count} entities, "
            f"{result.relation_count} relations in {result.duration_seconds:.2f}s"
        )
        
        return result
    
    def _extract_entities(self, chunks: List[Dict]) -> List[Entity]:
        """抽取实体"""
        all_entities = []
        
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            
            for chunk in batch:
                chunk_id = chunk.get("chunk_id", "")
                content = chunk.get("content", "")
                
                if not content:
                    continue
                
                # 使用组合抽取 (LLM + 规则)
                if self.extract_rules:
                    entities = self.entity_extractor.extract_combined(
                        content, chunk_id
                    )
                else:
                    entities = self.entity_extractor.extract_from_text(
                        content, chunk_id
                    )
                
                # 添加元数据
                metadata = chunk.get("metadata", {})
                for entity in entities:
                    entity.properties["section_path"] = metadata.get("section_path", "")
                    entity.properties["doc_id"] = metadata.get("doc_id", "")
                
                all_entities.extend(entities)
            
            logger.debug(f"Processed {min(i + self.batch_size, len(chunks))}/{len(chunks)} chunks")
        
        # 合并重复实体
        return self.entity_extractor._merge_entities(all_entities)
    
    def _build_relations(
        self, 
        entities: List[Entity], 
        chunks: List[Dict]
    ) -> List[Relation]:
        """构建关系"""
        return self.relation_builder.build_relations(entities, chunks)
    
    def update(
        self,
        chunks: List[Dict[str, Any]]
    ) -> GraphBuildResult:
        """
        增量更新图谱
        
        等同于 build(chunks, incremental=True)
        """
        return self.build(chunks, incremental=True)
    
    def rebuild(
        self,
        chunks: List[Dict[str, Any]]
    ) -> GraphBuildResult:
        """
        完全重建图谱
        
        清空现有图谱后重新构建
        """
        logger.info("Rebuilding graph from scratch...")
        
        # 清空图谱
        self.graph_store.clear()
        self._processed_chunks.clear()
        
        return self.build(chunks, incremental=False)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        stats = self.graph_store.get_stats()
        return {
            **stats.to_dict(),
            "processed_chunks": len(self._processed_chunks)
        }


class GraphBuildPipelineAsync:
    """
    异步图谱构建 Pipeline
    
    支持:
    - 异步并发处理
    - 进度回调
    - 可取消的长时间任务
    """
    
    def __init__(
        self,
        entity_extractor: EntityExtractor,
        relation_builder: RelationBuilder,
        graph_store: GraphStore,
        max_concurrency: int = 5
    ):
        self.entity_extractor = entity_extractor
        self.relation_builder = relation_builder
        self.graph_store = graph_store
        self.max_concurrency = max_concurrency
        
        self._cancelled = False
    
    async def build_async(
        self,
        chunks: List[Dict[str, Any]],
        progress_callback=None
    ) -> GraphBuildResult:
        """
        异步构建图谱
        
        Args:
            chunks: 文档 chunks
            progress_callback: 进度回调函数 (processed, total) -> None
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        result = GraphBuildResult()
        start_time = time.time()
        
        total = len(chunks)
        processed = 0
        all_entities = []
        
        # 使用线程池进行 LLM 调用
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            loop = asyncio.get_event_loop()
            
            # 创建任务
            tasks = []
            for chunk in chunks:
                if self._cancelled:
                    break
                    
                task = loop.run_in_executor(
                    executor,
                    self._process_chunk,
                    chunk
                )
                tasks.append(task)
            
            # 并发处理
            for task in asyncio.as_completed(tasks):
                if self._cancelled:
                    break
                    
                try:
                    entities = await task
                    all_entities.extend(entities)
                except Exception as e:
                    result.errors.append(str(e))
                
                processed += 1
                if progress_callback:
                    progress_callback(processed, total)
        
        if not self._cancelled:
            # 构建关系
            relations = self.relation_builder.build_relations(all_entities, chunks)
            
            # 存储
            result.entity_count = self.graph_store.add_entities_batch(all_entities)
            result.relation_count = self.graph_store.add_relations_batch(relations)
        
        result.duration_seconds = time.time() - start_time
        return result
    
    def _process_chunk(self, chunk: Dict) -> List[Entity]:
        """处理单个 chunk"""
        content = chunk.get("content", "")
        chunk_id = chunk.get("chunk_id", "")
        
        return self.entity_extractor.extract_combined(content, chunk_id)
    
    def cancel(self):
        """取消构建任务"""
        self._cancelled = True


def create_graph_pipeline(
    persist_path: str = "./data/knowledge_graph.json",
    llm_model: str = "gpt-4.1-mini",
    use_neo4j: bool = False,
    neo4j_config: Optional[Dict] = None
) -> GraphBuildPipeline:
    """
    创建图谱构建 Pipeline 的便捷函数
    
    Args:
        persist_path: 图谱持久化路径 (仅 InMemory 模式)
        llm_model: LLM 模型名称
        use_neo4j: 是否使用 Neo4j
        neo4j_config: Neo4j 配置 (uri, user, password)
        
    Returns:
        GraphBuildPipeline: 配置好的 Pipeline
    """
    from .graph_store import InMemoryGraphStore, Neo4jStore
    
    # 选择图存储
    if use_neo4j:
        config = neo4j_config or {}
        graph_store = Neo4jStore(
            uri=config.get("uri", "bolt://localhost:7687"),
            user=config.get("user", "neo4j"),
            password=config.get("password", "password")
        )
    else:
        graph_store = InMemoryGraphStore(persist_path=persist_path)
    
    return GraphBuildPipeline(
        graph_store=graph_store,
        model=llm_model,
        persist_path=persist_path if not use_neo4j else None
    )
