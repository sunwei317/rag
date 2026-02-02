"""
图片处理模块

功能：
1. 从 PDF 中提取图片并保存
2. 使用多模态 LLM 生成图片描述 (GPT-4V, Gemini Vision, Claude Vision)
3. 使用 CLIP 等模型生成图片向量
4. 支持图文混合检索

使用场景：
- 技术文档中的架构图、流程图、截图
- 扫描文档中的图表、示意图
- 产品手册中的设备照片、界面截图
"""
import os
import io
import base64
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json

from loguru import logger

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not installed, image processing limited")


class ImageType(Enum):
    """图片类型"""
    DIAGRAM = "diagram"           # 架构图、流程图
    SCREENSHOT = "screenshot"     # 界面截图
    PHOTO = "photo"              # 照片
    CHART = "chart"              # 图表 (柱状图、饼图等)
    TABLE_IMAGE = "table_image"  # 表格图片
    FORMULA = "formula"          # 公式
    CODE = "code"                # 代码截图
    LOGO = "logo"                # Logo/图标
    UNKNOWN = "unknown"


@dataclass
class ExtractedImage:
    """提取的图片"""
    image_id: str                    # 唯一标识
    source_doc: str                  # 来源文档
    page_num: int                    # 页码
    bbox: Tuple[float, float, float, float]  # 位置
    image_path: str                  # 保存路径
    image_bytes: Optional[bytes] = None  # 图片字节
    width: int = 0
    height: int = 0
    format: str = "png"
    
    # 生成的内容
    caption: str = ""                # 图片说明 (从文档提取)
    description: str = ""            # AI 生成的详细描述
    image_type: ImageType = ImageType.UNKNOWN
    
    # 向量
    text_embedding: Optional[List[float]] = None   # 描述文本的向量
    image_embedding: Optional[List[float]] = None  # 图片本身的向量 (CLIP)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_base64(self) -> str:
        """转换为 base64"""
        if self.image_bytes:
            return base64.b64encode(self.image_bytes).decode('utf-8')
        elif self.image_path and os.path.exists(self.image_path):
            with open(self.image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        return ""
    
    def to_data_uri(self) -> str:
        """转换为 data URI"""
        b64 = self.to_base64()
        mime_type = f"image/{self.format}"
        return f"data:{mime_type};base64,{b64}"


class ImageProcessor:
    """
    多模态图片处理器
    
    工作流程：
    1. 从 PDF 提取图片 → 保存到文件系统
    2. 使用多模态 LLM 分析图片 → 生成描述
    3. 使用 CLIP 生成图片向量 → 存入向量库
    4. 将描述文本也向量化 → 支持文本搜图
    """
    
    def __init__(
        self,
        output_dir: str = "data/images",
        vision_provider: str = "openai",  # openai, anthropic, google
        clip_model: str = "openai/clip-vit-base-patch32",
        min_image_size: int = 50,         # 最小像素，过滤小图标
        max_images_per_page: int = 20,
        enable_ocr_on_image: bool = True,  # 对图片中的文字进行 OCR
        generate_descriptions: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.vision_provider = vision_provider
        self.clip_model = clip_model
        self.min_image_size = min_image_size
        self.max_images_per_page = max_images_per_page
        self.enable_ocr_on_image = enable_ocr_on_image
        self.generate_descriptions = generate_descriptions
        
        # 懒加载的模型
        self._vision_client = None
        self._clip_model = None
        self._clip_processor = None
    
    # ==================== 图片提取 ====================
    
    def extract_images_from_pdf(
        self,
        pdf_path: str,
        doc_id: str = "",
        save_images: bool = True
    ) -> List[ExtractedImage]:
        """
        从 PDF 中提取所有图片
        
        Args:
            pdf_path: PDF 文件路径
            doc_id: 文档 ID (用于命名)
            save_images: 是否保存到文件系统
            
        Returns:
            提取的图片列表
        """
        import fitz  # PyMuPDF
        
        pdf_path = Path(pdf_path)
        if not doc_id:
            doc_id = hashlib.md5(pdf_path.name.encode()).hexdigest()[:8]
        
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_images = self._extract_page_images(
                doc, page, page_num + 1, doc_id, pdf_path.name, save_images
            )
            images.extend(page_images)
        
        doc.close()
        
        logger.info(f"从 {pdf_path.name} 提取了 {len(images)} 张图片")
        return images
    
    def _extract_page_images(
        self,
        doc,
        page,
        page_num: int,
        doc_id: str,
        source_doc: str,
        save_images: bool
    ) -> List[ExtractedImage]:
        """提取单页的图片"""
        import fitz
        
        images = []
        image_list = page.get_images(full=True)
        
        for img_index, img_info in enumerate(image_list[:self.max_images_per_page]):
            try:
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                
                if not base_image:
                    continue
                
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                width = base_image["width"]
                height = base_image["height"]
                
                # 过滤小图片
                if width < self.min_image_size or height < self.min_image_size:
                    continue
                
                # 生成唯一 ID
                image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
                image_id = f"{doc_id}_p{page_num}_img{img_index}_{image_hash}"
                
                # 保存图片
                image_path = ""
                if save_images:
                    image_path = self.output_dir / f"{image_id}.{image_ext}"
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)
                    image_path = str(image_path)
                
                # 获取边界框
                bbox = self._get_image_bbox(page, xref)
                
                extracted = ExtractedImage(
                    image_id=image_id,
                    source_doc=source_doc,
                    page_num=page_num,
                    bbox=bbox,
                    image_path=image_path,
                    image_bytes=image_bytes,
                    width=width,
                    height=height,
                    format=image_ext,
                    metadata={
                        "xref": xref,
                        "colorspace": base_image.get("colorspace", ""),
                        "bpc": base_image.get("bpc", 8),
                    }
                )
                images.append(extracted)
                
            except Exception as e:
                logger.warning(f"提取图片失败 (page {page_num}, img {img_index}): {e}")
                continue
        
        return images
    
    def _get_image_bbox(self, page, xref: int) -> Tuple[float, float, float, float]:
        """获取图片在页面中的位置"""
        try:
            for img in page.get_image_info():
                if img.get("xref") == xref:
                    bbox = img.get("bbox", (0, 0, 0, 0))
                    return tuple(bbox)
        except:
            pass
        return (0, 0, 0, 0)
    
    # ==================== 图片描述生成 ====================
    
    def generate_image_description(
        self,
        image: ExtractedImage,
        context: str = "",
        prompt_template: Optional[str] = None
    ) -> str:
        """
        使用多模态 LLM 生成图片描述
        
        Args:
            image: 图片对象
            context: 周围的文本上下文
            prompt_template: 自定义提示模板
            
        Returns:
            生成的描述文本
        """
        if not self.generate_descriptions:
            return ""
        
        # 默认提示模板
        if not prompt_template:
            prompt_template = """分析这张技术文档中的图片，生成详细的文字描述。

要求：
1. 描述图片的主要内容和目的
2. 如果是架构图/流程图，描述各组件和它们的关系
3. 如果是截图，描述界面元素和操作步骤
4. 如果是图表，总结数据趋势和关键数值
5. 如果包含文字，提取关键文字内容
6. 用中文回答，保持专业和准确

{context}

请描述这张图片："""
        
        prompt = prompt_template.format(
            context=f"图片周围的文本上下文：\n{context}" if context else ""
        )
        
        try:
            if self.vision_provider == "openai":
                description = self._describe_with_openai(image, prompt)
            elif self.vision_provider == "anthropic":
                description = self._describe_with_anthropic(image, prompt)
            elif self.vision_provider == "google":
                description = self._describe_with_google(image, prompt)
            else:
                logger.warning(f"Unknown vision provider: {self.vision_provider}")
                return ""
            
            image.description = description
            image.image_type = self._classify_image_type(description)
            return description
            
        except Exception as e:
            logger.error(f"生成图片描述失败: {e}")
            return ""
    
    def _describe_with_openai(self, image: ExtractedImage, prompt: str) -> str:
        """使用 OpenAI GPT-4V 生成描述"""
        import openai
        
        client = openai.OpenAI()
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image.to_data_uri(),
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def _describe_with_anthropic(self, image: ExtractedImage, prompt: str) -> str:
        """使用 Anthropic Claude Vision 生成描述"""
        import anthropic
        
        client = anthropic.Anthropic()
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": f"image/{image.format}",
                                "data": image.to_base64()
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        )
        
        return response.content[0].text
    
    def _describe_with_google(self, image: ExtractedImage, prompt: str) -> str:
        """使用 Google Gemini Vision 生成描述"""
        import google.generativeai as genai
        
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # 加载图片
        if PIL_AVAILABLE:
            if image.image_bytes:
                pil_image = Image.open(io.BytesIO(image.image_bytes))
            else:
                pil_image = Image.open(image.image_path)
            
            response = model.generate_content([prompt, pil_image])
            return response.text
        else:
            raise ImportError("PIL required for Google Vision")
    
    def _classify_image_type(self, description: str) -> ImageType:
        """根据描述分类图片类型"""
        description_lower = description.lower()
        
        if any(kw in description_lower for kw in ["架构", "流程", "diagram", "flow", "组件"]):
            return ImageType.DIAGRAM
        elif any(kw in description_lower for kw in ["截图", "界面", "screenshot", "ui", "按钮"]):
            return ImageType.SCREENSHOT
        elif any(kw in description_lower for kw in ["图表", "柱状", "饼图", "chart", "graph", "曲线"]):
            return ImageType.CHART
        elif any(kw in description_lower for kw in ["表格", "table", "行列"]):
            return ImageType.TABLE_IMAGE
        elif any(kw in description_lower for kw in ["公式", "formula", "equation"]):
            return ImageType.FORMULA
        elif any(kw in description_lower for kw in ["代码", "code", "函数"]):
            return ImageType.CODE
        elif any(kw in description_lower for kw in ["logo", "图标", "icon"]):
            return ImageType.LOGO
        elif any(kw in description_lower for kw in ["照片", "photo", "实物"]):
            return ImageType.PHOTO
        
        return ImageType.UNKNOWN
    
    # ==================== 图片向量化 ====================
    
    def generate_image_embedding(
        self,
        image: ExtractedImage,
        embed_type: str = "both"  # "image", "text", "both"
    ) -> ExtractedImage:
        """
        生成图片向量
        
        Args:
            image: 图片对象
            embed_type: 
                - "image": 只生成图片向量 (CLIP)
                - "text": 只生成描述文本向量
                - "both": 两者都生成
                
        Returns:
            更新后的图片对象
        """
        # 生成图片向量 (CLIP)
        if embed_type in ("image", "both"):
            try:
                image.image_embedding = self._embed_image_with_clip(image)
            except Exception as e:
                logger.warning(f"CLIP embedding failed: {e}")
        
        # 生成文本向量
        if embed_type in ("text", "both") and image.description:
            try:
                image.text_embedding = self._embed_text(image.description)
            except Exception as e:
                logger.warning(f"Text embedding failed: {e}")
        
        return image
    
    def _embed_image_with_clip(self, image: ExtractedImage) -> List[float]:
        """使用 CLIP 生成图片向量"""
        if self._clip_model is None:
            self._load_clip_model()
        
        import torch
        
        # 加载图片
        if image.image_bytes:
            pil_image = Image.open(io.BytesIO(image.image_bytes))
        else:
            pil_image = Image.open(image.image_path)
        
        # 处理图片
        inputs = self._clip_processor(images=pil_image, return_tensors="pt")
        
        with torch.no_grad():
            image_features = self._clip_model.get_image_features(**inputs)
            # 归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features[0].tolist()
    
    def _embed_text(self, text: str) -> List[float]:
        """生成文本向量 (使用与系统一致的 embedding 模型)"""
        from src.ingestion.embedder import Embedder
        
        embedder = Embedder()
        vectors = embedder.embed([text])
        return vectors[0]
    
    def _load_clip_model(self):
        """加载 CLIP 模型"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            model_name = "openai/clip-vit-base-patch32"
            self._clip_processor = CLIPProcessor.from_pretrained(model_name)
            self._clip_model = CLIPModel.from_pretrained(model_name)
            
            logger.info(f"CLIP model loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    # ==================== 批量处理 ====================
    
    def process_document_images(
        self,
        pdf_path: str,
        doc_id: str = "",
        generate_descriptions: bool = True,
        generate_embeddings: bool = True,
        context_extractor: Optional[callable] = None
    ) -> List[ExtractedImage]:
        """
        完整处理文档中的所有图片
        
        Args:
            pdf_path: PDF 路径
            doc_id: 文档 ID
            generate_descriptions: 是否生成 AI 描述
            generate_embeddings: 是否生成向量
            context_extractor: 可选的上下文提取函数
            
        Returns:
            处理后的图片列表
        """
        # 1. 提取图片
        images = self.extract_images_from_pdf(pdf_path, doc_id, save_images=True)
        
        if not images:
            return []
        
        # 2. 生成描述
        if generate_descriptions:
            for img in images:
                context = ""
                if context_extractor:
                    context = context_extractor(img.page_num, img.bbox)
                self.generate_image_description(img, context)
                logger.debug(f"Generated description for {img.image_id}")
        
        # 3. 生成向量
        if generate_embeddings:
            for img in images:
                self.generate_image_embedding(img, embed_type="both")
                logger.debug(f"Generated embeddings for {img.image_id}")
        
        return images
    
    # ==================== 存储接口 ====================
    
    def save_to_vector_store(
        self,
        images: List[ExtractedImage],
        vector_store,
        collection_name: str = "images"
    ):
        """
        将图片保存到向量库
        
        存储两类向量：
        1. 图片向量 (CLIP) - 用于图搜图
        2. 描述文本向量 - 用于文搜图
        """
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        for img in images:
            if img.text_embedding:
                # 使用文本向量 (更常用)
                documents.append(img.description)
                embeddings.append(img.text_embedding)
                metadatas.append({
                    "image_id": img.image_id,
                    "source_doc": img.source_doc,
                    "page_num": img.page_num,
                    "image_path": img.image_path,
                    "image_type": img.image_type.value,
                    "width": img.width,
                    "height": img.height,
                    "caption": img.caption,
                    "type": "image_description"
                })
                ids.append(f"{img.image_id}_text")
            
            if img.image_embedding:
                # 图片向量 (CLIP)
                documents.append(f"[Image: {img.caption or img.description[:100]}]")
                embeddings.append(img.image_embedding)
                metadatas.append({
                    "image_id": img.image_id,
                    "source_doc": img.source_doc,
                    "page_num": img.page_num,
                    "image_path": img.image_path,
                    "image_type": img.image_type.value,
                    "type": "image_clip"
                })
                ids.append(f"{img.image_id}_clip")
        
        if embeddings:
            vector_store.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
                collection_name=collection_name
            )
            logger.info(f"Saved {len(embeddings)} image vectors to store")
    
    def to_chunks(self, images: List[ExtractedImage]) -> List[Dict[str, Any]]:
        """
        将图片转换为可索引的 chunks
        
        每张图片生成一个 chunk，包含：
        - 描述文本作为 content
        - 图片路径和元数据
        """
        chunks = []
        
        for img in images:
            if not img.description:
                continue
            
            chunk = {
                "chunk_id": f"img_{img.image_id}",
                "content": img.description,
                "content_type": "image",
                "metadata": {
                    "image_id": img.image_id,
                    "image_path": img.image_path,
                    "image_type": img.image_type.value,
                    "source_doc": img.source_doc,
                    "page_num": img.page_num,
                    "width": img.width,
                    "height": img.height,
                    "caption": img.caption,
                    "bbox": list(img.bbox)
                }
            }
            
            if img.text_embedding:
                chunk["embedding"] = img.text_embedding
            
            chunks.append(chunk)
        
        return chunks


class ImageRetriever:
    """
    图片检索器
    
    支持：
    1. 文本查询图片 (Text-to-Image)
    2. 图片查询图片 (Image-to-Image)
    3. 混合查询 (文本 + 图片)
    """
    
    def __init__(
        self,
        vector_store,
        image_processor: Optional[ImageProcessor] = None,
        collection_name: str = "images"
    ):
        self.vector_store = vector_store
        self.image_processor = image_processor or ImageProcessor()
        self.collection_name = collection_name
    
    def search_by_text(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        用文本查询图片
        
        Args:
            query: 查询文本，如 "系统架构图"
            top_k: 返回数量
            filter_dict: 过滤条件
            
        Returns:
            匹配的图片列表
        """
        # 生成查询向量
        from src.ingestion.embedder import Embedder
        embedder = Embedder()
        query_embedding = embedder.embed([query])[0]
        
        # 在向量库中搜索
        filter_dict = filter_dict or {}
        filter_dict["type"] = "image_description"
        
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict,
            collection_name=self.collection_name
        )
        
        return self._format_results(results)
    
    def search_by_image(
        self,
        image_path: str,
        top_k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        用图片查询相似图片 (CLIP)
        
        Args:
            image_path: 查询图片路径
            top_k: 返回数量
            filter_dict: 过滤条件
            
        Returns:
            相似图片列表
        """
        # 创建临时图片对象
        query_image = ExtractedImage(
            image_id="query",
            source_doc="",
            page_num=0,
            bbox=(0, 0, 0, 0),
            image_path=image_path
        )
        
        # 生成 CLIP 向量
        self.image_processor._load_clip_model()
        query_embedding = self.image_processor._embed_image_with_clip(query_image)
        
        # 搜索 CLIP 向量
        filter_dict = filter_dict or {}
        filter_dict["type"] = "image_clip"
        
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_dict=filter_dict,
            collection_name=self.collection_name
        )
        
        return self._format_results(results)
    
    def _format_results(self, results: List[Dict]) -> List[Dict[str, Any]]:
        """格式化搜索结果"""
        formatted = []
        for r in results:
            formatted.append({
                "image_id": r.get("metadata", {}).get("image_id"),
                "image_path": r.get("metadata", {}).get("image_path"),
                "image_type": r.get("metadata", {}).get("image_type"),
                "source_doc": r.get("metadata", {}).get("source_doc"),
                "page_num": r.get("metadata", {}).get("page_num"),
                "description": r.get("document", ""),
                "score": r.get("score", 0)
            })
        return formatted


# ==================== 便捷函数 ====================

def process_pdf_images(
    pdf_path: str,
    output_dir: str = "data/images",
    vision_provider: str = "openai",
    generate_descriptions: bool = True
) -> List[ExtractedImage]:
    """
    处理 PDF 中的图片 (便捷函数)
    
    Example:
        images = process_pdf_images("manual.pdf")
        for img in images:
            print(f"{img.image_id}: {img.description[:100]}")
    """
    processor = ImageProcessor(
        output_dir=output_dir,
        vision_provider=vision_provider,
        generate_descriptions=generate_descriptions
    )
    return processor.process_document_images(pdf_path)


def create_image_processor(
    vision_provider: str = "openai",
    output_dir: str = "data/images"
) -> ImageProcessor:
    """创建图片处理器"""
    return ImageProcessor(
        output_dir=output_dir,
        vision_provider=vision_provider
    )
