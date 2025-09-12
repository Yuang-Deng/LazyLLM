import io
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

from .transform import NodeTransform
from .doc_node import DocNode
from lazyllm import LOG


@dataclass
class XMLChunk:
    """Represents a chunk of XML content with full tag path."""

    content: str  # The actual XML content of this chunk
    full_xml: str  # Complete XML from root to current node
    char_count: int
    tag_path: List[str]  # Path from root to current node
    trim_level: int = 0  # 从根节点开始计数


class XMLSplitter(NodeTransform):
    """
    XML document splitter that maintains complete tag hierarchy in each chunk.

    This splitter ensures that each chunk contains the full XML structure from
    the root element down to the current element, making each chunk self-contained
    and parseable as valid XML.

    Key improvements based on xml_to_block.py:
    1. String length calculation excludes closing tags
    2. Proper handling of whitespace and newlines in content
    3. Progressive building of tag context
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        trigger_ratio: float = 0.8,
        target_ratio: float = 0.3,
        num_workers: int = 0,
    ):
        super().__init__(num_workers=num_workers)

        assert chunk_size > 0, "chunk size should > 0"

        self.chunk_size = chunk_size
        self.trigger_ratio = trigger_ratio
        self.target_ratio = target_ratio
        self.nsmap = {}
        # 添加标签计数器，用于跟踪重复标签的文档顺序
        self.tag_counters = {}

    def transform(self, node: DocNode, **kwargs) -> List[str]:
        """Transform a DocNode containing XML content into chunks."""
        xml_text = node.get_text()
        if not xml_text.strip():
            return [""]

        try:
            # Parse XML and create chunks
            chunks = self._split_xml(xml_text)

            ret = []
            for chunk in chunks:
                text = ET.tostring(ET.fromstring(chunk.full_xml), encoding="unicode")
                metadata = {
                    'tag_path': chunk.tag_path,
                    'trim_level': chunk.trim_level,
                }
                ret.append(DocNode(text=text, metadata=metadata))
            return ret
        except Exception as e:
            LOG.warning(f"Failed to parse XML, falling back to text splitting: {e}")
            # Fallback to simple text splitting if XML parsing fails
            return self._fallback_text_split(xml_text)

    def _collect_ns_decls(self, xml_text: str):
        """
        用 iterparse 收集每个元素上的命名空间声明
        返回: {element: {prefix: uri}}
        """
        ns_decls_map = {}
        events = ("start-ns", "start")
        parser = ET.XMLParser()
        it = ET.iterparse(source=io.StringIO(xml_text), events=events, parser=parser)

        parent_path = []
        self.tag_counters = {}  # 计数器初始化
        ns_stack = []
        nsmap = {}
        for event, data in it:
            if event == "start-ns":
                prefix, uri = data
                ns_stack.append((prefix, uri))
                ET.register_namespace(prefix, uri)
                nsmap[prefix] = uri
            elif event == "start":
                elem = data
                # 生成带索引的路径
                indexed_path = self._get_indexed_tag_path(elem, parent_path)
                # 更新父路径堆栈
                parent_path.append(indexed_path[-1])

                if ns_stack:
                    ns_decls_map[tuple(indexed_path)] = {prefix: uri for prefix, uri in ns_stack}
                    ns_stack.clear()
        reverse_nsmap = {v: k for k, v in nsmap.items()}
        return ns_decls_map, reverse_nsmap

    def _split_xml(self, xml_text: str) -> List[XMLChunk]:
        """Split XML content into chunks while maintaining tag hierarchy."""
        try:
            self.ns_decls_map, self.reverse_nsmap = self._collect_ns_decls(xml_text)
            # Parse XML
            root = ET.fromstring(xml_text)

            # 使用新的简化方法
            chunks = self._simple_traverse_and_build_chunks(root)

            # Merge adjacent chunks if possible
            chunks = self._merge_chunks(chunks)

            return chunks

        except ET.ParseError as e:
            LOG.warning(f"XML parsing error: {e}, falling back to text splitting")
            return self._fallback_text_split(xml_text)

    def _simple_traverse_and_build_chunks(self, root: Element) -> List[XMLChunk]:  # noqa: C901
        """Build XML chunks with complete tag hierarchy using placeholders."""
        chunks = []
        # 重置标签计数器
        self.tag_counters = {}

        def attrs_to_string(node, xpath, ns_decls_map, reverse_nsmap):
            """
            把节点的属性拼成字符串，保留命名空间前缀 + 在定义处恢复 xmlns
            """
            attrs = []

            # 1. 恢复 xmlns 声明
            if xpath in ns_decls_map:
                for prefix, uri in ns_decls_map[xpath].items():
                    if prefix:
                        attrs.append(f'xmlns:{prefix}="{uri}"')
                    else:
                        attrs.append(f'xmlns="{uri}"')

            # 2. 处理普通属性
            for k, v in node.attrib.items():
                if k.startswith("{"):  # 带命名空间的属性
                    uri, local = k[1:].split("}")
                    prefix = reverse_nsmap.get(uri)
                    if prefix:
                        k = f"{prefix}:{local}"
                    else:
                        k = local
                attrs.append(f'{k}="{v}"')

            return " ".join(attrs)

        def qname_to_prefixed(tag: str, reverse_nsmap: Dict[str, str]) -> str:
            """
            将QName转换为带命名空间前缀的标签名
            """
            if tag.startswith("{"):
                uri, local = tag[1:].split("}")
                prefix = reverse_nsmap.get(uri)
                if prefix:
                    return f"{prefix}:{local}"
                else:
                    return local
            else:
                return tag

        def traverse(node: Element, prefix: str = "", parent_path: List[str] = None,
                     open_tag_stack: List[str] = None, closing_tag_stack: List[str] = None, trim_level: int = 0):
            """
            Traverse XML tree and build chunks with complete tag hierarchy.
            prefix format: "<tag {attrs}>{placeholder}</tag>" or empty string
            """
            nonlocal chunks  # noqa: F824

            # 初始化parent_path
            if parent_path is None:
                parent_path = []
            if open_tag_stack is None:
                open_tag_stack = []
            if closing_tag_stack is None:
                closing_tag_stack = []

            # # 生成带索引的标签路径
            indexed_tag_path = self._get_indexed_tag_path(node, parent_path)

            # 获取当前节点的标签名和属性
            attrs = attrs_to_string(node, tuple(indexed_tag_path), self.ns_decls_map, self.reverse_nsmap)
            tag_name = qname_to_prefixed(node.tag, self.reverse_nsmap)
            opening_tag = f"<{tag_name}{(' ' + attrs) if node.attrib else ''}>"
            closing_tag = f"</{tag_name}>"

            # 压栈
            open_tag_stack.append(opening_tag)
            closing_tag_stack = [closing_tag] + closing_tag_stack

            # 构建当前节点的完整 XML
            current_node_xml = ET.tostring(node, encoding="unicode")

            # 如果前缀为空，说明这是根节点
            if not prefix:
                full_xml = current_node_xml
            else:
                # 直接使用占位符替换，不去除外层标签
                # 这样可以保持完整的标签结构
                full_xml = prefix.replace("{placeholder}", current_node_xml)

            # 检查长度是否适合
            if len(full_xml) <= self.chunk_size:
                chunks.append(
                    XMLChunk(
                        content=full_xml,
                        full_xml=full_xml,
                        char_count=len(full_xml),
                        tag_path=indexed_tag_path,
                        trim_level=trim_level,
                    )
                )
            else:
                # 节点太长，需要分割
                # 构建新的前缀，包含当前节点的占位符格式
                if not prefix:
                    # 根节点，创建新的占位符格式
                    new_prefix = f"{opening_tag}{{placeholder}}{closing_tag}"
                else:
                    # 在现有前缀中添加新的占位符格式
                    # 将现有前缀中的占位符替换为新的标签结构
                    new_prefix = prefix.replace("{placeholder}", f"{opening_tag}{{placeholder}}{closing_tag}")

                # ✅ 裁剪逻辑：触发阈值 + 目标长度
                trigger_length = int(self.chunk_size * self.trigger_ratio)

                if len(new_prefix) > trigger_length:
                    new_prefix, trim_level = self._build_prefix_from_stack(open_tag_stack, closing_tag_stack, trim_level)

                # 1. 先处理当前节点的 text（出现在第一个子节点前的文本）
                if node.text and node.text.strip():
                    self._handle_text_content_with_placeholder(
                        node.text, new_prefix, chunks, indexed_tag_path, trim_level
                    )

                # 2. 递归处理子节点，传递新的前缀和当前路径，同时处理每个子节点的 tail
                for child in node:
                    traverse(child, new_prefix, indexed_tag_path, open_tag_stack, closing_tag_stack, trim_level)

                    # 处理子节点的 tail
                    if child.tail and child.tail.strip():
                        self._handle_text_content_with_placeholder(child.tail, new_prefix, chunks, indexed_tag_path)

                # 3. 如果既没有子节点也没有文本，那就是空标签
                if not node.text and len(node) == 0:
                    tag_only = new_prefix.replace("{placeholder}", "")
                    chunks.append(
                        XMLChunk(
                            content=tag_only,
                            full_xml=tag_only,
                            char_count=len(tag_only),
                            tag_path=indexed_tag_path,
                            trim_level=trim_level,
                        )
                    )
            # 弹栈
            open_tag_stack.pop()
            closing_tag_stack.pop(0)

        traverse(root, "", [])
        return chunks

    def _build_prefix_from_stack(self, open_tag_stack: List[str], closing_tag_stack: List[str], trim_level: int) -> str:
        """
        根据 open_tag_stack 和 closing_tag_stack 生成 prefix。
        :param open_tag_stack: 从根到当前节点标签路径
        :param closing_tag_stack: 从当前节点到根节点标签路径
        :param trim_level: 从根节点开始计数
        """
        assert len(open_tag_stack) == len(
            closing_tag_stack
        ), "open_tag_stack and closing_tag_stack must have the same length"
        assert trim_level >= 0, "trim_level must be >= 0"
        assert trim_level <= len(open_tag_stack), "trim_level must be <= len(open_tag_stack)"

        target_length = int(self.chunk_size * self.target_ratio)

        new_prefix = (
            "".join(open_tag_stack[trim_level:])
            + "{placeholder}"
            + "".join(closing_tag_stack[: len(closing_tag_stack) - trim_level])
        )

        while len(new_prefix) > target_length and trim_level < len(open_tag_stack) - 1:
            trim_level += 1
            new_prefix = (
                "".join(open_tag_stack[trim_level:])
                + "{placeholder}"
                + "".join(closing_tag_stack[: len(closing_tag_stack) - trim_level])
            )

        return new_prefix, trim_level

    def _handle_text_content_with_placeholder(
        self, text: str, prefix: str, chunks: List[XMLChunk], tag_path: List[str], trim_level: int = 0
    ):
        """Handle text content using placeholder format."""
        # 计算可用空间（排除占位符）
        placeholder_length = len("{placeholder}")
        available_space = self.chunk_size - len(prefix) + placeholder_length

        # 检查文本是否适合一个 chunk
        if len(text) <= available_space:
            # 文本适合一个 chunk
            text_block = prefix.replace("{placeholder}", text)
            chunks.append(
                XMLChunk(
                    content=text_block,
                    full_xml=text_block,
                    char_count=len(text_block),
                    tag_path=tag_path,
                    trim_level=trim_level,
                )
            )
        else:
            # 文本太长，需要分割
            self._split_text_content_with_placeholder(text, available_space, prefix, chunks, tag_path, trim_level)

    def _split_text_content_with_placeholder(
        self,
        text: str,
        available_space: int,
        prefix: str,
        chunks: List[XMLChunk],
        tag_path: List[str],
        trim_level: int = 0,
    ):
        """Split long text content using placeholder format."""
        # 计算可用空间（排除占位符）
        if re.search(r"[A-Za-z]", text) and " " in text:
            # 用空格切分英文字符串
            start = 0
            n = len(text)
            while start < n:
                # 当前 chunk 的结尾候选
                end = min(start + available_space, n)

                if end < n:  # 不是最后一个端， 需要找切分
                    # 在范围内找最后一个空格
                    split_index = text.rfind(" ", start, end)
                    if split_index == -1 or split_index <= start:
                        # 没有找到合适空格，直接硬切
                        split_index = end
                    else:
                        # 包含空格本身，保证原文完整
                        split_index += 1
                    text_chunk = text[start:split_index]
                    full_chunk = prefix.replace("{placeholder}", text_chunk)
                    chunks.append(
                        XMLChunk(
                            content=full_chunk,
                            full_xml=full_chunk,
                            char_count=len(full_chunk),
                            tag_path=tag_path,
                            trim_level=trim_level,
                        )
                    )
                    start = split_index
                else:
                    # 最后一个 chunk
                    text_chunk = text[start:end]
                    full_chunk = prefix.replace("{placeholder}", text_chunk)
                    chunks.append(
                        XMLChunk(
                            content=full_chunk,
                            full_xml=full_chunk,
                            char_count=len(full_chunk),
                            tag_path=tag_path,
                            trim_level=trim_level,
                        )
                    )
                    break
        else:
            # 没有空格或非英文，回退字符切分
            for i in range(0, len(text), available_space):
                text_chunk = text[i:i + available_space]
                full_chunk = prefix.replace("{placeholder}", text_chunk)

                chunks.append(
                    XMLChunk(
                        content=full_chunk,
                        full_xml=full_chunk,
                        char_count=len(full_chunk),
                        tag_path=tag_path,
                        trim_level=trim_level,
                    )
                )

    def _get_indexed_tag_path(self, node: Element, parent_path: List[str]) -> List[str]:
        """
        生成带文档顺序索引的标签路径。
        对于没有唯一属性的重复标签，添加索引以区分它们。
        """
        if not parent_path:
            # 根节点
            return [node.tag]

        # 检查当前节点是否有唯一标识符
        unique_attr = self._get_unique_identifier(node)

        # 没有唯一标识符，需要检查是否需要添加索引
        # 由于xml.etree.ElementTree没有getparent()方法，我们使用计数器来处理
        current_path = "/".join(parent_path) + "/" + node.tag

        current_path = current_path + f"[{unique_attr}]"

        # 使用计数器为重复标签添加索引
        if current_path not in self.tag_counters:
            self.tag_counters[current_path] = 0

        self.tag_counters[current_path] += 1

        indexed_tag = (
            f"{node.tag}[{unique_attr}][{self.tag_counters[current_path] - 1}]"
        )

        return parent_path + [indexed_tag]

    def _merge_chunks(self, chunks: List[XMLChunk]) -> List[XMLChunk]:
        """Merge adjacent chunks by removing duplicate outer tags and merging content."""
        if len(chunks) <= 1:
            return chunks

        merged_chunks = []
        i = 0

        while i < len(chunks):
            current_chunk = chunks[i]
            merged_content = [current_chunk.content]
            merged_tag_path = [current_chunk.tag_path.copy()]
            merged_trim_level = [current_chunk.trim_level]

            # 尝试与下一个 chunk 合并
            j = i + 1
            while j < len(chunks):
                # 检查合并后的长度是否会超出限制
                test_content = merged_content + [chunks[j].content]
                test_merged = self.merge_xml_chunks(
                    test_content,
                    merged_tag_path + [chunks[j].tag_path],
                    merged_trim_level + [chunks[j].trim_level],
                )
                test_size = len(test_merged)

                if test_size <= self.chunk_size:
                    # 可以合并，继续尝试下一个
                    merged_content.append(chunks[j].content)
                    # 使用最新的路径信息而不是公共前缀
                    merged_tag_path.append(chunks[j].tag_path.copy())
                    merged_trim_level.append(chunks[j].trim_level)
                    j += 1
                else:
                    # 超出长度限制，停止合并
                    break

            # 创建合并后的 chunk
            if len(merged_content) > 1:
                # 多个 chunk 被合并，使用 merge_xml_chunks
                merged_xml = self.merge_xml_chunks(
                    merged_content,
                    [chunk.tag_path for chunk in chunks[i:j]],
                    [chunk.trim_level for chunk in chunks[i:j]],
                )
                merged_chunk = XMLChunk(
                    content=merged_xml,
                    full_xml=merged_xml,
                    char_count=len(merged_xml),
                    tag_path=merged_tag_path,
                    trim_level=min(merged_trim_level),
                )
                merged_chunks.append(merged_chunk)
            else:
                # 没有合并发生
                merged_chunks.append(current_chunk)

            i = j

        return merged_chunks

    def _extract_tag_name_and_index(self, tag: str) -> tuple:
        """
        从标签字符串中提取标签名、属性串和索引。
        例如：
            "step[0]"                -> ("step", "@none", 0)
            "entry[align=A][0]"      -> ("entry", "align=A", 0)
            "entry[id=123,type=abc][2]" -> ("entry", "id=123,type=abc", 2)
            "row[@none][1]"          -> ("row", "@none", 1)
            "product"                -> ("product", None, None)
        """
        # 用正则获取所有中括号内容
        matches = re.findall(r"\[([^\]]*)\]", tag)

        # 标签名就是第一个 "[" 之前的部分
        name = tag.split("[", 1)[0]

        attrs, index = None, None
        if matches:
            if matches[-1].isdigit():  # 最后一个是数字，说明是索引
                index = int(matches[-1])
                if len(matches) > 1:
                    attrs = matches[-2]
            else:
                attrs = matches[-1]

        return name, attrs, index

    def merge_xml_chunks(self, xml_strings, tag_paths=None, trim_levels=None):
        """
        通用 XML 合并算法：根据索引信息智能合并XML内容
        Args:
            xml_strings: 要合并的XML字符串列表
            tag_paths: 对应的标签路径列表，用于索引信息判断
            trim_levels: 对应的从根节点开始计数列表，用于索引信息判断
        """
        if not xml_strings:
            return ""

        if len(xml_strings) == 1:
            return xml_strings[0]

        trees = [ET.fromstring(s) for s in xml_strings]

        # 如果没有提供标签路径信息，使用传统合并方式
        if tag_paths is None or len(tag_paths) != len(trees):
            return self._traditional_merge_xml(trees)

        # 使用索引信息进行智能合并
        return self._smart_merge_xml_with_index(trees, tag_paths, trim_levels)

    def _traditional_merge_xml(self, trees):
        """传统的XML合并方式"""
        base = trees[0]
        for tree in trees[1:]:
            self.merge_nodes(base, tree)
        return ET.tostring(base, encoding="unicode")

    def _smart_merge_xml_with_index(self, trees, tag_paths, trim_levels):
        """
        基于索引信息的智能XML合并
        由于所有chunk都包含完整的标签路径，它们必然有相同的根节点。
        从根节点开始往下比较，当遇见不相同的（包含标签和索引），就不用往下找了，直接合并到上一级标签即可
        """
        if len(trees) == 1:
            return ET.tostring(trees[0], encoding="unicode")

        # 所有树都有相同的根节点，以第一个为基准进行智能合并
        base = deepcopy(trees[0])

        # 对每个后续的树，执行基于索引的智能合并
        for i, tree in enumerate(trees[1:], 1):
            self._merge_tree_simple(base, tree, tag_paths[i - 1], tag_paths[i], trim_levels[i - 1], trim_levels[i])

        return ET.tostring(base, encoding="unicode")

    def _merge_tree_simple(self, target, source, target_path, source_path, target_trim_level, source_trim_level):
        """
        简化的合并逻辑：从根节点开始往下比较，当遇见不相同的就停止递归
        """
        # 如果根节点标签和属性以及从根节点开始计数相同，继续比较子节点
        if (
            target.tag == source.tag
            and target.attrib == source.attrib
            and target_trim_level == source_trim_level
        ) or (
            target_trim_level != source_trim_level
            and source_trim_level < len(target_path)
            and target_path[source_trim_level] == source_path[source_trim_level]
        ):
            # 从根节点开始，逐层比较路径
            self._merge_children_by_path_level(
                target, source, target_path, source_path, target_trim_level, source_trim_level, source_trim_level
            )
        else:
            # 根节点不匹配，直接添加
            target.append(deepcopy(source))

    def _merge_children_by_path_level(
        self,
        target,
        source,
        target_path,
        source_path,
        target_trim_level,
        source_trim_level,
        level
    ):
        """
        按路径层级合并子节点
        level: 当前比较的层级（0=根节点，1=第一层子节点，以此类推）
        """
        # 如果已经到达路径末尾，停止递归
        if level >= len(target_path) or level >= len(source_path):
            return

        # 比较当前层级的标签和索引
        target_tag_info = target_path[level]
        source_tag_info = source_path[level]

        # 提取标签名和索引
        target_tag_name, target_attrs, target_index = self._extract_tag_name_and_index(target_tag_info)
        source_tag_name, source_attrs, source_index = self._extract_tag_name_and_index(source_tag_info)

        # 如果标签名不同，停止递归，直接添加源节点的内容
        if target_tag_name != source_tag_name:
            self._add_source_content_to_target(
                target,
                source,
                level - 1,
                source_path,
                target_trim_level,
                source_trim_level
            )
            return

        # 如果索引不同，停止递归，直接添加源节点的内容
        if target_index != source_index or target_attrs != source_attrs:
            self._add_source_content_to_target(
                target,
                source,
                level - 1,
                source_path,
                target_trim_level,
                source_trim_level
            )
            return

        # 标签名和索引都相同，继续递归到下一层
        if level + 1 < len(target_path) and level + 1 < len(source_path):
            self._merge_children_by_path_level(
                target,
                source,
                target_path,
                source_path,
                target_trim_level,
                source_trim_level,
                level + 1,
            )
        else:
            # 已经到达路径末尾，直接添加源节点的内容（因为这一级是最后一级也是相同标签和索引，所以直接添加）
            self._add_source_content_to_target(target, source, level, source_path, target_trim_level, source_trim_level)

    def _add_source_content_to_target(self, target, source, level, source_path, target_trim_level, source_trim_level):
        """
        将源节点的内容添加到目标节点，而不是整个节点
        根据level信息，确定应该在哪一层级进行合并
        """
        # 根据level信息，确定合并策略
        if level == 0:
            # 根层级：只添加子节点，不添加整个dmodule标签
            for child in source:
                target.append(deepcopy(child))
        else:
            # 非根层级：根据level信息找到source中对应层级的节点
            if level < len(source_path):
                # 从source_path中提取对应层级的标签信息
                source_tag_info = source_path[level]
                source_tag_name, source_attrs, source_index = self._extract_tag_name_and_index(source_tag_info)

                # 在target中查找相同标签的节点
                # 在target中查找相同标签和索引的节点
                existing_node = self._find_node_at_level_with_index(
                    target, level, source_tag_name, source_attrs, source_index, target_trim_level
                )

                if existing_node is not None:
                    # 如果找到现有的相同标签节点，需要找到source中对应层级的节点
                    # 递归查找source中对应层级的节点
                    source_node = self._find_node_at_level_with_index(
                        source, level, source_tag_name, source_attrs, source_index, source_trim_level
                    )
                    if source_node is not None:
                        # 添加对应层级节点的子节点
                        for child in source_node:
                            existing_node.append(deepcopy(child))
                    else:
                        # 如果没找到对应层级的节点，添加整个source的子节点
                        for child in source:
                            existing_node.append(deepcopy(child))
                else:
                    # 如果没有找到，只添加子节点，不添加整个标签结构
                    for child in source:
                        target.append(deepcopy(child))
            else:
                # level超出范围，直接添加子节点
                for child in source:
                    target.append(deepcopy(child))

        # 如果源节点有文本内容，添加到目标节点
        if source.text and source.text.strip():
            if target.text:
                target.text += " " + source.text.strip()
            else:
                target.text = source.text.strip()

    def _find_node_at_level_with_index(
        self, element, target_level, target_tag_name, target_attrs, target_index, current_level=0
    ):
        """
        在element中查找指定层级、指定标签名和索引的节点
        """
        if current_level == target_level and element.tag == target_tag_name:
            # 如果没有索引要求，直接返回
            if target_index is None:
                return element

            unique_attr = self._get_unique_identifier(element)
            if unique_attr == target_attrs:
                return element

        # 从后向前遍历，优先找最新的节点
        for child in reversed(list(element)):
            result = self._find_node_at_level_with_index(
                child, target_level, target_tag_name, target_attrs, target_index, current_level + 1
            )
            if result is not None:
                return result

        return None

    def merge_nodes(self, target, source):
        """
        递归合并：如果标签和属性相同，就继续合并子节点；
        否则，直接追加到目标。
        利用索引信息进行更智能的合并决策。
        """
        if target.tag == source.tag and target.attrib == source.attrib:
            # 合并子节点
            for child in source:
                # 使用索引信息进行更智能的匹配
                matched = self._find_matching_child_by_index(target, child)
                if matched is not None:
                    self.merge_nodes(matched, child)
                else:
                    target.append(deepcopy(child))
        else:
            # 如果根不一致，直接挂到目标末尾（适配不同chunk结构）
            target.append(deepcopy(source))

    def _find_matching_child_by_index(self, target, child):
        """
        基于索引信息查找匹配的子节点。
        优先使用索引匹配，如果没有索引则回退到传统的标签+属性匹配。
        """
        # 由于xml.etree.ElementTree的限制，我们主要依赖传统的匹配方式
        # 但保留索引匹配的框架，以便将来扩展

        # 首先尝试基于索引的匹配（如果标签名包含索引）
        child_tag = child.tag
        if "[" in child_tag and "]" in child_tag:
            # 提取标签名和索引
            tag_name, attrs, index = self._extract_tag_name_and_index(child_tag)

            # 查找具有相同索引的节点
            for tgt_child in target:
                if tgt_child.tag == child_tag or (
                    tgt_child.tag == tag_name
                    and self._get_unique_identifier(tgt_child) == attrs
                    and self._extract_tag_name_and_index(tgt_child.tag)[1] == index
                ):
                    return tgt_child

        # 如果没有索引或索引匹配失败，回退到传统匹配
        for tgt_child in target:
            if tgt_child.tag == child.tag and tgt_child.attrib == child.attrib:
                return tgt_child

        return None

    def _get_unique_identifier(self, element: Element) -> Optional[str]:
        """
        返回元素的唯一标识符字符串，否则返回 None

        规则：
        1. 如果没有属性 -> 返回 None（交给索引保证唯一性）
        2. 如果有属性 -> 按属性名排序，拼接成字符串，确保顺序稳定
        格式示例: attr1=value1|attr2=value2
        """
        if not element.attrib:
            return "@none"

        # 按属性名排序，拼接成字符串
        sorted_attrs = sorted(element.attrib.items(), key=lambda kv: kv[0])
        unique_id = "|".join(
            f"{attr}={str(value).strip() if value is not None else ''}"
            for attr, value in sorted_attrs
        )

        return unique_id

    def _fallback_text_split(self, text: str) -> List[str]:
        """Fallback method for text that can't be parsed as XML."""
        # Simple character-based splitting
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks
