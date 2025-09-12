from pathlib import Path
from typing import List, Optional
from fsspec import AbstractFileSystem

from lazyllm import LOG
from .readerBase import LazyLLMReaderBase
from ..doc_node import DocNode


def is_xml_file(file: Path) -> bool:
    return file.suffix.lower() == ".xml"


class XMLReader(LazyLLMReaderBase):
    def _load_data(self, file: Path, fs: Optional[AbstractFileSystem] = None) -> List[DocNode]:
        if not isinstance(file, Path): file = Path(file)

        if is_xml_file(file):
            try:
                from lxml import etree

                tree = etree.parse(file)
                root = tree.getroot()
                xml_string = etree.tostring(
                    root, encoding="utf-8", pretty_print=True, xml_declaration=True
                ).decode("utf-8")
                return [DocNode(text=xml_string)]
            except ImportError:
                LOG.warning("lxml is not installed, using xml.etree.ElementTree instead")
                from xml.etree import ElementTree as ET
                tree = ET.parse(file)
                root = tree.getroot()
                xml_string = ET.tostring(root, encoding="utf-8", method="xml", xml_declaration=True).decode("utf-8")
                return [DocNode(text=xml_string)]
            except Exception as e:
                LOG.warning(f"Error parsing XML file {file}: {e}")
                raise e
        else:
            raise ValueError(f"Unsupported file type: {file.suffix}")
