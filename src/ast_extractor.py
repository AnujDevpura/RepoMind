from typing import Sequence, Any, List
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.core.graph_stores import EntityNode, Relation
from tree_sitter_languages import get_parser

# Monkeypatch EntityNode to support excluded_embed_metadata_keys
if not hasattr(EntityNode, "excluded_embed_metadata_keys"):
    def _get_excluded(self):
        if not hasattr(self, "_excluded_embed_metadata_keys"):
            self._excluded_embed_metadata_keys = []
        return self._excluded_embed_metadata_keys
        
    EntityNode.excluded_embed_metadata_keys = property(_get_excluded)

    def _entity_node_str(self) -> str:
        excluded = getattr(self, "excluded_embed_metadata_keys", [])
        props = {k: v for k, v in self.properties.items() if k not in excluded}
        if props:
            return f"{self.name} ({props})"
        return self.name
        
    EntityNode.__str__ = _entity_node_str

# LlamaIndex constants
KG_NODES_KEY = "nodes"
KG_RELATIONS_KEY = "relations"

class ASTPropertyGraphExtractor(TransformComponent):
    """
    Deterministically extracts architecture using Tree-Sitter ASTs.
    Supports 15+ backend, mobile, and frontend languages.
    """
    
    def __call__(
        self, nodes: Sequence[BaseNode], **kwargs: Any
    ) -> Sequence[BaseNode]:
        for node in nodes:
            self._extract_ast_graph(node)
        return nodes
        
    async def acall(
        self, nodes: Sequence[BaseNode], **kwargs: Any
    ) -> Sequence[BaseNode]:
        return self.__call__(nodes, **kwargs)

    def _extract_ast_graph(self, node: BaseNode):
        if KG_NODES_KEY not in node.metadata:
            node.metadata[KG_NODES_KEY] = []
        if KG_RELATIONS_KEY not in node.metadata:
            node.metadata[KG_RELATIONS_KEY] = []
            
        language_str = node.metadata.get("language")
        
        supported = [
            "python", "javascript", "typescript", "go", "java", "cpp", "c", "rust",
            "c_sharp", "ruby", "swift", "kotlin", "scala", "html", "css"
        ]
        
        if not language_str or language_str not in supported:
            return
            
        try:
            parser = get_parser(language_str)
            content = node.get_content()
            if not content:
                return
                
            tree = parser.parse(content.encode('utf-8'))
            
            entities = []
            relations = []
            
            self._traverse_ast(tree.root_node, language_str, node.metadata["file_name"], current_func=None, entities=entities, relations=relations)
            
            node.metadata[KG_NODES_KEY].extend(entities)
            node.metadata[KG_RELATIONS_KEY].extend(relations)
            
        except Exception as e:
            print(f"⚠️ AST Extraction failed for chunk in {node.metadata.get('file_path', 'unknown')}: {e}")

    def _get_clean_code(self, function_node, lang: str) -> str:
        ranges_to_remove = []
        
        def collect_ranges(node):
            if node.type in ["comment", "line_comment", "block_comment"]:
                ranges_to_remove.append((node.start_byte, node.end_byte))
            elif lang == "python" and node.type == "expression_statement":
                if len(node.children) > 0 and node.children[0].type == "string":
                    ranges_to_remove.append((node.start_byte, node.end_byte))
            
            for child in node.children:
                collect_ranges(child)
                
        collect_ranges(function_node)
        ranges_to_remove.sort(key=lambda x: x[0], reverse=True)
        
        raw_bytes = function_node.text
        base_start = function_node.start_byte
        
        for start, end in ranges_to_remove:
            local_start = max(0, start - base_start)
            local_end = max(0, end - base_start)
            raw_bytes = raw_bytes[:local_start] + raw_bytes[local_end:]
            
        return raw_bytes.decode('utf-8').strip()

    def _traverse_ast(self, ast_node, lang: str, file_name: str, current_func: str, entities: List[EntityNode], relations: List[Relation]):
        
        # ==========================================
        # 1. FRONTEND ARCHITECTURE (HTML/CSS)
        # ==========================================
        if lang == "html" and ast_node.type == "element":
            start_tag = next((c for c in ast_node.children if c.type == "start_tag"), None)
            if start_tag and len(start_tag.children) > 1:
                tag_name = start_tag.children[1].text.decode('utf-8')
                
                if tag_name == "script":
                    for child in start_tag.children:
                        if child.type == "attribute":
                            attr_name = child.children[0].text.decode('utf-8')
                            if attr_name == "src" and len(child.children) > 2:
                                js_file = child.children[2].text.decode('utf-8').strip('"\'')
                                entities.append(EntityNode(name=js_file, label="ASSET"))
                                relations.append(Relation(source_id=file_name, target_id=js_file, label="IMPORTS_SCRIPT"))
                                
                elif tag_name == "link":
                    is_stylesheet = False
                    href_val = None
                    for child in start_tag.children:
                        if child.type == "attribute":
                            attr_name = child.children[0].text.decode('utf-8')
                            if attr_name == "rel" and len(child.children) > 2 and "stylesheet" in child.children[2].text.decode('utf-8'):
                                is_stylesheet = True
                            if attr_name == "href" and len(child.children) > 2:
                                href_val = child.children[2].text.decode('utf-8').strip('"\'')
                    if is_stylesheet and href_val:
                        entities.append(EntityNode(name=href_val, label="ASSET"))
                        relations.append(Relation(source_id=file_name, target_id=href_val, label="IMPORTS_STYLE"))

        elif lang == "css":
            if ast_node.type == "id_selector":
                id_name = ast_node.text.decode('utf-8')
                entities.append(EntityNode(name=id_name, label="UI_ELEMENT"))
                relations.append(Relation(source_id=file_name, target_id=id_name, label="STYLES_UI"))
            elif ast_node.type == "class_selector":
                class_name = ast_node.text.decode('utf-8')
                if class_name not in [".container", ".row", ".col", ".flex", ".hidden"]:
                    entities.append(EntityNode(name=class_name, label="UI_CLASS"))


        # ==========================================
        # 2. IDENTIFY FUNCTION DEFINITIONS
        # ==========================================
        is_func_def = False
        func_name = None
        
        if lang == "python" and ast_node.type == "function_definition":
            is_func_def, name_node = True, ast_node.child_by_field_name("name")
        elif lang in ["javascript", "typescript"] and ast_node.type in ["function_declaration", "method_definition", "arrow_function"]:
            is_func_def, name_node = True, ast_node.child_by_field_name("name")
        elif lang == "go" and ast_node.type in ["function_declaration", "method_declaration"]:
            is_func_def, name_node = True, ast_node.child_by_field_name("name")
        elif lang == "java" and ast_node.type == "method_declaration":
            is_func_def, name_node = True, ast_node.child_by_field_name("name")
        elif lang in ["c", "cpp"] and ast_node.type == "function_definition":
            is_func_def = True
            decl = ast_node.child_by_field_name("declarator")
            name_node = decl.child_by_field_name("declarator") if decl and decl.type == "function_declarator" else None
        elif lang == "rust" and ast_node.type == "function_item":
            is_func_def, name_node = True, ast_node.child_by_field_name("name")
        elif lang == "c_sharp" and ast_node.type in ["method_declaration", "local_function_statement"]:
            is_func_def, name_node = True, ast_node.child_by_field_name("name")
        elif lang == "ruby" and ast_node.type in ["method", "singleton_method"]:
            is_func_def, name_node = True, ast_node.child_by_field_name("name")
        elif lang in ["swift", "kotlin"] and ast_node.type == "function_declaration":
            is_func_def, name_node = True, ast_node.child_by_field_name("name")
        elif lang == "scala" and ast_node.type == "function_definition":
            is_func_def, name_node = True, ast_node.child_by_field_name("name")

        if is_func_def and 'name_node' in locals() and name_node:
            func_name = name_node.text.decode('utf-8')
            code_block = self._get_clean_code(ast_node, lang)
            en = EntityNode(name=func_name, label="FUNCTION", properties={"code": code_block})
            en.excluded_embed_metadata_keys.append("code")
            entities.append(en)
            current_func = func_name
            
        # ==========================================
        # 3. IDENTIFY FUNCTION CALLS
        # ==========================================
        called_name = None
        
        if lang == "python" and ast_node.type == "call":
            func_node = ast_node.child_by_field_name("function")
            if func_node: called_name = func_node.text.decode('utf-8')
        elif lang in ["javascript", "typescript", "go", "c", "cpp", "rust"] and ast_node.type == "call_expression":
            func_node = ast_node.child_by_field_name("function")
            if func_node: called_name = func_node.text.decode('utf-8')
        elif lang == "java" and ast_node.type == "method_invocation":
            name_node = ast_node.child_by_field_name("name")
            if name_node: called_name = name_node.text.decode('utf-8')
        elif lang == "c_sharp" and ast_node.type == "invocation_expression":
            func_node = ast_node.child_by_field_name("function")
            if func_node: called_name = func_node.text.decode('utf-8')
        elif lang == "ruby" and ast_node.type == "call":
            func_node = ast_node.child_by_field_name("method") 
            if func_node: called_name = func_node.text.decode('utf-8')
        elif lang in ["swift", "kotlin", "scala"] and ast_node.type == "call_expression":
            func_node = ast_node.child_by_field_name("function")
            if func_node: called_name = func_node.text.decode('utf-8')

        if called_name and current_func:
            clean_call_name = called_name.split('.')[-1].strip()
            # Skip chained expressions (e.g., "toLowerCase().split('.').pop")
            # They produce generic collision nodes like "pop", "split", "toString"
            # that corrupt the graph's relation keys. Only track simple identifiers.
            _INVALID_CHARS = set('()[] \t\n"\'\\')
            if clean_call_name and not (_INVALID_CHARS & set(clean_call_name)):
                entities.append(EntityNode(name=clean_call_name, label="FUNCTION"))
                relations.append(Relation(source_id=current_func, target_id=clean_call_name, label="CALLS"))

        # Recurse
        for child in ast_node.children:
            self._traverse_ast(child, lang, file_name, current_func, entities, relations)