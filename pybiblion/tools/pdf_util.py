#!/usr/bin/env python3
"""Extract pdf structure in XML format"""
import json
import logging
import os.path
import re
import sys
import warnings
from collections import Counter
from typing import Any, Container, Dict, Iterable, List, Optional, TextIO, Union, cast
from argparse import ArgumentParser

import pdfminer
from pdfminer.pdfdocument import PDFDocument, PDFNoOutlines, PDFXRefFallback
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdftypes import PDFObjectNotFound, PDFValueError
from pdfminer.pdftypes import PDFStream, PDFObjRef, resolve1, stream_value
from pdfminer.psparser import PSKeyword, PSLiteral, LIT
from pdfminer.utils import isnumber




def get_subtitles(fpth: str):
    def escape(s: Union[str, bytes]) -> str:
        ESC_PAT = re.compile(r'[\000-\037&<>()"\042\047\134\177-\377]')
        if isinstance(s, bytes):
            us = str(s, "latin-1")
        else:
            us = s
        return ESC_PAT.sub(lambda m: "&#%d;" % ord(m.group(0)), us)
    #outfp: TextIO,
    fp = open(fpth, "rb")
    parser = PDFParser(fp)
    try:
        doc = PDFDocument(parser)
    except:
        return {}
    pages = {
        page.pageid: pageno
        for (pageno, page) in enumerate(PDFPage.create_pages(doc), 1)
    }

    def resolve_dest(dest: object) -> Any:
        if isinstance(dest, (str, bytes)):
            dest = resolve1(doc.get_dest(dest))
        elif isinstance(dest, PSLiteral):
            dest = resolve1(doc.get_dest(dest.name))
        if isinstance(dest, dict):
            dest = dest["D"]
        if isinstance(dest, PDFObjRef):
            dest = dest.resolve()
        return dest

    titles = {}
    try:

        outlines = doc.get_outlines()
        # outfp.write("<outlines>\n")
        for (level, title, dest, a, se) in outlines:
            pageno = None
            if dest:
                dest = resolve_dest(dest)
                pageno = pages[dest[0].objid]
            elif a:
                action = a
                if isinstance(action, dict):
                    subtype = action.get("S")
                    if subtype and repr(subtype) == "/'GoTo'" and action.get("D"):
                        dest = resolve_dest(action["D"])
                        pageno = pages[dest[0].objid]
            s = escape(title)
            if f'level{str(level)}' in titles:
                titles[f'level{str(level)}'].append({'title':s, 'pageno':pageno})
            else:
                titles[f'level{str(level)}'] = [{'title':s, 'pageno':pageno}]
            # print(s)
            # titles[f'level{str(level)}'] = titles[f'level{str(level)}'] if f'level{str(level)}' in titles else []
            # if level <= max_level:
            #     titles.append({'title':s,'level':level,'pageno':pageno})
    except PDFNoOutlines:
        warnings.warn(f'PDF NO OUTLINES: {fpth}')
        pass
    parser.close()
    fp.close()
    # print(key_word_freq)
    return titles
from pdfminer.high_level import extract_text
def get_section_title(title):
    pass
def extract_keyword_from_pdf(fpth):
    titles = get_subtitles(fpth)
    print(titles)
    text = extract_text(fpth)
    start_index = text.lower().find('abstract')
    end_index = text.lower().replace('\n', '').replace(' ', '').find(titles['level1'][0]['title'].replace(' ', ''))

    abstract_text = text[start_index:end_index]
    print(start_index,end_index)
    # print(abstract_text)

#     messages = [
#         {"role": "system",
#          "content": "You are a researcher, who is good at reading academic paper, and familiar with all of the "
#                     "citation style. Please note that the provided citation text may not have the correct line breaks "
#                     "or numbering identifiers."},
#
#         {"role": "user",
#          "content": f'''Extract the paper title only from the given reference text, and answer with the following format.
#                 [1] xxx
#                 [2] xxx
#                 [3] xxx
#             Reference text: {text}
# '''},
#     ]
#
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
# 
#         messages=messages,
#     )
#     result = ''
#     for choice in response.choices:
#         result += choice.message.content
#     result = result.split('\n')
#     return result
from markdown_it import MarkdownIt
from bs4 import BeautifulSoup
import re


def get_structure_md(md_pth):
    # 
    md = MarkdownIt()
    import md_toc
    # 
    with open(md_pth, 'r', encoding='utf-8') as md_file:
        markdown_text = md_file.read()
    # 
    # 
    html_content = md.render(markdown_text)

    # 
    soup = BeautifulSoup(html_content, 'html.parser')

    # 
    headings = soup.find_all(re.compile('^h[1-6]$'))

    # 
    content_dict = []

    # 
    for i, heading in enumerate(headings):
        level = int(heading.name[1])  # 
        title = heading.get_text()  # 

        # 
        content = []
        for sibling in heading.find_next_siblings():
            if sibling.name and sibling.name.startswith('h'):
                break
            content.append(sibling.get_text().strip())

        # 
        content_dict.append({
            'ID': i,
            'level': level,
            'title': title,
            'content': ' '.join(content).strip()
        })
    return content_dict


def get_toc_from_md(md_dict_list,remove_abs_ref = True):
    if md_dict_list:
        current_indent = 0
        previous_level = 0

        output = ""
        abstract_index = -1
        reference_index = len(md_dict_list)
        if remove_abs_ref:
            for i, item in enumerate(md_dict_list):
                if item['title'].lower() == 'abstract':
                    abstract_index = i
                if item['title'].lower() == 'references' or item['title'].lower() == 'reference':
                    reference_index = i


        for item in md_dict_list[abstract_index + 1:reference_index]:
            if item['level'] > previous_level:
                current_indent += 1
            elif item['level'] < previous_level:
                current_indent -= 1

            current_indent = max(current_indent, 0)

            indent = '\t' * item['level']

            output += f"{indent}[Heading. {item['level']}: {item['title']}]\n"

            previous_level = item['level']

        return output
    else:
        return None

def get_idtoc_from_md(md_dict_list,remove_abs_ref = True):
    if md_dict_list:
        current_indent = 0
        previous_level = 0

        output = ""
        abstract_index = -1
        reference_index = len(md_dict_list)
        if remove_abs_ref:
            for i, item in enumerate(md_dict_list):
                if item['title'].lower() == 'abstract':
                    abstract_index = i
                if item['title'].lower() == 'references' or item['title'].lower() == 'reference':
                    reference_index = i


        for item in md_dict_list[abstract_index + 1:reference_index]:
            if item['level'] > previous_level:
                current_indent += 1
            elif item['level'] < previous_level:
                current_indent -= 1

            current_indent = max(current_indent, 0)

            indent = '\t' * item['level']

            output += f"{indent}[Heading. {item['level']}: {item['title']}]\n"

            previous_level = item['level']

        return output
    else:
        return None


def extract_title_and_abstract(md_pth: str):
    with open(md_pth, 'r', encoding='utf-8') as md_file:
        md_content = md_file.read()
        # Split content into lines
        lines = md_content.splitlines()

        # Extract title (first non-empty line, remove leading #)
        title = next((line.lstrip('# ').strip() for line in lines if line.strip()), "")

        # Extract abstract (line after "###### Abstract")
        abstract_match = re.search(r"###### Abstract\s*\n(.+)", md_content)
        abstract = abstract_match.group(1).strip() if abstract_match else None

        return title, abstract


def get_toc_from_mineru_rst(rst_path: str):
    """
    从 mineru rst 的 json 文件中提取所有 text_level=1 的标题
    返回一个 list[str]
    """
    with open(rst_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 有的 json 最外层就是 list
    if not isinstance(data, list):
        raise ValueError("JSON 格式错误，应该是一个列表")

    titles = [item["text"] for item in data if item.get("text_level") == 1]
    return titles

if __name__ == "__main__":
    # titles = get_subtitles(r'xxx')
    # print(titles)
    toc = get_toc_from_mineru_rst(r'E:\SLR_minerU_RST\1905.06658v4\1905.06658v4_content_list.json')
    print(toc)
    pass