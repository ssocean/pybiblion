import json
import os

import tiktoken
from langchain_core.prompts import ChatPromptTemplate
from retry import retry
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langchain_community.document_loaders import UnstructuredMarkdownLoader, PDFMinerLoader
import requests_cache
import tiktoken
from langchain_core.exceptions import OutputParserException

from tools.pdf_util import get_structure_md, get_toc_from_md, extract_title_and_abstract

# 
requests_cache.install_cache('requests_cache')

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field
from typing import List
from typing import Dict

import langchain
from langchain.chains import LLMChain
from langchain_core.output_parsers import CommaSeparatedListOutputParser, StrOutputParser
# import langchain.chains.retrieval_qa.base
from langchain.prompts import PromptTemplate

from langchain.text_splitter import CharacterTextSplitter
from cfg.config import *
def _get_ref_list(text):
    messages = [
        SystemMessage(content="You are a researcher, who is good at reading academic paper, and familiar with all of the "
                    "citation style. Please note that the provided citation text may not have the correct line breaks "
                    "or numbering identifiers."),
        HumanMessage(content=f'''
        Extract the paper title only from the given reference text, and answer with the following format.
                [1] xxx
                [2] xxx
                [3] xxx 
            Reference text: {text}
        '''),
    ]

    chat = ChatOpenAI(model="gpt-3.5-turbo")


    return chat.batch([messages])[0].content


@retry(delay=6,)
def get_chatgpt_keyword(title, abstract):
    messages = [
        SystemMessage(content="You are a profound researcher in the field of artificial intelligence who is good at selecting "
                    "keywords for the paper with given title and abstract. Here are some guidelines for selecting keywords: 1. Represent the content of the title and abstract. 2. Be specific to the field or sub-field. "
                    "3. Keywords should be descriptive. 4. Keywords should reflect a collective understanding of the topic. 5. If the research paper involves a key method or technique, put the term in keywords"),
        HumanMessage(content=f'''Summarize 3-5 keywords only from the given title and abstract, and answer with the following format: xxx, xxx, ..., xxx,
            Given Title: {title}
            Given Abstract: {abstract}
'''),
    ]

    chat = ChatOpenAI(model="gpt-3.5-turbo")

    return chat.batch([messages])[0].content




from langchain.schema import SystemMessage, HumanMessage
@retry(delay=6,)
def check_PAMIreview(title, abstract):
    messages = [
        SystemMessage(
            content=(
                "You are a profound researcher in the field of artificial intelligence who is good at "
                "identifying whether a paper is a survey or literature review paper. "
                "Note that not all papers that have 'survey' or 'review' in the title are review papers. "
                "Here are some examples:\n"
                "- 'Transformers in Medical Image Analysis: A Review' is a survey paper.\n"
                "- 'Creating a Scholarly Knowledge Graph from Survey Article Tables' is not a survey.\n"
                "- 'Providing Insights for Open-Response Surveys via End-to-End Context-Aware Clustering' is not a survey.\n"
                "- 'SIFN: A Sentiment-Aware Interactive Fusion Network for Review-Based Item Recommendation' is not a review."
            )
        ),
        HumanMessage(
            content=(
                f"Based on the provided title and abstract, determine whether the paper is a literature review or survey paper. "
                f"Respond with 'Y' if it is a literature review or survey paper, and 'N' if it is not.\n"
                f"Given Title: {title}\n"
                f"Given Abstract: {abstract}\n"
                f"Answer with the exact following format: Y||N"
            )
        ),
    ]
    chat = ChatOpenAI(model_name="gpt-3.5-turbo")
    response = chat(messages)
    return 'y' in response.content.strip().lower()




def get_unnum_sectitle(sectitle):

    messages = [
        SystemMessage(
            content="You are a profound researcher in the field of artificial intelligence who have read a lot of "
                    "paper. You can figure out what is the title of section, irrespective of whether they are "
                    "numbered or unnumbered, and the specific numbering format utilized."),
        HumanMessage(content=f'This is the title of section, extract the title without chapter numbering(If chapter numbering exists). Answer with the following format: xxx. \n Section Title: {sectitle}'),
    ]
    chat = ChatOpenAI(model="gpt-3.5-turbo")


    return chat.batch([messages])[0].content

def get_chatgpt_field(title, abstract=None, sys_content=None, usr_prompt=None, extra_prompt=True):

    if not sys_content:
        sys_content = (
            "You are a profound researcher who is good at identifying the topic keyword from paper's title and "
            "abstract. The keyword will be used to retrieve related paper from online scholar search engines.")
    if not usr_prompt:
        usr_prompt = (
            "Identifying the topic of the paper based on the given title and abstract. So that I can use it as "
            "keyword to search highly related papers from Google Scholar.  Avoid using broad or overly general "
            "term like 'deep learning', 'taxonomy', or 'surveys'. Instead, focus on keyword that are unique "
            "and directly pertinent to the paper's subject.Answer with the word only in the"
            "following format: xxx")

    messages = [SystemMessage(content=sys_content)]

    extra_abs_content = '''
    Given Title: A Survey of Self-Supervised and Few-Shot Object Detection
    Given Abstract: Labeling data is often expensive and time-consuming, especially for tasks such as object detection and instance segmentation, which require dense labeling of the image. While few-shot object detection is about training a model on novel(unseen) object classeswith little data, it still requires prior training onmany labeled examples of base(seen) classes. On the other hand, self-supervisedmethods aimat learning representations fromunlabeled data which transfer well to downstream tasks such as object detection. Combining few-shot and self-supervised object detection is a promising research direction. In this survey, we reviewand characterize themost recent approaches on few-shot and self-supervised object detection. Then, we give our main takeaways and discuss future research directions. Project page: https://gabrielhuang.github.io/fsod-survey/''' if abstract else ''
    if extra_prompt:
        messages += [HumanMessage(content=f'''{usr_prompt}\n\n{extra_abs_content}'''), AIMessage(content='few-shot objection detection')]

    content = f'''{usr_prompt}
                Given Title: {title}
            '''
    if abstract:
        content += f'Given Abstract: {abstract}'
    messages.append(HumanMessage(content=content))

    chat = ChatOpenAI(model="gpt-3.5-turbo")


    return chat.batch([messages])[0].content

@retry(delay=6, )
def extract_keywords_from_article_with_gpt(text):
    messages = [
        SystemMessage(
            content="You are a profound researcher in the field of pattern recognition and machine intelligence. You "
                    "are aware of all types of keywords, such as keyword, index terms, etc.Please note: The text is "
                    "extracted from the PDF, so line breaks may appear anywhere, or even footnotes may appear between "
                    "consecutive lines of text."),
        HumanMessage(content= f'''I will give you the text in the first page of an academic paper, you should read it carefully. If there is no provided keywords, ask with None. If there does exist author provided keywords, answer with the extracted keywords (only keywords) in the following format: xxx,xxx,...,xxx. You should answer only with the keyword, do not answer with words like 'index terms'
         The text of the first page: Cryoelectron Microscopy as a Functional Instrument for Systems Biology, Structural Analysis &
        Experimental Manipulations with Living Cells
        (A comprehensive review of the current works).
        Oleg V. Gradov
        INEPCP RAS, Moscow, Russia
        Email: o.v.gradov@gmail.com
        Margaret A. Gradova
        ICP RAS, Moscow, Russia
        Email: m.a.gradova@gmail.com
        Abstract — The aim of this paper is to give an introductory
        review of the cryoelectron microscopy as a complex data source
        for the most of the system biology branches, including the most
        perspective non-local approaches known as "localomics" and
        "dynamomics". A brief summary of various cryoelectron microscopy methods and corresponding system biological approaches is given in the text. The above classification can be
        considered as a useful framework for the primary comprehensions about cryoelectron microscopy aims and instrumental
        tools
        Index Terms — cryo-electron microscopy, cryo-electron tomography, system biology, localomics, dynamomics, micromachining, structural analysis, in silico, molecular machines
        I. TECHNICAL APPLICATIONS OF
        CRYOELECTRON MICROSCOPY
        Since its development in early 1980s [31]
        cryo-electron microscopy has become one of
        the most functional research methods providing
        the study of physiological and biochemical
        changes in living matter at various hierarchical
        levels from single mammalian cell morphology
        [108] to nanostructures
    '''),
        AIMessage(content=f'''cryo-electron microscopy, cryo-electron tomography, system biology, localomics, dynamomics, micromachining, structural analysis, in silico, molecular machines'''),
        HumanMessage(content=f'''I will give you the text in the first page of another academic paper, you should read it carefully. If there is no provided keywords, ask with None. If there does exist author provided keywords, answer with the extracted keywords (only keywords) in the following format: xxx,xxx,...,xxx. You should answer only with the keyword, do not answer with words like 'index terms'
         The text of the first page:{text}''')
    ]
    chat = ChatOpenAI(model="gpt-3.5-turbo")


    return chat.batch([messages])[0].content

@retry()
def get_chatgpt_fields(title, abstract, extra_prompt=True,sys_content=None,usr_prompt=None):
    if not sys_content:
        sys_content = ("You are a profound researcher who is good at conduct a literature review based on given title and abstract.")
    if not usr_prompt:
        usr_prompt = ("Given title and abstract, please provide 5 seaching keywords for me so that I can use them as "
                      "keywords to search highly related papers from Google Scholar or Semantic Scholar. Please avoid "
                      "responding with overly general keywords such as deep learning, taxonomy, or surveys, etc., "
                      "and provide the output in descending order of relevance to the keywords. Answer with the words "
                      "only in the following format: xxx,xxx,xxx")

    if extra_prompt:
        messages = [SystemMessage(content=sys_content),HumanMessage(content=f'''{usr_prompt}\n Given Title: Diffusion Models in Vision: A Survey \nGiven Abstract: Denoising 
             diffusion models represent a recent emerging topic in computer vision, demonstrating remarkable results 
             in the area of generative modeling. A diffusion model is a deep generative model that is based on two 
             stages, a forward diffusion stage and a reverse diffusion stage. In the forward diffusion stage, 
             the input data is gradually perturbed over several steps by adding Gaussian noise. In the reverse stage, 
             a model is tasked at recovering the original input data by learning to gradually reverse the diffusion 
             process, step by step. Diffusion models are widely appreciated for the quality and diversity of the 
             generated samples, despite their known computational burdens, i.e., low speeds due to the high number of 
             steps involved during sampling. In this survey, we provide a comprehensive review of articles on 
             denoising diffusion models applied in vision, comprising both theoretical and practical contributions in 
             the field. First, we identify and present three generic diffusion modeling frameworks, which are based 
             on denoising diffusion probabilistic models, noise conditioned score networks, and stochastic 
             differential equations. We further discuss the relations between diffusion models and other deep 
             generative models, including variational auto-encoders, generative adversarial networks, energy-based 
             models, autoregressive models and normalizing flows. Then, we introduce a multi-perspective 
             categorization of diffusion models applied in computer vision. Finally, we illustrate the current 
             limitations of diffusion models and envision some interesting directions for future research.'''),
                    AIMessage(content='Denoising diffusion models,deep generative modeling,diffusion models,image generation,noise conditioned score networks'),
                    HumanMessage(content=f'''{usr_prompt}\n
                            Given Title: {title}\n
                            Given Abstract: {abstract}
                        ''')]
    else:
        messages = [SystemMessage(content=sys_content),HumanMessage(content=f'''{usr_prompt}\n
                Given Title: {title}\n
                Given Abstract: {abstract}
            ''')]
    chat = ChatOpenAI(model="gpt-3.5-turbo")


    return chat.batch([messages])[0].content

from pydantic import BaseModel, Field
from typing import List, Tuple
from typing import Dict

import langchain

from langchain.output_parsers.enum import EnumOutputParser

from enum import Enum

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
class ReviewTypes(Enum):
    MAPPING_REVIEW = "Mapping Review"
    META_ANALYSIS = "Meta-Analysis"
    CRITICAL_REVIEW = "Critical Review"
    SCOPING_REVIEW = "Scoping Review"
    SOTA_REVIEW = "SOTA Review"
    SYSTEMATIC_REVIEW = "Systematic Review"
    UMBRELLA_REVIEW = "Umbrella Review"
    NARRATIVE_REVIEW = "Narrative Review"
lt_parser = EnumOutputParser(enum=ReviewTypes)
from langchain_text_splitters import CharacterTextSplitter

from pydantic import BaseModel, Field

class ReviewFeatureParser(BaseModel):
    PROPOSE_TAXONOMY: bool = Field(
        default=False,
        description="True if the author has introduced a new taxonomy or classification system as a key contribution, false otherwise."
    )
    INCLUDE_EXCLUSION_CRITERIA: bool = Field(
        default=False,
        description="True if the article includes a clear section detailing the literature's inclusion and exclusion criteria, similar to PRISMA standards, false otherwise."
    )
    INCLUDE_PRELIMINARY: bool = Field(
        default=False,
        description="True if there is a separate chapter or section explaining foundational research and background information, not including the introduction, false otherwise."
    )
    INCLUDE_BENCHMARK: bool = Field(
        default=False,
        description="True if the review benchmarks the methods using specific metrics and includes quantitative performance comparisons, false otherwise."
    )
    INCLUDE_APPLICATION: bool = Field(
        default=False,
        description="True if there is a dedicated section on the applications of the techniques in real-world tasks or industrial settings, false otherwise."
    )
    INCLUDE_FUTURE: bool = Field(
        default=False,
        description="True if there is a distinct section discussing the future developments and current limitations of the field or methods, not just mentioned in the conclusion, false otherwise."
    )
class ReviewFeatureParser_FS1(BaseModel):#full sub 1
    CLS_METHODS: bool = Field(
        default=False,
        description="True if the author has introduced a new taxonomy or classification system as a key contribution, false otherwise."
    )
    SELECTION_CRITERIA: bool = Field(
        default=False,
        description="True if the article includes a clear section detailing the literature's inclusion and exclusion criteria, similar to PRISMA standards, false otherwise."
    )


class ReviewFeatureParser_FS2(BaseModel):
    BACKGROUND: bool = Field(
        default=False,
        description="True if there is a separate chapter or section explaining basic knowledge and background information, false otherwise."
    )
    DISSCUSSION: bool = Field(
        default=False,
        description="True if there is a distinct section discussing the future developments or current limitations of the field or methods, not just mentioned in the conclusion, false otherwise."
    )

    APPLICATION: bool = Field(
        default=False,
        description="True if there is a dedicated section on the applications of the techniques, false otherwise."#"True if there is a separate chapter or section explaining foundational research and background information, not including the introduction, false otherwise."
    )
    # INCLUDE_BENCHMARK: bool = Field(
    #     default=False,
    #     description="True if the review benchmarks the methods using specific metrics and includes quantitative performance comparisons, false otherwise."
    # )
class ReviewFeature():
    PROPOSE_TAXONOMY: bool = Field(
        description="Indicates whether the article introduces a novel taxonomy or typology for existing literature."
    )
    INCLUDE_EXCLUSION_CRITERIA: bool = Field(
        description="Indicates whether the article clearly outlines inclusion and exclusion criteria for the literature."
    )
    INCLUDE_PRELIMINARY: bool = Field(
        description="Indicates whether there is a section explaining preliminaries or background knowledge."
    )
    INCLUDE_BENCHMARK: bool = Field(
        description="Indicates whether the article includes a benchmark for existing methods."
    )
    INCLUDE_APPLICATION: bool = Field(
        description="Indicates whether there is a section exploring applications related to the field."
    )
    INCLUDE_FUTURE: bool = Field(
        description="Indicates whether there is a section discussing potential future developments in the field."
    )
class BenchRespParser(BaseModel):
    BENCHMARK: bool = Field(
        default=False,
        description="True if the review benchmarks the methods using specific metrics and includes quantitative performance comparisons, false otherwise."
    )


rfp = PydanticOutputParser(pydantic_object= ReviewFeatureParser)

rfp_full_sub_1 = PydanticOutputParser(pydantic_object= ReviewFeatureParser_FS1)
rfp_full_sub_2 = PydanticOutputParser(pydantic_object= ReviewFeatureParser_FS2)
brp = PydanticOutputParser(pydantic_object= BenchRespParser)
def get_content_by_section_id(target_id:int,dict_lst)->str:
    'Useful when you need to refer to the text of a certain chapter with specific ID.'
    for item in dict_lst:
        if item['ID'] == target_id:
            return f'{item.get("content")}'
def get_review_feature_full(pth, default_model="gpt-4o-mini", temperature=0):
    with open(pth, 'r', encoding='utf-8') as md_file:
        markdown_text = md_file.read()
        content_dict_list = get_structure_md(pth)
        toc = get_toc_from_md(content_dict_list, remove_abs_ref=False)
        fig_tab = ' || '.join(naive_figtab_retriver(markdown_text))
        intro = get_content_by_section_id(2,content_dict_list)
        title, abs = extract_title_and_abstract(pth)
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=200000, chunk_overlap=0)
        markdown_text = text_splitter.split_text(markdown_text)[0]
        instructions = rfp.get_format_instructions()
        prompt_template = """
        Please carefully review the provided literature survey and answer the following questions:

        1. **Taxonomy:** Examine whether the author proposes a novel taxonomy, classification, category, or typology for the surveyed domain. Focus on checking the abstract, introduction, and early sections to see if the author introduces new classification schemes or simply adopts existing ones. Pay attention to terms like 'taxonomy', 'classification', 'category', or 'typology'. 
        
        2. **Inclusion and Exclusion Criteria:** Identify if there is a clearly defined section that outlines the criteria for selecting or excluding literature, similar to standards like PRISMA. Review the table of contents and introduction for explicit mentions of 'Inclusion/Exclusion Criteria', 'Search Strategy', or related terms.
        
        3. **Preliminary Research:** Determine if the review includes a dedicated section for explaining preliminary research, task formulation, paradigms, definitions, or background information crucial for understanding the domain. These sections should be clearly separated from the introduction and may be labeled as 'Preliminaries', 'Task Formulation', 'Background', or similar headings.
        
        4. **Benchmark:** Check if the author has performed benchmarking on the reviewed methods using specific metrics, providing quantitative performance comparisons between different techniques. Look for performance tables or sections that discuss metric-based comparisons of the methods.
        
        5. **Applications:** Verify whether there is a distinct section focused on practical applications of the reviewed techniques in real-world tasks or industrial settings. This section should be clearly labeled, often appearing under headings such as 'Applications'.
        
        6. **Future Developments and Limitations:** Investigate whether the paper contains a dedicated chapter or section discussing future trends, emerging developments, and current limitations of the field. This discussion should be more extensive than a brief mention in the conclusion and provide detailed insights into the potential evolution of the domain.

        **Survey Content**: {content}

        **Format Instructions:** {instructions}
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)


        model = ChatOpenAI(model_name=default_model,temperature=temperature)

        chain = prompt | model | rfp
        # print(instructions)
        rst = chain.invoke({"abs": abs,"title": title,"toc": toc, "content":markdown_text, "intro":intro,"fig_tab":fig_tab,"instructions": instructions})
        return rst

def get_taxonomy_criteria(pth, default_model="gpt-4o-mini", temperature=0):
    with open(pth, 'r', encoding='utf-8') as md_file:
        markdown_text = md_file.read()
        content_dict_list = get_structure_md(pth)
        toc = get_toc_from_md(content_dict_list, remove_abs_ref=False)

        intro = None
        for item in content_dict_list:
            if (item['ID'] == 0 or item['ID'] == 1 or item['ID'] == 2 or item['ID'] == 3 ) and 'intro' in item.get('title').lower():
                intro = f'{item.get("content")}'
                break
        if intro is None:
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=4000, chunk_overlap=0)
            intro = text_splitter.split_text(markdown_text)[0]
            print('ERROR. USE first 4k token instead')
        # intro = get_content_by_section_id(2, content_dict_list)
        title, abs = extract_title_and_abstract(pth)
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=200000, chunk_overlap=0)

        instructions = rfp_full_sub_1.get_format_instructions()
        prompt_template = """
        Please carefully review the provided literature survey and answer the following questions:

        1. **Classify Existing Methods:** Check if the authors explicitly state in the abstract or introduction that proposing a new classification of methods is their contribution. Look for phrases like "propose a taxonomy" "categorize/classify methods into". A simple classification is not enough; the authors must clearly state that the classification itself is a key contribution of their work.

        2. **Literature Selection Criteria:** What criteria were used to select the literature in this survey? Identify if there is a specific section that describes the inclusion and exclusion criteria for the literature, similar to PRISMA standards.  Look for phrases like "inclusion criteria," "exclusion criteria," "selection process," "literature filtering," "eligibility criteria," "search strategy," or "screening process."

        **Title**: {title}
        **Abstract**: {abs}
        **Introduction**: {intro}
        **Table of Content**: {toc}

        **Format Instructions:** {instructions}
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)

        model = ChatOpenAI(model_name=default_model, temperature=temperature)

        chain = prompt | model | rfp_full_sub_1
        # print(instructions)
        rst = chain.invoke(
            {"abs": abs, "title": title, "toc": toc, "intro": intro,
             "instructions": instructions})
        return rst

def get_background_future_app(pth, default_model="gpt-4o-mini", temperature=0):
    with open(pth, 'r', encoding='utf-8') as md_file:
        markdown_text = md_file.read()
        content_dict_list = get_structure_md(pth)
        toc = get_toc_from_md(content_dict_list, remove_abs_ref=False)
        title, abs = extract_title_and_abstract(pth)
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=200000, chunk_overlap=0)
        instructions = rfp_full_sub_2.get_format_instructions()
        prompt_template = """
        Please carefully review the provided literature survey and answer the following questions:

        1. **Background Knowledge:** Check the Table of Contents to see if there is a section with any of the following keywords in the title: Preliminary, Background, Formulation, Paradigms, Definition, or Basic. This section should appear right after the introduction and is likely intended to provide background knowledge for readers.

        2. **Future, Challenge, Limitation Discussion:** Discuss the future directions, challenges, and limitations outlined by the authors. Verify if there is a separate section that addresses these aspects, using keywords like "future," "challenge," "limitation," or "discussion." Ensure this discussion goes beyond the Conclusion section.
        
        3. **Application:** Discuss potential applications or contexts for the methods reviewed. Verify if there is a dedicated section containing "Application" in the section title.
        
        **Table of Content**: {toc}
        
        **Format Instructions:** {instructions}
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)

        model = ChatOpenAI(model_name=default_model, temperature=temperature)

        chain = prompt | model | rfp_full_sub_2
        # print(instructions)
        rst = chain.invoke(
            {"abs": abs, "title": title, "toc": toc,"instructions": instructions})
        return rst


def check_abs_structure(abs, default_model="gpt-4o-mini", temperature=0):


    prompt_template = """
    You are an academic assistant. 
    I will give you the text of an abstract. 
    Your task is to decide whether it is a **structured abstract** (with explicit section labels such as "Background", "Objective", "Methods", "Results", "Conclusion", etc.) or just a normal narrative abstract.
    
    Rules:
    - If the abstract clearly has section headings or labeled segments (even partial ones like "Background:", "Aim:", "Methods:"), return 1.
    - If it is a single block of text without labeled subsections, return 0.
    - Do not check for full compliance with PRISMA; just judge based on visible structured segmentation.
    - Return only a single digit: 0 or 1.
    
    Abstract:
    {abs}
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    model = ChatOpenAI(model_name=default_model, temperature=temperature)

    chain = prompt | model | StrOutputParser()
    # print(instructions)
    result = chain.invoke({"abs": abs}).strip()
    if result not in ["0", "1"]:
        result = "0"  # fallback
    return int(result)


from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

def check_middle_typology(
    section_titles: List[str],
    default_model: str = "gpt-4o-mini",
    temperature: float = 0.0
) -> str:
    """
    依据“章节/小节标题”判断综述主体组织类型：
    - cluster   : 按方法家族/架构/组件分组（CNN/Transformer、template-based、voting-based、encoder-decoder、SVM…）
    - challenge : 按任务/问题/目标分组（localization、detection、segmentation、tracking、pose estimation、grasp、re-id、retrieval…）
    - hybrid    : 两种模式在主体中并行且都很明显（如按任务分章、章内再按方法家族细分）
    - other     : 不属于以上（纯时间线、纯应用域、强叙述无明显结构）
    """
    titles_str = "\n".join(f"- {t}" for t in section_titles)

    prompt_template = """
You are an academic assistant. Classify the ORGANIZATION TYPE of a survey's middle part using ONLY its section/subsection TITLES.

Labels:
- "cluster"   : grouped by method families/architectures/components (e.g., CNN/Transformer, encoder-decoder, attention-based, template-based, correspondence-based, voting-based, SVM-based, graph-based, point-cloud-based).
- "challenge" : grouped by tasks/problems/goals (e.g., localization, detection, segmentation, tracking, identification, pose estimation, interpretation, recognition, re-identification, retrieval, reconstruction, grasping, activity recognition, domain adaptation).
- "hybrid"    : both patterns are clearly present in the core review (e.g., sections by tasks and, within each task, subsections by method families; or separate sections mixing both extensively).

Guidelines:
- Treat task/goal nouns (localization/detection/segmentation/tracking/…/pose estimation/interpretation/recognition) as STRONG "challenge" signals.
- Treat method-family/architecture nouns (template-based/voting-based/correspondence-based/CNN/Transformer/encoder-decoder/attention/U-Net/SVM/graph-based) as STRONG "cluster" signals.
- Ignore utility sections: datasets, metrics, applications, related work, discussion, conclusion, future directions, acknowledgments, references.
- If top-level sections are tasks and most subsections are also tasks → "challenge".
- If both task sections and method-family subsections are widespread → "hybrid".
Return EXACTLY one word from: cluster, challenge, hybrid.

Section Titles:
{titles}
""".strip()

    prompt = ChatPromptTemplate.from_template(prompt_template)
    model = ChatOpenAI(model_name=default_model, temperature=temperature)
    chain = prompt | model | StrOutputParser()
    result = chain.invoke({"titles": titles_str}).strip().lower()

    # ---- Heuristic fallback if LLM misbehaves ----
    if result not in {"cluster", "challenge", "hybrid"}:
            result = "FAIL"

    return result




# print(check_abs_structure('''Context In a previous study, we reported on a systematic literature review (SLR), based on a manual search of 13 journals and conferences undertaken in the period 1st January 2004 to 30th June 2007. Objective The aim of this on-going research is to provide an annotated catalogue of SLRs available to software engineering researchers and practitioners. This study updates our previous study using a broad automated search. Method We performed a broad automated search to find SLRs published in the time period 1st January 2004 to 30th June 2008. We contrast the number, quality and source of these SLRs with SLRs found in the original study. Results Our broad search found an additional 35 SLRs corresponding to 33 unique studies. Of these papers, 17 appeared relevant to the undergraduate educational curriculum and 12 appeared of possible interest to practitioners. The number of SLRs being published is increasing. The quality of papers in conferences and workshops has improved as more researchers use SLR guidelines. Conclusion SLRs appear to have gone past the stage of being used solely by innovators but cannot yet be considered a main stream software engineering research methodology. They are addressing a wide range of topics but still have limitations, such as often failing to assess primary study quality.'''))
def get_benchmark(pth, default_model="gpt-4o-mini", temperature=0):
    with open(pth, 'r', encoding='utf-8') as md_file:
        fig_tab_resp = fig_tab_extractor(pth)
        figs = fig_tab_resp.FIGURE_CAPTION
        tabs = fig_tab_resp.TABLE_CAPTION
        instructions = brp.get_format_instructions()
        prompt_template = """
        Please carefully review the provided literature survey and answer the following questions:

        **Benchmark:** Ensure that the author has conducted benchmarking on the reviewed methods using quantifiable performance metrics. This requires precise numerical comparisons between different techniques. Pay attention to whether the metrics referenced in the table titles suggest quantitative performance data (e.g., accuracy, latency, precision, recall, or throughput). Simple comparisons without numerical values do not qualify as benchmarks.

        **Figure Caption**: {figs}
        **Table Caption**: {tabs}

        **Format Instructions:** {instructions}
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)

        model = ChatOpenAI(model_name=default_model, temperature=temperature)

        chain = prompt | model | brp
        # print(instructions)
        rst = chain.invoke(
            {"figs":figs,'tabs':tabs, "instructions": instructions})
        return rst


def get_review_feature_full_sub1(pth, default_model="gpt-4o-mini", temperature=0):
    with open(pth, 'r', encoding='utf-8') as md_file:
        markdown_text = md_file.read()
        content_dict_list = get_structure_md(pth)
        toc = get_toc_from_md(content_dict_list, remove_abs_ref=False)
        fig_tab = ' || '.join(naive_figtab_retriver(markdown_text))
        intro = get_content_by_section_id(2, content_dict_list)
        title, abs = extract_title_and_abstract(pth)
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=200000, chunk_overlap=0)
        markdown_text = text_splitter.split_text(markdown_text)[0]
        instructions = rfp_full_sub_1.get_format_instructions()
        prompt_template = """
        Please carefully review the provided literature survey and answer the following questions:

        1. **Taxonomy:** Investigate whether the author has divided the existing methods into specific categories. Determine if this classification is a novel taxonomy proposed by the author or if it merely uses an existing classification scheme. Pay attention to terms like 'taxonomy', 'category', 'classification', and ensure the author explicitly states that the introduction of the classification is a contribution.

        2. **Inclusion and Exclusion Criteria:** Identify if there is a clearly defined section that outlines the criteria for selecting or excluding literature, similar to standards like PRISMA. Review the table of contents and introduction for explicit mentions of 'Inclusion/Exclusion Criteria', 'Search Strategy', or related terms.

        3. **Preliminary Research:** Determine if the review includes a dedicated section for explaining preliminary research, task formulation, paradigms, definitions, or background information crucial for understanding the domain. These sections should be clearly separated from the introduction and may be labeled as 'Preliminaries', 'Task Formulation', 'Background', or similar headings.

        **Survey Content**: {content}

        **Format Instructions:** {instructions}
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)

        model = ChatOpenAI(model_name=default_model, temperature=temperature)

        chain = prompt | model | rfp_full_sub_1
        # print(instructions)
        rst = chain.invoke(
            {"abs": abs, "title": title, "toc": toc,  "intro": intro,
             "instructions": instructions})
        return rst


def get_review_feature_full_sub2(pth, default_model="gpt-4o-mini", temperature=0):
    with open(pth, 'r', encoding='utf-8') as md_file:
        markdown_text = md_file.read()
        content_dict_list = get_structure_md(pth)
        toc = get_toc_from_md(content_dict_list, remove_abs_ref=False)
        fig_tab = ' || '.join(naive_figtab_retriver(markdown_text))
        intro = get_content_by_section_id(2, content_dict_list)
        title, abs = extract_title_and_abstract(pth)
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=200000, chunk_overlap=0)
        markdown_text = text_splitter.split_text(markdown_text)[0]
        instructions = rfp_full_sub_2.get_format_instructions()
        prompt_template = """
        Please carefully review the provided literature survey and answer the following questions:
        
        1. **Benchmark:** Check if the author has performed benchmarking on the reviewed methods using specific metrics, providing quantitative performance comparisons between different techniques. Look for performance tables or sections that discuss metric-based comparisons of the methods.

        2. **Applications:** Verify if there is a distinct section or chapter specifically discussing the application of the reviewed techniques in real-world tasks or industrial settings. This section should be clearly marked, typically with titles like 'Applications' or similar. Focus on the section titles to ensure there is a dedicated part addressing how these methods are applied in practice.

        3. **Future Developments and Limitations:** Investigate whether the paper contains a dedicated chapter or section discussing future trends, emerging developments, and current limitations of the field. This discussion should be more extensive than a brief mention in the conclusion and provide detailed insights into the potential evolution of the domain.

        **Survey Content**: {content}

        **Format Instructions:** {instructions}
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)

        model = ChatOpenAI(model_name=default_model, temperature=temperature)

        chain = prompt | model | rfp_full_sub_2
        # print(instructions)
        rst = chain.invoke(
            {"abs": abs, "title": title, "toc": toc, "content": markdown_text, "intro": intro, "fig_tab": fig_tab,
             "instructions": instructions})
        return rst




# get_review_feature_sub3_agent(r'J:\md\output\1005.4270v1.Clustering_Time_Series_Data_Stream___A_Literature_Survey.mmd')
def naive_figtab_retriver(markdown_text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=0,
            separators=[',', "\n"]
        )
    # text_splitter = SemanticChunker(OpenAIEmbeddings())

        texts = text_splitter.create_documents([markdown_text])
    except Exception as e:
        return []

    def filter_list(input_list):
        substrings = ['Tab.', 'Table', 'Figure', 'Fig.',]
        output_set = set()
        output_list = []

        for item in input_list:
            if any(substring in item.page_content for substring in substrings):
                if item.page_content not in output_set:
                    output_list.append(item.page_content)
                    output_set.add(item.page_content)
        return output_list
    return filter_list(texts)



import re
def filter_list(input_list,pattern = r'\b(Fig\.?|Figure|FIG|FIGURE)\b'): # r'\b(Table|TAB|TABLE|Tab\.)\b'
    # 
    # pattern = re.compile(r'www\.|http:|https:|\.com|github|sway')
    pattern = re.compile(pattern, re.IGNORECASE)
    output_set = set()
    output_list = []

    for item in input_list:
        # 
        if pattern.search(item.page_content):
            # 
            if item.page_content not in output_set:
                output_list.append(item)
                output_set.add(item.page_content)
    return output_list
def naive_figtab_checker(pdf_pth):
    try:
        loader = PDFMinerLoader(pdf_pth)
        data = loader.load()
        raw_content = ''.join([d.page_content for d in data])
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=400,
            chunk_overlap=0,
            separators=[',', '.',"\n\n", ' ']
        )
        # text_splitter = SemanticChunker(OpenAIEmbeddings())
        texts = text_splitter.create_documents([raw_content])
    except Exception as e:
        return []
    rst = {'fig':filter_list(texts, r'\b(Fig\.?|Figure|FIG|FIGURE)\b'), 'tab': filter_list(texts,r'\b(Table|TAB|TABLE|Tab\.)\b')}
    return rst
class Fig_Response(BaseModel):
    FIGURE_CAPTION: List[str] = Field(
        description="Regardless of any formatting issues, list the captions of all figures in the order they actually appear in the document. Format your answer like ['Fig. 1: CAPTION', 'Fig. 2: CAPTION', ...].")
    TABLE_CAPTION: List[str] = Field(
        description="Regardless of any formatting issues, list the captions of all tables in the order they actually appear in the document.  Format your answer like ['Tab. 1: CAPTION', 'Tab. 2: CAPTION', ...]. ")

fig_parser = PydanticOutputParser(pydantic_object=Fig_Response)
# @retry()
def fig_tab_extractor(pdf_pth):
    try:


        top_ss_chunks = naive_figtab_checker(pdf_pth)
        figs = top_ss_chunks['fig']
        tabs = top_ss_chunks['tab']
        top_fig_chunks = [i.page_content for i in figs]
        top_tab_chunks = [i.page_content for i in tabs]
        if len(top_ss_chunks) > 0:
            prompt_template = """Below are sentences extracted from an academic paper PDF file that contain mentions of figures and tables. Please parse the content and use it to extract all the figure or table captions present in the paper. Focus specifically on any text chunks containing the terms 'fig', 'figure', or 'table', 'tab'. 

            Text Chunks (Figure): {fig_chunks}
            Text Chunks (Table): {tab_chunks}
            Question: Based on the provided sentences, extract all figure or table captions from the text. Ensure only the captions belonging to figures, charts, or tables from the current paper are included.


            Format Instruction:
            {format_instructions}
            """

            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=['fig_chunks','tab_chunks'],
                partial_variables={"format_instructions": fig_parser.get_format_instructions()},
            )
            chain = LLMChain(llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
                             prompt=PROMPT)

            rst = chain.run(fig_chunks=' || '.join(top_fig_chunks),tab_chunks=' || '.join(top_tab_chunks))

            try:
                p_rst = fig_parser.parse(rst)
            except OutputParserException as e:
                print(f'OA Checker Parser Error: {e}. || {top_ss_chunks} || {rst}')
                return False
            if len(p_rst.FIGURE_CAPTION + p_rst.TABLE_CAPTION) > 0:
                return p_rst
            else:
                return None
        else:
            return None
    except Exception as e:

        raise Exception('Error occurs in github_checker:', e)
        return None


from typing import List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class TOC(BaseModel):
    top_level_titles: List[str] = Field(
        description="Extracted top-level (first-level) section titles exactly as they appear"
    )


def extract_top_level_titles(
    section_titles: List[str],
    default_model: str = "gpt-4o-mini",
    temperature: float = 0.0
) -> List[str]:
    """
    输入：所有章节/小节标题 (扁平list)
    输出：仅一级标题（list[str]），一字不拉
    """
    titles_str = "\n".join(f"- {t}" for t in section_titles)

    parser = PydanticOutputParser(pydantic_object=TOC)

    prompt_template = """
You are an academic assistant. 
Your task: From the following list of ALL section/subsection titles, 
IDENTIFY ONLY the top-level (first-level) section titles.

Requirements:
- Output EXACTLY in the required JSON format.
- Do not add explanations.
- Copy the titles EXACTLY as they appear, character by character.
- Ignore abstract, acknowledgments, references, appendices.
Format Instruction:
{format_instructions}

All Titles:
{titles}
""".strip()

    prompt = ChatPromptTemplate.from_template(
        prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    model = ChatOpenAI(model_name=default_model, temperature=temperature)

    chain = prompt | model | parser
    result: TOC = chain.invoke({"titles": titles_str})

    return result.top_level_titles