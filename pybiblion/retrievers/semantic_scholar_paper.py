import sys
import os
import re
import math
import logging
import statistics
from collections import OrderedDict
from datetime import datetime, timedelta
from threading import Lock

import requests
from retry import retry

from cfg.config import s2api
from tools.gpt_util import get_chatgpt_field, get_chatgpt_fields
from retrievers.Author import Author
from retrievers.Publication import Document

from retrievers.metric_util import get_TNCSI, get_IEI, get_RQM, get_RUI

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

logger = logging.getLogger(__name__)

# 你原来代码里依赖的 disk_cache：我假设它在别处初始化
# 如果你在本文件里就是用全局 dict，也能兼容：
try:
    disk_cache
except NameError:
    disk_cache = {}


class S2paper(Document):
    def __init__(self, ref_obj, ref_type='title', filled_authors=True, force_return=False, use_cache=True, **kwargs):
        """
        :param ref_obj: search keywords OR entity dict when ref_type='entity'
        :param ref_type: 'title' or 'entity'
        :param filled_authors: retrieve detailed info about authors?
        :param force_return: even title is not mapping, still return the result
        """
        super().__init__(ref_obj, **kwargs)

        self.ref_obj = ref_obj
        self.ref_type = ref_type

        # Expectation: A typical program is unlikely to create more than 5 of these.
        self.S2_PAPER_URL = "https://api.semanticscholar.org/v1/paper/"
        self.S2_QUERY_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.CACHE_FILE = ".ppicache"
        self.DEFAULT_TIMEOUT = 3.05  # 100 requests per 5 minutes

        self._entity = None  # dict | False | None  (工程约束：绝不能是 S2paper)
        self._entity_loaded = False  # ⭐ 新增：避免重复请求（工程优化，不改变行为）

        # a few settings
        self.filled_authors = filled_authors
        self.force_return = force_return
        self.use_cache = use_cache

        # gpt
        self._gpt_keyword = None
        self._gpt_keywords = None
        self._gpt_lock = Lock()  # ⭐ 新增：线程安全（工程优化）

        # metrics cache
        self._TNCSI = None
        self._TNCSI_S = None
        self._IEI = None
        self._RQM = None
        self._RUI = None

        # related cache
        self._authors_cache = None
        self._references_cache = None
        self._citations_cache = None

    def __eq__(self, other):
        if isinstance(other, S2paper):
            return self.s2id == other.s2id
        return False

    def __hash__(self):
        return hash(self.s2id)

    # -----------------------------
    # Entity (工程加固：锁死类型不变量)
    # -----------------------------
    @property
    @retry()
    def entity(self):
        """
        保持你原始逻辑，但做工程加固：
        - ref_type == 'entity' 时，只允许 dict / False / None
        - 增加 _entity_loaded，避免重复请求
        """
        if self._entity_loaded:
            return self._entity

        if self.ref_type == 'entity':
            # ⭐ 关键修复：只允许 dict/False/None，禁止 S2paper 对象污染
            if self.ref_obj is None or self.ref_obj is False:
                self._entity = self.ref_obj
            elif isinstance(self.ref_obj, dict):
                self._entity = self.ref_obj
            else:
                raise TypeError(
                    f"S2paper(ref_type='entity') expects dict/False/None, got {type(self.ref_obj)}"
                )
            self._entity_loaded = True
            return self._entity

        # ref_type == 'title'
        if self._entity is None:
            url = (
                f"{self.S2_QUERY_URL}?query={self.ref_obj}"
                f"&fieldsOfStudy=Computer Science"
                f"&fields=url,title,abstract,authors,venue,externalIds,referenceCount,tldr,openAccessPdf,"
                f"citationCount,influentialCitationCount,influentialCitationCount,fieldsOfStudy,s2FieldsOfStudy,"
                f"publicationTypes,publicationDate,publicationVenue"
                f"&offset=0&limit=1"
            )

            if url in disk_cache and self.use_cache:
                response = disk_cache[url]
            else:
                session = requests.Session()
                headers = {'x-api-key': s2api} if s2api is not None else None
                reply = session.get(url, headers=headers)
                response = reply.json()
                disk_cache[url] = response

            if "data" not in response:
                self._entity = False
                self._entity_loaded = True
                return self._entity

            # 走到这里，data 存在
            ent = response['data'][0]
            # title 严格对齐检查（保留你原逻辑）
            if (
                self.ref_type == 'title'
                and re.sub(r'\W+', '', ent['title'].lower()) != re.sub(r'\W+', '', str(self.ref_obj).lower())
            ):
                if self.force_return:
                    self._entity = ent
                else:
                    print(ent['title'].lower())
                    self._entity = False
            else:
                self._entity = ent

            self._entity_loaded = True
            return self._entity

        # _entity 已经不是 None（可能是 False 或 dict）
        self._entity_loaded = True
        return self._entity

    # -----------------------------
    # Basic fields (加一点 dict 防御，不改行为)
    # -----------------------------
    @property
    def title(self):
        if self.ref_type == 'title':
            return str(self.ref_obj).lower()
        ent = self.entity
        if isinstance(ent, dict):
            t = ent.get('title')
            return t.lower() if isinstance(t, str) else None
        return None

    @property
    def publication_date(self):
        """The date of publication."""
        ent = self.entity
        if isinstance(ent, dict):
            if ent.get('publicationDate') is not None:
                try:
                    return datetime.strptime(ent['publicationDate'], "%Y-%m-%d")
                except Exception:
                    return None
        return None

    @property
    def s2id(self):
        """The DocumentIdentifier of this document."""
        ent = self.entity
        return ent.get('paperId') if isinstance(ent, dict) else None

    @property
    def tldr(self):
        ent = self.entity
        if isinstance(ent, dict) and ent.get('tldr') is not None:
            try:
                return ent['tldr']['text']
            except Exception:
                return None
        return None

    @property
    def DOI(self):
        ent = self.entity
        if isinstance(ent, dict):
            return ent.get('DOI')
        return None

    # -----------------------------
    # Authors (保留原逻辑 + 轻微缓存)
    # -----------------------------
    @property
    @retry()
    def authors(self):
        """The authors of this document."""
        if self._authors_cache is not None:
            return self._authors_cache

        ent = self.entity
        if not ent or not isinstance(ent, dict):
            return None

        authors = []
        if 'authors' in ent:
            if not self.filled_authors:
                for item in ent['authors']:
                    author = Author(item['name'], _s2_id=item.get('authorId'))
                    authors.append(author)
                self._authors_cache = authors
                return authors
            else:
                url = (
                    f'https://api.semanticscholar.org/graph/v1/paper/{self.s2id}/authors'
                    f'?fields=authorId,externalIds,name,affiliations,homepage,paperCount,'
                    f'citationCount,hIndex,url'
                )
                if url in disk_cache and self.use_cache:
                    response = disk_cache[url]
                else:
                    headers = {'x-api-key': s2api} if s2api else None
                    reply = requests.get(url, headers=headers)
                    response = reply.json()
                    disk_cache[url] = response

                for item in response.get('data', []):
                    author = Author(
                        item.get('name'),
                        _s2_id=item.get('authorId'),
                        _s2_url=item.get('url'),
                        _h_index=item.get('hIndex'),
                        _citationCount=item.get('citationCount'),
                        _paperCount=item.get('paperCount'),
                    )
                    authors.append(author)

                self._authors_cache = authors
                return authors

        return None

    @property
    def affiliations(self):
        if self.authors:
            affiliations = []
            for author in self.authors:
                if author.affiliations is not None:
                    affiliations.append(author.affiliations)
            return ';'.join(list(set(affiliations)))
        return None

    @property
    def publisher(self):
        ent = self.entity
        if isinstance(ent, dict):
            return ent.get('publicationVenue')
        return None

    @property
    def publication_source(self):
        ent = self.entity
        if isinstance(ent, dict):
            return ent.get('venue')
        return None

    @property
    def source_type(self):
        ent = self.entity
        if isinstance(ent, dict):
            return ent.get('publicationTypes')
        return None

    @property
    def abstract(self):
        ent = self.entity
        if isinstance(ent, dict):
            return ent.get('abstract')
        return None

    @property
    def pub_url(self):
        ent = self.entity
        if isinstance(ent, dict):
            return ent.get('openAccessPdf')
        return None

    @property
    def citation_count(self):
        ent = self.entity
        if isinstance(ent, dict):
            return ent.get('citationCount')
        return None

    @property
    def reference_count(self):
        ent = self.entity
        if isinstance(ent, dict):
            return ent.get('referenceCount')
        return None

    @property
    def field(self):
        ent = self.entity
        if isinstance(ent, dict) and ent.get('s2FieldsOfStudy') is not None:
            fields = []
            for fdict in ent.get('s2FieldsOfStudy', []):
                category = fdict.get('category')
                if category:
                    fields.append(category)
            fields = ','.join(list(set(fields)))
            return fields
        return None

    @property
    def influential_citation_count(self):
        ent = self.entity
        if isinstance(ent, dict):
            return ent.get('influentialCitationCount')
        return None

    # -----------------------------
    # References (关键：构造子 paper 时保证 entity 是 dict，不污染)
    # -----------------------------
    @property
    @retry()
    def references(self):
        if self._references_cache is not None:
            return self._references_cache

        ent = self.entity
        if not isinstance(ent, dict):
            return None

        references = []
        url = (
            f'https://api.semanticscholar.org/graph/v1/paper/{self.s2id}/references'
            f'?fields=authors,contexts,intents,isInfluential,venue,title,authors,citationCount,'
            f'influentialCitationCount,publicationDate,venue&limit=999'
        )

        if url in disk_cache and self.use_cache:
            response = disk_cache[url]
        else:
            session = requests.Session()
            headers = {'x-api-key': s2api} if s2api else None
            reply = session.get(url, headers=headers)
            response = reply.json()
            disk_cache[url] = response

        if 'data' not in response:
            self._references_cache = []
            return self._references_cache

        for item in response['data']:
            cited = item.get('citedPaper') or {}
            if not isinstance(cited, dict):
                continue

            info = {
                'paperId': cited.get('paperId'),
                'contexts': item.get('contexts'),
                'intents': item.get('intents'),
                'isInfluential': item.get('isInfluential'),
                'title': cited.get('title'),
                'venue': cited.get('venue'),
                'citationCount': cited.get('citationCount'),
                'influentialCitationCount': cited.get('influentialCitationCount'),
                'publicationDate': cited.get('publicationDate'),
                'authors': cited.get('authors'),
            }

            # ⭐ 工程修复：用 ref_type='entity' 直接传 dict，避免 title 构造 + 再塞 _entity 的混合污染
            ref = S2paper(info, ref_type='entity', filled_authors=False, force_return=True, use_cache=self.use_cache)
            references.append(ref)

        self._references_cache = references
        return references

    # -----------------------------
    # Citations detail (加硬上限防假死 + 保证 entity dict)
    # -----------------------------
    @property
    @retry()
    def citations_detail(self):
        if self._citations_cache is not None:
            return self._citations_cache

        ent = self.entity
        if not isinstance(ent, dict):
            return None

        results = []
        offset = 0
        session = requests.Session()
        headers = {'x-api-key': s2api} if s2api else None

        MAX_PAGES = 10      # <= 10 * 1000 = 1w citations
        MAX_TOTAL = 10_000  # 兜底

        pages = 0
        while True:
            if pages >= MAX_PAGES or len(results) >= MAX_TOTAL:
                break

            url = (
                f'https://api.semanticscholar.org/graph/v1/paper/{self.s2id}/citations'
                f'?fields=authors,contexts,intents,isInfluential,venue,title,authors,'
                f'citationCount,influentialCitationCount,publicationDate,venue'
                f'&limit=1000&offset={offset}'
            )
            offset += 1000
            pages += 1

            if url in disk_cache and self.use_cache:
                response = disk_cache[url]
            else:
                reply = session.get(url, headers=headers)
                response = reply.json()
                disk_cache[url] = response

            data = response.get('data', [])
            if not data:
                break

            for item in data:
                citing_paper = item.get('citingPaper') or {}
                if not isinstance(citing_paper, dict):
                    continue

                info = {
                    'paperId': citing_paper.get('paperId'),
                    'contexts': item.get('contexts'),
                    'intents': item.get('intents'),
                    'isInfluential': item.get('isInfluential'),
                    'title': citing_paper.get('title'),
                    'venue': citing_paper.get('venue'),
                    'citationCount': citing_paper.get('citationCount'),
                    'influentialCitationCount': citing_paper.get('influentialCitationCount'),
                    'publicationDate': citing_paper.get('publicationDate'),
                    'authors': citing_paper.get('authors'),
                }

                # ⭐ 同样用 entity dict 构造，避免污染
                ref = S2paper(info, ref_type='entity', filled_authors=True, force_return=True, use_cache=self.use_cache)
                results.append(ref)

                if len(results) >= MAX_TOTAL:
                    break

        self._citations_cache = results
        return results

    # -----------------------------
    # GPT keywords (加锁：工程优化)
    # -----------------------------
    @property
    def gpt_keyword(self):
        if self._gpt_keyword is None:
            with self._gpt_lock:
                if self._gpt_keyword is None:
                    self._gpt_keyword = get_chatgpt_field(self.title, self.abstract, extra_prompt=True)
        return self._gpt_keyword

    @property
    def gpt_keywords(self):
        if self._gpt_keywords is None:
            with self._gpt_lock:
                if self._gpt_keywords is None:
                    self._gpt_keywords = get_chatgpt_fields(self.title, self.abstract, extra_prompt=True)
        return self._gpt_keywords

    # -----------------------------
    # Metrics (保持你原来的 property + retry 行为不变)
    # -----------------------------
    @property
    @retry(tries=5)
    def TNCSI(self):
        if self._TNCSI is None:
            self._TNCSI = get_TNCSI(self, show_PDF=False)
        return self._TNCSI

    @property
    @retry(tries=5)
    def TNCSI_S(self):
        if self._TNCSI_S is None:
            kwd = self.gpt_keyword
            self._TNCSI_S = get_TNCSI(self, topic_keyword=kwd, show_PDF=False, same_year=True)
        return self._TNCSI_S

    @property
    @retry(tries=5)
    def IEI(self):
        if self.publication_date is not None and self.citation_count != 0:
            if self._IEI is None:
                self._IEI = get_IEI(self.title, normalized=False, exclude_last_n_month=1, show_img=False)
            return self._IEI
        rst = {'L6': float('-inf'), 'I6': float('-inf')}
        return rst

    @property
    @retry(tries=5)
    def RQM(self):
        if self.publication_date is not None and self.reference_count != 0:
            if self._RQM is None:
                self._RQM = get_RQM(self, tncsi_rst=self.TNCSI, beta=5)
            return self._RQM
        return {}

    @property
    @retry(tries=5)
    def RUI(self):
        if self.publication_date is not None and self.reference_count != 0:
            if self._RUI is None:
                self._RUI = get_RUI(self)
            return self._RUI
        return {}




s2paper = S2paper('A survey on segment anything model (sam): Vision foundation model meets prompt engineering')
print(s2paper.TNCSI)

# if s2paper.citation_count is not None:
#     print(f'Paper Title: {s2paper.title}, Topic Keyword: {s2paper.gpt_keyword}, TNCSI: {s2paper.TNCSI.get("TNCSI")}, IEI: {s2paper.IEI.get("L6")}, RQM: {s2paper.RQM.get("RQM")}, RUI: {s2paper.RUI.get("RUI")}')