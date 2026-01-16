# retrievers/metric_util.py
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import math
import statistics
from collections import OrderedDict
from datetime import datetime, timedelta
from functools import lru_cache
from urllib.parse import urlencode

from cfg.config import s2api
from CACHE.cache_request import cached_get

try:
    from scipy.integrate import cumtrapz
except ImportError:
    from scipy.integrate import cumulative_trapezoid as cumtrapz

from scipy import stats
from retry import retry
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb


S2_PAPER_URL = "https://api.semanticscholar.org/v1/paper/"
S2_QUERY_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

# ============================================================
# Engineering limits (DO NOT affect formulas)
# ============================================================

# 用于“topic 分布拟合”时最多抓多少篇（原来 topk=1000/2000 太容易卡）
_DEFAULT_DIST_TOTAL_NUM = 300        # <= 300 papers
_DEFAULT_DIST_PAGE_SIZE = 100        # S2 search limit
_DEFAULT_DIST_MAX_PAGES = _DEFAULT_DIST_TOTAL_NUM // _DEFAULT_DIST_PAGE_SIZE  # 3 pages

# IEI citations 拉取上限（每页1000）
_DEFAULT_CITATION_PAGE_SIZE = 1000
_DEFAULT_CITATION_MAX_PAGES = 5      # <= 5000 citations

# bulk search 兜底页数（token 翻页）
_DEFAULT_BULK_MAX_PAGES = 5

# ============================================================
# TNCSI core (FORMULA UNCHANGED)
# ============================================================

def _get_TNCSI_score(citation: int, loc, scale):
    import math

    def exponential_cdf(x, loc, scale):
        if x < loc:
            return 0
        else:
            z = (x - loc) / scale
            cdf = 1 - math.exp(-z)
            return cdf

    TNCSI = exponential_cdf(citation, loc, scale)
    return TNCSI


# ============================================================
# PDF curve params (same formula, add cache + hard limit)
# ============================================================

def _cache_key_pubdate(pub_date):
    if pub_date is None:
        return None
    # 只用于缓存 key，不改变任何计算
    return pub_date.strftime("%Y-%m-%d")


@lru_cache(maxsize=256)
def _cached_pdf_curve_params(topic: str, topk: int, show_img: bool, pub_date_key, mode: int):
    pub_date = datetime.strptime(pub_date_key, "%Y-%m-%d") if pub_date_key else None
    return _get_PDF_curve_params_impl(topic, topk=topk, show_img=show_img, pub_date=pub_date, mode=mode)


def get_PDF_curve_params(topic, topk=2000, show_img=False, pub_date: datetime = None, mode=1):
    """
    ⚠️ 公式不动，只做工程改造：
    - 加缓存（同 topic/pub_date/mode 复用结果）
    - 加 hard limit：topk 太大时会很慢/卡，自动截断到 _DEFAULT_DIST_TOTAL_NUM
    """
    safe_topk = min(int(topk or 0), _DEFAULT_DIST_TOTAL_NUM)
    pub_key = _cache_key_pubdate(pub_date)
    # show_img=True 不建议缓存（因为绘图），但你代码结构是“可选”，仍放入 key
    return _cached_pdf_curve_params(str(topic), safe_topk, bool(show_img), pub_key, int(mode))


def _get_PDF_curve_params_impl(topic, topk=2000, show_img=False, pub_date: datetime = None, mode=1):
    citation, _ = get_citation_discrete_distribution(
        topic,
        total_num=topk,
        pub_date=pub_date,
        mode=mode
    )
    citation = np.array(citation)

    try:
        params = stats.expon.fit(citation)
    except Exception:
        return None, None

    loc, scale = params
    if len(citation) <= 1:
        return None, None

    x = np.linspace(np.min(citation), np.max(citation), 100)
    pdf = stats.expon.pdf(x, loc, scale)

    if show_img:
        plt.clf()
        plt.figure(figsize=(6, 4))
        plt.hist(citation, bins=1000, density=True, alpha=0.5)
        plt.plot(x, pdf, 'r', label='Fitted Exponential Distribution')
        plt.xlabel('Number of Citations')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

    return loc, scale


# ============================================================
# Citation distribution for topic fitting (FORMULA UNCHANGED)
# ============================================================

@retry(delay=6)
def get_citation_discrete_distribution(keyword: str, total_num=1000, pub_date: datetime = None, mode=1):
    """
    ⚠️ 公式/统计口径不动，只做工程防卡：
    - 限制最多抓 _DEFAULT_DIST_MAX_PAGES 页（默认3页=300篇）
    - 一旦 data 为空立刻 break
    - API 错误时返回当前已累积结果
    """
    citation_count = []
    influentionCC = []

    date_six_months_ago = None
    date_six_months_later = None
    publicationDateOrYear = ""

    if pub_date:
        six_months = timedelta(days=183)
        date_six_months_ago = pub_date - six_months
        date_six_months_later = pub_date + six_months
        publicationDateOrYear = (
            f"&publicationDateOrYear={date_six_months_ago.strftime('%Y-%m')}:"
            f"{date_six_months_later.strftime('%Y-%m')}"
        )

    # hard limit
    total_num = min(int(total_num or 0), _DEFAULT_DIST_TOTAL_NUM)
    max_pages = min(int(total_num // _DEFAULT_DIST_PAGE_SIZE), _DEFAULT_DIST_MAX_PAGES)

    if mode == 1:  # default search
        for i in range(max_pages):
            url = (
                f"https://api.semanticscholar.org/graph/v1/paper/search"
                f"?query={keyword}{publicationDateOrYear}"
                f"&fieldsOfStudy=Computer Science"
                f"&fields=title,year,citationCount,influentialCitationCount"
                f"&offset={_DEFAULT_DIST_PAGE_SIZE * i}&limit={_DEFAULT_DIST_PAGE_SIZE}"
            )

            headers = {"x-api-key": s2api} if s2api is not None else None
            r = cached_get(url, headers=headers)

            try:
                response = r.json()
            except Exception:
                return citation_count, influentionCC

            data = response.get("data", [])
            if not data:
                break

            for item in data:
                try:
                    cc = int(item.get("citationCount", -1))
                    icc = int(item.get("influentialCitationCount", -1))
                except Exception:
                    continue

                if cc >= 0:
                    citation_count.append(cc)
                    influentionCC.append(max(icc, 0))

            # 如果已经够了，提前结束
            if len(citation_count) >= total_num:
                break

        return citation_count, influentionCC

    elif mode == 2:  # bulk search
        query = f'"{keyword}"~3'
        continue_token = None

        for _ in range(_DEFAULT_BULK_MAX_PAGES):
            if continue_token is None:
                response = request_query(
                    query,
                    early_date=date_six_months_ago,
                    later_date=date_six_months_later
                )
            else:
                response = request_query(
                    query,
                    continue_token=continue_token,
                    early_date=date_six_months_ago,
                    later_date=date_six_months_later
                )

            if "token" in response:
                continue_token = response["token"]

            data = response.get("data", [])
            if not data:
                break

            for item in data:
                try:
                    cc = int(item.get("citationCount", -1))
                    icc = int(item.get("influentialCitationCount", -1))
                except Exception:
                    continue

                if cc >= 0:
                    citation_count.append(cc)
                    influentionCC.append(max(icc, 0))

            if len(citation_count) >= total_num:
                break

            if not continue_token:
                break

        return citation_count, influentionCC

    return citation_count, influentionCC


# ============================================================
# TNCSI public API (FORMULA UNCHANGED, recursion-safe)
# ============================================================

@retry(tries=3)
def get_TNCSI(ref_obj, ref_type="entity", topic_keyword=None, same_year=False, mode=1, show_PDF=False):
    """
    兼容原调用方式，但工程上更安全：
    - 强烈建议 ref_type='entity' 且 ref_obj 是 S2paper 实例（避免递归 new）
    - 公式不改
    """
    from .semantic_scholar_paper import S2paper

    if ref_type == "title":
        # 兼容，但可能导致链式递归；尽量少用
        s2paper = S2paper(ref_obj)
    elif ref_type == "entity":
        s2paper = ref_obj
        if not isinstance(s2paper, S2paper):
            # 兼容旧传法：给的是 dict entity
            s2paper = S2paper(s2paper, ref_type="entity")
    else:
        return None

    if s2paper.citation_count is None:
        return {"TNCSI": -1, "topic": "NONE"}

    topic = s2paper.gpt_keyword if topic_keyword is None else topic_keyword

    if same_year:
        loc, scale = get_PDF_curve_params(
            topic,
            topk=1000,
            show_img=show_PDF,
            pub_date=s2paper.publication_date,
            mode=mode,
        )
    else:
        loc, scale = get_PDF_curve_params(
            topic,
            topk=1000,
            show_img=show_PDF,
            pub_date=None,
            mode=mode,
        )

    if loc is not None and scale is not None:
        try:
            TNCSI = _get_TNCSI_score(s2paper.citation_count, loc, scale)
        except ZeroDivisionError:
            return {"TNCSI": -1, "topic": "NaN"}

        return {"TNCSI": TNCSI, "topic": topic, "loc": loc, "scale": scale}

    return {"TNCSI": -1, "topic": topic, "loc": loc, "scale": scale}


# ============================================================
# IEI (FORMULA UNCHANGED, bounded citations fetch)
# ============================================================

@retry()
def get_s2citaions_per_month(title, total_num=1000):
    from .semantic_scholar_paper import S2paper

    s2paper = S2paper(title)

    if s2paper.publication_date is None:
        print("No publication date recorded")
        return []

    # 你原来用 s2paper.s2id，但你的 S2paper 类里未必有；这里用 entity.paperId 更稳
    if not s2paper.entity or "paperId" not in s2paper.entity:
        print("No paperId recorded")
        return []

    s2id = s2paper.entity["paperId"]

    citation_count = {}
    missing_count = 0
    OFFSET = 0

    # hard limit pages
    max_pages = min(int(total_num / _DEFAULT_CITATION_PAGE_SIZE), _DEFAULT_CITATION_MAX_PAGES)

    url_template = (
        f"https://api.semanticscholar.org/graph/v1/paper/{s2id}/citations"
        f"?fields=paperId,title,venue,year,referenceCount,citationCount,publicationDate,publicationTypes&offset="
    )

    for _ in range(max_pages):
        url = f"{url_template}{OFFSET}&limit={_DEFAULT_CITATION_PAGE_SIZE}"
        OFFSET += _DEFAULT_CITATION_PAGE_SIZE

        headers = {"x-api-key": s2api} if s2api is not None else None
        r = cached_get(url, headers=headers)
        resp = r.json()
        data = resp.get("data", [])

        if not data:
            break

        for item in data:
            cp = item.get("citingPaper") or {}
            info = {
                "paperId": cp.get("paperId"),
                "title": cp.get("title"),
                "citationCount": cp.get("citationCount"),
                "publicationDate": cp.get("publicationDate"),
            }

            cite_entry = S2paper(info, ref_type="entity")
            cite_entry.filled_authors = False

            try:
                if (
                    s2paper.publication_date
                    and cite_entry.publication_date
                    and s2paper.publication_date <= cite_entry.publication_date <= datetime.now()
                ):
                    dict_key = f"{cite_entry.publication_date.year}.{cite_entry.publication_date.month}"
                    citation_count[dict_key] = citation_count.get(dict_key, 0) + 1
                else:
                    missing_count += 1
            except Exception:
                missing_count += 1
                continue

    sorted_data = OrderedDict(
        sorted(citation_count.items(), key=lambda x: datetime.strptime(x[0], "%Y.%m"), reverse=True)
    )

    latest_month = datetime.now()
    earliest_month = s2paper.publication_date

    all_months = [datetime.strftime(latest_month, "%Y.%m")]
    while latest_month > earliest_month:
        latest_month = latest_month.replace(day=1)
        latest_month -= timedelta(days=1)
        all_months.append(datetime.strftime(latest_month, "%Y.%m"))

    result = {month: sorted_data.get(month, 0) for month in all_months}
    result = OrderedDict(sorted(result.items(), key=lambda x: datetime.strptime(x[0], "%Y.%m"), reverse=True))
    return result


@retry()
def get_IEI(title, show_img=False, save_img_pth=None, exclude_last_n_month=1, normalized=False):
    spms = get_s2citaions_per_month(title, 2000)

    actual_len = 6 if len(spms) >= 6 + exclude_last_n_month else len(spms) - exclude_last_n_month
    if actual_len < 6:
        return {"L6": float("-inf"), "I6": float("-inf")}

    x = [i for i in range(actual_len)]
    subset = list(spms.items())[exclude_last_n_month : exclude_last_n_month + actual_len][::-1]
    y = [item[1] for item in subset]

    if normalized:
        min_y = min(y)
        max_y = max(y)
        range_y = max_y - min_y
        if range_y == 0:
            y = [0 for _ in y]
        else:
            y = [(y_i - min_y) / range_y for y_i in y]

    t = np.linspace(0, 1, 100)
    n = len(x) - 1
    curve_x = np.zeros_like(t)
    curve_y = np.zeros_like(t)

    for i in range(n + 1):
        curve_x += comb(n, i) * (1 - t) ** (n - i) * t ** i * x[i]
        curve_y += comb(n, i) * (1 - t) ** (n - i) * t ** i * y[i]

    if show_img or save_img_pth:
        print("IEI绘图执行")
        plt.clf()
        fig = plt.figure(figsize=(6, 4), dpi=300)
        plt.style.use("seaborn-v0_8")
        plt.plot(x, y, "o", color="darkorange", label="Data Point")
        plt.plot(curve_x, curve_y, color="steelblue", label="Bezier Curve")
        plt.legend()
        plt.xlabel("Month")
        plt.ylabel("Received Citation")
        plt.grid(True)
        if save_img_pth:
            plt.savefig(save_img_pth, dpi=300, bbox_inches="tight")
        if show_img:
            plt.show()

    dx_dt = np.zeros_like(t)
    dy_dt = np.zeros_like(t)

    for i in range(n):
        dx_dt += comb(n - 1, i) * (1 - t) ** (n - i - 1) * t ** i * (x[i + 1] - x[i])
        dy_dt += comb(n - 1, i) * (1 - t) ** (n - i - 1) * t ** i * (y[i + 1] - y[i])

    I6 = dy_dt[-1] / dx_dt[-1]

    slope_avg = []
    for i in range(0, 100, 20):
        slope_avg.append(dy_dt[i] / dx_dt[i])
    slope_avg.append(I6)

    return {
        "L6": sum(slope_avg) / 6 if not math.isnan(sum(slope_avg)) else float("-inf"),
        "I6": I6 if not math.isnan(I6) else float("-inf"),
    }


# ============================================================
# RQM / RUI helpers (FORMULA UNCHANGED, recursion-safe)
# ============================================================

def get_pubdate_stat(refs):
    pub_dates = [i.publication_date for i in refs if i.publication_date and i.publication_date >= datetime(1970, 1, 1)]
    if not pub_dates:
        return None

    timestamps = [d.timestamp() for d in sorted(pub_dates, reverse=True)]
    median_timestamp = statistics.median(timestamps)
    median_value = datetime.fromtimestamp(median_timestamp)

    return {"med": median_value, "latest": pub_dates[0], "pub_dates": pub_dates}


def get_RQM(ref_obj, ref_type="entity", tncsi_rst=None, beta=20, topic_keyword=None):
    from .semantic_scholar_paper import S2paper

    if ref_type == "title":
        s2paper = S2paper(ref_obj)
    elif ref_type == "entity":
        s2paper = ref_obj
        if not isinstance(s2paper, S2paper):
            s2paper = S2paper(s2paper, ref_type="entity")
    else:
        return None

    if not tncsi_rst:
        tncsi_rst = get_TNCSI(s2paper, ref_type="entity", topic_keyword=topic_keyword, show_PDF=False)

    loc = tncsi_rst.get("loc")
    scale = tncsi_rst.get("scale")

    # 这里不改你的判定逻辑
    if len(getattr(s2paper, "references", [])) == 0 or (loc is None or scale is None) or (loc < 0 or scale < 0):
        print('''Error: Assert len(s2paper.references) == 0 or (loc<0 or scale<0) is Ture.''')
        return {"RQM": None, "ARQ": None, "S_mp": None, "loc": loc, "scale": scale}

    pub_dates = []
    for i in s2paper.references:
        if i.publication_date:
            pub_dates.append(i.publication_date)

    sorted_dates = sorted(pub_dates, reverse=True)
    date_index = len(sorted_dates) // 2
    index_date = sorted_dates[date_index]

    pub_time = s2paper.publication_date
    months_difference = (pub_time - index_date) // timedelta(days=30)
    S_mp = (months_difference // 6) + 1

    N_R = len(s2paper.references)

    score = 0
    for item in s2paper.references:
        try:
            score += _get_TNCSI_score(item.citation_count, loc, scale)
        except Exception:
            N_R = N_R - 1
            continue

    try:
        ARQ = score / N_R
    except ZeroDivisionError:
        ARQ = 0

    rst = {}
    rst["RQM"] = 1 - math.exp(-beta * math.exp(-(1 - ARQ) * S_mp))
    rst["ARQ"] = ARQ
    rst["S_mp"] = S_mp
    rst["loc"] = loc
    rst["scale"] = scale
    return rst


def get_RAD(M_pc):
    x = M_pc / 12
    coefficients = np.array([-0.0025163, 0.00106611, 0.12671325, 0.01288683])

    polynomial_function = np.poly1d(coefficients)
    x_pdf = np.linspace(0, 7.36, 200)
    fitted_y_pdf = polynomial_function(x_pdf)
    pdf_normalized = fitted_y_pdf / np.trapz(fitted_y_pdf, x_pdf)

    cdf = cumtrapz(pdf_normalized, x_pdf, initial=0)
    cdf = np.where(cdf > 1.0, 1.0, cdf)

    if x < x_pdf[0]:
        return 0
    if x > x_pdf[-1]:
        return 1

    index = np.searchsorted(x_pdf, x, side="left")
    return cdf[index]


def get_RUI(s2paper, p=10, q=10):
    PC = request_query(s2paper.gpt_keyword, early_date=s2paper.publication_date)

    if s2paper.publication_date:
        stat = get_pubdate_stat(s2paper.references)
        ref_median_pubdate = stat.get("med") if stat else None
        if ref_median_pubdate:
            if s2paper.publication_date > ref_median_pubdate:
                t = (datetime.now() - s2paper.publication_date) // timedelta(days=30)
                RAD = get_RAD(t)
                MP = request_query(
                    s2paper.gpt_keyword,
                    early_date=ref_median_pubdate,
                    later_date=s2paper.publication_date,
                )

                N_pc = PC["total"]
                N_mp = MP["total"]
                if N_mp == 0:
                    return {"RAD": RAD, "CDR": float("-inf"), "RUI": float("-inf")}

                CDR = N_pc / N_mp
                return {"RAD": RAD, "CDR": CDR, "RUI": p * CDR + q * RAD}

            else:
                ref_latest_date = stat.get("latest") if stat else None
                print(
                    f'''Caution: Publication date of paper "{s2paper.title}" is no longer accurate. Using the pubdate of the latest reference instead.'''
                )
                t = (datetime.now() - ref_latest_date) // timedelta(days=30)
                RAD = get_RAD(t)

                MP = request_query(
                    s2paper.gpt_keyword,
                    early_date=ref_median_pubdate,
                    later_date=ref_latest_date,
                )

                N_pc = PC["total"]
                N_mp = MP["total"]
                if N_mp == 0:
                    return {"RAD": RAD, "CDR": float("-inf"), "RUI": float("-inf")}

                CDR = N_pc / N_mp
                return {"RAD": RAD, "CDR": CDR, "RUI": p * CDR + q * RAD}

    print(f'''Publication date of paper "{s2paper.title}" is not available. This is a remote server issue. ''')
    return {"RAD": None, "CDR": None, "RUI": None}


# ============================================================
# request_query (ENGINEERING BUGFIX ONLY: keep s2api usable)
# ============================================================

@retry()
def request_query(
    query,
    sort_rule=None,
    continue_token=None,
    early_date: datetime = None,
    later_date: datetime = None,
):
    """
    你的 docstring/逻辑保持原样，只修一个工程 bug：
    - 你原来函数开头写了 `s2api = None`，导致永远不带 key。
    - 这里改为使用 cfg.config.s2api（不影响公式）
    """
    p_dict = dict(query=query)

    if early_date and later_date is None:
        p_dict["publicationDateOrYear"] = f'{early_date.strftime("%Y-%m-%d")}:'
    elif later_date and early_date is None:
        p_dict["publicationDateOrYear"] = f':{later_date.strftime("%Y-%m-%d")}'
    elif later_date and early_date:
        p_dict["publicationDateOrYear"] = f'{early_date.strftime("%Y-%m-%d")}:{later_date.strftime("%Y-%m-%d")}'
    else:
        pass

    if continue_token:
        p_dict["token"] = continue_token
    if sort_rule:
        p_dict["sort"] = sort_rule

    params = urlencode(p_dict)
    url = (
        f"{S2_QUERY_URL}?{params}&fields=url,title,abstract,authors,venue,externalIds,referenceCount,"
        f"openAccessPdf,citationCount,influentialCitationCount,influentialCitationCount,fieldsOfStudy,"
        f"s2FieldsOfStudy,publicationTypes,publicationDate"
    )

    headers = {"x-api-key": s2api} if s2api is not None else None
    reply = cached_get(url, headers=headers)
    response = reply.json()

    if "data" not in response:
        msg = response.get("error") or response.get("message") or "unknown"
        raise Exception(f"error while fetching {reply.url}: {msg}")

    return response
