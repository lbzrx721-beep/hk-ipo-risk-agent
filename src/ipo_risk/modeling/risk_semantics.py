"""把开放式 Agent 风险结论转换为稳定、可审计的语义特征。

该模块只读取已经生成的风险标题、结论和证据，不调用模型，也不修改原始报告。
开放风险的 ``open_xxx`` 只是运行内 ID，不能作为跨公司机器学习特征；这里用稳定的
业务分类词典将其转换为一级风险族和细分风险主题。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Any

from ipo_risk.domain import AgentRole, Evidence, FeatureValue, RiskFinding, RiskSeverity


RISK_SEMANTIC_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SemanticRiskRule:
    subtype: str
    family: str
    label_zh: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class RiskSemanticClassification:
    finding_id: str
    original_risk_type: str
    primary_family: str
    primary_subtype: str
    primary_label_zh: str
    secondary_subtypes: tuple[str, ...]
    method: str
    mapping_confidence: float
    matched_keywords: tuple[str, ...]


_RULES = (
    # 财务风险
    SemanticRiskRule("cash_burn_liquidity", "financial", "现金消耗与流动性", (
        "经营现金流出", "经营活动现金净流出", "经营活动所用现金", "负经营现金流",
        "现金消耗", "现金短缺", "流动负债净额", "流动性压力", "外部融资支撑",
        "持续融资", "资金链", "现金流为负",
    )),
    SemanticRiskRule("persistent_loss", "financial", "持续亏损", (
        "持续亏损", "连续亏损", "累计亏损", "净亏损", "尚未盈利", "未盈利",
        "盈利时间不确定", "资不抵债", "负债净额状态",
    )),
    SemanticRiskRule("accounting_estimation", "financial", "会计判断与财务可靠性", (
        "收入确认", "收益确认", "总额法", "净额法", "会计估计", "重大判断",
        "未经审计", "未經審計", "公允价值计量", "第三层级", "会计政策",
    )),
    SemanticRiskRule("receivables_collection", "financial", "应收款与回款", (
        "贸易应收", "应收账款", "应收款项", "应收票据", "合约资产", "回款风险",
        "回收风险", "周转天数", "信贷期", "坏账", "预期信用损失",
    )),
    SemanticRiskRule("inventory_obsolescence", "financial", "存货与减值", (
        "存货", "库存", "陈旧", "过时", "滞销", "存货减值", "库存减值",
    )),
    SemanticRiskRule("debt_refinancing", "financial", "债务与再融资", (
        "银行借款", "计息借款", "借款总额", "公司债券", "偿债压力", "再融资",
        "债务到期", "债务偿还", "负债率", "资本负债", "抵押借款",
    )),
    SemanticRiskRule("capital_commitment", "financial", "资本承诺与投资开支", (
        "资本承诺", "资本开支", "投资活动现金流出", "建设投入", "资金需求",
        "未来投入", "扩产投入",
    )),
    SemanticRiskRule("asset_impairment", "financial", "资产及商誉减值", (
        "商誉", "资产减值", "无形资产减值", "物业减值", "模具减值", "减值风险",
        "可收回金额",
    )),
    SemanticRiskRule("tax_uncertainty", "financial", "税务与优惠资格", (
        "递延所得税", "税务亏损", "企业所得税", "优惠税率", "高新技术企业资格",
        "税务风险", "税负", "所得税率", "全球最低税",
    )),
    SemanticRiskRule("foreign_exchange", "financial", "汇率与资金跨境", (
        "外汇风险", "汇率波动", "汇兑", "外汇管制", "资金汇出", "货币兑换",
    )),
    SemanticRiskRule("margin_cost_pressure", "financial", "毛利与成本压力", (
        "毛利率", "盈利能力", "成本上升", "价格下降", "价格大幅下降", "费用增长",
        "开支增长", "授权费", "特许权使用费", "利润率", "定价压力",
    )),
    SemanticRiskRule("revenue_growth_uncertainty", "financial", "收入增长不确定性", (
        "收入增长放缓", "收益增长放缓", "收入下降", "收益下降", "收入波动",
        "收益波动", "增长可持续性", "未来收入", "收入高度波动", "业绩下滑",
    )),
    SemanticRiskRule("dividend_cash_distribution", "financial", "分红与现金分配", (
        "股息分配", "利润分配", "分红", "派息", "现金分派",
    )),
    # 经营风险
    SemanticRiskRule("customer_concentration", "operational", "客户集中度", (
        "客户集中", "最大客户", "五大客户", "单一客户", "客户a", "主要客户",
        "收益集中于", "收入集中于",
    )),
    SemanticRiskRule("supplier_concentration", "operational", "供应商集中度", (
        "供应商集中", "最大供应商", "五大供应商", "单一供应商", "主要供应商",
        "采购集中", "代工厂集中",
    )),
    SemanticRiskRule("product_concentration", "operational", "单一产品或业务依赖", (
        "单一产品", "核心产品", "产品收入占比", "收入高度依赖单一产品",
        "单一业务", "业务收入占比高", "ip收入占比",
    )),
    SemanticRiskRule("key_partner_dependency", "operational", "关键合作方依赖", (
        "合作伙伴", "合资伙伴", "依赖阿斯利康", "依赖礼来", "战略合作方",
        "单一合作", "合作协议可被终止", "承购协议",
    )),
    SemanticRiskRule("license_dependency", "operational", "牌照及授权依赖", (
        "牌照依赖", "品牌许可", "授权到期", "授权终止", "许可被终止", "维持资质",
        "资质维持", "医保定点", "双通道资质", "特许经营权",
    )),
    SemanticRiskRule("commercialization_pipeline", "operational", "研发管线与商业化", (
        "商业化不确定", "商业化存在不确定", "审批结果", "nda审批", "临床试验",
        "候选药物", "研发管线", "药物开发", "产品获批", "监管批准存在延迟",
    )),
    SemanticRiskRule("competition_growth", "operational", "竞争与增长持续性", (
        "市场竞争", "竞争激烈", "行业竞争", "市场份额", "增长放缓", "需求放缓",
        "业务扩张", "增长持续性", "市场规模",
    )),
    SemanticRiskRule("supply_chain", "operational", "供应链与原材料", (
        "供应链", "供应中断", "原材料", "材料短缺", "延迟交付", "分包商",
        "封装及测试", "第三方代工", "采购价格", "最低采购承诺",
    )),
    SemanticRiskRule("project_execution", "operational", "项目交付与履约", (
        "项目制", "中标率", "项目延期", "延迟风险", "成本超支", "未能准确估计",
        "里程碑", "履约义务", "项目验收", "项目表现", "服务特许权安排",
    )),
    SemanticRiskRule("capacity_utilization", "operational", "产能与扩产", (
        "产能利用率", "产能扩张", "产能过剩", "生产设施", "扩产", "闲置产能",
    )),
    SemanticRiskRule("channel_dependency", "operational", "销售渠道与分销依赖", (
        "分销商", "经销商", "销售渠道", "渠道调整", "线上渠道", "线下渠道",
    )),
    SemanticRiskRule("geographic_dependency", "operational", "地域与海外经营", (
        "海外业务", "国际化扩张", "地域集中", "单一地区", "海外市场依赖",
        "地缘政治",
    )),
    SemanticRiskRule("acquisition_integration", "operational", "收购与整合", (
        "收购及投资", "收购整合", "整合风险", "并购", "被收购", "收购项目",
    )),
    # 法律与合规风险
    SemanticRiskRule("related_party_transaction", "legal_compliance", "关联交易与利益冲突", (
        "关联方交易", "关连交易", "关联采购", "关联销售", "资金往来", "利益冲突",
        "关联方借款", "控股股东租赁", "定价公允", "利益输送",
    )),
    SemanticRiskRule("redemption_vam", "legal_compliance", "赎回、对赌与回购义务", (
        "赎回权", "赎回义务", "强制回购", "回购义务", "对赌", "利润保证",
        "股份补偿", "市值承诺", "清算优先权", "优先股赎回",
    )),
    SemanticRiskRule("litigation_penalty", "legal_compliance", "诉讼、调查与处罚", (
        "诉讼", "仲裁", "处罚", "罚款", "监管措施", "自律监管", "调查",
        "法律程序", "被吊销", "或有事项",
    )),
    SemanticRiskRule("regulatory_compliance", "legal_compliance", "监管许可与一般合规", (
        "监管合规", "法规不确定", "监管政策", "合规风险", "未完全遵守",
        "法律法规", "许可要求", "监管批准", "监管变化", "监管要求",
    )),
    SemanticRiskRule("data_privacy", "legal_compliance", "数据隐私与网络合规", (
        "数据隐私", "个人信息", "数据合规", "网络安全法", "数据跨境",
        "网络安全审查", "数据保护",
    )),
    SemanticRiskRule("intellectual_property", "legal_compliance", "知识产权", (
        "知识产权", "专利", "商标", "版权", "侵权", "技术保护期限",
        "共同持有", "知识产权归属",
    )),
    SemanticRiskRule("contract_vie", "legal_compliance", "合约安排与VIE", (
        "合约安排", "vie", "可变利益实体", "外资持股限制", "控制协议",
    )),
    SemanticRiskRule("property_permit", "legal_compliance", "物业、土地及经营许可", (
        "租赁物业", "物业业权", "未完成备案", "土地使用权", "无证物业",
        "节能审查", "环评批准", "竣工验收", "经营许可",
    )),
    SemanticRiskRule("labor_compliance", "legal_compliance", "劳动与社保合规", (
        "社会保险", "住房公积金", "劳动合规", "员工福利", "劳务派遣",
    )),
    SemanticRiskRule("trade_export_control", "legal_compliance", "贸易及出口管制", (
        "实体清单", "出口管制", "国际贸易政策", "贸易限制", "制裁",
    )),
    # 公司治理风险
    SemanticRiskRule("controlling_shareholder", "governance", "控股股东与控制权集中", (
        "控股股东", "股权高度集中", "控制权集中", "最大股东", "重大控制权",
        "控制公司", "小股东权益",
    )),
    SemanticRiskRule("weighted_voting_rights", "governance", "不同投票权架构", (
        "不同投票权", "双重股权", "a类/b类", "特别投票权", "投票权架构",
    )),
    SemanticRiskRule("board_governance", "governance", "董事会与治理机制", (
        "董事会", "独立非执行董事", "自由裁量权", "缺乏独立监督", "公司秘书资格",
        "董事薪酬", "股份转让登记",
    )),
    SemanticRiskRule("minority_shareholder", "governance", "少数股东与附属公司控制", (
        "少数股东", "少数股东权益", "子公司控制权", "合资企业控制",
    )),
    # 发行结构风险
    SemanticRiskRule("share_incentive_dilution", "issuance_structure", "股权激励摊薄", (
        "购股权计划", "股份激励", "受限制股份", "股份支付", "esop", "行使价极低",
        "尚未行使的购股权", "奖励计划",
    )),
    SemanticRiskRule("equity_dilution", "issuance_structure", "潜在股权摊薄", (
        "股权稀释", "摊薄现有股东", "摊薄股东", "认股权证", "可转换优先股",
        "可交换票据", "额外股份发行", "配发新股",
    )),
    SemanticRiskRule("offer_structure", "issuance_structure", "发售及配售结构", (
        "超额配股权", "发售量调整权", "回补机制", "公开发售", "国际发售",
        "优先发售", "全球发售架构", "发售规模", "发售失效", "未能定价",
    )),
    SemanticRiskRule("use_of_proceeds", "issuance_structure", "募集资金用途", (
        "所得款项用途", "募集资金用途", "净募集资金", "所得款项净额",
        "上市开支", "发行开支",
    )),
    SemanticRiskRule("preipo_investor_rights", "issuance_structure", "上市前投资者特殊权利", (
        "首次公开发售前投资者", "上市前投资者", "反摊薄权", "注册权",
        "优先股股东", "特殊权利",
    )),
    # 市场与流动性风险
    SemanticRiskRule("free_float_liquidity", "market_liquidity", "公众持股与流动性", (
        "公众持股", "自由买卖", "流通量", "流通盘", "股份流动性", "流动性风险",
        "发售股份占", "全球发售规模较小",
    )),
    SemanticRiskRule("cornerstone_concentration", "market_liquidity", "基石投资者集中度", (
        "基石投资者", "基石认购", "基石投資者",
    )),
    SemanticRiskRule("lockup_selloff", "market_liquidity", "禁售期与潜在抛压", (
        "禁售期", "解禁", "减持", "抛售压力", "抛压", "不受禁售限制",
        "锁定期届满", "持股成本极低", "发售价折让",
    )),
    SemanticRiskRule("price_stabilization", "market_liquidity", "稳价安排", (
        "稳定价格", "稳价", "借股安排", "价格稳定期", "股份购回授权",
    )),
    SemanticRiskRule("market_volatility", "market_liquidity", "上市后价格波动", (
        "股价波动", "价格波动", "市场差异", "市场供求", "股价稳定性",
        "上市后股价",
    )),
    # 技术风险
    SemanticRiskRule("cybersecurity_system", "technology", "系统与网络安全", (
        "系统故障", "系统中断", "安全漏洞", "网络攻击", "信息系统", "技术基础设施",
    )),
    SemanticRiskRule("technology_obsolescence", "technology", "技术迭代与替代", (
        "技术迭代", "技术过时", "技术替代", "研发失败", "新产品开发",
        "技术路线", "缺乏技术壁垒",
    )),
    SemanticRiskRule("research_development", "technology", "研发投入与成果转化", (
        "研发开支", "研发投入", "研发费用", "研发活动", "研发成果",
    )),
)


_CANONICAL_MAP = {
    "cash_burn": "cash_burn_liquidity",
    "persistent_loss": "persistent_loss",
    "customer_concentration": "customer_concentration",
    "supplier_concentration": "supplier_concentration",
    "single_product_dependency": "product_concentration",
    "related_party_transaction": "related_party_transaction",
    "redemption_or_vam": "redemption_vam",
    "litigation_or_penalty": "litigation_penalty",
    "license_dependency": "license_dependency",
    "data_privacy_compliance": "data_privacy",
    "commercialization_uncertainty": "commercialization_pipeline",
    "pipeline_uncertainty": "commercialization_pipeline",
    "issuance_structure": "offer_structure",
}

_RULE_BY_SUBTYPE = {rule.subtype: rule for rule in _RULES}
_FAMILY_FALLBACK_BY_ROLE = {
    AgentRole.FINANCE: ("financial", "financial_other", "其他财务风险"),
    AgentRole.LEGAL: ("legal_compliance", "legal_other", "其他法律合规风险"),
    AgentRole.BUSINESS: ("operational", "operational_other", "其他经营风险"),
    AgentRole.MARKET: ("market_liquidity", "market_other", "其他市场风险"),
}

RISK_FAMILIES = (
    "financial",
    "operational",
    "legal_compliance",
    "governance",
    "issuance_structure",
    "market_liquidity",
    "technology",
)
RISK_SUBTYPES = tuple(rule.subtype for rule in _RULES) + (
    "financial_other",
    "operational_other",
    "legal_other",
    "market_other",
)
_FALLBACK_SUBTYPES = frozenset(
    {"financial_other", "operational_other", "legal_other", "market_other"}
)


def classify_risk_finding(
    finding: RiskFinding,
    *,
    evidence_by_id: Mapping[str, Evidence],
) -> RiskSemanticClassification:
    """按稳定词典对单条风险做可复现、多标签语义映射。"""
    canonical = _CANONICAL_MAP.get(finding.risk_type)
    if canonical is not None:
        rule = _RULE_BY_SUBTYPE[canonical]
        return RiskSemanticClassification(
            finding_id=finding.finding_id,
            original_risk_type=finding.risk_type,
            primary_family=rule.family,
            primary_subtype=rule.subtype,
            primary_label_zh=rule.label_zh,
            secondary_subtypes=(),
            method="canonical_risk_type",
            mapping_confidence=1.0,
            matched_keywords=(),
        )

    title = _normalize_text(finding.title)
    claim = _normalize_text(finding.claim)
    evidence_text = _normalize_text(
        " ".join(
            evidence_by_id[evidence_id].quote
            for evidence_id in finding.evidence_ids
            if evidence_id in evidence_by_id
        )
    )
    matches: list[tuple[int, int, SemanticRiskRule, tuple[str, ...], str]] = []
    for priority, rule in enumerate(_RULES):
        normalized_keywords = _normalized_rule_keywords()[rule.subtype]
        title_hits = tuple(keyword for keyword in normalized_keywords if keyword in title)
        claim_hits = tuple(keyword for keyword in normalized_keywords if keyword in claim)
        evidence_hits = tuple(
            keyword for keyword in normalized_keywords if keyword in evidence_text
        )
        if title_hits:
            score, method, hits = 60 + min(len(title_hits), 5), "title_keyword", title_hits
        elif claim_hits:
            score, method, hits = 40 + min(len(claim_hits), 5), "claim_keyword", claim_hits
        elif evidence_hits:
            score, method, hits = 20 + min(len(evidence_hits), 5), "evidence_keyword", evidence_hits
        else:
            continue
        matches.append((score, -priority, rule, hits, method))

    # 证据片段往往包含同页其他语义。已有标题/结论命中时，不再引入仅凭证据命中的标签。
    strong = [item for item in matches if item[0] >= 40]
    selected_pool = strong if strong else matches
    selected_pool.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = selected_pool[:3]
    if selected:
        score, _, primary, hits, method = selected[0]
        secondary = tuple(
            item[2].subtype for item in selected[1:] if item[2].subtype != primary.subtype
        )
        confidence = 0.90 if score >= 60 else 0.80 if score >= 40 else 0.65
        return RiskSemanticClassification(
            finding_id=finding.finding_id,
            original_risk_type=finding.risk_type,
            primary_family=primary.family,
            primary_subtype=primary.subtype,
            primary_label_zh=primary.label_zh,
            secondary_subtypes=secondary,
            method=method,
            mapping_confidence=confidence,
            matched_keywords=tuple(dict.fromkeys(hits)),
        )

    family, subtype, label = _FAMILY_FALLBACK_BY_ROLE[finding.created_by]
    return RiskSemanticClassification(
        finding_id=finding.finding_id,
        original_risk_type=finding.risk_type,
        primary_family=family,
        primary_subtype=subtype,
        primary_label_zh=label,
        secondary_subtypes=(),
        method="agent_role_fallback",
        mapping_confidence=0.35,
        matched_keywords=(),
    )


def build_risk_semantic_feature_snapshot(
    findings: Sequence[RiskFinding],
    evidences: Sequence[Evidence],
    *,
    available_at: date,
    source_ref: str,
) -> tuple[dict[str, FeatureValue], tuple[RiskSemanticClassification, ...]]:
    """将一家公司全部风险聚合为固定列的数值特征，并返回逐风险审计记录。"""
    evidence_by_id = {evidence.evidence_id: evidence for evidence in evidences}
    classifications = tuple(
        classify_risk_finding(finding, evidence_by_id=evidence_by_id)
        for finding in findings
    )
    finding_by_id = {finding.finding_id: finding for finding in findings}
    severity_score = {
        RiskSeverity.LOW: 1,
        RiskSeverity.MEDIUM: 2,
        RiskSeverity.HIGH: 3,
        RiskSeverity.CRITICAL: 4,
    }
    family_counts: Counter[str] = Counter()
    family_severities: defaultdict[str, list[int]] = defaultdict(list)
    family_supported_counts: Counter[str] = Counter()
    subtype_counts: Counter[str] = Counter()
    subtype_severities: defaultdict[str, list[int]] = defaultdict(list)
    subtype_supported_counts: Counter[str] = Counter()
    specific_count = 0
    multilabel_count = 0
    supported_count = 0
    technical_failure_count = 0
    for classification in classifications:
        finding = finding_by_id[classification.finding_id]
        score = severity_score[finding.severity]
        supported = _finding_is_supported(finding)
        technical_failure = _finding_is_technical_failure(finding)
        family_counts[classification.primary_family] += 1
        family_severities[classification.primary_family].append(score)
        family_supported_counts[classification.primary_family] += int(supported)
        subtypes = (classification.primary_subtype, *classification.secondary_subtypes)
        for subtype in dict.fromkeys(subtypes):
            subtype_counts[subtype] += 1
            subtype_severities[subtype].append(score)
            subtype_supported_counts[subtype] += int(supported)
        if classification.method != "agent_role_fallback":
            specific_count += 1
        if classification.secondary_subtypes:
            multilabel_count += 1
        supported_count += int(supported)
        technical_failure_count += int(technical_failure)

    count = len(findings)
    values: dict[str, int | float] = {
        "agent_semantic_specific_classified_count": specific_count,
        "agent_semantic_specific_classified_share": (
            round(specific_count / count, 8) if count else 0.0
        ),
        "agent_semantic_multilabel_finding_count": multilabel_count,
        "agent_semantic_distinct_subtype_count": len(subtype_counts),
        "agent_semantic_supported_finding_count": supported_count,
        "agent_semantic_supported_finding_share": (
            round(supported_count / count, 8) if count else 0.0
        ),
        "agent_semantic_technical_failure_count": technical_failure_count,
        "agent_semantic_technical_failure_share": (
            round(technical_failure_count / count, 8) if count else 0.0
        ),
    }
    for family in RISK_FAMILIES:
        family_count = family_counts[family]
        values[f"agent_semantic_family_{family}_count"] = family_count
        values[f"agent_semantic_family_{family}_share"] = (
            round(family_count / count, 8) if count else 0.0
        )
        values[f"agent_semantic_family_{family}_max_severity"] = max(
            family_severities[family], default=0
        )
        values[f"agent_semantic_family_{family}_supported_count"] = (
            family_supported_counts[family]
        )
    # ``*_other`` 只用于审计覆盖率，不进入模型白名单。
    for subtype in (rule.subtype for rule in _RULES):
        subtype_count = subtype_counts[subtype]
        values[f"agent_semantic_has_{subtype}"] = int(subtype_count > 0)
        values[f"agent_semantic_risk_{subtype}_count"] = subtype_count
        values[f"agent_semantic_risk_{subtype}_max_severity"] = max(
            subtype_severities[subtype], default=0
        )
        values[f"agent_semantic_risk_{subtype}_supported_count"] = (
            subtype_supported_counts[subtype]
        )

    return (
        {
            name: FeatureValue(
                value=value,
                available_at=available_at,
                source_ref=source_ref,
            )
            for name, value in values.items()
        },
        classifications,
    )


def semantic_model_feature_names() -> tuple[str, ...]:
    """返回允许进入模型筛选的稳定语义字段；排除分类器覆盖率等质量字段。"""
    names: list[str] = []
    for family in RISK_FAMILIES:
        names.extend(
            (
                f"agent_semantic_family_{family}_count",
                f"agent_semantic_family_{family}_share",
                f"agent_semantic_family_{family}_max_severity",
                f"agent_semantic_family_{family}_supported_count",
            )
        )
    # 角色兜底的 ``*_other`` 仅用于审计覆盖率，语义不够具体，不能作为训练特征。
    for subtype in (item for item in RISK_SUBTYPES if item not in _FALLBACK_SUBTYPES):
        names.extend(
            (
                f"agent_semantic_has_{subtype}",
                f"agent_semantic_risk_{subtype}_count",
                f"agent_semantic_risk_{subtype}_max_severity",
                f"agent_semantic_risk_{subtype}_supported_count",
            )
        )
    names.append("agent_semantic_distinct_subtype_count")
    return tuple(names)


def risk_semantic_catalog() -> tuple[dict[str, str], ...]:
    """返回稳定风险字典，供审计、报告和后续版本比较。"""
    return tuple(
        {
            "family": rule.family,
            "subtype": rule.subtype,
            "label_zh": rule.label_zh,
        }
        for rule in _RULES
    )


@lru_cache(maxsize=1)
def _opencc_converter() -> Any | None:
    try:
        from opencc import OpenCC
    except ImportError:
        return None
    return OpenCC("t2s")


@lru_cache(maxsize=1)
def _normalized_rule_keywords() -> dict[str, tuple[str, ...]]:
    return {
        rule.subtype: tuple(_normalize_text(keyword) for keyword in rule.keywords)
        for rule in _RULES
    }


@lru_cache(maxsize=100_000)
def _normalize_text(value: str) -> str:
    text = str(value or "").lower().replace("－", "-").replace("／", "/")
    converter = _opencc_converter()
    if converter is not None:
        text = converter.convert(text)
    return "".join(text.split())


def _finding_is_supported(finding: RiskFinding) -> bool:
    if finding.specialist_status == "supported":
        return True
    if finding.specialist_status is not None:
        return False
    uncertainty = _normalize_text(finding.uncertainty or "")
    failure_markers = (
        "失败",
        "error",
        "insufficient",
        "未生成结论",
        "未找到",
        "未召回",
        "待补证",
    )
    return finding.confidence > 0 and not any(
        marker in uncertainty for marker in failure_markers
    )


def _finding_is_technical_failure(finding: RiskFinding) -> bool:
    if finding.specialist_status == "technical_failure":
        return True
    uncertainty = _normalize_text(finding.uncertainty or "")
    return any(
        marker in uncertainty
        for marker in ("失败", "error", "429", "overload", "technicalfailure")
    )


__all__ = [
    "RISK_FAMILIES",
    "RISK_SEMANTIC_SCHEMA_VERSION",
    "RISK_SUBTYPES",
    "RiskSemanticClassification",
    "build_risk_semantic_feature_snapshot",
    "classify_risk_finding",
    "risk_semantic_catalog",
    "semantic_model_feature_names",
]
