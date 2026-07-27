"""
AgenticLoop — hypothesis-driven active learning.

Extends beyond passive reading with:
1. Hypothesis generation from knowledge gaps
2. Multi-source evidence gathering (books + web + internal)
3. Source comparison & conflict detection
4. Hypothesis testing & outcome evaluation
5. Confidence update based on evidence strength
"""

import json
import logging
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse, quote_plus

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Hypothesis:
    statement: str
    subject: str = ""
    relation: str = ""
    object: str = ""
    confidence: float = 0.5
    source: str = "generated"
    evidence_for: List[Dict[str, Any]] = field(default_factory=list)
    evidence_against: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "untested"  # untested, testing, confirmed, rejected, inconclusive


@dataclass(slots=True)
class EvidenceSource:
    source_type: str = ""  # book, web, internal, code_execution, tool
    source_name: str = ""
    content: str = ""
    relevance: float = 0.0
    credibility: float = 0.5
    url: str = ""  # source URL for web sources
    page_title: str = ""  # page title for web sources


@dataclass(slots=True)
class Outcome:
    hypothesis_id: str = ""
    confirmed: bool = False
    confidence_delta: float = 0.0
    evidence_summary: str = ""
    sources_used: List[str] = field(default_factory=list)


class HypothesisGenerator:
    """Generates testable hypotheses from knowledge gaps."""

    def __init__(self, memory=None):
        self.memory = memory

    def generate_from_goal(self, goal: str) -> List[Hypothesis]:
        hypotheses: List[Hypothesis] = []
        concepts = self._extract_concepts(goal)
        for c in concepts:
            rels = self._get_relationships(c)
            if not rels:
                hypotheses.append(Hypothesis(
                    statement=f"{c} may have unknown relationships",
                    subject=c, relation="has", object="?",
                    confidence=0.3, source="gap"))
            else:
                for r in rels[:3]:
                    hypotheses.append(Hypothesis(
                        statement=f"{c} --[{r}]--> ? implies further connections",
                        subject=c, relation=r.get("relation", ""),
                        object=r.get("target", "?"),
                        confidence=0.4, source="inference"))
        if not hypotheses:
            words = re.findall(r"\b\w{4,}\b", goal.lower())
            for w in words[:3]:
                hypotheses.append(Hypothesis(
                    statement=f"{w} may be relevant to '{goal}'",
                    subject=w, relation="related_to",
                    object=goal, confidence=0.3, source="keyword"))
        return hypotheses

    def generate_contradictory(self, hypothesis: Hypothesis) -> Hypothesis:
        return Hypothesis(
            statement=f"NOT({hypothesis.statement})",
            subject=hypothesis.subject,
            relation=hypothesis.relation,
            object=hypothesis.object,
            confidence=0.3,
            source="counter_hypothesis",
        )

    def _extract_concepts(self, text: str) -> List[str]:
        if self.memory:
            try:
                return self.memory.search_concepts(text, limit=5)
            except Exception:
                pass
        words = re.findall(r"'([^']+)'|\"([^\"]+)\"|([A-Z][a-z]+)", text)
        flat = []
        for w in words:
            for sub in w:
                if sub and len(sub) > 2:
                    flat.append(sub.lower())
        return flat or [w for w in re.split(r"[^a-z]+", text.lower())
                        if len(w) > 3][:5]

    def _get_relationships(self, concept: str) -> List[Dict[str, Any]]:
        if not self.memory:
            return []
        try:
            graph = self.memory.get_concept_graph(concept)
            return graph.get("relationships", [])
        except Exception:
            return []


class EvidenceGatherer:
    """Gathers evidence from multiple sources."""

    def __init__(self, memory=None, tool_registry=None):
        self.memory = memory
        self.tools = tool_registry
        self._research_log: List[Dict[str, Any]] = []

    def gather(self, hypothesis: Hypothesis,
               search_web: bool = True) -> List[EvidenceSource]:
        sources: List[EvidenceSource] = []
        query = f"{hypothesis.subject} {hypothesis.relation} {hypothesis.object}"

        internal = self._search_internal(query)
        if internal:
            sources.append(internal)

        if search_web and self.tools:
            shallow = self._search_web(query)
            if shallow:
                sources.append(shallow)

            deep_results = self._deep_research(query)
            sources.extend(deep_results)

        return sources

    @staticmethod
    def _extract_domain(url: str) -> str:
        try:
            return urlparse(url).netloc
        except Exception:
            return url

    def _fetch_web_content(self, url: str) -> Optional[EvidenceSource]:
        if not self.tools:
            return None
        try:
            result = self.tools.execute("web_fetch", url=url)
            if result.get("success"):
                content = result.get("content", "")
                title = result.get("title", "") or ""
                return EvidenceSource(
                    source_type="web",
                    source_name=self._extract_domain(url),
                    content=content[:6000],
                    relevance=0.7,
                    credibility=0.6,
                    url=url,
                    page_title=title,
                )
        except Exception as e:
            logger.debug("Web fetch failed for %s: %s", url, e)
        return None

    def _deep_research(self, query: str, max_pages: int = 3) -> List[EvidenceSource]:
        if not self.tools:
            return []
        sources: List[EvidenceSource] = []
        try:
            result = self.tools.execute("web_search", query=query, num_results=max_pages)
            if not result.get("success"):
                return []
            results_list = result.get("results", [])
            if not results_list:
                return []
            for item in results_list[:max_pages]:
                url = item.get("url", "") or item.get("link", "")
                if not url:
                    continue
                page = self._fetch_web_content(url)
                if page:
                    sources.append(page)
                    self._research_log.append({
                        "query": query, "url": url, "title": page.page_title,
                    })
            if sources:
                logger.info("Deep research: %d pages fetched for '%s'", len(sources), query)
        except Exception as e:
            logger.debug("Deep research failed: %s", e)
        return sources

    def _search_internal(self, query: str) -> Optional[EvidenceSource]:
        if not self.memory:
            return None
        try:
            concepts = self.memory.search_concepts(query, limit=3)
            passages = []
            for c in concepts:
                rels = self.memory.get_concept_graph(c)
                for r in rels.get("relationships", [])[:3]:
                    passages.append(
                        f"{r.get('subject', c)} --[{r.get('relation', '?')}]--> "
                        f"{r.get('target', '?')} "
                        f"(conf={r.get('confidence', 0.5):.2f})")
            if passages:
                return EvidenceSource(
                    source_type="internal",
                    source_name="knowledge_graph",
                    content="\n".join(passages),
                    relevance=0.7,
                    credibility=0.8)
        except Exception as e:
            logger.debug("Internal search failed: %s", e)
        return None

    def _search_web(self, query: str) -> Optional[EvidenceSource]:
        if not self.tools:
            return None
        try:
            result = self.tools.execute("web_search", query=query)
            if result.get("success"):
                content = result.get("content", "")
                count = result.get("total_results", 0)
                urls = result.get("results", [])
                url_list = "; ".join(
                    u.get("url", "") or u.get("link", "")
                    for u in (urls or [])[:5]
                )
                return EvidenceSource(
                    source_type="web",
                    source_name=f"web_search ({count} results)",
                    content=content[:3000],
                    relevance=0.6,
                    credibility=0.5,
                    url=url_list,
                    page_title="search results",
                )
        except Exception as e:
            logger.debug("Web search failed: %s", e)
        return None

    def get_research_log(self) -> List[Dict[str, Any]]:
        return list(self._research_log)


class OutcomeEvaluator:
    """Evaluate hypothesis against gathered evidence and update confidence."""

    def evaluate(self, hypothesis: Hypothesis,
                 sources: List[EvidenceSource]) -> Outcome:
        if not sources:
            return Outcome(
                hypothesis_id=hypothesis.statement[:50],
                confirmed=False,
                confidence_delta=-0.1,
                evidence_summary="No evidence found",
                sources_used=[])

        for_score = 0.0
        against_score = 0.0
        source_names: List[str] = []
        web_domains: set = set()
        shared_domain_bonus = 0.0

        for src in sources:
            label = f"{src.source_type}:{src.source_name}"
            if src.url and src.source_type == "web":
                domain = EvidenceGatherer._extract_domain(src.url)
                web_domains.add(domain)
                if domain:
                    label = f"{domain}:{src.page_title or src.source_name}"
            source_names.append(label)

            relevance = src.relevance
            credibility = src.credibility
            weight = relevance * credibility

            content_lower = src.content.lower()
            subj = hypothesis.subject.lower()
            obj = hypothesis.object.lower()
            rel = hypothesis.relation.lower()

            match = (subj in content_lower and obj in content_lower
                     and rel in content_lower)
            partial = (subj in content_lower or obj in content_lower)

            if match:
                for_score += weight
            elif partial:
                for_score += weight * 0.3
            else:
                against_score += weight * 0.2

        source_diversity_bonus = min(len(web_domains) * 0.03, 0.12)

        total = for_score + against_score
        net = for_score - against_score

        if total == 0:
            return Outcome(
                hypothesis_id=hypothesis.statement[:50],
                confirmed=False,
                confidence_delta=-0.05,
                evidence_summary="No relevant evidence found",
                sources_used=source_names)

        support_ratio = for_score / total
        adjusted_ratio = min(support_ratio + source_diversity_bonus, 1.0)
        confidence_delta = round((adjusted_ratio - 0.5) * 0.4, 4)
        confirmed = adjusted_ratio >= 0.65

        return Outcome(
            hypothesis_id=hypothesis.statement[:50],
            confirmed=confirmed,
            confidence_delta=confidence_delta,
            evidence_summary=(
                f"Support ratio: {support_ratio:.2f} "
                f"(adj: {adjusted_ratio:.2f}, {len(web_domains)} web domains, "
                f"{len(sources)} sources, {for_score:.2f} for / {against_score:.2f} against)"),
            sources_used=source_names,
        )


class ConfidenceUpdater:
    """Update memory confidence based on outcome evaluation."""

    def __init__(self, memory=None):
        self.memory = memory

    def update(self, hypothesis: Hypothesis, outcome: Outcome) -> bool:
        if not self.memory:
            return False
        try:
            new_conf = max(0.0, min(1.0,
                            hypothesis.confidence + outcome.confidence_delta))

            if outcome.confirmed and hypothesis.subject and hypothesis.object:
                self.memory.add_fact_triple(
                    hypothesis.subject,
                    hypothesis.relation or "related_to",
                    hypothesis.object,
                    source=f"agentic:{hypothesis.source}",
                    confidence=new_conf,
                    source_type="inferred",
                    truth_confidence=new_conf * 0.85,
                )

            self.memory.record_experience(
                event=f"hypothesis:{hypothesis.source}",
                result=outcome.evidence_summary,
                lesson=(f"{hypothesis.statement} "
                        f"{'confirmed' if outcome.confirmed else 'not supported'} "
                        f"(Δ={outcome.confidence_delta:+.3f})"),
                outcome_score=0.5 + outcome.confidence_delta,
                domain="agentic_learning",
            )
            return True
        except Exception as e:
            logger.debug("Confidence update failed: %s", e)
            return False


class AgenticLearningLoop:
    """Full agentic cycle: generate -> gather -> evaluate -> update.

    For each goal:
    1. Generate testable hypotheses
    2. Gather evidence (books + web + internal)
    3. Evaluate each hypothesis against evidence
    4. Update memory confidence
    5. Log outcomes for future learning

    Optionally:
    - Execute code to test hypotheses numerically
    - Generate counter-hypotheses for balance
    - Self-improve by reviewing past outcomes
    """

    def __init__(self, memory=None, tool_registry=None,
                 code_memory=None, code_sandbox=None):
        self.memory = memory
        self.tools = tool_registry
        self.code_memory = code_memory
        self.code_sandbox = code_sandbox
        self.generator = HypothesisGenerator(memory)
        self.gatherer = EvidenceGatherer(memory, tool_registry)
        self.evaluator = OutcomeEvaluator()
        self.updater = ConfidenceUpdater(memory)
        self._history: List[Dict[str, Any]] = []

    def run(self, goal: str, max_hypotheses: int = 3,
            search_web: bool = True,
            generate_counter: bool = True) -> Dict[str, Any]:
        start = time.time()

        hypotheses = self.generator.generate_from_goal(goal)
        hypotheses = hypotheses[:max_hypotheses]

        if not hypotheses:
            return {
                "goal": goal,
                "status": "no_hypotheses",
                "hypotheses_tested": 0,
                "confirmed": 0,
                "total_confidence_delta": 0.0,
                "duration": round(time.time() - start, 2),
            }

        if generate_counter:
            counter = [self.generator.generate_contradictory(h)
                       for h in hypotheses]
            hypotheses.extend(counter)

        results: List[Dict[str, Any]] = []

        for h in hypotheses:
            sources = self.gatherer.gather(h, search_web=search_web)

            code_result = self._try_code_execution(h)
            if code_result:
                sources.append(code_result)

            outcome = self.evaluator.evaluate(h, sources)
            self.updater.update(h, outcome)

            h.status = "confirmed" if outcome.confirmed else "rejected"
            h.confidence = max(0.0, min(1.0,
                              h.confidence + outcome.confidence_delta))

            results.append({
                "hypothesis": h.statement,
                "initial_confidence": round(h.confidence - outcome.confidence_delta, 3),
                "final_confidence": round(h.confidence, 3),
                "delta": outcome.confidence_delta,
                "confirmed": outcome.confirmed,
                "evidence": outcome.evidence_summary,
                "sources": outcome.sources_used,
            })

        total_delta = sum(r["delta"] for r in results)
        confirmed_count = sum(1 for r in results if r["confirmed"])

        summary = {
            "goal": goal,
            "status": "ok",
            "hypotheses_tested": len(results),
            "confirmed": confirmed_count,
            "rejected": len(results) - confirmed_count,
            "total_confidence_delta": round(total_delta, 4),
            "avg_delta": (round(total_delta / max(len(results), 1), 4)),
            "results": results,
            "duration": round(time.time() - start, 2),
        }

        self._history.append(summary)
        return summary

    def run_with_code(self, goal: str, code_snippet: str,
                      timeout: float = 10.0) -> Dict[str, Any]:
        hypotheses = self.generator.generate_from_goal(goal)
        if not hypotheses:
            return {"goal": goal, "status": "no_hypotheses"}

        for h in hypotheses[:2]:
            sources = self.gatherer.gather(h, search_web=True)

            exec_result = None
            if self.code_sandbox:
                exec_result = self.code_sandbox.run_code(code_snippet,
                                                         timeout=timeout)
                if exec_result["success"]:
                    sources.append(EvidenceSource(
                        source_type="code_execution",
                        source_name="sandbox",
                        content=f"Output: {exec_result['output'][:2000]}",
                        relevance=0.9,
                        credibility=0.9))
                else:
                    sources.append(EvidenceSource(
                        source_type="code_execution",
                        source_name="sandbox",
                        content=f"Error: {exec_result['error'][:1000]}",
                        relevance=0.7,
                        credibility=0.5))

            outcome = self.evaluator.evaluate(h, sources)
            self.updater.update(h, outcome)

            return {
                "goal": goal,
                "hypothesis": h.statement,
                "confirmed": outcome.confirmed,
                "confidence_delta": outcome.confidence_delta,
                "evidence": outcome.evidence_summary,
                "code_result": exec_result,
            }

        return {"goal": goal, "status": "no_hypotheses_processed"}

    def _try_code_execution(self, hypothesis: Hypothesis
                            ) -> Optional[EvidenceSource]:
        if not self.code_memory or not self.code_sandbox:
            return None
        query = f"{hypothesis.relation} {hypothesis.subject} {hypothesis.object}"
        artifacts = self.code_memory.search(query, top_k=3)
        if not artifacts:
            return None
        code = artifacts[0].source
        if len(code) > 2000:
            code = code[:2000]
        test_code = f"{code}\n\nprint('Code pattern found and executed successfully')"
        result = self.code_sandbox.run_code(test_code, timeout=3.0)
        if result["success"]:
            return EvidenceSource(
                source_type="code_execution",
                source_name=artifacts[0].name,
                content=result["output"][:1000],
                relevance=0.8,
                credibility=0.85)
        return None

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self._history[-limit:]

    def get_research_log(self) -> List[Dict[str, Any]]:
        return self.gatherer.get_research_log()

    def get_stats(self) -> Dict[str, Any]:
        if not self._history:
            return {"total_cycles": 0}
        total = len(self._history)
        confirmed = sum(h["confirmed"] for h in self._history)
        rejected = sum(h["rejected"] for h in self._history)
        deltas = [h["total_confidence_delta"] for h in self._history]
        return {
            "total_cycles": total,
            "total_hypotheses": sum(h["hypotheses_tested"] for h in self._history),
            "total_confirmed": confirmed,
            "total_rejected": rejected,
            "avg_confidence_delta": round(sum(deltas) / len(deltas), 4) if deltas else 0.0,
        }
