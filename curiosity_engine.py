"""
Jafar — Curiosity Engine
Proactively identifies weak knowledge, generates questions, and seeks answers
using available tools. No longer waits for books — it asks and explores.
"""
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
from datetime import datetime, timezone
from urllib.parse import urlparse

from memory_system import MemorySystem
from reflection_system import ReflectionSystem

logger = logging.getLogger(__name__)

# Domain credibility ratings (0.0-1.0) used to seed truth_confidence
# for facts learned from web sources.
DOMAIN_CREDIBILITY: Dict[str, float] = {
    "nature.com": 0.95, "science.org": 0.95, "cell.com": 0.90,
    "pnas.org": 0.90, "cambridge.org": 0.90, "oxford.ac.uk": 0.90,
    "springer.com": 0.85, "ieee.org": 0.85, "acm.org": 0.85,
    "arxiv.org": 0.80, "pubmed.ncbi.nlm.nih.gov": 0.95,
    "ncbi.nlm.nih.gov": 0.90, "sciencedirect.com": 0.85,
    "nasa.gov": 0.95, "nih.gov": 0.95, "cdc.gov": 0.95,
    "who.int": 0.90, "whitehouse.gov": 0.85,
    "wikipedia.org": 0.75, "britannica.com": 0.85,
    "developer.mozilla.org": 0.90, "docs.python.org": 0.90,
    "learn.microsoft.com": 0.80,
    "reuters.com": 0.80, "apnews.com": 0.80,
    "bbc.com": 0.75, "bbc.co.uk": 0.75,
    "economist.com": 0.75, "wsj.com": 0.70,
    "nytimes.com": 0.70, "washingtonpost.com": 0.70,
    "theguardian.com": 0.65,
    "stackoverflow.com": 0.55, "github.com": 0.55,
    "medium.com": 0.35, "substack.com": 0.35,
    "wordpress.com": 0.30, "blogspot.com": 0.30,
    "reddit.com": 0.20, "twitter.com": 0.15,
    "x.com": 0.15, "facebook.com": 0.10,
}

CREDIBILITY_TLD_BOOSTS: Dict[str, float] = {
    "edu": 0.80, "gov": 0.85, "org": 0.55, "com": 0.40,
}

CURIOSITY_MIN_CONFIDENCE = 0.4
CURIOSITY_MIN_RELATIONSHIPS = 3
CURIOSITY_MAX_QUESTIONS_PER_CYCLE = 5


class CuriosityEngine:
    """Identifies knowledge gaps and actively tries to fill them."""

    def __init__(self, memory: MemorySystem, reflection: ReflectionSystem,
                 tool_registry=None):
        self.memory = memory
        self.reflection = reflection
        self.tools = tool_registry
        self._asked: set = set()
        logger.debug("CuriosityEngine initialized")

    # ---------------------------------------------------------------
    #  Identify
    # ---------------------------------------------------------------

    def identify_opportunities(self, reflection_report: Optional[Dict] = None
                               ) -> List[Dict[str, Any]]:
        """Score all concepts by curiosity potential — weak, isolated,
        contradictory, or epistemically uncertain."""
        report = reflection_report or self.reflection.run_full_reflection()

        seen = set()
        opportunities = []

        for w in report.get("weak_concepts", []):
            key = w["name"]
            if key in seen:
                continue
            seen.add(key)
            score = self._score_weak(w)
            opportunities.append({
                "concept": w["name"],
                "score": score,
                "reason": f"Weak: {w.get('rel_count', 0)} rels, "
                          f"avg conf {w.get('avg_confidence', 0):.2f}",
                "type": "weak",
            })

        for name in report.get("isolated_concepts", []):
            if name in seen:
                continue
            seen.add(name)
            opportunities.append({
                "concept": name,
                "score": 0.9,
                "reason": "Isolated: concept has no relationships",
                "type": "isolated",
            })

        for c in report.get("contradictions", []):
            name = c["source_concept"]
            if name in seen:
                continue
            seen.add(name)
            opportunities.append({
                "concept": name,
                "score": 0.8,
                "reason": f"Contradiction: {c['targets']}",
                "type": "contradiction",
            })

        opportunities.sort(key=lambda o: o["score"], reverse=True)
        return opportunities

    def _score_weak(self, weak: Dict) -> float:
        rels = weak.get("rel_count", 0)
        conf = weak.get("avg_confidence", 0)
        if rels == 0:
            return 0.85
        base = max(0.0, 1.0 - conf)
        penalty = min(0.3, rels * 0.1)
        return min(1.0, base + 0.3 - penalty)

    # ---------------------------------------------------------------
    #  Web Source Credibility
    # ---------------------------------------------------------------

    @staticmethod
    def _extract_domain(url: str) -> str:
        try:
            parsed = urlparse(url)
            return parsed.netloc or parsed.path.split("/")[0]
        except Exception:
            return url

    def _score_domain_credibility(self, domain: str) -> float:
        domain = domain.lower()
        # Exact match
        if domain in DOMAIN_CREDIBILITY:
            return DOMAIN_CREDIBILITY[domain]
        # Subdomain match — check if any known domain is a suffix
        for known, score in DOMAIN_CREDIBILITY.items():
            if domain.endswith("." + known):
                return score
        # TLD fallback
        parts = domain.rsplit(".", 2)
        for i in range(len(parts) - 1, 0, -1):
            suffix = ".".join(parts[i:])
            if suffix in CREDIBILITY_TLD_BOOSTS:
                return CREDIBILITY_TLD_BOOSTS[suffix]
        return 0.40

    def _extract_web_sources(self, tool_result: Dict) -> List[Dict]:
        if tool_result.get("tool_name") != "web_search":
            return []
        sources = []
        for r in tool_result.get("results", []):
            url = r.get("url", "")
            domain = self._extract_domain(url)
            sources.append({
                "title": r.get("title", ""),
                "url": url,
                "domain": domain,
                "credibility": self._score_domain_credibility(domain),
                "snippet": r.get("snippet", ""),
            })
        return sources

    def _compute_aggregate_credibility(
        self, web_sources: List[Dict], answer_text: str = ""
    ) -> float:
        if not web_sources:
            return 0.5
        creds = [s.get("credibility", 0.5) for s in web_sources]
        base = max(creds)
        agreeing = sum(1 for c in creds if c >= 0.35)
        agreement_boost = min(0.15, max(0, agreeing - 1) * 0.05)
        return min(0.95, base + agreement_boost)

    # ---------------------------------------------------------------
    #  Question
    # ---------------------------------------------------------------

    def generate_question(self, opportunity: Dict) -> str:
        """Turn an opportunity into a natural-language question."""
        concept = opportunity["concept"]
        typ = opportunity.get("type", "weak")
        if typ == "isolated":
            return f"What is {concept} and how does it work?"
        if typ == "contradiction":
            return f"Can you explain what {concept} really means?"
        if typ == "weak":
            return f"What is {concept}?"
        return f"Tell me about {concept}?"

    # ---------------------------------------------------------------
    #  Seek
    # ---------------------------------------------------------------

    def _deep_research(self, question: str, web_sources: List[Dict],
                       max_pages: int = 2) -> Tuple[str, List[Dict]]:
        """Fetch full content from top web sources and return enriched answer text."""
        if not self.tools or not web_sources:
            return "", web_sources
        fetched_parts: List[str] = []
        enriched = list(web_sources)
        for ws in web_sources[:max_pages]:
            url = ws.get("url", "")
            if not url:
                continue
            try:
                result = self.tools.execute("web_fetch", url=url)
                if result.get("success"):
                    content = result.get("content", "")[:3000]
                    title = result.get("title", "") or ws.get("title", "")
                    ws["page_content"] = content[:1000]
                    fetched_parts.append(f"From {title} ({ws['domain']}):\n{content.strip()[:2000]}")
            except Exception as e:
                logger.debug("Deep research fetch failed for %s: %s", url, e)
        return "\n\n".join(fetched_parts), enriched

    def seek_answer(self, question: str, concept: str) -> Dict[str, Any]:
        """Search for answers using available tools.

        Priority: 1) knowledge graph search, 2) registered tools.
        Returns structured metadata including web sources with credibility scores
        when web search is used.
        Does deep follow-up fetches on top web results for richer evidence.
        """
        answer_parts = []
        found_source = None
        web_sources = []

        known = self.memory.search_concepts(concept)
        if known:
            graph = self.memory.get_concept_graph(concept)
            if graph.get("relationships"):
                summary = self._summarize_graph(graph)
                answer_parts.append(summary)
                found_source = "knowledge_graph"

        if self.tools is not None:
            result = self.tools.decide_and_execute(question, min_score=0.3)
            if result.get("success"):
                tool_name = result.get("tool_name", "")
                if tool_name == "web_search":
                    web_sources = self._extract_web_sources(result)
                if result.get("results"):
                    answer_parts.append(self._render_tool_result(result))
                elif result.get("content"):
                    answer_parts.append(result["content"])
                if not found_source:
                    found_source = "tool"

        deep_content, enriched_sources = self._deep_research(question, web_sources)
        if deep_content:
            answer_parts.append(deep_content)
            found_source = "deep_web"

        if not answer_parts:
            return {"found": False, "answer": None, "source": None,
                    "web_sources": [], "aggregate_credibility": 0.0}

        answer_text = "\n".join(answer_parts)
        agg_cred = self._compute_aggregate_credibility(enriched_sources, answer_text)
        return {
            "found": True,
            "answer": answer_text,
            "source": found_source,
            "web_sources": enriched_sources,
            "aggregate_credibility": agg_cred,
        }

    def _summarize_graph(self, graph: Dict) -> str:
        lines = [f"Known about {graph['name']}:"]
        for r in graph.get("relationships", [])[:5]:
            direction = r.get("direction", "out")
            if direction == "out":
                lines.append(f"  {r['relation']} -> {r['target']}")
            else:
                lines.append(f"  ← {r['relation']} ({r.get('source', '?')})")
        return "\n".join(lines)

    def _render_tool_result(self, result: Dict) -> str:
        parts = []
        for r in result.get("results", []):
            concept = r.get("concept", "?")
            rels = r.get("relationships", [])
            for rel in rels[:3]:
                parts.append(
                    f"{concept} {rel.get('relation', '?')} "
                    f"{rel.get('target', '?')}"
                )
        return "\n".join(parts)

    # ---------------------------------------------------------------
    #  Learn
    # ---------------------------------------------------------------

    def learn_from_answer(self, question: str, answer: str,
                          concept: str,
                          web_sources: Optional[List[Dict]] = None
                          ) -> Dict[str, Any]:
        """Extract triples from answer text and add to memory.

        When web_sources are provided, truth_confidence is seeded from
        domain credibility with an agreement boost for corroborating sources.
        Source metadata (URLs, domains, credibility) is stored in tags
        for downstream truth system scoring.
        """
        source = f"curiosity: {question}"
        tags: Dict[str, Any] = {"curiosity": "true", "question": question}

        if web_sources:
            creds = [s.get("credibility", 0.5) for s in web_sources]
            base_cred = max(creds)
            agreeing = sum(1 for c in creds if c >= 0.35)
            agreement_boost = min(0.15, max(0, agreeing - 1) * 0.05)
            truth_confidence = min(0.95, base_cred + agreement_boost)
            tags["web_sources"] = json.dumps([
                {"title": s.get("title", ""), "domain": s.get("domain", ""),
                 "credibility": s.get("credibility", 0.5),
                 "page_content": s.get("page_content", "")[:500] if s.get("page_content") else ""}
                for s in web_sources
            ])
            tags["source_count"] = str(len(web_sources))
            tags["avg_credibility"] = f"{sum(creds) / len(creds):.3f}"
            source_quality = "web"
        else:
            truth_confidence = 0.5
            source_quality = None

        result = self.memory.learn_from_text(
            answer, source=source, source_type="idea",
            tags=tags,
            truth_confidence=truth_confidence,
            source_quality=source_quality,
        )
        logger.info(
            "Learned %d triples from curiosity about '%s' (tc=%.2f)",
            result.get("triples_processed", 0), concept, truth_confidence,
        )
        return result

    # ---------------------------------------------------------------
    #  Full Cycle
    # ---------------------------------------------------------------

    def run_curiosity_cycle(self, max_questions: int = 3
                            ) -> Dict[str, Any]:
        """Full curiosity cycle:
        1. Identify weak/isolated/contradictory concepts
        2. Generate questions about the most promising
        3. Seek answers using available tools
        4. Learn from what was found
        """
        logger.info("Starting curiosity cycle...")
        opportunities = self.identify_opportunities()
        results = []

        for opp in opportunities[:max_questions]:
            concept = opp["concept"]
            if concept in self._asked:
                continue
            self._asked.add(concept)

            question = self.generate_question(opp)
            logger.debug("Curious about '%s' -> %s", concept, question)

            seek_result = self.seek_answer(question, concept)

            item = {
                "concept": concept,
                "question": question,
                "score": opp["score"],
                "reason": opp["reason"],
                "found": seek_result.get("found", False),
                "source": seek_result.get("source"),
            }

            if seek_result.get("found") and seek_result.get("answer"):
                learn = self.learn_from_answer(
                    question, seek_result["answer"], concept,
                    web_sources=seek_result.get("web_sources", []),
                )
                item["triples_learned"] = learn.get("triples_processed", 0)

            web_sources = seek_result.get("web_sources", [])
            item["web_source_count"] = len(web_sources)
            item["aggregate_credibility"] = seek_result.get(
                "aggregate_credibility", 0.0)

            results.append(item)
            logger.info(
                "Curiosity: %s -> %s (found=%s)",
                concept, question, seek_result.get("found"),
            )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "opportunities_evaluated": len(opportunities),
            "questions_asked": len(results),
            "results": results,
        }
