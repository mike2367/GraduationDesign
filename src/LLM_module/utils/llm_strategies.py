from __future__ import annotations

from dataclasses import dataclass

from LLM_module import eval_config as ecfg
from LLM_module.utils.llm_client import LLMClientProtocol, LLMResponse


@dataclass
class StrategyTrace:
	name: str
	initial: LLMResponse
	feedback: LLMResponse | None = None
	refined: LLMResponse | None = None
	questions: LLMResponse | None = None
	answers: LLMResponse | None = None
	final: LLMResponse | None = None


def _vprint(msg: str) -> None:
	if bool(getattr(ecfg, "EVAL_VERBOSE", False)):
		print(msg)


def run_baseline(
	client: LLMClientProtocol,
	prompt: str,
	*,
	system_prompt: str | None = None,
	temperature: float | None = None,
	top_p: float | None = None,
	max_tokens: int | None = None,
) -> StrategyTrace:
	_vprint("=" * 20 + " LLM baseline: start " + "=" * 20)
	initial = client.complete(prompt, system_prompt=system_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
	_vprint("=" * 20 + " LLM baseline: done " + "=" * 20)
	return StrategyTrace(name="baseline", initial=initial, final=initial)


def _join_system_user(system_prompt: str | None, user_prompt: str) -> str:
	return user_prompt if not system_prompt else (str(system_prompt).rstrip() + "\n\n" + str(user_prompt).lstrip()).strip()

def _build_prompt(*parts: str) -> str:
	"""Join prompt parts with newlines, filtering empty strings."""
	return "\n\n".join(p for p in parts if p)

REQUIRED_SECTION_HEADERS = (
	"You MUST include these section headers verbatim (in order), using the exact numbering token `N)` (not `N.` or `N:`):\n"
	"  1) Mechanism Name: ...\n"
	"  2) Mechanistic Summary: ...\n"
	"  3) Evidence Chains: ...\n"
	"  4) Key Claims ...\n"
	"  5) Competing Hypotheses ...\n"
	"  6) Caveats ...\n"
	"  7) Confidence: ...\n"
	"  8) Suggested validations ...\n"
)


def run_self_refine(
	client: LLMClientProtocol,
	prompt: str,
	*,
	system_prompt: str | None = None,
	rounds: int | None = None,
	temperature: float | None = None,
	top_p: float | None = None,
	max_tokens: int | None = None,
) -> StrategyTrace:
	rounds = max(1, int(ecfg.SELF_REFINE_ROUNDS if rounds is None else rounds))
	_vprint("=" * 20 + f" LLM self_refine: start (rounds={rounds}) " + "=" * 20)
	trace = StrategyTrace(
		name="self_refine",
		initial=client.complete(prompt, system_prompt=system_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens),
	)
	current = trace.initial

	for i in range(rounds):
		_vprint('-' * 20 + f" self_refine round {i + 1}/{rounds} " + '-' * 20)
		feedback_prompt = _build_prompt(
			"You are a strict scientific reviewer.\n"
			"Given the ORIGINAL USER PROMPT and the MODEL RESPONSE, produce actionable feedback to improve factual grounding with MINIMAL semantic drift.\n\n"
			"Rules:\n"
			"- Follow the ORIGINAL USER PROMPT rules exactly (including whether BACKGROUND knowledge is allowed).\n"
			"- Identify fabricated/missing KG citations, evidence-vs-inference mixing, and any formatting violations.\n"
			"- Check for canonical-function anchoring: if the response collapses the mechanism into a single famous function (e.g., DNA repair only), require revision to cover any additional KG-supported functional axes (e.g., metabolic/energy/redox stress) as distinct contributing mechanisms, not merely upstream preconditions.\n"
			"- If the ORIGINAL USER PROMPT explicitly requires multiple distinct functional domains in evidence chains, verify this requirement is satisfied; otherwise require deletion/weakening of unsupported claims or addition of KG-grounded chains (no invention).\n"
			"- If the prompt allows BACKGROUND knowledge: require it to be explicitly labeled as BACKGROUND and to have NO KG citations.\n"
			"- Do NOT introduce new KG edges or sources that are not present in the prompt tables.\n"
			"- Do NOT suggest adding new biological claims; prefer deletion, weakening, or explicit labeling (SPECULATION/BACKGROUND) over invention.\n"
			"- Output 6-12 bullets, each starting with: ISSUE:, FIX:, WHY: (one line each).",
			"ORIGINAL USER PROMPT:\n" + _join_system_user(system_prompt, prompt),
			"MODEL RESPONSE:\n" + current.text
		)
		feedback = client.complete(feedback_prompt, system_prompt=system_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

		rewrite_prompt = _build_prompt(
			"Rewrite the MODEL RESPONSE to fully address the feedback.\n"
			"Rules:\n"
			"- Follow the ORIGINAL USER PROMPT rules exactly.\n"
			"- MINIMAL-CHANGE POLICY: keep as much of the original wording as possible; only edit what is necessary to fix the listed issues.\n"
			"- Do NOT add new mechanistic claims, new entities, new pathways, or new causal links. Prefer deleting or weakening claims.\n"
			"- Never fabricate KG citations; cite only edges that appear in the prompt tables.\n"
			"- If the prompt allows BACKGROUND knowledge, label it explicitly as BACKGROUND and do not attach KG citations to it.\n"
			"- Keep the output format exactly as required by the original prompt.\n"
			"- " + REQUIRED_SECTION_HEADERS + "- Any non-supported statement must be labeled SPECULATION or removed (per the original prompt rules).",
			"ORIGINAL USER PROMPT:\n" + _join_system_user(system_prompt, prompt),
			"MODEL RESPONSE (to rewrite):\n" + current.text,
			"FEEDBACK:\n" + feedback.text
		)
		refined = client.complete(rewrite_prompt, system_prompt=system_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
		trace.feedback = feedback
		trace.refined = refined
		current = refined

	trace.final = current
	_vprint("=" * 20 + " LLM self_refine: done " + "=" * 20)
	return trace


def run_cove(
	client: LLMClientProtocol,
	prompt: str,
	*,
	system_prompt: str | None = None,
	verification_mode: str | None = None,
	answer_citation_style: str | None = None,
	n_questions: int | None = None,
	question_instructions: str | None = None,
	temperature: float | None = None,
	top_p: float | None = None,
	max_tokens: int | None = None,
) -> StrategyTrace:
	n_questions = max(1, int(ecfg.COVE_NUM_QUESTIONS if n_questions is None else n_questions))
	verification_mode = (verification_mode or "prompt_grounding").strip().lower()
	answer_citation_style = (answer_citation_style or "keys").strip().lower()
	_vprint("=" * 20 + f" LLM cove: start (n_questions={n_questions}) " + "=" * 20)
	trace = StrategyTrace(
		name="cove",
		initial=client.complete(prompt, system_prompt=system_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens),
	)

	if verification_mode in {"mechanistic_wetlab", "mechanistic-wetlab", "mechanistic"}:
		mode_instructions = (
			"Verification mode: mechanistic + wet-lab focused.\n"
			"Write questions that: (1) force a concrete causal chain with intermediate events, "
			"(2) explicitly tie each step to KG evidence present in the prompt, and "
			"(3) propose discriminating wet-lab assays with expected outcomes to test key links.\n"
			"If a question would require external/background biology, phrase it to ask whether the prompt contains supporting evidence; "
			"if not, it must be marked NOT VERIFIABLE in the answers step."
		)
	else:
		mode_instructions = (
			"Verification mode: prompt-grounded.\n"
			"Questions must be answerable ONLY from the original prompt content (tables, edge lines, constraints)."
		)

	if answer_citation_style in {"full_edges", "full", "edges"}:
		citation_instructions = "full human-readable KG edge citation lines copied from the prompt (verbatim)"
	else:
		citation_instructions = "comma-separated key=... identifiers from the prompt (verbatim)"

	q_prompt = _build_prompt(
		(
			"You are performing Chain-of-Verification (CoVe).\n"
			"Given the ORIGINAL USER PROMPT and the DRAFT RESPONSE, produce a numbered list of verification questions.\n\n"
			"Rules:\n"
			f"- Write exactly {n_questions} questions.\n"
			f"- {mode_instructions}\n"
			"- Target ONLY high-impact, checkable items: (a) each evidence-chain link, (b) each causal claim, (c) each KG citation usage, (d) any named pathway/entity not present in the prompt.\n"
			+ (
				"- Additional instructions: " + question_instructions.strip() + "\n"
				if question_instructions and question_instructions.strip()
				else ""
			)
			+ "- Output ONLY the numbered questions (no extra text)."
		),
		"ORIGINAL USER PROMPT:\n" + _join_system_user(system_prompt, prompt),
		"DRAFT RESPONSE:\n" + trace.initial.text
	)
	questions = client.complete(q_prompt, system_prompt=system_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

	a_prompt = _build_prompt(
		"Answer the verification questions using ONLY the ORIGINAL USER PROMPT.\n"
		"Rules:\n"
		"- For each question, provide exactly two lines:\n"
		"  A: <answer>\n"
		f"  Citations: <{citation_instructions} or NONE>\n"
		"- If the answer cannot be verified from the prompt, say: A: NOT VERIFIABLE.\n"
		"- If you cite, you MUST cite exact evidence that appears in the prompt.\n"
		"- Do NOT add external facts.",
		"ORIGINAL USER PROMPT:\n" + _join_system_user(system_prompt, prompt),
		"VERIFICATION QUESTIONS:\n" + questions.text
	)
	answers = client.complete(a_prompt, system_prompt=system_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

	revise_prompt = _build_prompt(
		"Revise the DRAFT RESPONSE using the verification answers.\n"
		"Rules:\n"
		"- MINIMAL-CHANGE POLICY: keep as much of the draft wording as possible; only edit what verification forces you to change.\n"
		"- Do NOT add any new claims. You may only delete claims, weaken claims, or add missing citations to existing verifiable claims.\n"
		"- Drop or relabel any claim that is NOT VERIFIABLE (label as SPECULATION/BACKGROUND only if allowed by the original prompt).\n"
		"- Ensure every KG-supported claim has valid KG edge citations from the prompt.\n"
		"- If the ORIGINAL USER PROMPT allows BACKGROUND knowledge, keep it explicitly labeled as BACKGROUND and without KG citations.\n"
		"- Keep the original required output format exactly.\n"
		"- " + REQUIRED_SECTION_HEADERS,
		"ORIGINAL USER PROMPT:\n" + _join_system_user(system_prompt, prompt),
		"DRAFT RESPONSE:\n" + trace.initial.text,
		"VERIFICATION ANSWERS:\n" + answers.text
	)
	final = client.complete(revise_prompt, system_prompt=system_prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)

	trace.questions = questions
	trace.answers = answers
	trace.final = final
	_vprint("=" * 20 + " LLM cove: done " + "=" * 20)
	return trace
