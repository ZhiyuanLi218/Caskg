"""Prompt templates for CaSKG causal reasoning with LLM judges."""

CAUSAL_LLM_JUDGE_SYSTEM = (
    "You are an expert in analyzing causal dependencies between skills in "
    "autonomous agent task graphs. Given a pair of skills, you assess how "
    "strongly one skill causally enables or is a prerequisite for another. "
    "You reason about interventional relationships: if the source skill were "
    "removed or disabled, how much would performance on tasks requiring the "
    "target skill degrade? Provide a numeric score between 0 and 1."
)

CAUSAL_LLM_JUDGE_PROMPT = (
    "Consider the following two skills in an agent's skill graph:\n\n"
    "Source skill: {source_skill}\n"
    "Target skill: {target_skill}\n\n"
    "Rate the causal dependency strength from the source skill to the target "
    "skill on a scale of 0 to 1, where:\n"
    "- 0 means the target skill functions completely independently of the "
    "source skill\n"
    "- 1 means the target skill entirely depends on the source skill and "
    "cannot function without it\n\n"
    "Consider:\n"
    "1. Would removing the source skill cause the target skill to fail?\n"
    "2. Does the source skill produce outputs consumed by the target skill?\n"
    "3. Is there a logical ordering where the source must precede the target?\n\n"
    "Respond with a JSON object: {{\"score\": <float>, \"reasoning\": \"<brief explanation>\"}}"
)

CAUSAL_MERGE_PROPOSAL_SYSTEM = (
    "You are an expert in skill graph optimization for autonomous agents. "
    "Your task is to evaluate whether two skills are interventionally "
    "substitutable -- meaning that in any context where one skill is used, "
    "the other could replace it without degrading task outcomes. You reason "
    "about functional equivalence, input/output compatibility, and contextual "
    "interchangeability."
)

CAUSAL_MERGE_PROPOSAL_PROMPT = (
    "Consider the following two skills in an agent's skill graph:\n\n"
    "Skill A: {skill_a}\n"
    "Skill B: {skill_b}\n\n"
    "Evaluate whether these skills are interventionally substitutable:\n"
    "1. If Skill A were replaced by Skill B in all task executions, would "
    "outcomes remain equivalent?\n"
    "2. If Skill B were replaced by Skill A in all task executions, would "
    "outcomes remain equivalent?\n"
    "3. Do they share the same causal parents and children in typical task "
    "graphs?\n"
    "4. Are there any contexts where one skill succeeds but the other would "
    "fail?\n\n"
    "Respond with a JSON object: {{\"substitutable\": <bool>, \"confidence\": "
    "<float 0-1>, \"merge_recommendation\": \"<merge|keep_separate|needs_more_data>\", "
    "\"reasoning\": \"<brief explanation>\"}}"
)

CAUSAL_SPLIT_ANALYSIS_SYSTEM = (
    "You are an expert in identifying context-dependent behavior in agent "
    "skill graphs. Your task is to analyze execution traces for a single skill "
    "and determine whether it behaves as multiple distinct sub-skills depending "
    "on context. You look for clusters of usage patterns that suggest the skill "
    "should be split into specialized variants."
)

CAUSAL_SPLIT_ANALYSIS_PROMPT = (
    "Analyze the following skill and its execution trace summary:\n\n"
    "Skill: {skill}\n\n"
    "Execution traces summary:\n{traces_summary}\n\n"
    "Identify whether this skill exhibits distinct behavioral clusters that "
    "suggest it should be split into separate specialized skills. Consider:\n"
    "1. Does the skill behave differently across environments or task types?\n"
    "2. Are there subsets of traces where it co-occurs with very different "
    "companion skills?\n"
    "3. Do success rates vary significantly by context?\n"
    "4. Would splitting improve causal clarity in the graph?\n\n"
    "Respond with a JSON object: {{\"should_split\": <bool>, \"num_clusters\": "
    "<int>, \"clusters\": [{{\"label\": \"<name>\", \"description\": \"<context>\", "
    "\"distinguishing_features\": [\"<feature>\"]}}], \"reasoning\": \"<brief explanation>\"}}"
)
