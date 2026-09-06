from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CASKG_",
        extra="ignore",
    )

    WORKING_DIR: str = "./caskg_workspace"
    PREBUILT_WORKING_DIR: str | None = None
    DOMAIN: str = "Agent Skills and Tool Dependencies"

    LLM_MODEL: str = "openrouter/google/gemini-2.5-flash"
    EMBEDDING_MODEL: str = "openai/text-embedding-3-large"
    EMBEDDING_DIM: int = 3072
    GEMINI_API_KEY: SecretStr | None = Field(default=None, alias="GEMINI_API_KEY")
    OPENAI_API_KEY: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    OPENROUTER_API_KEY: SecretStr | None = Field(default=None, alias="OPENROUTER_API_KEY")
    OPENAI_BASE_URL: str | None = Field(default=None, alias="OPENAI_BASE_URL")

    LINK_TOP_K: int = 8
    SEED_TOP_K: int = 5
    RETRIEVAL_TOP_N: int = 8
    USE_FULL_MARKDOWN: bool = True
    ENABLE_SEMANTIC_LINKING: bool = True
    DEPENDENCY_MATCH_THRESHOLD: float = 0.6

    PPR_DAMPING: float = 0.2
    PPR_MAX_ITER: int = 50
    PPR_TOLERANCE: float = 1e-6

    SNIPPET_CHARS: int = 900
    MAX_SKILL_CHARS: int = 2400
    MAX_CONTEXT_CHARS: int = 12000
    RERANK_CANDIDATE_MULTIPLIER: int = 4
    SEED_CANDIDATE_TOP_K_SEMANTIC: int = 20
    SEED_CANDIDATE_TOP_K_LEXICAL: int = 20
    ENABLE_QUERY_REWRITE: bool = False

    SKILL_FILENAME: str = "SKILL.md"
    ALLOW_FRONTMATTER_DOCS: bool = True

    # When set, graphskills-query rewrites Source: paths to {SKILLS_DIR}/{skill_name}/SKILL.md
    # so agents in containerised environments can find skill scripts at the mounted path.
    SKILLS_DIR: str = ""

    # CaSKG: Counterfactual Validation
    CAUSAL_CONFIRM_THRESHOLD: float = 0.6
    CAUSAL_REJECT_THRESHOLD: float = 0.2
    MAX_INTERVENTION_BUDGET: int = 1000
    P_EXPLORE_INITIAL: float = 0.3
    P_EXPLORE_MIN: float = 0.05
    EVOLUTION_CYCLE_INTERVAL: int = 50
    TRANSPORTABILITY_THRESHOLD: float = 0.7

    # CaSKG: Candidate Induction Signal Weights
    LAMBDA_SEM: float = 0.25
    LAMBDA_LEX: float = 0.10
    LAMBDA_IO: float = 0.25
    LAMBDA_COOCCUR: float = 0.20
    LAMBDA_REPAIR: float = 0.10
    LAMBDA_LLM_JUDGE: float = 0.10

    # CaSKG: LLM-judge gate. The judge runs during Phase-1 induction only for
    # pairs whose preliminary association score (over active signals) exceeds
    # this threshold. Candidate pairs are pre-filtered to high-similarity pairs,
    # so the preliminary median is ~0.3; a 0.4 gate restricts the judge to the
    # top ~10% strongest candidates (keeps indexing fast and judge calls cheap),
    # while Phase-2 counterfactual probes do the real causal validation.
    LLM_JUDGE_GATE: float = 0.4

    # Phase 1 LLM judge concurrency (asyncio semaphore). Set via env
    # CASKG_JUDGE_SEMAPHORE or .env. Lower = safer against rate limits.
    JUDGE_SEMAPHORE: int = 1

    # Phase 1 LLM judge global start-to-start delay in seconds. Set via env
    # CASKG_JUDGE_CALL_DELAY. With delay=1.05, judge starts stay near 57/min
    # even when JUDGE_SEMAPHORE allows multiple in-flight requests.
    JUDGE_CALL_DELAY: float = 0.6

    # CaSKG: Retrieval Weights
    CAUSAL_RETRIEVAL_LAMBDA_REL: float = 0.3
    CAUSAL_RETRIEVAL_LAMBDA_STRUCT: float = 0.25
    CAUSAL_RETRIEVAL_LAMBDA_CAUSALNEC: float = 0.35
    CAUSAL_RETRIEVAL_LAMBDA_UNC: float = 0.1

    # CaSKG: Scheduler Weights
    SCHEDULER_ALPHA_USAGE: float = 0.25
    SCHEDULER_BETA_CENTRALITY: float = 0.20
    SCHEDULER_GAMMA_UNCERTAINTY: float = 0.30
    SCHEDULER_DELTA_FAILURE: float = 0.15
    SCHEDULER_EPSILON_MAINTENANCE: float = 0.10

    # CaSKG: Maintenance Thresholds
    MERGE_SUBSTITUTABILITY_THRESHOLD: float = 0.85
    RETIRE_MARGINAL_THRESHOLD: float = 0.05
    SPLIT_VARIANCE_THRESHOLD: float = 0.3
    PRUNE_AGE_THRESHOLD: int = 100
    STABLE_REINFORCEMENT_FACTOR: float = 1.05


settings = Settings()
