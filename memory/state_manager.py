from typing import Dict, TypedDict, List, Optional


class AgentState(TypedDict):
    run_id: str
    user_goal: str
    current_task: str
    urls_visited: List[str]
    extracted_jobs: List[dict]
    failed_attempts: int
    status: str
    search_results: List[dict]
    reflection_notes: Optional[str]
    human_question: Optional[str]
    pages: List[dict]
    search_query: str
    extraction_fields: List[str]
    max_results: int
    crawl_depth: int
    capture_screenshots: bool
    screenshots: Dict[str, str]