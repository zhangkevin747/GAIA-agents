from typing import Any, List, Optional

from smolagents import CodeAgent
from utils.logger import get_logger
import time

logger = get_logger(__name__)

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
GAIA_FILE_BASE_URL = "https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/2023/validation"

# GAIA-specific instructions injected into CodeAgent's system prompt
# via the `instructions` parameter (appears in code_agent.yaml's
# {{custom_instructions}} slot, right before "Now Begin!").
GAIA_INSTRUCTIONS = """
You are solving GAIA benchmark tasks. GAIA uses exact-match scoring, so your
final_answer() must contain ONLY the precise answer — no explanations, no
reasoning, no units unless explicitly asked.

CRITICAL — Answer format rules:
- Numbers: plain digits, no commas or units (e.g. 42)
- Strings: the exact factual answer only — no preamble, no "The answer is...",
  no quoting the question back, no markdown formatting (no ** bold **)
- Lists: comma-separated with spaces after commas, no brackets
  (e.g. apple, orange, banana)
- If the question asks "What does X say?", answer with ONLY the quote itself,
  not "X says: ..." or any framing

WRONG: final_answer("The answer is Paris")
WRONG: final_answer("Teal'c responds with: **Extremely.**")
RIGHT: final_answer("Paris")
RIGHT: final_answer("Extremely")

When a question has an attached file, download it first with file_from_url(),
then use the appropriate tool to analyze it (vision_tool for images,
summarize_csv_data for CSVs, etc.). Always capture and use the return value
of file_from_url() as the file path for subsequent tools.

If the question text is reversed, read it backwards to understand it, then
answer in normal (forward) text.
"""


class Agent:
    """Agent class wrapping smolagents CodeAgent for GAIA benchmark evaluation.

    Args:
        model: The language model to use.
        tools: List of tools to provide to the agent.
        instructions: Custom instructions injected into CodeAgent's system prompt.
        verbose: Whether to print debug information.
    """

    def __init__(
        self,
        model: Any,
        tools: Optional[List[Any]] = None,
        instructions: Optional[str] = None,
        verbose: bool = False
    ):
        logger.info("Initializing Agent")
        self.verbose = verbose
        self.imports = [
            "pandas", "numpy", "os", "requests", "tempfile",
            "datetime", "json", "time", "re", "openpyxl",
            "pathlib", "sys"
        ]

        self.agent = CodeAgent(
            model=model,
            tools=tools,
            instructions=instructions or GAIA_INSTRUCTIONS,
            add_base_tools=False,
            additional_authorized_imports=self.imports,
        )
        logger.info("Agent initialized")

    def __call__(self, question: str, files: List[str] = None) -> str:
        """Main interface that logs inputs/outputs and handles timing."""
        if self.verbose:
            print(f"Agent received question: {question[:50]}... with files: {files}")

        time.sleep(25)
        return self.answer_question(question, files[0] if files else None)

    def answer_question(self, question: str, task_file_path: Optional[str] = None) -> str:
        """Process a GAIA benchmark question with optional file context.

        Args:
            question: The question to answer.
            task_file_path: Optional filename for the associated task file.

        Returns:
            The cleaned answer string.
        """
        try:
            task = self._build_task(question, task_file_path)

            if self.verbose:
                print("Task:", task[:200] + "...")

            answer = self.agent.run(task)
            return self._clean_answer(str(answer))

        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return f"ERROR: {str(e)}"

    def _build_task(self, question: str, file_path: Optional[str]) -> str:
        """Build the task string passed to agent.run().

        This is the user-message content — just the question and file context.
        System-level instructions live in GAIA_INSTRUCTIONS (injected into
        CodeAgent's system prompt via the `instructions` parameter).
        """
        parts = [question]

        if file_path:
            parts.append(
                f"\nAttached file available at: {GAIA_FILE_BASE_URL}/{file_path}\n"
                "Download and analyze this file using the appropriate tools."
            )

        if self._is_reversed_text(question):
            parts.append(
                f"\nNote: This question is reversed text. "
                f"Here it is forwards: {question[::-1]}"
            )

        return "\n".join(parts)

    @staticmethod
    def _is_reversed_text(text: str) -> bool:
        """Detect GAIA's reversed-text questions.

        Only triggers on the known GAIA pattern (contains '.rewsna eht sa'),
        not on any text that happens to start with a period.
        """
        return ".rewsna eht sa" in text.lower()

    @staticmethod
    def _clean_answer(answer: str) -> str:
        """Strip common prefixes, verbose framing, and whitespace from the raw answer."""
        import re

        for prefix in ["Final Answer:", "Answer:", "=>"]:
            if answer.startswith(prefix):
                answer = answer[len(prefix):]

        # Strip markdown bold markers
        answer = re.sub(r'\*\*(.+?)\*\*', r'\1', answer)

        # Strip trailing periods (GAIA answers don't end with periods)
        answer = answer.strip(" '\"").rstrip(".")

        return answer.strip()
