import asyncio
from dotenv import load_dotenv
from graphs.workflow import run_workflow
from tools.logging_config import configure_logging

load_dotenv()
configure_logging()


async def main():
    print("🤖 Autonomous Browser + API Workflow Agent")
    print("==========================================")
    user_goal = input("Enter your goal: ").strip()

    if not user_goal:
        user_goal = "Find top remote AI internships 2025"

    print(f"\n🚀 Starting agent for: {user_goal}\n")
    result = await run_workflow(user_goal)

    # Human-in-the-loop: the planner can ask a clarifying question instead of guessing.
    # Answer it and re-run (up to 3 rounds) rather than proceeding with a bad plan.
    rounds = 0
    while result.get("status") == "needs_input" and rounds < 3:
        rounds += 1
        question = result.get("human_question", "Could you clarify your goal?")
        print(f"\n🤔 {question}")
        answer = input("Your answer: ").strip()
        if not answer:
            print("No answer given — stopping.")
            return
        user_goal = f"{user_goal}\n\nAdditional detail from user: {answer}"
        result = await run_workflow(user_goal)

    print("\n==========================================")
    print("📊 FINAL RESULTS")
    print("==========================================")
    print(f"Status   : {result.get('status')}")
    print(f"Pages visited : {len(result.get('urls_visited', []))}")
    print(f"Jobs found    : {len(result.get('extracted_jobs', []))}")
    print("\n--- Extracted Jobs ---")

    for i, job in enumerate(result.get("extracted_jobs", []), 1):
        print(f"\n#{i}")
        print(f"  Role     : {job.get('role')}")
        print(f"  Company  : {job.get('company')}")
        print(f"  Location : {job.get('location')}")
        print(f"  Salary   : {job.get('salary')}")
        print(f"  Apply    : {job.get('apply_url')}")
        print(f"  Source   : {job.get('source_url')}")


if __name__ == "__main__":
    asyncio.run(main())
