
# %%writefile mcp_project/research_server.py

import arxiv
import json
import os
from typing import List
from mcp.server.fastmcp import FastMCP


PAPER_DIR = "papers"


# Initialize FastMCP server
mcp = FastMCP("research")

@mcp.tool()
def search_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    arXiv에서 주제에 따른 논문을 검색하고 해당 정보를 저장합니다.

    Args:
        topic: The topic to search for
        max_results: 조회할 논문 최대 정수 값 (default: 5)

    Returns:
        List of paper IDs found in the search
    """
    max_results = int(max_results)
    # Use arxiv to find the papers
    client = arxiv.Client()
    print("max_results == ", max_results)

    # Search for the most relevant articles matching the queried topic
    search = arxiv.Search(
        query = topic,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.Relevance
    )

    results = client.results(search)

    # Create directory for this topic
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)

    file_path = os.path.join(path, "papers_info.json")

    # Try to load existing papers info
    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    # Process each paper and add to papers_info
    paper_ids = []
    for paper in results:
        paper_id = paper.get_short_id()
        paper_ids.append(paper_id)
        paper_info = {
            # 'title': paper.title,    
            'heading': paper.title,
            'authors': [author.name for author in paper.authors],
            'summary': paper.summary,
            'pdf_url': paper.pdf_url,
            'published': str(paper.published.date())
        }
        papers_info[paper_id] = paper_info

    # Save updated papers_info to json file
    with open(file_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=2)

    print(f"Results are saved in: {file_path}")

    return paper_ids



@mcp.tool()
def extract_info(paper_id: str) -> str:
    """
    모든 렉토리에서 특정 논문에 대한 정보를 검색합니다.

    Args:
        paper_id: The ID of the paper to look for

    Returns:
        JSON string with paper information if found, error message if not found
    """

    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue

    return f"There's no saved information related to paper {paper_id}."



@mcp.resource("papers://folders")
def get_available_folders() -> str:
    """
논문 디렉토리에서 사용 가능한 모든 주제 폴더를 보여줍니다.
이 리소스는 사용 가능한 모든 주제 폴더의 간단한 목록을 제공합니다.
    """
    folders = []

    # Get all topic directories
    if os.path.exists(PAPER_DIR):
        for topic_dir in os.listdir(PAPER_DIR):
            topic_path = os.path.join(PAPER_DIR, topic_dir)
            if os.path.isdir(topic_path):
                papers_file = os.path.join(topic_path, "papers_info.json")
                if os.path.exists(papers_file):
                    folders.append(topic_dir)

    # Create a simple markdown list
    content = "# Available Topics\n\n"
    if folders:
        for folder in folders:
            content += f"- {folder}\n"
        content += f"\nUse @{folder} to access papers in that topic.\n"
    else:
        content += "No topics found.\n"

    return content

@mcp.resource("papers://{topic}")
def get_topic_papers(topic: str) -> str:
    """
특정 주제에 대한 논문에 대한 자세한 정보를 보여줍니다.

    Args:
        topic: 검색할 연구 주제
    """
    topic_dir = topic.lower().replace(" ", "_")
    papers_file = os.path.join(PAPER_DIR, topic_dir, "papers_info.json")

    if not os.path.exists(papers_file):
        return f"# No papers found for topic: {topic}\n\nTry searching for papers on this topic first."

    try:
        with open(papers_file, 'r') as f:
            papers_data = json.load(f)

        # Create markdown content with paper details
        content = f"# Papers on {topic.replace('_', ' ').title()}\n\n"
        content += f"Total papers: {len(papers_data)}\n\n"

        for paper_id, paper_info in papers_data.items():
            content += f"## {paper_info['title']}\n"
            content += f"- **Paper ID**: {paper_id}\n"
            content += f"- **Authors**: {', '.join(paper_info['authors'])}\n"
            content += f"- **Published**: {paper_info['published']}\n"
            content += f"- **PDF URL**: [{paper_info['pdf_url']}]({paper_info['pdf_url']})\n\n"
            content += f"### Summary\n{paper_info['summary'][:500]}...\n\n"
            content += "---\n\n"

        return content
    except json.JSONDecodeError:
        return f"# Error reading papers data for {topic}\n\nThe papers data file is corrupted."

@mcp.prompt()
def generate_search_prompt(topic: str, num_papers: int = 5) -> str:
    """LLM이 특정 주제에 대한 학술 논문을 찾아 논의할 수 있도록 프롬프트를 생성합니다.."""
    return f"""'{topic}'에 대한 {num_papers}개의 학술 논문을 검색하려면 검색 도구를 사용하세요.

다음 지침을 따르세요.
1. 먼저, search_papers(topic='{topic}', max_results={num_papers})를 사용하여 논문을 검색합니다.
2. 검색된 각 논문에 대해 다음 정보를 추출하여 정리합니다.
- 논문 제목
- 저자
- 출판일
- 주요 연구 결과의 간략한 요약
- 주요 기여 또는 혁신
- 사용된 방법론
- '{topic}' 와의 관련성

3. 다음을 포함하는 포괄적인 요약을 제공합니다.
- '{topic}' 연구의 현재 상태 개요
- 논문 전반의 공통 주제 및 동향
- 주요 연구 격차 또는 향후 연구 분야
- 이 분야에서 가장 영향력 있는 논문

4. 읽기 쉬운 제목과 요점을 사용하여 연구 결과를 명확하고 체계적인 형식으로 정리합니다.

{topic}의 각 논문에 대한 자세한 정보와 연구 환경에 대한 개략적인 요약을 모두 제시해 주세요.
"""


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')