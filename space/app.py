"""Sheikh-Freemium Visual CoT Demo API."""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json

app = FastAPI(
    title="Sheikh-Freemium Visual CoT Demo",
    description="Interactive demo for Zebra-CoT Visual Chain of Thought dataset",
    version="1.0.0"
)

# Sample data for demo
SAMPLE_DATA = {
    "scientific": {
        "id": "geo_triangle_001",
        "category": "scientific",
        "subcategory": "geometry",
        "question": "In triangle ABC, angle A = 60°, angle B = 45°. If side AB = 10 units, find the length of side BC.",
        "reasoning_steps": [
            {"step": 1, "type": "text", "content": "Find angle C: C = 180° - 60° - 45° = 75°"},
            {"step": 2, "type": "interleaved", "content": "Draw triangle with labeled angles"},
            {"step": 3, "type": "text", "content": "Apply Law of Sines: BC = AB × sin(A)/sin(C)"},
            {"step": 4, "type": "text", "content": "Calculate: BC = 10 × 0.866/0.966 ≈ 8.97"}
        ],
        "answer": "BC ≈ 8.97 units"
    },
    "logic_games": {
        "id": "chess_mate_001",
        "category": "logic_games",
        "subcategory": "chess",
        "question": "White to move. Find checkmate in 2 moves.",
        "reasoning_steps": [
            {"step": 1, "type": "interleaved", "content": "Analyze position: Black king restricted by pawns"},
            {"step": 2, "type": "text", "content": "Consider Qd8: attacks rook, threatens back rank mate"},
            {"step": 3, "type": "interleaved", "content": "After 1.Qd8 Rxd8, play 2.Rxf8#"}
        ],
        "answer": "1.Qd8+ Rxd8 2.Rxf8#"
    }
}

STATISTICS = {
    "total_samples": 182384,
    "categories": [
        {"name": "Visual Logic & Games", "samples": 66854, "percentage": 36.7},
        {"name": "2D Visual Reasoning", "samples": 51899, "percentage": 28.5},
        {"name": "3D Visual Reasoning", "samples": 39610, "percentage": 21.7},
        {"name": "Scientific Reasoning", "samples": 24021, "percentage": 13.2}
    ],
    "performance": {
        "before": 4.2,
        "after": 16.9,
        "gain": 12.7
    }
}


@app.get("/", response_class=HTMLResponse)
async def home():
    """Render home page."""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sheikh-Freemium Visual CoT Demo</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 2rem;
                color: #333;
            }
            .container { max-width: 1200px; margin: 0 auto; }
            h1 { color: white; text-align: center; margin-bottom: 0.5rem; font-size: 2.5rem; }
            .subtitle { color: rgba(255,255,255,0.9); text-align: center; margin-bottom: 2rem; }
            .card {
                background: white;
                border-radius: 16px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }
            .card h2 { color: #667eea; margin-bottom: 1rem; }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
            }
            .stat-box {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 12px;
                text-align: center;
            }
            .stat-box .number { font-size: 2rem; font-weight: bold; }
            .stat-box .label { opacity: 0.9; font-size: 0.9rem; }
            .category-bar {
                display: flex;
                align-items: center;
                margin: 0.5rem 0;
            }
            .category-bar .name { width: 200px; }
            .category-bar .bar {
                flex: 1;
                height: 24px;
                background: #e0e0e0;
                border-radius: 12px;
                overflow: hidden;
                margin: 0 1rem;
            }
            .category-bar .fill {
                height: 100%;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                border-radius: 12px;
            }
            .sample {
                background: #f8f9fa;
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
            }
            .sample h3 { color: #764ba2; margin-bottom: 0.5rem; }
            .step {
                display: flex;
                align-items: flex-start;
                margin: 0.5rem 0;
                padding: 0.5rem;
                background: white;
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }
            .step-num {
                background: #667eea;
                color: white;
                width: 24px;
                height: 24px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 0.8rem;
                margin-right: 0.75rem;
                flex-shrink: 0;
            }
            .step.interleaved { border-left-color: #764ba2; }
            .step.interleaved .step-num { background: #764ba2; }
            .answer {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1rem;
                border-radius: 8px;
                margin-top: 1rem;
            }
            .links { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1rem; }
            .links a {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 0.75rem 1.5rem;
                border-radius: 8px;
                text-decoration: none;
                font-weight: 500;
            }
            .links a:hover { opacity: 0.9; }
            .badge {
                display: inline-block;
                padding: 0.25rem 0.75rem;
                border-radius: 9999px;
                font-size: 0.75rem;
                font-weight: 500;
                margin-right: 0.5rem;
            }
            .badge.text { background: #e3f2fd; color: #1976d2; }
            .badge.interleaved { background: #f3e5f5; color: #7b1fa2; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🦓 Sheikh-Freemium</h1>
            <p class="subtitle">Visual Chain of Thought Dataset Demo</p>
            
            <div class="card">
                <h2>Dataset Statistics</h2>
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="number">182,384</div>
                        <div class="label">Total Samples</div>
                    </div>
                    <div class="stat-box">
                        <div class="number">4</div>
                        <div class="label">Categories</div>
                    </div>
                    <div class="stat-box">
                        <div class="number">+12.7%</div>
                        <div class="label">Accuracy Gain</div>
                    </div>
                    <div class="stat-box">
                        <div class="number">58.9 GB</div>
                        <div class="label">Dataset Size</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Category Distribution</h2>
                <div class="category-bar">
                    <span class="name">Visual Logic & Games</span>
                    <div class="bar"><div class="fill" style="width: 36.7%"></div></div>
                    <span>36.7%</span>
                </div>
                <div class="category-bar">
                    <span class="name">2D Visual Reasoning</span>
                    <div class="bar"><div class="fill" style="width: 28.5%"></div></div>
                    <span>28.5%</span>
                </div>
                <div class="category-bar">
                    <span class="name">3D Visual Reasoning</span>
                    <div class="bar"><div class="fill" style="width: 21.7%"></div></div>
                    <span>21.7%</span>
                </div>
                <div class="category-bar">
                    <span class="name">Scientific Reasoning</span>
                    <div class="bar"><div class="fill" style="width: 13.2%"></div></div>
                    <span>13.2%</span>
                </div>
            </div>
            
            <div class="card">
                <h2>Sample: Geometry Problem</h2>
                <div class="sample">
                    <h3>📐 Triangle Side Length</h3>
                    <p><strong>Question:</strong> In triangle ABC, angle A = 60°, angle B = 45°. If side AB = 10 units, find BC.</p>
                    
                    <div style="margin-top: 1rem;">
                        <div class="step">
                            <span class="step-num">1</span>
                            <div><span class="badge text">text</span> Find angle C: C = 180° - 60° - 45° = 75°</div>
                        </div>
                        <div class="step interleaved">
                            <span class="step-num">2</span>
                            <div><span class="badge interleaved">visual</span> Draw triangle with labeled angles</div>
                        </div>
                        <div class="step">
                            <span class="step-num">3</span>
                            <div><span class="badge text">text</span> Apply Law of Sines: BC = AB × sin(A)/sin(C)</div>
                        </div>
                        <div class="step">
                            <span class="step-num">4</span>
                            <div><span class="badge text">text</span> Calculate: BC = 10 × 0.866/0.966 ≈ 8.97</div>
                        </div>
                    </div>
                    
                    <div class="answer">
                        <strong>Answer:</strong> BC ≈ 8.97 units
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Sample: Chess Puzzle</h2>
                <div class="sample">
                    <h3>♚ Mate in 2</h3>
                    <p><strong>Question:</strong> White to move. Find checkmate in 2 moves.</p>
                    
                    <div style="margin-top: 1rem;">
                        <div class="step interleaved">
                            <span class="step-num">1</span>
                            <div><span class="badge interleaved">visual</span> Analyze: Black king restricted by pawns</div>
                        </div>
                        <div class="step">
                            <span class="step-num">2</span>
                            <div><span class="badge text">text</span> Consider Qd8: attacks rook, threatens back rank mate</div>
                        </div>
                        <div class="step interleaved">
                            <span class="step-num">3</span>
                            <div><span class="badge interleaved">visual</span> After 1.Qd8 Rxd8, play 2.Rxf8#</div>
                        </div>
                    </div>
                    
                    <div class="answer">
                        <strong>Answer:</strong> 1.Qd8+ Rxd8 2.Rxf8# (Back rank mate)
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Resources</h2>
                <div class="links">
                    <a href="https://huggingface.co/datasets/shk-bd/Sheikh-Freemium" target="_blank">🤗 Dataset</a>
                    <a href="https://github.com/isshiki-dev/Sheikh-Freemium" target="_blank">💻 GitHub</a>
                    <a href="https://arxiv.org/abs/2507.16746" target="_blank">📄 Paper</a>
                    <a href="https://huggingface.co/datasets/multimodal-reasoning-lab/Zebra-CoT" target="_blank">🦓 Zebra-CoT</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html


@app.get("/api/stats")
async def get_stats():
    """Get dataset statistics."""
    return STATISTICS


@app.get("/api/samples")
async def get_samples():
    """Get sample data."""
    return SAMPLE_DATA


@app.get("/api/sample/{category}")
async def get_sample(category: str):
    """Get sample by category."""
    if category in SAMPLE_DATA:
        return SAMPLE_DATA[category]
    return {"error": "Category not found"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "sheikh-freemium-demo"}
