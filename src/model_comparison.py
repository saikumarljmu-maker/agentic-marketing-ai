"""
Multi-Model Comparison — 6 Models
Runs the same campaign scenarios through all models
and compares outputs for thesis model comparison section.
"""

import json
import time
import subprocess
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
import os
import anthropic

load_dotenv()

RESULTS_PATH = Path("results")
RESULTS_PATH.mkdir(exist_ok=True)

TEST_SCENARIOS = [
    {
        "id": "SC-01",
        "name": "High ROAS Winner",
        "description": "Campaign performing above threshold — scaling decision",
        "data": {"campaign_id": 32135670, "day": 10, "spend": 0.012,
                 "conversions": 4, "clicks": 89, "impressions": 2100,
                 "roas": 2.8, "cpa": 0.003, "ctr": 0.042, "days_active": 10}
    },
    {
        "id": "SC-02",
        "name": "Zero Conversion Spender",
        "description": "Campaign spending with zero conversions — pause decision",
        "data": {"campaign_id": 442617, "day": 7, "spend": 0.008,
                 "conversions": 0, "clicks": 0, "impressions": 450,
                 "roas": 0.0, "cpa": 0.0, "ctr": 0.0, "days_active": 7}
    },
    {
        "id": "SC-03",
        "name": "Declining Performance",
        "description": "Campaign with declining ROAS over 5 days",
        "data": {"campaign_id": 73327, "day": 15, "spend": 0.015,
                 "conversions": 1, "clicks": 142, "impressions": 3200,
                 "roas": 0.4, "cpa": 0.015, "ctr": 0.044, "days_active": 15,
                 "roas_trend": "declining from 1.8 to 0.4 over 5 days"}
    },
    {
        "id": "SC-04",
        "name": "Ambiguous Signal",
        "description": "Campaign with mixed signals requiring contextual judgment",
        "data": {"campaign_id": 9100693, "day": 20, "spend": 0.006,
                 "conversions": 18, "clicks": 201, "impressions": 4100,
                 "roas": 1.2, "cpa": 0.0003, "ctr": 0.049, "days_active": 20,
                 "note": "Low ROAS but highest conversion volume in portfolio"}
    },
    {
        "id": "SC-05",
        "name": "Dormant High-ROAS",
        "description": "Campaign with historical high ROAS but zero current activity",
        "data": {"campaign_id": 14104561, "day": 25, "spend": 0.0,
                 "conversions": 0, "clicks": 0, "impressions": 0,
                 "roas": 0.0, "cpa": 0.0, "ctr": 0.0, "days_active": 0,
                 "historical_roas": 27300,
                 "note": "Dormant campaign with extremely high historical ROAS"}
    }
]

PROMPT_TEMPLATE = """You are an expert digital marketing campaign analyst.

Analyse this campaign performance data and provide a specific recommendation.

CAMPAIGN DATA:
{data}

Respond in this EXACT format with no extra text:
ACTION: [one of: increase_budget / decrease_budget / pause / maintain / expand_audience / narrow_audience]
CONFIDENCE: [high / medium / low]
REASON: [one sentence explaining why]
RISK: [one sentence about the main risk]
HUMAN_REVIEW_NEEDED: [yes / no]"""

MODELS = [
    {"id": "claude", "name": "Claude Sonnet 4.6", "type": "api"},
    {"id": "qwen3:8b", "name": "Qwen3 8b", "type": "ollama"},
    {"id": "gemma3:4b", "name": "Gemma3 4b", "type": "ollama"},
    {"id": "llama3.2:3b", "name": "Llama3.2 3b", "type": "ollama"},
    {"id": "mistral:7b", "name": "Mistral 7b", "type": "ollama"},
    {"id": "deepseek-r1:8b", "name": "DeepSeek-R1 8b", "type": "ollama"},
    {"id": "phi4:14b", "name": "Phi4 14b", "type": "ollama"},
]


def run_ollama(model_id, prompt, timeout=180):
    try:
        start = time.time()
        result = subprocess.run(
            ["ollama", "run", model_id, prompt],
            capture_output=True, text=True, timeout=timeout
        )
        duration = time.time() - start
        return {
            "output": result.stdout.strip(),
            "duration_seconds": round(duration, 2),
            "success": result.returncode == 0,
            "error": result.stderr.strip() if result.returncode != 0 else None
        }
    except subprocess.TimeoutExpired:
        return {"output": None, "duration_seconds": timeout,
                "success": False, "error": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"output": None, "duration_seconds": 0,
                "success": False, "error": str(e)}


def run_claude(prompt):
    try:
        client = anthropic.Anthropic(
            default_headers={
                "anthropic-workspace-id": os.getenv("ANTHROPIC_WORKSPACE_ID")
            }
        )
        start = time.time()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        duration = time.time() - start
        return {
            "output": response.content[0].text.strip(),
            "duration_seconds": round(duration, 2),
            "success": True,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "error": None
        }
    except Exception as e:
        return {"output": None, "duration_seconds": 0,
                "success": False, "error": str(e)}


def parse_output(output):
    if not output:
        return {}
    result = {}
    # Remove thinking sections from DeepSeek
    if "...done thinking." in output:
        output = output.split("...done thinking.")[-1].strip()
    for line in output.strip().split('\n'):
        line = line.strip()
        if line.startswith('ACTION:'):
            val = line.replace('ACTION:', '').strip().lower()
            # Normalise action names
            if 'increase' in val or 'scale' in val:
                val = 'increase_budget'
            elif 'decrease' in val or 'reduce' in val:
                val = 'decrease_budget'
            elif 'pause' in val or 'stop' in val:
                val = 'pause'
            elif 'maintain' in val or 'keep' in val:
                val = 'maintain'
            elif 'expand' in val or 'broaden' in val:
                val = 'expand_audience'
            elif 'narrow' in val or 'tighten' in val:
                val = 'narrow_audience'
            result['action'] = val
        elif line.startswith('CONFIDENCE:'):
            result['confidence'] = line.replace('CONFIDENCE:', '').strip().lower()
        elif line.startswith('REASON:'):
            result['reason'] = line.replace('REASON:', '').strip()
        elif line.startswith('RISK:'):
            result['risk'] = line.replace('RISK:', '').strip()
        elif line.startswith('HUMAN_REVIEW_NEEDED:'):
            result['human_review'] = line.replace(
                'HUMAN_REVIEW_NEEDED:', ''
            ).strip().lower()
    return result


def run_full_comparison():
    logger.info("Starting Full 6-Model Comparison")
    logger.info(f"Models: {len(MODELS)} | Scenarios: {len(TEST_SCENARIOS)}")

    all_results = []

    for scenario in TEST_SCENARIOS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Scenario: {scenario['id']} — {scenario['name']}")

        prompt = PROMPT_TEMPLATE.format(
            data=json.dumps(scenario['data'], indent=2)
        )

        scenario_result = {
            "scenario_id": scenario['id'],
            "scenario_name": scenario['name'],
            "description": scenario['description'],
            "input_data": scenario['data'],
            "models": {}
        }

        for model in MODELS:
            logger.info(f"  Running {model['name']}...")

            if model['type'] == 'api':
                result = run_claude(prompt)
            else:
                result = run_ollama(model['id'], prompt)

            result['parsed'] = parse_output(result.get('output', ''))
            scenario_result['models'][model['id']] = {
                "model_name": model['name'],
                **result
            }

            action = result['parsed'].get('action', 'N/A')
            conf = result['parsed'].get('confidence', 'N/A')
            dur = result['duration_seconds']
            status = "✅" if result['success'] else "❌"
            logger.info(f"  {status} {model['name']}: {action} ({conf}) — {dur}s")

        # Check agreement
        actions = [
            v['parsed'].get('action', '')
            for v in scenario_result['models'].values()
            if v.get('success') and v['parsed'].get('action')
        ]
        unique_actions = set(actions)
        scenario_result['actions_summary'] = list(unique_actions)
        scenario_result['full_agreement'] = len(unique_actions) == 1
        scenario_result['successful_models'] = len(actions)

        all_results.append(scenario_result)

    # Save
    output_path = RESULTS_PATH / "full_model_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "="*100)
    print("FULL 6-MODEL COMPARISON RESULTS")
    print("="*100)

    headers = ["Scenario"] + [m['name'] for m in MODELS] + ["Agree"]
    col_w = 16
    print(f"{'Scenario':<20}" + "".join(f"{h:<{col_w}}" for h in headers[1:]))
    print("-"*100)

    for r in all_results:
        row = f"{r['scenario_name']:<20}"
        for model in MODELS:
            model_data = r['models'].get(model['id'], {})
            action = model_data.get('parsed', {}).get('action', 'N/A')
            if not model_data.get('success'):
                action = 'TIMEOUT'
            row += f"{action:<{col_w}}"
        agree = "✅ YES" if r['full_agreement'] else "❌ NO"
        row += agree
        print(row)

    print("\n" + "="*100)
    print("RESPONSE TIMES (seconds)")
    print("="*100)
    print(f"{'Scenario':<20}" + "".join(
        f"{m['name']:<{col_w}}" for m in MODELS
    ))
    print("-"*100)

    for r in all_results:
        row = f"{r['scenario_name']:<20}"
        for model in MODELS:
            model_data = r['models'].get(model['id'], {})
            dur = model_data.get('duration_seconds', 'N/A')
            row += f"{str(dur)+'s':<{col_w}}"
        print(row)

    print("\n" + "="*100)
    print("CONFIDENCE LEVELS")
    print("="*100)
    print(f"{'Scenario':<20}" + "".join(
        f"{m['name']:<{col_w}}" for m in MODELS
    ))
    print("-"*100)

    for r in all_results:
        row = f"{r['scenario_name']:<20}"
        for model in MODELS:
            model_data = r['models'].get(model['id'], {})
            conf = model_data.get('parsed', {}).get('confidence', 'N/A')
            row += f"{conf:<{col_w}}"
        print(row)

    agree_count = sum(1 for r in all_results if r['full_agreement'])
    print(f"\nFull agreement: {agree_count}/{len(all_results)} scenarios")
    print(f"Results saved to: {output_path}")
    print("\n✅ Full comparison complete!")

    return all_results


if __name__ == "__main__":
    run_full_comparison()
