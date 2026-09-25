"""
Memory Layer - ChromaDB based vector storage
Stores historical decisions and retrieves relevant context
at each simulation decision point.
"""

import chromadb
import json
import hashlib
from pathlib import Path
from loguru import logger
from datetime import datetime

CHROMA_PATH = Path("data/processed/chroma_db")


class MemoryLayer:
    def __init__(self, collection_name="campaign_decisions"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialise()

    def _initialise(self):
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Campaign management decisions and outcomes"}
        )
        logger.success(f"Memory Layer initialised — {self.collection.count()} existing records")

    def store(self, day, campaign, decision_type, decision, context, outcome=None):
        record_id = hashlib.md5(
            f"{day}_{campaign}_{decision_type}_{datetime.now().isoformat()}".encode()
        ).hexdigest()

        document = f"""
Day: {day}
Campaign: {campaign}
Decision Type: {decision_type}
Decision: {json.dumps(decision)}
Context: {json.dumps(context)}
Outcome: {json.dumps(outcome) if outcome else 'pending'}
""".strip()

        metadata = {
            "day": day,
            "campaign": str(campaign),
            "decision_type": decision_type,
            "has_outcome": outcome is not None,
            "timestamp": datetime.now().isoformat()
        }

        self.collection.add(
            documents=[document],
            metadatas=[metadata],
            ids=[record_id]
        )

        logger.info(f"Stored memory: Day {day}, Campaign {campaign}, Type {decision_type}")
        return record_id

    def retrieve(self, query, n_results=5, campaign=None):
        where = None
        if campaign:
            where = {"campaign": str(campaign)}

        count = self.collection.count()
        if count == 0:
            logger.info("Memory is empty — no historical context available")
            return []

        n_results = min(n_results, count)

        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )

        retrieved = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                retrieved.append({
                    "document": doc,
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                })

        logger.info(f"Retrieved {len(retrieved)} relevant memories")
        return retrieved

    def update_outcome(self, record_id, outcome):
        existing = self.collection.get(ids=[record_id])
        if not existing["documents"]:
            logger.warning(f"Record {record_id} not found")
            return

        old_doc = existing["documents"][0]
        new_doc = old_doc.replace(
            "Outcome: pending",
            f"Outcome: {json.dumps(outcome)}"
        )

        self.collection.update(
            ids=[record_id],
            documents=[new_doc],
            metadatas=[{**existing["metadatas"][0], "has_outcome": True}]
        )
        logger.info(f"Updated outcome for record {record_id}")

    def get_campaign_history(self, campaign, last_n_days=7):
        results = self.collection.get(
            where={"campaign": str(campaign)}
        )
        if not results["documents"]:
            return []

        history = []
        for i, doc in enumerate(results["documents"]):
            meta = results["metadatas"][i]
            history.append({"document": doc, "metadata": meta})

        history.sort(
            key=lambda x: x["metadata"].get("day", 0),
            reverse=True
        )
        return history[:last_n_days]

    def count(self):
        return self.collection.count()

    def clear(self):
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )
        logger.warning("Memory cleared")


if __name__ == "__main__":
    logger.info("Testing Memory Layer...")

    memory = MemoryLayer()

    id1 = memory.store(
        day=5,
        campaign=12345,
        decision_type="BUDGET_INCREASE",
        decision={
            "action": "increase_budget",
            "amount_increase": 0.002,
            "reason": "ROAS above 2.0 threshold for 3 consecutive days"
        },
        context={
            "roas": 2.5,
            "cpa": 0.003,
            "spend": 0.01,
            "conversions": 3,
            "ctr": 0.04
        },
        outcome={
            "roas_after": 2.8,
            "conversions_after": 4,
            "spend_after": 0.012
        }
    )
    print(f"Stored decision 1: {id1[:8]}...")

    id2 = memory.store(
        day=6,
        campaign=12345,
        decision_type="PAUSE_RECOMMENDATION",
        decision={
            "action": "pause_ad_set",
            "reason": "Zero conversions for 3 consecutive days with spend"
        },
        context={
            "roas": 0.0,
            "cpa": 0.0,
            "spend": 0.005,
            "conversions": 0,
            "ctr": 0.01
        }
    )
    print(f"Stored decision 2: {id2[:8]}...")

    id3 = memory.store(
        day=7,
        campaign=99999,
        decision_type="BUDGET_DECREASE",
        decision={
            "action": "decrease_budget",
            "reason": "CPA too high — 3x above target"
        },
        context={
            "roas": 0.5,
            "cpa": 0.02,
            "spend": 0.02,
            "conversions": 1,
            "ctr": 0.02
        }
    )
    print(f"Stored decision 3: {id3[:8]}...")

    print(f"\nTotal memories stored: {memory.count()}")

    print("\n--- Testing Retrieval ---")
    results = memory.retrieve(
        query="campaign with poor ROAS and high CPA needs budget reduction",
        n_results=2
    )
    print(f"Retrieved {len(results)} relevant memories:")
    for r in results:
        print(f"  Day {r['metadata']['day']}: "
              f"{r['metadata']['decision_type']} "
              f"(distance: {r['distance']:.3f})")

    print("\n--- Testing Campaign History ---")
    history = memory.get_campaign_history(campaign=12345)
    print(f"Campaign 12345 history: {len(history)} decisions")
    for h in history:
        print(f"  Day {h['metadata']['day']}: "
              f"{h['metadata']['decision_type']}")

    print("\n--- Testing Outcome Update ---")
    memory.update_outcome(
        id2,
        {"result": "spend_saved", "amount": 0.005, "status": "confirmed"}
    )
    print("Outcome updated successfully")

    print("\n✅ Memory Layer working correctly!")
    print(f"Final memory count: {memory.count()}")
