#!/usr/bin/env python3
"""
Test script to demonstrate Sanjaya RAG evaluation metrics.
Run this to see hypothetical scores and test the evaluation system.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.evaluation_metrics import EvaluationMetrics, SanjayaHypotheticalScores

def main():
    print("\n" + "="*80)
    print(" 🎯 SANJAYA VIDEO ANALYTICS - RAG EVALUATION SYSTEM DEMO")
    print("="*80 + "\n")
    
    # Show hypothetical scores for the system
    print("📊 SHOWING HYPOTHETICAL PERFORMANCE SCORES")
    print("   (Based on architecture analysis: FAISS + BGE-M3 + Neo4j + Qwen3-VL)\n")
    
    SanjayaHypotheticalScores.print_score_card()
    
    # Demo individual metric calculations
    print("\n" + "="*80)
    print(" 🧪 TESTING INDIVIDUAL METRICS")
    print("="*80 + "\n")
    
    evaluator = EvaluationMetrics()
    
    # Test Case 1: Surveillance query
    print("TEST CASE 1: Loitering Detection Query")
    print("-" * 80)
    
    generated = "Person P1 was detected loitering in Zone1 for 45 seconds carrying a red backpack. The person wore a blue shirt and jeans. Risk assessment: medium priority due to prolonged stationary behavior in restricted area."
    reference = "Person P1 loitered in Zone1 for 45 seconds with a red backpack. Wearing blue shirt and jeans. Medium risk."
    
    rouge = evaluator.rouge_score(generated, reference)
    bert = evaluator.bert_score(generated, reference)
    completeness = evaluator.answer_completeness(generated, ["P1", "Zone1", "loitering", "backpack", "medium"])
    coherence = evaluator.answer_coherence(generated)
    
    print(f"\n📝 Generated Answer:")
    print(f"   {generated[:100]}...")
    print(f"\n📊 Scores:")
    print(f"   ROUGE-1 F1:     {rouge['rouge1_f']:.4f}")
    print(f"   ROUGE-2 F1:     {rouge['rouge2_f']:.4f}")
    print(f"   ROUGE-L F1:     {rouge['rougeL_f']:.4f}")
    print(f"   BERTScore F1:   {bert['bert_f1']:.4f}")
    print(f"   Completeness:   {completeness:.4f}")
    print(f"   Coherence:      {coherence:.4f}")
    
    # Test Case 2: Person-Object interaction query
    print("\n\nTEST CASE 2: Person-Object Interaction Query")
    print("-" * 80)
    
    generated2 = "Three persons were detected. P1 carries a laptop bag and is using a silver laptop on the table. P2 stands near P1 holding a phone. P3 is walking in Zone2."
    reference2 = "Three people detected: P1 with laptop and bag, P2 with phone near P1, P3 walking in Zone2."
    
    rouge2 = evaluator.rouge_score(generated2, reference2)
    bert2 = evaluator.bert_score(generated2, reference2)
    completeness2 = evaluator.answer_completeness(generated2, ["P1", "P2", "P3", "laptop", "phone", "bag", "Zone2"])
    
    print(f"\n📝 Generated Answer:")
    print(f"   {generated2}")
    print(f"\n📊 Scores:")
    print(f"   ROUGE-1 F1:     {rouge2['rouge1_f']:.4f}")
    print(f"   ROUGE-2 F1:     {rouge2['rouge2_f']:.4f}")
    print(f"   ROUGE-L F1:     {rouge2['rougeL_f']:.4f}")
    print(f"   BERTScore F1:   {bert2['bert_f1']:.4f}")
    print(f"   Completeness:   {completeness2:.4f}")
    
    # Test Case 3: Retrieval metrics
    print("\n\nTEST CASE 3: Retrieval Quality Assessment")
    print("-" * 80)
    
    retrieved = ["frame_43_cv", "frame_111_vlm", "frame_135_cv", "event_222", "event_109", "event_55"]
    relevant = ["frame_43_cv", "frame_111_vlm", "event_109"]
    
    p_at_5 = evaluator.precision_at_k(retrieved, relevant, k=5)
    r_at_5 = evaluator.recall_at_k(retrieved, relevant, k=5)
    mrr = evaluator.mean_reciprocal_rank(retrieved, relevant)
    map_score = evaluator.mean_average_precision(retrieved, relevant)
    
    print(f"\n🔍 Retrieved: {len(retrieved)} documents")
    print(f"   Relevant: {len(relevant)} documents")
    print(f"\n📊 Retrieval Scores:")
    print(f"   Precision@5:    {p_at_5:.4f}")
    print(f"   Recall@5:       {r_at_5:.4f}")
    print(f"   MRR:            {mrr:.4f}")
    print(f"   MAP:            {map_score:.4f}")
    
    # Test Case 4: Performance metrics
    print("\n\nTEST CASE 4: Performance Evaluation")
    print("-" * 80)
    
    latencies = [1500, 2800, 3500, 2200, 4100]
    for i, lat in enumerate(latencies, 1):
        lat_eval = evaluator.latency_score(lat)
        print(f"\n   Query {i}: {lat}ms → Score: {lat_eval['latency_score']:.2f} ({lat_eval['category']})")
    
    # Comprehensive evaluation
    print("\n\n" + "="*80)
    print(" 🎯 COMPREHENSIVE RAG EVALUATION")
    print("="*80 + "\n")
    
    comprehensive = evaluator.evaluate_rag_response(
        question="What suspicious activities were detected in the last 5 minutes?",
        generated_answer="Two suspicious activities: Person P1 loitered in Zone1 for 45s with unattended red backpack (medium risk). Person P2 accessed restricted Zone3 without authorization (high risk).",
        reference_answer="Two suspicious events: P1 loitering with backpack in Zone1 (medium risk), P2 unauthorized access to Zone3 (high risk).",
        retrieved_docs=["frame_43", "frame_111", "event_109", "event_222", "zone_analysis_1"],
        relevant_docs=["frame_43", "event_109", "event_222"],
        evidence_snippets=[
            "Person P1 stationary in Zone1 for 45 seconds",
            "Red backpack detected near P1, unattended",
            "Person P2 entered Zone3 without proper clearance"
        ],
        expected_entities=["P1", "P2", "Zone1", "Zone3", "loitering", "backpack", "unauthorized"],
        latency_ms=2800
    )
    
    print("📝 Question:", comprehensive['question'])
    print(f"\n🎯 COMPOSITE SCORE: {comprehensive['scores']['composite_score']:.4f}")
    print("\n📊 Detailed Scores:")
    print(f"   ROUGE-1 F1:         {comprehensive['scores']['rouge_scores']['rouge1_f']:.4f}")
    print(f"   ROUGE-L F1:         {comprehensive['scores']['rouge_scores']['rougeL_f']:.4f}")
    print(f"   BERTScore F1:       {comprehensive['scores']['bert_scores']['bert_f1']:.4f}")
    print(f"   Precision@5:        {comprehensive['scores']['retrieval']['precision_at_5']:.4f}")
    print(f"   Recall@5:           {comprehensive['scores']['retrieval']['recall_at_5']:.4f}")
    print(f"   MRR:                {comprehensive['scores']['retrieval']['mean_reciprocal_rank']:.4f}")
    print(f"   Completeness:       {comprehensive['scores']['answer_quality']['completeness']:.4f}")
    print(f"   Coherence:          {comprehensive['scores']['answer_quality']['coherence']:.4f}")
    print(f"   Factuality:         {comprehensive['scores']['answer_quality']['factuality']:.4f}")
    print(f"   Latency Score:      {comprehensive['scores']['performance']['latency_score']:.2f}")
    
    print(f"\n📈 Metadata:")
    print(f"   Generated Length:   {comprehensive['metadata']['generated_answer_length']} chars")
    print(f"   Retrieved Docs:     {comprehensive['metadata']['num_retrieved_docs']}")
    print(f"   Relevant Docs:      {comprehensive['metadata']['num_relevant_docs']}")
    print(f"   Evaluation Time:    {comprehensive['metadata']['evaluation_time_ms']:.2f} ms")
    
    # Score interpretation
    print("\n\n" + "="*80)
    print(" 📖 SCORE INTERPRETATION GUIDE")
    print("="*80 + "\n")
    
    print("ROUGE Scores (0.0-1.0):")
    print("  0.85-1.00: Excellent - Very high text overlap")
    print("  0.70-0.84: Good - Strong text similarity")
    print("  0.50-0.69: Moderate - Acceptable overlap")
    print("  <0.50:     Poor - Low text similarity\n")
    
    print("BERTScore (0.0-1.0):")
    print("  0.85-1.00: Excellent - Semantically equivalent")
    print("  0.70-0.84: Good - Minor semantic differences")
    print("  0.50-0.69: Moderate - Some semantic divergence")
    print("  <0.50:     Poor - Semantically different\n")
    
    print("Retrieval Metrics (0.0-1.0):")
    print("  Precision@5: Fraction of top-5 results that are relevant")
    print("  Recall@5:    Fraction of relevant docs in top-5")
    print("  MRR:         1.0 = first result relevant, 0.5 = second, 0.33 = third")
    print("  MAP:         Overall retrieval quality across all ranks\n")
    
    print("Answer Quality (0.0-1.0):")
    print("  Completeness: Coverage of expected entities/facts")
    print("  Coherence:    Linguistic quality and structure")
    print("  Factuality:   Grounding in provided evidence\n")
    
    print("Performance Scores:")
    print("  <1000ms:      Excellent (1.0)")
    print("  1000-2000ms:  Good (0.8)")
    print("  2000-4000ms:  Acceptable (0.6)")
    print("  >4000ms:      Slow (0.4)\n")
    
    print("\n" + "="*80)
    print(" ✅ EVALUATION COMPLETE")
    print("="*80 + "\n")
    
    print("💡 To use in your pipeline:")
    print("   from rag.evaluation_metrics import EvaluationMetrics")
    print("   evaluator = EvaluationMetrics()")
    print("   scores = evaluator.evaluate_rag_response(...)")
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
